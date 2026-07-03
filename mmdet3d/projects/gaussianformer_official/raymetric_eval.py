# Copyright (c) OpenMMLab. All rights reserved.
"""RayMetric evaluation for official OpenOcc ``BEVSegmentor`` checkpoints (indoor).

``torchrun --nproc_per_node>1`` 时：本模块**不**初始化 ``torch.distributed``，仅
``LOCAL_RANK==0`` 的进程跑全量 val；其余进程立刻返回，避免 NCCL 与单进程推理混用
导致的卡住/超时。真正多卡并行推理需另接 DDP 数据并行（未在本路径实现）。
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.config import ConfigDict
from mmseg.models import build_segmentor

# 避免循环导入，从当前包相对导入
from . import ensure_registered
from mmdet3d.registry import METRICS

import mmdet3d.evaluation.metrics.ray_metric  # noqa: F401 — register RayMetric


def _strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        out[nk] = v
    return out


def _copy_checkpoint_into_model(model: torch.nn.Module, raw_sd: dict) -> None:
    """Load weights without ``load_state_dict`` (spconv ``SubMConv3d`` can false-trigger size mismatch).

    Applies :func:`_adapt_spconv_subm_weights` then copies each tensor in-place into ``model.state_dict()``.
    """
    sd = _strip_module_prefix(raw_sd)
    sd = _adapt_spconv_subm_weights(sd, model)
    target = model.state_dict()
    skipped: list[str] = []
    with torch.no_grad():
        for k, src in sd.items():
            if k not in target:
                continue
            dst = target[k]
            if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
                continue
            if src.shape != dst.shape:
                skipped.append(k)
                continue
            dst.copy_(src)
    if skipped:
        preview = ', '.join(skipped[:5])
        more = '...' if len(skipped) > 5 else ''
        print(
            f'Warning: {len(skipped)} checkpoint tensors skipped (shape mismatch after layout fix): '
            f'{preview}{more}',
        )


def _adapt_spconv_subm_weights(ckpt_sd: dict, model: torch.nn.Module) -> dict:
    """Match ``SubMConv3d`` layouts: some checkpoints store (O, I, K, K, K); spconv2 uses (O, K, K, K, I)."""
    model_sd = model.state_dict()
    fixed: dict[str, torch.Tensor] = {}
    for k, v in ckpt_sd.items():
        if not isinstance(v, torch.Tensor) or v.dim() != 5:
            fixed[k] = v
            continue
        if k not in model_sd:
            fixed[k] = v
            continue
        target = model_sd[k]
        if v.shape == target.shape:
            fixed[k] = v
            continue
        o, c_in, k0, k1, k2 = v.shape
        if target.shape == (o, k0, k1, k2, c_in):
            fixed[k] = v.permute(0, 2, 3, 4, 1).contiguous()
            continue
        if v.shape == (o, k0, k1, k2, c_in) and target.shape == (o, c_in, k0, k1, k2):
            fixed[k] = v.permute(0, 4, 1, 2, 3).contiguous()
            continue
        fixed[k] = v
    return fixed


def _occ_tensor_to_dense_hwd(t: torch.Tensor | np.ndarray, h: int, w: int, d: int) -> torch.Tensor:
    """``RayMetric`` expects dense ``(H, W, D)`` labels; ``final_occ`` may be ``(B, H*W*D)`` or flat ``(N,)``."""
    if isinstance(t, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(t))
    else:
        x = t
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.is_sparse:
        x = x.to_dense()
    x = x.detach()
    if not x.dtype.is_floating_point:
        x = x.long()
    while x.dim() > 3 and x.size(0) == 1:
        x = x.squeeze(0)
    flat = h * w * d
    if x.dim() == 1:
        if x.numel() != flat:
            raise RuntimeError(f'Expected {flat} flat voxels, got {x.numel()}')
        return x.reshape(h, w, d)
    if x.dim() == 2:
        if x.size(-1) != flat:
            raise RuntimeError(
                f'Expected last dim {flat} for flattened occupancy, got shape={tuple(x.shape)}',
            )
        x = x.view(x.size(0), h, w, d)
        if x.size(0) != 1:
            raise RuntimeError(f'RayMetric path expects batch 1, got shape={tuple(x.shape)}')
        return x.squeeze(0)
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        if x.size(0) == 1:
            return x[0]
        raise RuntimeError(f'Unexpected 4D occupancy without leading singleton: {tuple(x.shape)}')
    raise RuntimeError(f'Unexpected occupancy ndim={x.dim()} shape={tuple(x.shape)}')


def _to_plain_dict(obj):
    if isinstance(obj, (ConfigDict, Config)):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_plain_dict(x) for x in obj)
    return obj


def run_official_gaussianformer_raymetric_test(
    cfg: Config,
    checkpoint: str,
    work_dir: str,
    launcher: str,
) -> None:
    """仅 ``LOCAL_RANK==0`` 跑全量 RayMetric；多进程时其余 rank 直接返回（不初始化 NCCL）。"""
    _ = launcher  # 与 ``tools/test.py`` 调用签名一致；本路径不启分布式。
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1 and local_rank != 0:
        return

    _run_raymetric_worker(
        cfg,
        checkpoint,
        work_dir,
        grid_h=200,
        grid_w=200,
        grid_d=24,
    )


def _run_raymetric_worker(
    cfg: Config,
    checkpoint: str,
    work_dir: str,
    grid_h: int,
    grid_w: int,
    grid_d: int,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

    ensure_registered()
    import model  # noqa: F401 — side effect: registers mmseg custom modules

    from dataset import get_dataloader

    os.makedirs(work_dir, exist_ok=True)
    os.environ['eval'] = 'true'

    # 主进程已 ``set_device`` 后，DataLoader ``num_workers>0`` 在 Linux 上 fork 子进程易与 CUDA 死锁。
    vl = cfg.val_loader
    if isinstance(vl, ConfigDict):
        vl = vl.to_dict()
    elif not isinstance(vl, dict):
        vl = dict(vl)
    else:
        vl = dict(vl)
    vl = {**vl, 'num_workers': 0}

    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        vl,
        dist=False,
        val_only=True,
    )
    n_batches = len(val_loader)
    _ws = int(os.environ.get('WORLD_SIZE', 1))
    if _ws > 1:
        print(
            '[gaussianformer_official] torchrun：仅 LOCAL_RANK=0 跑评测，其余进程已退出。',
            flush=True,
        )
    print(f'[gaussianformer_official] 开始 RayMetric 评测，共 {n_batches} 个 batch', flush=True)

    model_cfg = cfg.model
    if isinstance(model_cfg, ConfigDict):
        model_cfg = model_cfg.to_dict()
    my_model = build_segmentor(model_cfg)
    ckpt = torch.load(checkpoint, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    if not isinstance(sd, dict):
        raise RuntimeError(f'Unexpected checkpoint format: {checkpoint}')
    _copy_checkpoint_into_model(my_model, sd)
    print(
        '[gaussianformer_official] Checkpoint weights applied via inplace copy '
        '(spconv SubMConv3d–safe). If you do not see this line, refresh/sync '
        '`mmdet3d/projects/gaussianformer_official/raymetric_eval.py`.',
        flush=True,
    )
    my_model.cuda()
    my_model.eval()

    metric_cfg = cfg.get('raymetric_eval')
    if isinstance(metric_cfg, ConfigDict):
        metric_cfg = metric_cfg.to_dict()
    if metric_cfg is None:
        metric_cfg = dict(
            type='RayMetric',
            num_classes=len(cfg['class_names']),
            class_names=list(cfg['class_names']),
            point_cloud_range=list(cfg['point_cloud_range']),
            occupancy_size=[0.1, 0.1, 0.1],
            use_image_mask=False,
        )
    ray_metric = METRICS.build(metric_cfg)

    # 进度条：batch 少时每步都打；多则约 50 次汇总，避免刷屏。
    log_every = 1 if n_batches <= 50 else max(1, n_batches // 50)
    t_loop0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)

            final_list = result_dict['final_occ']
            if not isinstance(final_list, (list, tuple)):
                final_list = [final_list]
            pred = _occ_tensor_to_dense_hwd(final_list[0], grid_h, grid_w, grid_d)

            gt = _occ_tensor_to_dense_hwd(data['occ_label'], grid_h, grid_w, grid_d)

            lo = data['lidar_origins']
            if lo.dim() == 3:
                lo = lo[0]

            sample = dict(
                pred_occupancy=pred.cpu(),
                gt_occupancy=gt.cpu(),
                lidar_origins=lo.cpu(),
            )
            ray_metric.process({}, [sample])

            done = batch_idx + 1
            if done == 1 or done == n_batches or done % log_every == 0:
                elapsed = time.perf_counter() - t_loop0
                pct = 100.0 * done / n_batches
                rate = done / max(elapsed, 1e-9)
                remain = n_batches - done
                eta_s = remain / max(rate, 1e-9)
                print(
                    f'[gaussianformer_official] 评测进度 {done}/{n_batches} ({pct:.1f}%) | '
                    f'已用 {elapsed:.0f}s | {rate:.2f} batch/s | ETA≈{eta_s:.0f}s',
                    flush=True,
                )

    # 避免 ``RayMetric.evaluate`` 内 ``broadcast_object_list``（依赖已初始化的进程组）。
    metrics = ray_metric.compute_metrics()
    if getattr(ray_metric, 'prefix', None):
        metrics = {
            '/'.join((ray_metric.prefix, k)): v
            for k, v in metrics.items()
        }
    if hasattr(ray_metric, 'results'):
        ray_metric.results.clear()

    out_path = osp.join(work_dir, 'raymetric_eval.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_to_plain_dict(metrics), f, indent=2, ensure_ascii=False)
    print(f'RayMetric results saved to {out_path}')
