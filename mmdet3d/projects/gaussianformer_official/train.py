"""Training wrapper for official GaussianFormer to be called from tools/train.py."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def run_official_gaussianformer_training(
    cfg,
    work_dir: str,
    launcher: str,
    resume_from: str | None = None,
) -> None:
    """Run official GaussianFormer training from standard MMDet tools/train.py interface.

    This function is called by tools/train.py when it detects a GaussianFormer config
    (model.type == 'BEVSegmentor' with train_dataset_config but no train_dataloader).

    Args:
        cfg: Loaded mmengine Config object.
        work_dir: Working directory for outputs.
        launcher: Job launcher ('pytorch', 'slurm', etc.).
        resume_from: Optional checkpoint path to resume from.
    """
    # 避免循环导入，从当前包相对导入
    from . import ensure_registered

    bench_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)

    if importlib.util.find_spec('mmseg') is None:
        raise RuntimeError(
            '缺少 mmsegmentation：官方 GaussianFormer 使用 ``from mmseg.models import build_segmentor``。'
            '请安装: pip install "mmsegmentation>=1.2.0,<1.3.0" '
            '（需与当前 mmengine / torch 版本兼容；装好后重试）。')

    gf_root = ensure_registered()
    os.environ['GAUSSIANFORMER_ROOT'] = gf_root
    os.environ['OCCBENCH_ROOT'] = bench_root

    # Get config path from cfg object
    cfg_path = cfg.filename
    if not cfg_path:
        raise ValueError('Config object must have filename attribute')
    cfg_path = os.path.abspath(cfg_path)

    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    train_path = os.path.join(gf_root, 'train.py')
    spec = importlib.util.spec_from_file_location('gf_train_internal', train_path)
    gf_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf_train)

    def wrapped_main(local_rank, run_args):
        ensure_registered()
        return gf_train.main(local_rank, run_args)

    import torch

    torchrun = 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if torchrun:
        world = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        distributed_backend = 'torchrun'
        gpus = world
    else:
        local_rank = 0
        distributed_backend = 'spawn_tcp'
        gpus = torch.cuda.device_count()

    run_args = argparse.Namespace(
        py_config=cfg_path,
        work_dir=work_dir,
        resume_from=resume_from or '',
        iter_resume=False,
        seed=42,
        gradient_accumulation=1,
        dataset='nuscenes',
        gpus=gpus,
        distributed_backend=distributed_backend,
    )

    prev = os.getcwd()
    os.chdir(gf_root)
    try:
        print(f'[GaussianFormer] Starting training with args: {run_args}')
        if torchrun:
            wrapped_main(local_rank, run_args)
        elif run_args.gpus > 1:
            torch.multiprocessing.spawn(wrapped_main, args=(run_args,), nprocs=run_args.gpus)
        else:
            wrapped_main(0, run_args)
    finally:
        os.chdir(prev)
