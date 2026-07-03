#!/usr/bin/env python3
"""Train vendored GaussianFormer (official OpenOcc ``train.py`` / BEVSegmentor).

This is the **recommended** indoor entry — same model code as upstream, integrated only via
``mmdet3d.projects.gaussianformer_official`` (XHumanoid OpenOcc dataset).

Examples::

    export PYTHONPATH=/path/to/Occupancy_Giga-benchmark:$PYTHONPATH

    # Multi-GPU（可省略 --work-dir，默认为 output/gaussianformer_indoor）
    torchrun --nproc_per_node=4 tools/train_gaussianformer_official.py \\
        --py-config configs/exp/gaussianformer_indoor.py

    # Legacy: internal torch.multiprocessing.spawn (no torchrun)
    python tools/train_gaussianformer_official.py \\
        --py-config configs/exp/gaussianformer_indoor.py

依赖（官方栈基于 MMSegmentation，需单独安装）::

    pip install 'mmsegmentation>=1.2.0,<1.3.0'

CUDA extensions::

    cd mmdet3d/models/GaussianFormer/model/head/localagg && pip install -e .

    # DeformableAggregation：若本仓库 mmdet3d 已编译 deformable_aggregation，可跳过；
    # 否则在官方目录安装：cd mmdet3d/models/GaussianFormer/model/encoder/gaussian_encoder/ops && pip install -e .

MMDet Runner 版（实验）见 ``configs/exp/gaussianformer_mmdet_indoor.py`` + ``tools/train.py``。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def main():
    bench_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)

    if importlib.util.find_spec('mmseg') is None:
        raise RuntimeError(
            '缺少 mmsegmentation：官方 GaussianFormer/train.py 使用 ``from mmseg.models import build_segmentor``。'
            '请安装: pip install "mmsegmentation>=1.2.0,<1.3.0" '
            '（需与当前 mmengine / torch 版本兼容；装好后重试 torchrun）。')

    parser = argparse.ArgumentParser(
        description='Launch official GaussianFormer train loop with Occ-benchmark OpenOcc hooks.',
    )
    parser.add_argument('--py-config', required=True)
    parser.add_argument(
        '--work-dir',
        default=None,
        help='日志与权重目录；默认 <repo>/output/gaussianformer_indoor',
    )
    parser.add_argument('--resume-from', default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument(
        '--gf-root',
        default=None,
        help='GaussianFormer repo root (default: env GAUSSIANFORMER_ROOT or vendored path).',
    )
    args, unknown = parser.parse_known_args()

    from mmdet3d.projects.gaussianformer_official import ensure_registered

    gf_root = ensure_registered(args.gf_root)
    os.environ['GAUSSIANFORMER_ROOT'] = gf_root
    os.environ['OCCBENCH_ROOT'] = bench_root

    cfg_path = args.py_config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(bench_root, cfg_path)
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)

    work_dir = args.work_dir
    if not work_dir:
        work_dir = os.path.join(bench_root, 'output', 'gaussianformer_indoor')
    work_dir = os.path.abspath(work_dir)

    train_path = os.path.join(gf_root, 'train.py')
    spec = importlib.util.spec_from_file_location('gf_train_internal', train_path)
    gf_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf_train)

    def wrapped_main(local_rank, run_args):
        ensure_registered(gf_root)
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
        resume_from=args.resume_from,
        iter_resume=args.iter_resume,
        seed=args.seed,
        gradient_accumulation=args.gradient_accumulation,
        dataset=args.dataset,
        gpus=gpus,
        distributed_backend=distributed_backend,
    )
    if unknown:
        print('Ignoring unknown args (pass through not supported):', unknown)

    prev = os.getcwd()
    os.chdir(gf_root)
    try:
        print(run_args)
        if torchrun:
            wrapped_main(local_rank, run_args)
        elif run_args.gpus > 1:
            torch.multiprocessing.spawn(wrapped_main, args=(run_args,), nprocs=run_args.gpus)
        else:
            wrapped_main(0, run_args)
    finally:
        os.chdir(prev)


if __name__ == '__main__':
    main()
