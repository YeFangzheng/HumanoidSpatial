"""
保存 SparseOcc 预测的 occ 为 .npz 格式（与 GT occ 格式一致）

用法：
    python tools/save_pred_occ.py \
        configs/exp/sparseocc_indoor.py \
        output/sparseocc_indoor/epoch_19.pth \
        --save-dir vis_results/pred_occ \
        --scene-tokens 69d1da25feca98469f9d3e309f4601b5 16cf8c0ff7275f2840128d60ab3f84db

输出：
    vis_results/pred_occ/
    ├── pred/
    │   └── {scene_token}/frame_000000.npz ...
    └── gt/
        └── {scene_token}/frame_000000.npz ...

之后用 rename_pred_occ.py 把 frame_XXXXXX.npz 重命名为与 GT 一致的文件名。
"""

import argparse
import os
import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS


def bevdet_to_original(occ):
    """
    反转 BEVDet 坐标变换，还原到原始坐标系。
    """
    if isinstance(occ, np.ndarray):
        occ = torch.from_numpy(occ)
    occ = occ.permute(2, 0, 1)
    occ = torch.flip(occ, [1])
    occ = torch.rot90(occ, -1, [1, 2])
    occ = occ.permute(1, 2, 0)
    return occ.numpy()


@METRICS.register_module(force=True)
class SaveOccMetric(BaseMetric):
    def __init__(self, save_dir='.', scene_tokens=None,
                 collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.save_dir = save_dir
        self.scene_tokens = set(scene_tokens) if scene_tokens else None
        self.saved_count = 0
        self.frame_counters = {}
        os.makedirs(save_dir, exist_ok=True)

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            # data_sample 是 sparseocc.predict() 返回的 dict
            # 里面没有 metainfo，但如果不过滤 scene_token 就全部保存
            
            # 尝试从 data_batch 获取 scene_token
            scene_token = None
            batch_metas = data_batch.get('data_samples', [])
            for m in batch_metas:
                if hasattr(m, 'metainfo'):
                    scene_token = m.metainfo.get('scene_token', None)
                elif hasattr(m, 'scene_token'):
                    scene_token = m.scene_token
                if scene_token:
                    break

            # 如果指定了 scene_tokens 但获取不到或不匹配，跳过
            if self.scene_tokens:
                if scene_token is None or scene_token not in self.scene_tokens:
                    return

            # 用 scene_token 或 "unknown" 做目录名
            token_dir = scene_token if scene_token else "unknown"

            if token_dir not in self.frame_counters:
                self.frame_counters[token_dir] = 0
            frame_idx = self.frame_counters[token_dir]
            self.frame_counters[token_dir] += 1

            # 获取预测和GT
            pred_occ = data_sample['pred_occupancy']
            gt_occ = data_sample['gt_occupancy']

            if isinstance(pred_occ, torch.Tensor):
                pred_occ = pred_occ.cpu().numpy()
            if isinstance(gt_occ, torch.Tensor):
                gt_occ = gt_occ.cpu().numpy()

            # 反转 BEVDet 变换
            pred_occ_orig = bevdet_to_original(pred_occ)
            gt_occ_orig = bevdet_to_original(gt_occ)

            filename = f'frame_{frame_idx:06d}.npz'

            # 保存 pred
            pred_dir = os.path.join(self.save_dir, 'pred', token_dir)
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, filename)
            np.savez_compressed(pred_path, occ=pred_occ_orig.astype(np.uint8))

            # 保存 GT
            gt_dir = os.path.join(self.save_dir, 'gt', token_dir)
            os.makedirs(gt_dir, exist_ok=True)
            gt_path = os.path.join(gt_dir, filename)
            np.savez_compressed(gt_path, occ=gt_occ_orig.astype(np.uint8))

            self.saved_count += 1
            labels = np.unique(pred_occ_orig[(pred_occ_orig > 0) & (pred_occ_orig < 255)])
            print(f"  [{self.saved_count}] scene={token_dir[:12]}... "
                  f"frame={frame_idx} "
                  f"pred_labels={labels} "
                  f"-> {pred_path}")

    def compute_metrics(self, results=None):
        print(f"\nDone! Saved {self.saved_count} pred/gt pairs to {self.save_dir}/")
        print(f"Scenes: {list(self.frame_counters.keys())}")
        print(f"Frames per scene: {dict(self.frame_counters)}")
        return {'saved_count': self.saved_count}

    def evaluate(self, size):
        _metrics = self.compute_metrics()
        self.results.clear()
        return _metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Save SparseOcc predictions as npz')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--save-dir', type=str, default='vis_results/pred_occ')
    parser.add_argument('--scene-tokens', nargs='+', default=None,
                        help='Only save these scene tokens (default: save all)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.save_dir

    cfg.val_evaluator = dict(
        type='SaveOccMetric',
        save_dir=args.save_dir,
        scene_tokens=args.scene_tokens,
    )
    cfg.test_evaluator = cfg.val_evaluator

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()