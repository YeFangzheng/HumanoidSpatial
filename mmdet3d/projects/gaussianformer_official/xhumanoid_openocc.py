# Copyright (c) OpenMMLab. All rights reserved.
"""XHumanoid dataset + transforms for the official GaussianFormer OpenOcc training stack."""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from typing import List, Sequence

import mmengine
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from mmdet3d.datasets.transforms.loading import PrepareImageInputs


def _bevdet_xyz_grid(
    pc_range: Sequence[float],
    grid: Sequence[int],
    reso: float,
) -> np.ndarray:
    """Voxel centers in lidar frame, shape (H, W, D, 3), same layout as ``LoadOccupancyXHumanoid``."""
    ranges = list(pc_range)
    gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])
    xxx = torch.arange(gx, dtype=torch.float32) * reso + 0.5 * reso + ranges[0]
    yyy = torch.arange(gy, dtype=torch.float32) * reso + 0.5 * reso + ranges[1]
    zzz = torch.arange(gz, dtype=torch.float32) * reso + 0.5 * reso + ranges[2]

    xxx = xxx[:, None, None].expand(gx, gy, gz)
    yyy = yyy[None, :, None].expand(gx, gy, gz)
    zzz = zzz[None, None, :].expand(gx, gy, gz)

    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    xyz = xyz.permute(2, 0, 1, 3)
    xyz = torch.rot90(xyz, 1, [1, 2])
    xyz = torch.flip(xyz, [1])
    xyz = xyz.permute(1, 2, 0, 3).contiguous()
    return xyz.numpy().astype(np.float32)


@OPENOCC_DATASET.register_module()
class XHumanoidOpenOccDataset(Dataset):
    """Frame list JSON (same as ``XHumanoidDataset``) for official ``train.py``."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        data_config: dict,
        occupancy_path: str,
        class_names: List[str],
        pipeline: List[dict] | None = None,
        phase: str = 'train',
        return_keys: List[str] | None = None,
        filter_scenes: List[str] | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.occupancy_path = occupancy_path
        self.class_names = class_names
        self.phase = phase
        self.data_config = data_config
        self.filter_scenes = set(filter_scenes) if filter_scenes else None

        raw_list = mmengine.load(ann_file)
        self.data_list: List[dict] = []
        for raw in raw_list:
            if self.filter_scenes and raw.get('scene_token') not in self.filter_scenes:
                continue
            self.data_list.append(raw)

        self.pipeline = [OPENOCC_TRANSFORMS.build(t) for t in (pipeline or [])]

        self.return_keys = return_keys or [
            'img',
            'projection_mat',
            'image_wh',
            'occ_label',
            'occ_xyz',
            'occ_cam_mask',
            'ori_img',
            'cam_positions',
            'focal_positions',
        ]

    def __len__(self):
        return len(self.data_list)

    def _parse_data_info(self, info: dict) -> dict:
        info = copy.deepcopy(info)
        info['lidar_points'] = info['lidars']['LIDAR_TOP']
        info['lidar_points']['lidar_path'] = os.path.join(
            self.data_root, info['lidar_points']['lidar_path'])
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        info['images'] = info['cameras']
        for _cam_id, img_info in info['images'].items():
            img_info['img_path'] = os.path.join(self.data_root, img_info['img_path'])
            intrinsic = np.array(img_info['intrinsic'])
            cam2img = np.array(
                [[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]],
                dtype=np.float32,
            )
            img_info['cam2img'] = cam2img
            cam2lidar = np.array(img_info['cam2lidar'])
            cam2ego = lidar2ego @ cam2lidar
            img_info['cam2ego'] = cam2ego
        info['frame_idx'] = info['frame_id']
        return info

    def __getitem__(self, index: int):
        info = self._parse_data_info(self.data_list[index])
        input_dict: dict = info
        input_dict['scene_token'] = info['scene_token']
        input_dict['timestamp'] = 0.0

        for t in self.pipeline:
            input_dict = t(input_dict)

        return {k: input_dict[k] for k in self.return_keys if k in input_dict}


@OPENOCC_TRANSFORMS.register_module()
class XHumanoidOpenOccLidarOrigins(object):
    """Match ``XHumanoidDataset.get_data_info`` (test_mode) RayIoU origins in current lidar frame."""

    def __init__(self, ann_file: str):
        raw_list = mmengine.load(ann_file)
        scene_frames: dict[str, list] = defaultdict(list)
        for raw in raw_list:
            scene_frames[raw['scene_token']].append(raw)
        for _tok, frames in scene_frames.items():
            frames.sort(key=lambda x: int(x['frame_id']) if not isinstance(x['frame_id'], str) else int(x['frame_id']))
        self.scene_frames = dict(scene_frames)

    def __call__(self, results: dict) -> dict:
        ref_ego2global = np.array(results['ego2global'], dtype=np.float64)
        ref_lidar2ego = np.array(results['lidar_points']['lidar2ego'], dtype=np.float64)
        ref_lidar2global = ref_lidar2ego @ ref_ego2global

        scene_frame = self.scene_frames[results['scene_token']]
        output_origin_list: list[np.ndarray] = []
        for curr_info in scene_frame:
            curr_ego2global = np.array(curr_info['ego2global'], dtype=np.float64)
            curr_lidar2ego = np.array(curr_info['lidars']['LIDAR_TOP']['lidar2ego'], dtype=np.float64)
            curr_lidar2global = curr_lidar2ego @ curr_ego2global
            curr2ref = curr_lidar2global @ np.linalg.inv(ref_lidar2global)
            origin_tf = np.array(curr2ref[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones(4, dtype=np.float64)
            origin_tf_pad[:3] = origin_tf
            origin_tf = np.dot(ref_lidar2ego[:3], origin_tf_pad.T).T.astype(np.float32)

            if np.abs(origin_tf[0]) < 9 and np.abs(origin_tf[1]) < 9:
                output_origin_list.append(origin_tf)

        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        if not output_origin_list:
            output_origin_list = [np.zeros(3, dtype=np.float32)]

        results['lidar_origins'] = torch.from_numpy(np.stack(output_origin_list))
        return results


@OPENOCC_TRANSFORMS.register_module()
class XHumanoidPrepareImageInputs(object):
    """Reuses benchmark ``PrepareImageInputs`` (undistort + aug + normalize)."""

    def __init__(self, data_config: dict, is_train: bool = True, undistort: bool = True):
        self.t = PrepareImageInputs(data_config, is_train=is_train, undistort=undistort)

    def __call__(self, results: dict) -> dict:
        return self.t(results)


@OPENOCC_TRANSFORMS.register_module()
class XHumanoidOpenOccProjection(object):
    """Build ``projection_mat`` / ``image_wh`` expected by official GaussianFormer (ego2img)."""

    def __call__(self, results: dict) -> dict:
        intrinsic = results['intrinsic']
        cam2ego = results['cam2ego']
        post_trans = results['post_trans']
        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0)
            cam2ego = cam2ego.unsqueeze(0)
            post_trans = post_trans.unsqueeze(0)
        n = intrinsic.shape[0]
        cam2img = torch.eye(4, dtype=intrinsic.dtype, device=intrinsic.device).unsqueeze(0).repeat(n, 1, 1)
        cam2img[:, :3, :3] = intrinsic[:, :3, :3]
        ego2cam = torch.inverse(cam2ego)
        ego2img = cam2img @ ego2cam
        mat = torch.eye(4, dtype=intrinsic.dtype, device=intrinsic.device).unsqueeze(0).repeat(n, 1, 1)
        mat[:, :2, :2] = post_trans[:, :2, :2]
        mat[:, :2, 3] = post_trans[:, :2, 3]
        ego2img = mat @ ego2img
        results['projection_mat'] = ego2img.cpu().numpy().astype(np.float32)

        img = results['img']
        if img.dim() == 4:
            _, _, h, w = img.shape
            results['image_wh'] = np.array([[float(w), float(h)]] * n, dtype=np.float32)
        else:
            raise ValueError('Expected img tensor (N,C,H,W) after PrepareImageInputs')

        results['cam_positions'] = np.zeros((n, 3), dtype=np.float32)
        results['focal_positions'] = np.zeros((n, 3), dtype=np.float32)
        return results


@OPENOCC_TRANSFORMS.register_module()
class XHumanoidOpenOccLoadOccupancy(object):
    """Load npz occupancy + voxel centers aligned with ``LoadOccupancyXHumanoid``."""

    def __init__(
        self,
        occupancy_path: str,
        class_names: List[str],
        remap_labels: dict | None = None,
        pc_range: Sequence[float] | None = None,
        grid_size: Sequence[int] | None = None,
        voxel_size: float = 0.1,
    ):
        self.occupancy_path = occupancy_path
        self.class_names = class_names
        self.remap_labels = remap_labels or {}
        self.pc_range = list(pc_range) if pc_range is not None else [-10, -10, -1.5, 10, 10, 0.9]
        self.grid_size = list(grid_size) if grid_size is not None else [200, 200, 24]
        self.voxel_size = voxel_size
        self._occ_xyz = _bevdet_xyz_grid(self.pc_range, self.grid_size, self.voxel_size)

    def __call__(self, results: dict) -> dict:
        occ_name = os.path.basename(results['lidar_points']['lidar_path']).replace('pcd', 'npz')
        occ_file = os.path.join(self.occupancy_path, results['scene_token'], occ_name)
        occ = np.load(occ_file)['occ'][:, :, :24]
        label = torch.tensor(occ, dtype=torch.long)

        label = label.permute(2, 0, 1)
        label = torch.rot90(label, 1, [1, 2])
        label = torch.flip(label, [1])
        label = label.permute(1, 2, 0)

        for src, dst in self.remap_labels.items():
            label[label == int(src)] = int(dst)

        occ_label = label.numpy().astype(np.int64)
        results['occ_label'] = occ_label
        results['occ_xyz'] = self._occ_xyz.copy()
        results['occ_cam_mask'] = (occ_label != 255).astype(np.bool_)
        return results
