# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, List, Union
import copy
import pickle

import torch
import numpy as np
import math

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.fileio import load
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class XHumanoidDataset(BaseDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 seq_split_num=1,
                 filter=[],
                 **kwargs) -> None:
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.filter = filter

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
        
        self.seq_split_num = seq_split_num
        self._set_sequence_group_flag()

        if self.test_mode:
            self.scene_frames = {}

            for info in self.data_list:
                scene_token = info['scene_token']
                if scene_token not in self.scene_frames:
                    self.scene_frames[scene_token] = []
                self.scene_frames[scene_token].append(info)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        raw_data_list = load(self.ann_file)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            if len(self.filter) > 0 and raw_data_info['scene_token'] not in self.filter:
                continue
            data_info = self.parse_data_info(raw_data_info)
            data_list.append(data_info)

        return data_list
    
    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_list)):
            if idx != 0 and self.data_list[idx]['frame_idx'] == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_list)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)
        
        prev_flag = None
        for i in range(len(self.flag)):
            if self.flag[i] != prev_flag:
                self.data_list[i]['prev_exists'] = False
            else:
                self.data_list[i]['prev_exists'] = True
            prev_flag = self.flag[i]

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        bbox_path = os.path.join(self.data_root, info['bbox_path'])
        with open(bbox_path, 'r') as f:
            lines = f.readlines()
        kitti_bboxes = [line.strip().split() for line in lines]

        gt_bboxes_3d = []
        for kitti_bbox in kitti_bboxes:
            obj = {
                'type': kitti_bbox[0],
                'dimensions': [float(x) for x in kitti_bbox[1:4][::-1]],
                'location': [float(x) for x in kitti_bbox[4:7]],
                'yaw': float(kitti_bbox[7]),
                'track_id': int(kitti_bbox[8]), 
            }
            gt_bboxes_3d.append(obj['location'] + obj['dimensions'] + [obj['yaw']])
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, origin=(0.5, 0.5, 0.5))
        info['gt_bboxes_3d'] = gt_bboxes_3d
        return info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['lidar_points'] = info['lidars']['LIDAR_TOP']
        info['lidar_points']['lidar_path'] = os.path.join(self.data_root, info['lidar_points']['lidar_path'])
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        info['images'] = info['cameras']
        for cam_id, img_info in info['images'].items():
            img_info['img_path'] = os.path.join(self.data_root, img_info['img_path'])
            intrinsic = np.array(img_info['intrinsic'])
            cam2img = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
            img_info['cam2img'] = cam2img
            cam2lidar = np.array(img_info['cam2lidar'])
            cam2ego = lidar2ego @ cam2lidar
            img_info['cam2ego'] = cam2ego
        info['frame_idx'] = info['frame_id']
        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        return info

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['adjacent'] = []

        adj_id_list = []
        for select_id in adj_id_list:
            adj_idx = max(idx - select_id, 0)
            
            if self.serialize_data:
                start_addr = 0 if adj_idx == 0 else self.data_address[adj_idx - 1].item()
                end_addr = self.data_address[adj_idx].item()
                bytes = memoryview(
                    self.data_bytes[start_addr:end_addr])  # type: ignore
                adj_info = pickle.loads(bytes)  # type: ignore
            else:
                adj_info = copy.deepcopy(self.data_list[adj_idx])

            # Some codebase needs `sample_idx` of data information. Here we convert
            # the idx to a positive number and save it in data information.
            if adj_idx >= 0:
                adj_info['sample_idx'] = adj_idx
            else:
                adj_info['sample_idx'] = len(self) + adj_idx

            if not adj_info['scene_token'] == data_info['scene_token']:
                data_info['adjacent'].append(data_info)
            else:
                data_info['adjacent'].append(adj_info)

        # prepare for RayIoU
        if self.test_mode:
            ref_ego2global = np.array(data_info['ego2global'])
            ref_lidar2ego = np.array(data_info['lidar_points']['lidar2ego'])
            ref_lidar2global = ref_lidar2ego @ ref_ego2global

            scene_frame = self.scene_frames[data_info['scene_token']]

            # NOTE: getting output frames
            output_origin_list = []
            for curr_info in scene_frame:
                # transform from the current lidar frame to global and then to the reference lidar frame
                curr_ego2global = np.array(curr_info['ego2global'])
                curr_lidar2ego = np.array(curr_info['lidar_points']['lidar2ego'])
                curr_lidar2global = curr_lidar2ego @ curr_ego2global
                curr2ref = curr_lidar2global @ np.linalg.inv(ref_lidar2global)
                origin_tf = np.array(curr2ref[:3, 3], dtype=np.float32)

                origin_tf_pad = np.ones([4])
                origin_tf_pad[:3] = origin_tf  # pad to [4]
                origin_tf = np.dot(ref_lidar2ego[:3], origin_tf_pad.T).T  # [3]

                # origin
                if np.abs(origin_tf[0]) < 9 and np.abs(origin_tf[1]) < 9:
                    output_origin_list.append(origin_tf)
            
            # select 8 origins
            if len(output_origin_list) > 8:
                select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
                output_origin_list = [output_origin_list[i] for i in select_idx]

            data_info['lidar_origins'] = torch.from_numpy(np.stack(output_origin_list))

        return data_info