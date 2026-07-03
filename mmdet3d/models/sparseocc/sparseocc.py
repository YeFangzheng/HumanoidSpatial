# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
from mmengine.runner import autocast
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet3d.models.sparseocc.utils import sparse2dense

@MODELS.register_module()
class SparseOcc(MVXTwoStageDetector):

    def __init__(self, 
                 bev_h=200,
                 bev_w=200,
                 use_grid_mask=False,
                 grid_config=None,
                 occupancy_head=None,
                 single_bev_dims=80,
                  **kwargs):
        super(SparseOcc, self).__init__(**kwargs)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.3, mode=1, prob=0.5)
        self.use_grid_mask = use_grid_mask

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.grid_config = grid_config
        self.occupancy_head = MODELS.build(occupancy_head)

        self.single_bev_dims = single_bev_dims

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        if self.use_grid_mask:
            img = self.grid_mask(imgs)

        x = self.img_backbone(imgs)
       
        if self.with_img_neck:
            x = self.img_neck(x)
            if isinstance(x, tuple):
                x = list(x)
        
            for i in range(len(x)):
                _, output_dim, ouput_H, output_W = x[i].shape
                x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
      
        return x

    def extract_feat(self, imgs, img_metas):
        """Extract features from images and points."""
        img_feats = self.image_encoder(imgs)
        return img_feats

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        imgs = torch.stack(batch_inputs_dict['img'])
        voxel_semantics = torch.stack([item.gt_pts_seg.voxel_semantics for item in batch_data_samples], dim=0)
        voxel_instances = torch.stack([item.gt_pts_seg.voxel_instances for item in batch_data_samples], dim=0)
        instance_class_ids = [item.gt_pts_seg.instance_class_ids for item in batch_data_samples]
        for i, instance in enumerate(instance_class_ids):
            if len(instance) == 0:
                print(batch_data_samples[i].scene_token)
                print(batch_data_samples[i].frame_idx)
                print(voxel_instances[i])

        mlvl_feats = self.extract_feat(imgs, img_metas=batch_input_metas)
        outs = self.occupancy_head(mlvl_feats, batch_input_metas)

        loss_inputs = [voxel_semantics, voxel_instances, instance_class_ids, outs]
        with autocast('cuda', enabled=False):
            losses = self.occupancy_head.loss(*loss_inputs)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Test function without augmentaiton."""

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        imgs = torch.stack(batch_inputs_dict['img'])
        voxel_semantics = torch.stack([item.gt_pts_seg.voxel_semantics for item in batch_data_samples], dim=0)

        lidar_origins = torch.stack([item.lidar_origins for item in batch_data_samples], dim=0)
        bbox_list = [dict() for _ in range(len(batch_input_metas))]

        mlvl_feats = self.extract_feat(imgs, img_metas=batch_input_metas)
        outs = self.occupancy_head(mlvl_feats, batch_input_metas)
        outs = self.occupancy_head.merge_occ_pred(outs)

        sem_pred = outs['sem_pred']  # [B, N]
        occ_loc = outs['occ_loc']  # [B, N, 3]
        num_cls = len(self.occupancy_head.class_names)   # 21
        free_model_id = num_cls - 1                       # 20

        pred_occupancy, _ = sparse2dense(
            occ_loc, sem_pred, dense_shape=[200, 200, 24], 
            empty_value=free_model_id)                     # 空缺体素填free(20)

        for i, result_dict in enumerate(bbox_list):
            pred_occ = pred_occupancy[i]
            pred_occ[pred_occ < 255] += 1                  # 0→1, ..., 20→21
            pred_occ[pred_occ == free_model_id + 1] = 0    # 21→0 (还原free到GT label 0)
            
            gt_occ = voxel_semantics[i]
            gt_occ[gt_occ < 255] += 1
            gt_occ[gt_occ == free_model_id + 1] = 0
            
            result_dict['pred_occupancy'] = pred_occ
            result_dict['gt_occupancy'] = gt_occ
            result_dict['lidar_origins'] = lidar_origins[i]
        return bbox_list