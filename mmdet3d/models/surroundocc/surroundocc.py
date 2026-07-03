# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
from mmengine.runner import autocast
from mmdet3d.models.utils.grid_mask import GridMask


@MODELS.register_module()
class SurroundOcc(MVXTwoStageDetector):

    def __init__(self, 
                 bev_h=100,
                 bev_w=100,
                 bev_z=12,
                 use_grid_mask=False,
                 grid_config=None,
                 occupancy_head=None,
                 memory_len=1,
                 single_bev_dims=80,
                  **kwargs):
        super(SurroundOcc, self).__init__(**kwargs)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.3, mode=1, prob=0.5)
        self.use_grid_mask = use_grid_mask

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.grid_config = grid_config
        self.occupancy_head = MODELS.build(occupancy_head)

        self.single_bev_dims = single_bev_dims
        self.memory_len = memory_len

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        if self.use_grid_mask:
            img = self.grid_mask(imgs)

        x = self.img_backbone(imgs)
       
        if self.with_img_neck:
            x = self.img_neck(x)
        
            for i in range(len(x)):
                _, output_dim, ouput_H, output_W = x[i].shape
                x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
      
        return x

    def extract_img_feat(self, imgs, img_metas):
        """Extract features of images."""
        img_feat = self.image_encoder(imgs)
        return img_feat

    def extract_feat(self, imgs, img_metas):
        """Extract features from images and points."""
        img_feat = self.extract_img_feat(imgs, img_metas)
        return img_feat

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

        img_metas = [item.metainfo for item in batch_data_samples]
        imgs = torch.stack(batch_inputs_dict['img'])
        img_feats = self.extract_feat(imgs, img_metas=img_metas)

        losses = dict()
        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        with autocast('cuda', enabled=False):
            outs = self.occupancy_head(img_feats, img_metas)
            loss_inputs = [gt_occupancy, outs]
            losses = self.occupancy_head.loss(*loss_inputs, img_metas=img_metas)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Test function without augmentaiton."""

        img_metas = [item.metainfo for item in batch_data_samples]
        imgs = torch.stack(batch_inputs_dict['img'])
        img_feats = self.extract_feat(imgs, img_metas=img_metas)

        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        # visible_mask = torch.stack([item.visible_mask for item in batch_data_samples], dim=0)
        lidar_origins = torch.stack([item.lidar_origins for item in batch_data_samples], dim=0)

        bbox_list = [dict() for _ in range(len(img_metas))]

        pred_occupancy = self.occupancy_head(img_feats, img_metas)['occ_preds'][-1]

        pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
        pred_occupancy = pred_occupancy.softmax(-1)
            
        pred_occupancy = pred_occupancy.argmax(-1) 

        for i, result_dict in enumerate(bbox_list):
            result_dict['pred_occupancy'] = pred_occupancy
            result_dict['gt_occupancy'] = gt_occupancy[i]
            result_dict['lidar_origins'] = lidar_origins[i]
            # result_dict['visible_mask'] = visible_mask[i]
        return bbox_list