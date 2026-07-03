# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.models.fusionocc import BEVDetOcc
from mmengine.runner import autocast
from mmdet3d.models.utils.grid_mask import GridMask


@MODELS.register_module()
class COTR(BEVDetOcc):

    def __init__(self, 
                 group_split=None,
                 occupancy_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        pts_train_cfg = train_cfg.pts if train_cfg else None
        occupancy_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        occupancy_head.update(test_cfg=pts_test_cfg)
        super(COTR, self).__init__(occupancy_head=occupancy_head, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        self.group_split = torch.tensor(group_split, dtype=torch.uint8)

    def generate_mask(self, semantics):
        """Convert semantics to semantic mask for each instance
        Args:
            semantics: [W, H, Z]
        Return:
            classes: [N]
                N unique class in semantics
            masks: [N, W, H, Z]
                N instance masks
        """
        
        w, h, z = semantics.shape
        classes = torch.unique(semantics)
        # # remove ignore region
        # if self.ignore_label is not None:
        #     classes = classes[classes != self.ignore_label]
        gt_classes = classes.long()

        masks = []
        for class_id in classes:
            masks.append(semantics == class_id)
        
        if len(masks) == 0:
            masks = torch.zeros(0, w, h, z)
        else:
            masks = torch.stack([x.clone() for x in masks])

        return gt_classes, masks.long()
        
    def generate_group(self, voxel_semantics):
        group_classes = []
        group_masks = []
        for i in range(len(self.group_split)+1):
            gt_classes = []
            sem_masks = []
            for voxel_semantic in voxel_semantics:
                voxel_semantic[voxel_semantic == 255] = 0 # unknown as free
                if not i < 1:
                    w, h, z = voxel_semantic.shape
                    group_split = self.group_split[i-1].to(voxel_semantic)
                    voxel_semantic = group_split[voxel_semantic.flatten().long()].reshape(w, h, z)
                gt_class, sem_mask = self.generate_mask(voxel_semantic)
                gt_classes.append(gt_class.to(voxel_semantic.device))
                sem_masks.append(sem_mask.to(voxel_semantic.device))
            
            group_classes.append(gt_classes)
            group_masks.append(sem_masks)

        return group_classes, group_masks


    def extract_feat(self, inputs, img_metas):
        """Extract features from images and points."""
        imgs = inputs['imgs']

        prev_exists = torch.tensor([meta['prev_exists'] for meta in img_metas], dtype=torch.float32).to(imgs.device)
        self.pre_update_memory(prev_exists)
        curr_bev_feat, depth, mlvl_feats = self.extract_bev_feat(imgs, img_metas)

        # Fuse History
        ego_pose = torch.stack([meta['ego_pose'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        ego_pose_inv = torch.stack([meta['ego_pose_inv'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        with autocast('cuda', enabled=False):
            bev_feat = self.fuse_history(curr_bev_feat, ego_pose)
            bev_feat = bev_feat.permute(0, 1, 4, 2, 3) # B, C, X, Y, Z - > B, C, Z, X, Y
            x, feats = self.bev_encoder(bev_feat)

        self.post_update_memory(curr_bev_feat, ego_pose_inv)

        return [x, feats, mlvl_feats], depth
    
    def bev_encoder(self, x):
        x = self.bev_encoder_backbone(x)
        x, feats = self.bev_encoder_neck(x)
        return x, feats
    
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
        device = batch_inputs_dict['imgs'].device
        img_metas = [item.metainfo for item in batch_data_samples]
        cam_params = []
        for key in ['cam2ego', 'intrinsic', 'distortion', 'post_trans', 'bda']:
            cam_params.append(torch.stack([meta[key] for meta in img_metas], dim=0).to(device))
        img_feats, depth = self.extract_feat(
            batch_inputs_dict, img_metas=img_metas)

        losses = dict()

        gt_depth = torch.stack([item.gt_pts_seg.depth for item in batch_data_samples], dim=0)
        voxel_semantics = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        gt_classes, sem_mask = self.generate_group(voxel_semantics)
        with autocast('cuda', enabled=False):
            outs = self.occupancy_head(img_feats, img_metas, cam_params)
            loss_inputs = [voxel_semantics, gt_classes, sem_mask, outs]
            losses_occupancy = self.occupancy_head.loss(*loss_inputs, img_metas=img_metas)
            losses.update(losses_occupancy)

            loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
            losses.update(loss_depth)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Test function without augmentaiton."""
        device = batch_inputs_dict['imgs'].device
        img_metas = [item.metainfo for item in batch_data_samples]
        cam_params = []
        for key in ['cam2ego', 'intrinsic', 'distortion', 'post_trans', 'bda']:
            cam_params.append(torch.stack([meta[key] for meta in img_metas], dim=0).to(device))
        img_feats, depth = self.extract_feat(
            batch_inputs_dict, img_metas=img_metas)

        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        # visible_mask = torch.stack([item.visible_mask for item in batch_data_samples], dim=0)
        lidar_origins = torch.stack([item.lidar_origins for item in batch_data_samples], dim=0)

        bbox_list = [dict() for _ in range(len(img_metas))]

        outs = self.occupancy_head(img_feats, img_metas, cam_params)
        pred_occupancy = self.occupancy_head.get_occ(outs, img_metas=img_metas)['occ']
        pred_occupancy = pred_occupancy[0]

        for i, result_dict in enumerate(bbox_list):
            result_dict['pred_occupancy'] = pred_occupancy
            result_dict['gt_occupancy'] = gt_occupancy[i]
            result_dict['lidar_origins'] = lidar_origins[i]
            # result_dict['visible_mask'] = visible_mask[i]
        return bbox_list