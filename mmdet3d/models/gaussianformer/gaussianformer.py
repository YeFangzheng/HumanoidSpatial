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
class GaussianFormer(MVXTwoStageDetector):

    def __init__(self, 
                 use_grid_mask=False,
                 lifter=None,
                 encoder=None,
                 head=None,
                  **kwargs):
        super(GaussianFormer, self).__init__(**kwargs)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.3, mode=1, prob=0.5)
        self.use_grid_mask = use_grid_mask

        self.lifter = MODELS.build(lifter)
        self.encoder = MODELS.build(encoder)
        self.head = MODELS.build(head)

    def prepare_img_metas(self, img_metas, batch_data_samples):
        intrinsic = torch.stack([meta['intrinsic'] for meta in img_metas])
        cam2ego = torch.stack([meta['cam2ego'] for meta in img_metas])
        post_trans = torch.stack([meta['post_trans'] for meta in img_metas])
        device = intrinsic.device

        B, N, _, _ = cam2ego.shape
        cam2img = torch.eye(4)[None, None].repeat(B, N, 1, 1)
        cam2img[..., :3, :3] = intrinsic
        ego2cam = torch.inverse(cam2ego)
        ego2img = cam2img @ ego2cam
        mat = torch.eye(4)[None, None].repeat(B, N, 1, 1)
        mat[..., :2, :2] = post_trans[..., :2, :2]
        mat[..., :2, 2] = post_trans[..., :2, 3]
        ego2img = mat @ ego2img
        for i in range(B):
            img_metas[i]['projection_mat'] = ego2img[i]

        # IMPORTANT:
        # `LoadOccupancyXHumanoid` converts occupancy to a BEVDet-style layout
        # using rot90 + flip on the BEV plane. `GaussianHead` supervises by
        # flattening `occ_xyz` and `occ_label` elementwise, so `occ_xyz` must
        # follow the same layout; otherwise supervision becomes spatially
        # misaligned and training/metrics will collapse.
        grid = [200, 200, 24]
        reso = 0.1
        ranges = [-10, -10, -1.5, 10, 10, 0.9]
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1)  # (H, W, D, 3)

        # Match `LoadOccupancyXHumanoid` BEVDet-format transform:
        # occupancy = occ.permute(2, 0, 1); rot90(1,[1,2]); flip([1]); permute(1,2,0)
        # Apply the same transform to xyz so that (x_idx, y_idx, z_idx) aligns.
        xyz = xyz.permute(2, 0, 1, 3)                 # (D, H, W, 3)
        xyz = torch.rot90(xyz, 1, [1, 2])             # rot in (H, W)
        xyz = torch.flip(xyz, [1])                    # flip H
        xyz = xyz.permute(1, 2, 0, 3).contiguous()    # (H, W, D, 3)
        # img_metas[0]['occ_xyz'] = xyz.to(device)
        # img_metas[0]['occ_label'] = batch_data_samples[0].gt_pts_seg.occupancy
        for i in range(B):
            img_metas[i]['occ_xyz'] = xyz.to(device)
            img_metas[i]['occ_label'] = batch_data_samples[i].gt_pts_seg.occupancy

        return img_metas

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

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        img_feat = self.image_encoder(imgs)
        result = {'ms_img_feats': img_feat}
        return result

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

        img_metas = self.prepare_img_metas(img_metas, batch_data_samples)        
        
        results = {
            'imgs': imgs,
            'metas': img_metas
        }
        outs = self.extract_img_feat(**results)
        results.update(outs)

        outs = self.lifter(**results)
        results.update(outs)

        outs = self.encoder(**results)
        results.update(outs)

        losses = dict()
        with autocast('cuda', enabled=False):
            outs = self.head(**results)
            loss_inputs = [outs[key] for key in ['pred_occ', 'sampled_label']]
            losses = self.head.loss(*loss_inputs)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Test function without augmentaiton."""

        img_metas = [item.metainfo for item in batch_data_samples]
        imgs = torch.stack(batch_inputs_dict['img'])

        img_metas = self.prepare_img_metas(img_metas, batch_data_samples)        
        
        results = {
            'imgs': imgs,
            'metas': img_metas
        }
        outs = self.extract_img_feat(**results)
        results.update(outs)

        outs = self.lifter(**results)
        results.update(outs)

        outs = self.encoder(**results)
        results.update(outs)

        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        # visible_mask = torch.stack([item.visible_mask for item in batch_data_samples], dim=0)
        lidar_origins = torch.stack([item.lidar_origins for item in batch_data_samples], dim=0)

        bbox_list = [dict() for _ in range(len(img_metas))]

        outs = self.head(**results)

        pred_occupancy = outs['final_occ'].reshape(-1, 200, 200, 24)

        for i, result_dict in enumerate(bbox_list):
            result_dict['pred_occupancy'] = pred_occupancy[i]
            result_dict['gt_occupancy'] = gt_occupancy[i]
            result_dict['lidar_origins'] = lidar_origins[i]
            # result_dict['visible_mask'] = visible_mask[i]
        return bbox_list