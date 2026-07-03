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
class FusionOcc(MVXTwoStageDetector):

    def __init__(self, 
                 bev_h=200,
                 bev_w=200,
                 use_grid_mask=False,
                 grid_config=None,
                 pre_process=None,
                 bev_encoder_backbone=None,
                 bev_encoder_neck=None,
                 backward_projection=None,
                 occupancy_head=None,
                 memory_len=1,
                 single_bev_dims=80,
                 pts_feat_dims=192,
                  **kwargs):
        super(FusionOcc, self).__init__(**kwargs)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.3, mode=1, prob=0.5)
        self.use_grid_mask = use_grid_mask

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.grid_config = grid_config

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = MODELS.build(pre_process)
        self.bev_encoder_backbone = MODELS.build(bev_encoder_backbone)
        self.bev_encoder_neck = MODELS.build(bev_encoder_neck)
        self.backward_projection = MODELS.build(backward_projection)
        self.occupancy_head = MODELS.build(occupancy_head)

        self.single_bev_dims = single_bev_dims
        self.memory_len = memory_len

        self.pts_feat_conv = nn.Sequential(
            nn.Conv2d(
                      pts_feat_dims,
                      single_bev_dims,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(single_bev_dims),
            nn.ReLU(inplace=True))

        self.temporal_fuse_conv = nn.Sequential(
            nn.Conv2d(single_bev_dims * (self.memory_len + 1),
                      single_bev_dims,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(single_bev_dims),
            nn.ReLU(inplace=True))

        self.with_prev = True
        self.gen_grid()
        self.init_memory()
    
    def init_memory(self):
        self.memory_bev_feats = None
        self.memory_egopose = None

    def memory_refresh(self, memory, prev_exist):
        memory_shape = memory.shape
        view_shape = [1 for _ in range(len(memory_shape))]
        prev_exist = prev_exist.view(-1, *view_shape[1:]) 
        return memory * prev_exist

    def pre_update_memory(self, prev_exists):
        B = prev_exists.shape[0]
        if not self.with_prev or (self.memory_bev_feats is None or len(self.memory_bev_feats) != B):
            self.memory_bev_feats = prev_exists.new_zeros(B, self.memory_len, self.single_bev_dims, self.bev_h, self.bev_w)
            self.memory_egopose_inv = torch.eye(4).to(prev_exists).reshape(1, 1, 4, 4).expand(B, self.memory_len, 4, 4)
        
        self.memory_bev_feats = self.memory_refresh(self.memory_bev_feats, prev_exists)
        self.memory_egopose_inv = self.memory_refresh(self.memory_egopose_inv, prev_exists)

    def post_update_memory(self, bev_feats, ego_pose_inv):
        self.memory_bev_feats = torch.cat([self.memory_bev_feats[:, 1:], bev_feats.detach().unsqueeze(1)], dim=1)
        self.memory_egopose_inv = torch.cat([self.memory_egopose_inv[:, 1:], ego_pose_inv.detach().unsqueeze(1)], dim=1)

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

    def bev_encoder(self, x):
        x = self.bev_encoder_backbone(x)
        x = self.bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x
    
    def gen_grid(self):
        w = self.bev_w
        h = self.bev_h

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=torch.float32).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=torch.float32).view(h, 1).expand(h, w)
        self.grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        
    def fuse_history(self, curr_bev, ego_pose):        
        w = self.bev_w
        h = self.bev_h
        b = curr_bev.shape[0]

        grid_3d = self.grid.to(curr_bev).view(1, h, w, 3).expand(b, h, w, 3).view(b, h, w, 3, 1)

        # This converts BEV indices to meters
        feat2bev = torch.eye(3, dtype=grid_3d.dtype).to(grid_3d)
        feat2bev[0, 0] = self.grid_config['x'][2]
        feat2bev[1, 1] = self.grid_config['y'][2]
        feat2bev[0, 2] = self.grid_config['x'][0] + self.grid_config['x'][2] / 2
        feat2bev[1, 2] = self.grid_config['y'][0] + self.grid_config['y'][2] / 2
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)

        bev_feats = [curr_bev]
        for i in range(self.memory_len):
            memory_bev_feats = self.memory_bev_feats[:, i]
            memory_egopose_inv = self.memory_egopose_inv[:, i]

            curr2prev = memory_egopose_inv @ ego_pose
            curr2prev = curr2prev[:, [0, 1, 3], :][:, :,[0, 1, 3]]
            tf = (torch.inverse(feat2bev) @ curr2prev @ feat2bev)

            grid = tf.view(b, 1, 1, 3, 3) @ grid_3d

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
            aligned_bev = F.grid_sample(memory_bev_feats, grid.to(curr_bev.dtype), align_corners=True, mode='bilinear')

            no_prev = curr2prev.view(b, -1).sum(-1) == 0
            aligned_bev[no_prev] = curr_bev[no_prev]
            aligned_bev = aligned_bev.detach()
            bev_feats.append(aligned_bev)

        bev_feats = torch.cat(bev_feats, dim=1) # B x C x H x W
        bev_feats = self.temporal_fuse_conv(bev_feats) # B x C x H x W
        
        return bev_feats

    def extract_img_feat(self, imgs, img_metas, pts_feat=None):
        """Extract features of images."""
        device = imgs.device
        cam_params = []
        for key in ['cam2ego', 'intrinsic', 'distortion', 'post_trans', 'bda']:
            cam_params.append(torch.stack([meta[key] for meta in img_metas], dim=0).to(device))

        img_feat = self.image_encoder(imgs)

        bev_feat = self.backward_projection(
            img_feat,
            lss_bev=pts_feat,
            cam_params=cam_params,
            bev_mask=None,
            pred_img_depth=None)  

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat

    def extract_pts_feat(self, voxels):
        coors = voxels['coors']
        num_points = voxels['num_points']
        voxels = voxels['voxels']
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        pts_feat = self.pts_backbone(pts_feat)
        pts_feat = self.pts_neck(pts_feat)[0]
        pts_feat = self.pts_feat_conv(pts_feat)
        return pts_feat

    def extract_bev_feat(self, points, img, img_metas):
        pts_feat = self.extract_pts_feat(points)
        bev_feat = self.extract_img_feat(img, img_metas, pts_feat=pts_feat)
        return bev_feat

    def extract_feat(self, inputs, img_metas):
        """Extract features from images and points."""
        imgs = inputs['imgs']
        voxels = inputs['voxels']

        prev_exists = torch.tensor([meta['prev_exists'] for meta in img_metas], dtype=torch.float32).to(imgs.device)
        self.pre_update_memory(prev_exists)
        curr_bev_feat = self.extract_bev_feat(voxels, imgs, img_metas)

        # Fuse History
        ego_pose = torch.stack([meta['ego_pose'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        ego_pose_inv = torch.stack([meta['ego_pose_inv'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        with autocast('cuda', enabled=False):
            bev_feat = self.fuse_history(curr_bev_feat, ego_pose)
            bev_feat = self.bev_encoder(bev_feat)

        self.post_update_memory(curr_bev_feat, ego_pose_inv)

        return bev_feat


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
        bev_feat = self.extract_feat(
            batch_inputs_dict, img_metas=batch_input_metas)
        losses = dict()

        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        # gt_occupancy.append(torch.stack([item.gt_pts_seg.occupancy_2x for item in batch_data_samples], dim=0))
        with autocast('cuda', enabled=False):
            losses_occupancy, pred_occupancy = self.occupancy_head.forward_train(bev_feat, gt_occupancy=gt_occupancy)
            losses.update(losses_occupancy)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Test function without augmentaiton."""

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        bev_feat = self.extract_feat(
            batch_inputs_dict, img_metas=batch_input_metas)

        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        # visible_mask = torch.stack([item.visible_mask for item in batch_data_samples], dim=0)
        lidar_origins = torch.stack([item.lidar_origins for item in batch_data_samples], dim=0)

        bbox_list = [dict() for _ in range(len(batch_input_metas))]

        pred_occupancy = self.occupancy_head(bev_feat)['out_voxels']

        pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
        pred_occupancy = pred_occupancy.softmax(-1)
            
        pred_occupancy = pred_occupancy.argmax(-1) 

        for i, result_dict in enumerate(bbox_list):
            result_dict['pred_occupancy'] = pred_occupancy
            result_dict['gt_occupancy'] = gt_occupancy[i]
            result_dict['lidar_origins'] = lidar_origins[i]
            # result_dict['visible_mask'] = visible_mask[i]
        return bbox_list


@MODELS.register_module()
class FusionOccV2(FusionOcc):
    def __init__(self, 
                 **kwargs):
        super(FusionOccV2, self).__init__(**kwargs)

        self.feat_fuse_conv = nn.Sequential(
            nn.Conv2d(512,
                      256,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True))
        
        self.temporal_fuse_conv = nn.Sequential(
            nn.Conv2d(256 * (self.memory_len + 1),
                      256,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(inplace=True))
        
        self.with_prev = False

    def extract_img_feat(self, imgs, img_metas, pts_feat=None, **kwargs):
        """Extract features of images."""
        device = imgs.device
        cam_params = []
        for key in ['cam2ego', 'intrinsic', 'post_trans', 'bda']:
            cam_params.append(torch.stack([meta[key] for meta in img_metas], dim=0).to(device))

        img_feat = self.image_encoder(imgs)

        img_feat = self.backward_projection(
            img_feat,
            img_metas,
            cam_params=cam_params,
            bev_mask=None,
            pred_img_depth=None)  

        bev_feat = torch.cat([img_feat, pts_feat], dim=1)
        bev_feat = self.feat_fuse_conv(bev_feat)
        return bev_feat