# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2 as cv 
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
import numpy as np
from mmengine.runner import autocast
import matplotlib.pyplot as plt
import time

@MODELS.register_module()
class FusionOccONNX(MVXTwoStageDetector):

    def __init__(self, 
                 voxel_layer=None,
                 img_shape=None,
                 bev_h=200,
                 bev_w=200,
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
        super(FusionOccONNX, self).__init__(**kwargs)
        self.voxel_layer = voxel_layer
        self.img_shape = img_shape

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

        self.gen_grid()

    def pre_process_imgs(self, imgs):
        B, N, _, H, W = imgs.shape
        imgs = imgs.to(torch.float16)

        mean = torch.tensor([123.675, 116.28, 103.53], device='cuda').reshape(1,1,3,1,1)
        std = torch.tensor([58.395, 57.12, 57.375], device='cuda').reshape(1,1,3,1,1)
        imgs = imgs[:,:,[2,1,0]]
        imgs = (imgs - mean) / std
        return imgs

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
      
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

        feat2bev = torch.eye(3)
        feat2bev[0, 0] = self.grid_config['x'][2]
        feat2bev[1, 1] = self.grid_config['y'][2]
        feat2bev[0, 2] = self.grid_config['x'][0] + self.grid_config['x'][2] / 2
        feat2bev[1, 2] = self.grid_config['y'][0] + self.grid_config['y'][2] / 2
        feat2bev[2, 2] = 1

        self.feat2bev = feat2bev
        self.bev2feat = torch.inverse(feat2bev)
        
    def fuse_history(self, curr_bev, curr2prev, memory_bev):        
        w = self.bev_w
        h = self.bev_h
        b = curr_bev.shape[0]

        grid_3d = self.grid.to(curr_bev).view(1, h, w, 3).expand(b, h, w, 3).view(b, h, w, 3, 1)
        feat2bev = self.feat2bev.to(curr_bev).view(1, 3, 3)
        bev2feat = self.bev2feat.to(curr_bev).view(1, 3, 3)

        bev_feats = [curr_bev]
        for i in range(curr2prev.shape[1]):
            memory_bev_feats = memory_bev[:, i]

            tf = (bev2feat @ curr2prev[:, i] @ feat2bev)
            grid = tf.view(b, 1, 1, 3, 3) @ grid_3d

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

            aligned_bev = F.grid_sample(memory_bev_feats, grid.to(curr_bev.dtype), align_corners=True, mode='bilinear')
            bev_feats.append(aligned_bev)

        bev_feats = torch.cat(bev_feats, dim=1) # B x C x H x W
        bev_feats = self.temporal_fuse_conv(bev_feats) # B x C x H x W
        
        return bev_feats

    def extract_img_feat(self, imgs, cam_params, pts_feat):
        """Extract features of images."""
        img_feat = self.image_encoder(imgs)

        bev_feat = self.backward_projection(
            img_feat,
            lss_bev=pts_feat,
            cam_params=cam_params)  

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat
    
    def extract_pts_feat(self, voxel_inputs):
        voxels, num_points_per_voxel, coors, voxel_masks = voxel_inputs
        voxel_features = self.pts_voxel_encoder(voxels, num_points_per_voxel, coors)
        voxel_features = torch.where(voxel_masks.unsqueeze(1) > 0, voxel_features, torch.zeros_like(voxel_features))
        batch_size = coors[-1, 0] + 1
        pts_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        pts_feat = self.pts_backbone(pts_feat)
        pts_feat = self.pts_neck(pts_feat)[0]
        pts_feat = self.pts_feat_conv(pts_feat)
        return pts_feat

    def extract_bev_feat(self, points, img, cam_params):
        pts_feat = self.extract_pts_feat(points)
        bev_feat = self.extract_img_feat(img, cam_params, pts_feat)
        return bev_feat

    def extract_feat(self, imgs, points, cam_params, curr2prev, memory_bev):
        """Extract features from images and points."""
        curr_bev_feat = self.extract_bev_feat(points, imgs, cam_params)

        # Fuse History
        with autocast('cuda', enabled=False):
            bev_feat = self.fuse_history(curr_bev_feat, curr2prev, memory_bev)
            bev_feat = self.bev_encoder(bev_feat)

        return bev_feat, curr_bev_feat

    def forward(self, voxels, num_points, coors, voxel_masks, imgs, post_trans, ego2cam, distortion, intrinsic, curr2prev, memory_bev):
        """Test function without augmentaiton."""
        voxel_inputs = (voxels, num_points, coors, voxel_masks)
        imgs = self.pre_process_imgs(imgs)

        cam_params = [ego2cam, distortion, intrinsic, post_trans]
        bev_feat, curr_bev_feat = self.extract_feat(imgs, voxel_inputs, cam_params, curr2prev, memory_bev)
        pred_occ = self.occupancy_head(bev_feat)['out_voxels']

        pred_occ = pred_occ.permute(0, 3, 2, 4, 1)[0]
        pred_occ = pred_occ.softmax(-1)

        memory_bank = torch.cat([memory_bev[:, 1:], curr_bev_feat.unsqueeze(1)], dim=1)
        return pred_occ, memory_bank