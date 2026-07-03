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
class FBOcc(MVXTwoStageDetector):

    def __init__(self, 
                 bev_h=100,
                 bev_w=100,
                 bev_z=12,
                 use_grid_mask=False,
                 grid_config=None,
                 pre_process=None,
                 bev_encoder_backbone=None,
                 bev_encoder_neck=None,
                 depth_net=None,
                 forward_projection=None,
                 backward_projection=None,
                 occupancy_head=None,
                 memory_len=1,
                 single_bev_dims=80,
                  **kwargs):
        super(FBOcc, self).__init__(**kwargs)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.3, mode=1, prob=0.5)
        self.use_grid_mask = use_grid_mask

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.grid_config = grid_config

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = MODELS.build(pre_process)
        self.bev_encoder_backbone = MODELS.build(bev_encoder_backbone)
        self.bev_encoder_neck = MODELS.build(bev_encoder_neck)
        self.depth_net = MODELS.build(depth_net) if depth_net else None
        self.forward_projection = MODELS.build(forward_projection) if forward_projection else None
        self.backward_projection = MODELS.build(backward_projection) if backward_projection else None
        self.occupancy_head = MODELS.build(occupancy_head)

        self.single_bev_dims = single_bev_dims
        self.memory_len = memory_len

        self.temporal_fuse_conv = nn.Sequential(
            nn.Conv3d(single_bev_dims * (self.memory_len + 1),
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
            self.memory_bev_feats = prev_exists.new_zeros(B, self.memory_len, self.single_bev_dims, self.bev_h, self.bev_w, self.bev_z)
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
        z = self.bev_z

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=torch.float32).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=torch.float32).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=torch.float32).view(1, 1, z).expand(h, w, z)
        self.grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1)
        
    def fuse_history(self, curr_bev, ego_pose):
        curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        b, c_, z, h, w = curr_bev.shape

        grid_3d = self.grid.to(curr_bev).view(1, h, w, z, 4).expand(b, h, w, z, 4).view(b, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid_3d.dtype).to(grid_3d)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        bev_feats = [curr_bev]
        for i in range(self.memory_len):
            memory_bev_feats = self.memory_bev_feats[:, i].permute(0, 1, 4, 2, 3) # n, c, z, h, w
            memory_egopose_inv = self.memory_egopose_inv[:, i]

            curr2prev = memory_egopose_inv @ ego_pose
            tf = (torch.inverse(feat2bev) @ curr2prev @ feat2bev)

            grid = tf.view(b, 1, 1, 1, 4, 4) @ grid_3d

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
            aligned_bev = F.grid_sample(memory_bev_feats, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode='bilinear')

            no_prev = curr2prev.view(b, -1).sum(-1) == 0
            aligned_bev[no_prev] = curr_bev[no_prev]
            aligned_bev = aligned_bev.detach()
            bev_feats.append(aligned_bev)

        bev_feats = torch.cat(bev_feats, dim=1).permute(0, 1, 3, 4, 2) # B x C x H x W x Z
        bev_feats = self.temporal_fuse_conv(bev_feats) # B x C x H x W x Z
        return bev_feats

    def extract_img_feat(self, imgs, img_metas):
        """Extract features of images."""
        device = imgs.device
        cam2ego = torch.stack([meta['cam2ego'] for meta in img_metas], dim=0).to(device)
        intrinsic = torch.stack([meta['intrinsic'] for meta in img_metas], dim=0).to(device)
        post_trans = torch.stack([meta['post_trans'] for meta in img_metas], dim=0).to(device)
        bda = torch.stack([meta['bda'] for meta in img_metas], dim=0).to(device)
        cam_params = [cam2ego[:, :, :3, :3], cam2ego[:, :, :3, 3], intrinsic, post_trans[:, :, :3, :3], post_trans[:, :, :3, 3], bda]

        img_feat = self.image_encoder(imgs)[0]
        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        context, depth = self.depth_net(img_feat, mlp_input)

        bev_feat = self.forward_projection(cam_params, context, depth)
        
        if self.backward_projection:
            cam_params_bevformer = []
            for key in ['cam2ego', 'intrinsic', 'distortion', 'post_trans', 'bda']:
                cam_params_bevformer.append(torch.stack([meta[key] for meta in img_metas], dim=0).to(device))
            bev_feat_refined = self.backward_projection(
                [context],
                lss_bev=bev_feat.mean(-1),
                cam_params=cam_params_bevformer,
                bev_mask=None,
                pred_img_depth=depth)  

            bev_feat = bev_feat_refined[..., None] + bev_feat

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_bev_feat(self, img, img_metas):
        bev_feat, depth = self.extract_img_feat(img, img_metas)
        return bev_feat, depth

    def extract_feat(self, inputs, img_metas):
        """Extract features from images and points."""
        imgs = inputs['imgs']

        prev_exists = torch.tensor([meta['prev_exists'] for meta in img_metas], dtype=torch.float32).to(imgs.device)
        self.pre_update_memory(prev_exists)
        curr_bev_feat, depth = self.extract_bev_feat(imgs, img_metas)

        # Fuse History
        ego_pose = torch.stack([meta['ego_pose'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        ego_pose_inv = torch.stack([meta['ego_pose_inv'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        with autocast('cuda', enabled=False):
            bev_feat = self.fuse_history(curr_bev_feat, ego_pose)
            bev_feat = self.bev_encoder(bev_feat)

        self.post_update_memory(curr_bev_feat, ego_pose_inv)

        return bev_feat, depth


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
        bev_feat, depth = self.extract_feat(
            batch_inputs_dict, img_metas=batch_input_metas)

        losses = dict()

        gt_depth = torch.stack([item.gt_pts_seg.depth for item in batch_data_samples], dim=0)
        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0)
        with autocast('cuda', enabled=False):
            losses_occupancy, pred_occupancy = self.occupancy_head.forward_train(bev_feat, gt_occupancy=gt_occupancy)
            losses.update(losses_occupancy)

            loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
            losses.update(loss_depth)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Test function without augmentaiton."""

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        bev_feat, depth = self.extract_feat(
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
class BEVFusion(FBOcc):
    def __init__(self, 
                pts_feat_dims=192,
                **kwargs):
        super(BEVFusion, self).__init__(**kwargs)

        self.pts_feat_conv = nn.Sequential(
            nn.Conv2d(
                      pts_feat_dims,
                      self.single_bev_dims,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(self.single_bev_dims),
            nn.ReLU(inplace=True))
        
        self.feat_fuse_conv = nn.Sequential(
            nn.Conv2d(self.single_bev_dims * 2,
                      self.single_bev_dims,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(self.single_bev_dims),
            nn.ReLU(inplace=True))
        
        self.temporal_fuse_conv = nn.Sequential(
            nn.Conv2d(self.single_bev_dims * (self.memory_len + 1),
                      self.single_bev_dims,
                      kernel_size=1,
                      padding=0,
                      stride=1),
            nn.SyncBatchNorm(self.single_bev_dims),
            nn.ReLU(inplace=True))

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
        img_feat, depth = self.extract_img_feat(img, img_metas)

        bev_feat = torch.cat([img_feat.mean(-1), pts_feat], dim=1)
        bev_feat = self.feat_fuse_conv(bev_feat)
        return bev_feat, depth
    
    def extract_feat(self, inputs, img_metas):
        """Extract features from images and points."""
        imgs = inputs['imgs']
        voxels = inputs['voxels']

        prev_exists = torch.tensor([meta['prev_exists'] for meta in img_metas], dtype=torch.float32).to(imgs.device)
        self.pre_update_memory(prev_exists)
        curr_bev_feat, depth = self.extract_bev_feat(voxels, imgs, img_metas)

        # Fuse History
        ego_pose = torch.stack([meta['ego_pose'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        ego_pose_inv = torch.stack([meta['ego_pose_inv'] for meta in img_metas], dim=0).to(curr_bev_feat.device)
        with autocast('cuda', enabled=False):
            bev_feat = self.fuse_history(curr_bev_feat, ego_pose)
            bev_feat = self.bev_encoder(bev_feat)

        self.post_update_memory(curr_bev_feat, ego_pose_inv)

        return bev_feat, depth
    
    def pre_update_memory(self, prev_exists):
        B = prev_exists.shape[0]
        if not self.with_prev or (self.memory_bev_feats is None or len(self.memory_bev_feats) != B):
            self.memory_bev_feats = prev_exists.new_zeros(B, self.memory_len, self.single_bev_dims, self.bev_h, self.bev_w)
            self.memory_egopose_inv = torch.eye(4).to(prev_exists).reshape(1, 1, 4, 4).expand(B, self.memory_len, 4, 4)
        
        self.memory_bev_feats = self.memory_refresh(self.memory_bev_feats, prev_exists)
        self.memory_egopose_inv = self.memory_refresh(self.memory_egopose_inv, prev_exists)

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