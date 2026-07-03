# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d.models.fusionocc.losses import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.fusionocc.losses import nusc_class_frequencies, nusc_class_names, humanoid_industry_frequencies
from mmdet3d.models.fusionocc.losses import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from torch.utils.checkpoint import checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

@MODELS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        soft_weights=False,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_3d_cfg=dict(type='Conv3d'),
        norm_3d_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        occ_size=None,
        empty_idx=0,
        balance_cls_weight=True,
        use_focal_loss=False,
        use_dice_loss=False,
        use_deblock=False,
    ):
        super(OccHead, self).__init__()

        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.use_deblock = use_deblock
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = MODELS.build(dict(type='CustomFocalLoss'))
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()
        self.Z = occ_size[2]

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        
        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i]
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_3d_cfg, in_channels=mid_channel // self.Z, 
                        out_channels=mid_channel // self.Z, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_3d_cfg, mid_channel // self.Z)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_3d_cfg, in_channels=mid_channel // self.Z, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = MODELS.build(dict(type='DiceLoss', loss_weight=2))

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(humanoid_industry_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel) / out_channel  # FIXME hardcode 17

        if self.use_deblock:
            upsample_cfg = dict(type='deconv2d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=self.in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))


        self.empty_idx = empty_idx
    
    def forward(self, voxel_feats):
        occ_feats = []
        output = {}

        if self.use_deblock:
            x0 = self.deblock(voxel_feats[0])
            occ_feats.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            x = occ_conv(feats)
            occ_feats.append(x)

        B, C, H, W = occ_feats[0].shape
        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(occ_feats[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([B, self.num_point_sampling_feat, 1, 1],).to(occ_feats[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        for feats, weights in zip(occ_feats, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W], mode='bilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
            
        out_voxel_feats = out_voxel_feats.reshape(B, -1, self.Z, H, W).permute(0, 1, 3, 4, 2)
        output['out_voxel_feats'] = out_voxel_feats

        out_voxels = self.occ_pred_conv(out_voxel_feats)

        output['out_voxels'] = out_voxels

        return output
    
    def forward_train(self, voxel_feats, gt_occupancy):
        output = self.forward(voxel_feats)
        output_voxels = output['out_voxels']
        loss = dict()
        loss.update(self.loss(target_voxels=gt_occupancy, output_voxels=output_voxels, tag=0))

        return loss, output_voxels


    def loss(self, output_voxels, target_voxels, tag):
        # output_voxels = torch.log(output_voxels * 0) + output_voxels/0 # debug !!!!!!!!

        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        if self.use_focal_loss:
            loss_dict[f'losses/loss_voxel_ce_{tag}'] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        else:
            loss_dict[f'losses/loss_voxel_ce_{tag}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        loss_dict[f'losses/loss_voxel_sem_scal_{tag}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict[f'losses/loss_voxel_geo_scal_{tag}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict[f'losses/loss_voxel_lovasz_{tag}'] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


        if self.use_dice_loss:
            visible_mask = target_voxels != 255
            visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
            visible_target_voxels = target_voxels[visible_mask]
            visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
            loss_dict[f'losses/loss_voxel_dice_{tag}'] = self.dice_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict
    

@MODELS.register_module()
class OccHead3D(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        soft_weights=False,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        occ_size=None,
        empty_idx=0,
        balance_cls_weight=True,
        use_focal_loss=False,
        use_dice_loss=False,
        use_deblock=True,
    ):
        super(OccHead3D, self).__init__()

        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.use_deblock = use_deblock
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = MODELS.build(dict(type='CustomFocalLoss'))
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()
        self.Z = occ_size[2]

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        
        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                        out_channels=mid_channel//2, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = MODELS.build(dict(type='DiceLoss', loss_weight=2))

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel) / out_channel  # FIXME hardcode 17

        if self.use_deblock:
            upsample_cfg = dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=self.in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))


        self.empty_idx = empty_idx
    
    def forward(self, voxel_feats):
        output_occs = []
        output = {}

        if self.use_deblock:
            x0 = self.deblock(voxel_feats[0])
            output_occs.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            x = occ_conv(feats)
            output_occs.append(x)
        B, C, H, W, D = output_occs[0].shape

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([B, self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        
        output['out_voxel_feats'] = out_voxel_feats

        out_voxel = self.occ_pred_conv(out_voxel_feats)

        output['out_voxels'] = out_voxel

        return output
    
    def forward_train(self, voxel_feats, gt_occupancy):
        output = self.forward(voxel_feats)
        output_voxels = output['out_voxels']
        loss = self.loss(target_voxels=gt_occupancy, output_voxels=output_voxels)

        return loss, output_voxels

    def loss(self, output_voxels, target_voxels):
        # output_voxels = torch.log(output_voxels * 0) + output_voxels/0 # debug !!!!!!!!

        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        if self.use_focal_loss:
            loss_dict['losses/loss_voxel_ce'] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        else:
            loss_dict['losses/loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        loss_dict['losses/loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['losses/loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['losses/loss_voxel_lovasz'] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


        if self.use_dice_loss:
            visible_mask = target_voxels != 255
            visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
            visible_target_voxels = target_voxels[visible_mask]
            visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
            loss_dict['losses/loss_voxel_dice'] = self.dice_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict