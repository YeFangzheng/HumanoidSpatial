# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from mmdet3d.models.backbones.resnet import ConvModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 reverse=False,
                 size=(16, 50, 50)):
        super().__init__()
        self.reverse = reverse
        self.size = size
        if not reverse:
            self.up1 =  nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.up2 =  nn.Upsample(
                scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        old_x_8 = x_8
        old_x_16 = x_16
        # print(f"x_8.shape:{x_8.shape}")
        # print(f"x_16.shape:{x_16.shape}")
        # print(f"x_32.shape:{x_32.shape}")
        if not self.reverse:
            x_16 = self.up1(x_16)
            x_32 = self.up2(x_32)
        else:
            x_8 = F.interpolate(x_8, size=self.size,
                                 mode='trilinear', align_corners=True)
            x_16 = F.interpolate(x_16, size=self.size,
                                 mode='trilinear', align_corners=True)
            # x_32 = F.interpolate(x_32, size=(z, h, w),
            #                      mode='trilinear', align_corners=True)
        
        if x_32.shape[-3:] != x_8.shape[-3:]:
            x_32 = F.interpolate(x_32, size=x_8.shape[-3:], mode='trilinear')
        # print(f"x_8.shape:{x_8.shape}")
        # print(f"x_16.shape:{x_16.shape}")
        # print(f"x_32.shape:{x_32.shape}")
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        if self.reverse:
            return x, (old_x_8, old_x_16, x_32)
        return x, x