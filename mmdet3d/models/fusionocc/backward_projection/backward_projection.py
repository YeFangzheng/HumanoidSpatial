# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class BackwardProjection(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 transformer=None,
                 positional_encoding=None,
                 pc_range=None,
                 bev_h=30,
                 bev_w=30):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
       
        self.positional_encoding = MODELS.build(
            positional_encoding)
        self.transformer = MODELS.build(transformer)
        self.embed_dims = self.transformer.embed_dims

        self._init_layers()

    def _init_layers(self):
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    # @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, lss_bev=None, cam_params=None, pred_img_depth=None, bev_mask=None, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        
        if lss_bev is not None:
            lss_bev = lss_bev.flatten(2).permute(2, 0, 1)
            bev_queries = bev_queries + lss_bev
        
        if bev_mask is not None:
            bev_mask = bev_mask.reshape(bs, -1)

        bev_pos = self.positional_encoding(bs, self.bev_h, self.bev_w, bev_queries.device).to(dtype)

        bev = self.transformer(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                cam_params=cam_params,
                pred_img_depth=pred_img_depth,
                prev_bev=None,
                bev_mask=bev_mask,
                **kwargs
            )

        bev = bev.permute(0, 2, 1).view(bs, -1, self.bev_h, self.bev_w).contiguous()


        return bev

