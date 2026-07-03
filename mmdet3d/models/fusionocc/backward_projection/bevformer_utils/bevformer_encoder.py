# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet3d.registry import MODELS
import numpy as np
import torch
from mmcv.utils import ext_loader
from mmengine.runner import autocast

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@MODELS.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, grid_config=None, data_config=None, return_intermediate=False, distortion=True,
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.distortion = distortion

    def get_reference_points(self, H, W, Z=8, dim='3d', bs=1, device='cuda', dtype=torch.float32):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':

            X = torch.arange(*self.x_bound, dtype=torch.float32) + self.x_bound[-1]/2
            Y = torch.arange(*self.y_bound, dtype=torch.float32) + self.y_bound[-1]/2
            Z = torch.arange(*self.z_bound, dtype=torch.float32) + self.z_bound[-1]/2
            Y, X, Z = torch.meshgrid([Y, X, Z])
            coords = torch.stack([X, Y, Z], dim=-1)
            coords = coords.to(dtype).to(device)
            # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)
            return coords

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    
    # @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, cam_params):
        cam2egos, intrins, distortions, post_trans, bda = cam_params
        rots = cam2egos[..., :3, :3]
        trans = cam2egos[..., :3, 3]
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.final_dim
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        # reference_points -= trans.view(B, N, 1, 1, 1, 3)
        # combine = rots.matmul(torch.inverse(intrins)).inverse()
        # reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)

        ego2cams = torch.inverse(cam2egos)
        rots = ego2cams[..., :3, :3]
        trans = ego2cams[..., :3, 3]
        reference_points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)) + trans.view(B, N, 1, 1, 1, 3, 1)
        if self.distortion:
            k1, k2, p1, p2, k3, k4, k5, k6 = torch.unbind(distortions.view(B, N, 1, 1, 1, 8, 1), dim=-2)
            x, y, z = torch.unbind(reference_points, dim=-2)
            x = x / z
            y = y / z
            r2 = x**2 + y**2
            max_r2 = 10.0
            r2[r2 > max_r2] = max_r2
            r4 = r2 * r2
            r6 = r4 * r2

            # 径向畸变系数 (增加高阶项)
            radial = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
            
            # 切向畸变
            xy = x * y
            tangential_x = 2 * p1 * xy + p2 * (r2 + 2 * x**2)
            tangential_y = p1 * (r2 + 2 * y**2) + 2 * p2 * xy
            
            # 应用畸变
            x_dist = x * radial + tangential_x
            y_dist = y * radial + tangential_y
            reference_points = torch.stack([x_dist*z, y_dist*z, z], dim=-2)

        reference_points_cam = intrins.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points).squeeze(-1)

        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        reference_points_cam = post_trans[..., :3, :3].view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans[..., :3, 3].view(B, N, 1, 1, 1, 3) 
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > 0.05) 
                 & (reference_points_cam[..., 0:1] < (1.0-0.05)) 
                 & (reference_points_cam[..., 1:2] > eps) 
                 & (reference_points_cam[..., 1:2] < (1.0-eps)))
        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 1).squeeze(-1)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]


    # @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                cam_params=None,
                pred_img_depth=None,
                bev_mask=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        with autocast('cuda', enabled=False):
            ref_3d = self.get_reference_points(
                bev_h, bev_w, self.pc_range[5]-self.pc_range[2], dim='3d', bs=bev_query.size(1),  device=bev_query.device)
            ref_2d = self.get_reference_points(
                bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

            ref_3d, reference_points_cam, per_cam_mask_list, bev_query_depth = self.point_sampling(
                ref_3d, cam_params)

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        for lid, layer in enumerate(self.layers):
           
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                prev_bev=prev_bev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                per_cam_mask_list=per_cam_mask_list,
                bev_mask=bev_mask,
                bev_query_depth=bev_query_depth,
                pred_img_depth=pred_img_depth,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@MODELS.register_module()
class BEVFormerEncoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default: 2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=512,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) in {2, 4, 6}
        # assert set(operation_order) in set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    # @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_mask=None,
                bev_query_depth=None,
                per_cam_mask_list=None,
                pred_img_depth=None, 
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    None,
                    None,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=bev_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_query_depth=bev_query_depth,
                    pred_img_depth=pred_img_depth,
                    bev_mask=bev_mask,
                    per_cam_mask_list=per_cam_mask_list,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

