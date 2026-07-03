
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from mmdet3d.models.fusionocc.backward_projection.bevformer_utils.custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet3d.registry import MODELS
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer


@MODELS.register_module()
class OccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, return_intermediate=False, 
                 **kwargs):

        super(OccEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range

    def get_reference_points(self, H, W, Z, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        ref_3d = ref_3d[None]

        pc_range = self.pc_range
        ref_3d[..., 0:1] = ref_3d[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        return ref_3d # D, num_query, 3

    def point_sampling(self, reference_points, img_metas):
        device = reference_points.device
        cam2egos = torch.stack([meta['cam2ego'] for meta in img_metas], dim=0).to(device)
        intrins = torch.stack([meta['intrinsic'] for meta in img_metas], dim=0).to(device)
        post_trans = torch.stack([meta['post_trans'] for meta in img_metas], dim=0).to(device)
        bda = torch.stack([meta['bda'] for meta in img_metas], dim=0).to(device)

        rots = cam2egos[..., :3, :3]
        trans = cam2egos[..., :3, 3]
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = 768, 960 # HARDCODE
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)

        ego2cams = torch.inverse(cam2egos)
        rots = ego2cams[..., :3, :3]
        trans = ego2cams[..., :3, 3]
        reference_points = rots.view(B, N, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)) + trans.view(B, N, 1, 1, 3, 1)

        reference_points_cam = intrins.view(B, N, 1, 1, 3, 3).matmul(reference_points).squeeze(-1)

        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 4
            )
        reference_points_cam = post_trans[..., :3, :3].view(B, N, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans[..., :3, 3].view(B, N, 1, 1, 3) 
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > 0.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > 0.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
        reference_points_cam = reference_points_cam.permute(1, 0, 3, 2, 4) # N, B, num_query, D, 3
        mask = mask.permute(1, 0, 3, 2, 4).squeeze(-1)

        return reference_points_cam[..., 0:2], mask

    def forward(self,
                volume_query,
                key,
                value,
                *args,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            volume_query (Tensor): Input 3D volume query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = volume_query
        intermediate = []

        ref_3d = self.get_reference_points(
                    volume_h, volume_w, volume_z, bs=volume_query.size(1),  device=volume_query.device, dtype=volume_query.dtype)

        reference_points_cam, volume_mask = self.point_sampling(
            ref_3d, kwargs['img_metas'])

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        volume_query = volume_query.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                volume_query,
                key,
                value,
                *args,
                ref_3d=ref_3d,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                per_cam_mask_list=volume_mask,
                **kwargs)

            volume_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@MODELS.register_module()
class OccLayer(MyCustomBaseTransformerLayer):
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
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 embed_dims,
                 ffn_dropout=0.0,
                 operation_order=None,
                 conv_num=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(OccLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

        self.deblock = nn.ModuleList()
        conv_cfg=dict(type='Conv3d', bias=False)
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        for i in range(conv_num):
            conv_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            deblock = nn.Sequential(conv_layer,
                                    build_norm_layer(norm_cfg, embed_dims)[1],
                                    nn.ReLU(inplace=True))
            self.deblock.append(deblock)
        #assert len(operation_order) == 6
        #assert set(operation_order) == set(
        #    ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
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
            if layer == 'conv':
                bs = query.shape[0]
                identity = query
                query = query.reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)
                for i in range(len(self.deblock)):
                    query = self.deblock[i](query)
                query = query.permute(0, 4, 3, 2, 1).reshape(bs, volume_z*volume_h*volume_w, -1)
                query = query + identity
    
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
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
            

        return query