# ---------------------------------------------
#  Modified by Qihang Ma
# ---------------------------------------------

from mmdet3d.models.fusionocc.backward_projection.bevformer_utils.custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet3d.registry import MODELS
import numpy as np
import torch


@MODELS.register_module()
class OccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, pc_range=None, grid_config=None, data_config=None,
                 fix_bug=False, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(OccEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        
        if grid_config is not None:
            self.x_bound = grid_config['x']
            self.y_bound = grid_config['y']
            self.z_bound = grid_config['z']
        if data_config is not None:
            self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.fp16_enabled = False

    def get_reference_points(self,H, W, Z=8, bs=1, device='cuda', dtype=torch.float):
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
        X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1]/2
        Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1]/2
        Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1]/2
        Y, X, Z = torch.meshgrid([Y, X, Z])
        coords = torch.stack([X, Y, Z], dim=-1)
        coords = coords.to(dtype).to(device)
        # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)
        return coords

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
        ego2cams = torch.inverse(cam2egos)
        rots = ego2cams[..., :3, :3]
        trans = ego2cams[..., :3, 3]
        reference_points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)) + trans.view(B, N, 1, 1, 1, 3, 1)
        reference_points_cam = intrins.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        reference_points_cam = post_trans[..., :3, :3].view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans[..., :3, 3].view(B, N, 1, 1, 1, 3) 
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > 0.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > 0.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))

        B, N, W, H, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 4, 3, 2, 5).reshape(N, B, D*H*W, 3)
        reference_points_cam = reference_points_cam[:, :, :, None, :]   # shape: (num_cam,bs,z*h*w,num_level,2)
        # (B, N, W, H, D, 1) --> (N, B, D*H*W, 1)
        mask = mask.permute(1, 0, 4, 3, 2, 5).reshape(N, B, D*H*W, 1)

        return reference_points_cam[..., :2], mask

    def forward(self,
                occ_query,
                key,
                value,
                *args,
                occ_h=None,
                occ_w=None,
                occ_z=None,
                occ_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                cam_params=None,
                valid_ratios=None,
                prev_occ=None,
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

        output = occ_query
        intermediate = []

        ref_3d = self.get_reference_points(
            occ_h, occ_w, occ_z, bs=occ_query.size(1), 
            device=occ_query.device, dtype=occ_query.dtype
        )
        reference_points_cam, occ_mask = self.point_sampling(
            ref_3d, cam_params)
        
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        occ_query = occ_query.permute(1, 0, 2)
        occ_pos = occ_pos.permute(1, 0, 2)

        ref_3d = ref_3d.permute(3, 2, 1, 0).flatten(1).permute(1, 0) 
        ref_3d = ref_3d[None, None].repeat(occ_query.shape[0], 1, 1, 1) # [w, h, z, 3] -> [bs, num_level, z*h*w, 3]

        for lid, layer in enumerate(self.layers):
            output = layer(
                occ_query,
                key,
                value,
                *args,
                occ_pos=occ_pos,
                ref_3d=ref_3d, # [bs, num_level, z*h*w, 3]
                occ_h=occ_h,
                occ_w=occ_w,
                occ_z=occ_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam, # [num_cam, bs, num_level, 2]
                occ_mask=occ_mask,                         # [num_cam, bs, num_level, 1]
                prev_occ=prev_occ,
                **kwargs)

            occ_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@MODELS.register_module()
class OccFormerLayer(MyCustomBaseTransformerLayer):
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
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(OccFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                occ_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                occ_h=None,
                occ_w=None,
                occ_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_occ=None,
                occ_mask=None,
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
            # deformable self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity=query,
                    query_pos=occ_pos,
                    key_pos=occ_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_3d.permute(0, 2, 1, 3),
                    spatial_shapes=torch.tensor(
                        [[occ_h, occ_w, occ_z]], device=query.device),
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
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_mask=None,
                    per_cam_mask_list=occ_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
