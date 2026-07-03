from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet3d.registry import MODELS
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_attention
from mmengine.runner import autocast
from mmengine.model import BaseModule


@MODELS.register_module()
class BEVFormerEncoderONNX(TransformerLayerSequence):

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

        super(BEVFormerEncoderONNX, self).__init__(*args, **kwargs)
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
    
    def point_sampling(self, reference_points, cam_params):
        ego2cams, distortions, intrins, post_trans = cam_params
        rots = ego2cams[..., :3, :3]
        trans = ego2cams[..., :3, 3]
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.final_dim
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
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


        # import cv2 as cv
        # import matplotlib

        # cmap = matplotlib.colormaps["turbo_r"]
        # norm = matplotlib.colors.Normalize(
        #     vmin=0.5,
        #     vmax=5.0,
        # )

        # big_img = np.zeros((1536*2, 1920*3, 3), dtype=np.uint8)

        # for i in range(2):
        #     for j in range(3):
        #         k = i * 3 + j
        #         points_2d = reference_points_cam[k,..., :2][mask[k]].cpu().numpy()
        #         points_z = reference_points_cam[k, ..., 2][mask[k]].cpu().numpy()

        #         for p,z in zip(points_2d, points_z):
        #             cv.circle(big_img[1536 * i:1536 * (i + 1), 1920 * j:1920 * (j + 1)], (int(p[0] * 1920), int(p[1] * 1536)), 3, [c * 255 for c in cmap(norm(z))[:3]])
        # cv.imwrite(f'proj.jpg', big_img)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]


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
class SpatialCrossAttentionONNX(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttentionONNX, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
    
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                level_start_index=None,
                flag='encoder',
                bev_mask=None,
                per_cam_mask_list=None,                
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = [[] for _ in range(bs)]

        if bev_mask is not None:
            per_cam_mask_list_ = per_cam_mask_list & bev_mask[None, :, :, None]
        else:
            per_cam_mask_list_ = per_cam_mask_list

        max_len = 14000
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = per_cam_mask_list_[i][j].sum(-1).nonzero().squeeze(-1)
                indexes[j].append(index_query_per_img)
                if index_query_per_img.shape[0] > max_len:
                    index_query_per_img = index_query_per_img[:max_len]
                queries_rebatch[j, i, :index_query_per_img.shape[0]] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :index_query_per_img.shape[0]] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,\
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,\
                                            level_start_index=level_start_index
                                            ).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :index_query_per_img.shape[0]]

        count = per_cam_mask_list_.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual