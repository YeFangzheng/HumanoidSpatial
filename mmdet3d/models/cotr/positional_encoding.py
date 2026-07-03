# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class CustomLearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=256,
                 col_num_embed=256,
                 tub_num_embed=32,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        
        super(CustomLearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats[0])
        self.col_embed = nn.Embedding(col_num_embed, num_feats[1])
        self.tub_embed = nn.Embedding(tub_num_embed, num_feats[2])
        
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.tub_num_embed = tub_num_embed

    def forward(self, mask, stride):
        """Forward function for `LearnedPositionalEncoding3D`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        X, Y, Z = mask.shape[-3:]
        
        x = torch.arange(0, X, step=stride, device=mask.device)
        y = torch.arange(0, Y, step=stride, device=mask.device)
        z = torch.arange(0, Z, step=stride, device=mask.device)
        
        x_embed = self.row_embed(x).view(X, 1, 1, -1).expand(X, Y, Z, self.num_feats[0])
        y_embed = self.col_embed(y).view(1, Y, 1, -1).expand(X, Y, Z, self.num_feats[1])
        z_embed = self.tub_embed(z).view(1, 1, Z, -1).expand(X, Y, Z, self.num_feats[2])

        # [X, Y, Z, num_feat * 3] ==> [C, X, Y, Z] ==> [1, C, X, Y, Z] ==> [B, C, X, Y, Z]
        pos = torch.cat((x_embed, y_embed, z_embed), dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1, 1)
        
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        
        return repr_str