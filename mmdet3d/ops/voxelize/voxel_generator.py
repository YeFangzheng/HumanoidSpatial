# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.autograd import Function

class VoxelGenerator(Function):

    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            num_points: torch.Tensor,
            max_points: int,
            max_voxels: int,
            coors_range: Union[tuple, float],
            voxel_feature_num: int,
            voxel_size: Union[tuple, float],
            ) -> Union[Tuple[torch.Tensor], Tuple]:

            batch_size = points.shape[0]
            voxels = torch.zeros((batch_size, max_voxels, max_points, voxel_feature_num), dtype=torch.float32).to(points.device)
            coors = torch.zeros((batch_size, max_voxels, 4), dtype=torch.int32).to(points.device)
            voxel_num = torch.zeros([batch_size], dtype=torch.int32).to(points.device)
            return voxels, coors, voxel_num

    @staticmethod
    def symbolic(g, points, num_points, max_points, max_voxels, coors_range, voxel_feature_num, voxel_size):
        return g.op("VoxelGeneratorPlugin", 
                    points,
                    num_points,
                    max_num_points_per_voxel_i=max_points,
                    max_voxels_i=max_voxels,
                    point_cloud_range_f=coors_range,
                    voxel_feature_num_i=voxel_feature_num,
                    voxel_size_f=voxel_size,
                    outputs=3)