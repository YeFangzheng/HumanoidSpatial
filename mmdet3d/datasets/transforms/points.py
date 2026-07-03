from typing import List, Optional, Tuple, Union

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import points_cam2img, points_img2cam
from mmdet3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class ConvertRGBDToPoints(BaseTransform):
    """Convert depth map to point clouds.

    Args:
        coord_type (str): The type of point coordinates. Defaults to 'CAMERA'.
        use_color (bool): Whether to use color as additional features
            when converting the image to points. Generally speaking, if False,
            only return xyz points. Otherwise, return xyzrgb points.
            Defaults to False.
    """

    def __init__(self,
                 coord_type: str = 'CAMERA',
                 use_color: bool = False) -> None:
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.use_color = use_color

    def transform(self, input_dict: dict) -> dict:
        """Call function to normalize color of points.

        Args:
            input_dict (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        depth_img = input_dict['depth_img']
        depth_cam2img = input_dict['depth_cam2img']
        ws = np.arange(depth_img.shape[1])
        hs = np.arange(depth_img.shape[0])
        us, vs = np.meshgrid(ws, hs)
        grid = np.stack(
            [us.astype(np.float32),
             vs.astype(np.float32), depth_img], axis=-1).reshape(-1, 3)
        nonzero_indices = depth_img.reshape(-1).nonzero()[0]
        grid3d = points_img2cam(grid, depth_cam2img)
        points = grid3d[nonzero_indices]

        attribute_dims = None
        if self.use_color:
            img = input_dict['img']
            h, w = img.shape[0], img.shape[1]
            cam2img = input_dict['cam2img']
            points2d = np.round(points_cam2img(points,
                                               cam2img)).astype(np.int32)
            us = np.clip(points2d[:, 0], a_min=0, a_max=w - 1)
            vs = np.clip(points2d[:, 1], a_min=0, a_max=h - 1)
            rgb_points = img[vs, us]
            points = np.concatenate([points, rgb_points], axis=-1)

            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(points,
                              points_dim=points.shape[-1],
                              attribute_dims=attribute_dims)
        input_dict['points'] = points

        return input_dict
