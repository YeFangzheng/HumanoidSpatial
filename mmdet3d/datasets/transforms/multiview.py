from typing import Optional
import numpy as np
import torch
import mmengine
from mmcv.transforms import BaseTransform, Compose

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type


@TRANSFORMS.register_module()
class LoadOccupancyScannet(object):
    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        occ_filename = results['ann_info']['occ_filename']
        mask_filename = results['ann_info']['mask_filename']
        if occ_filename is None:
            gt_occ = np.zeros((0, 4), dtype=np.int64)
        else:
            gt_occ = np.load(occ_filename)
            for i in range(gt_occ.shape[0]):
                cls_id = self.occ_label_mapping[gt_occ[i][3]]
                if cls_id < 0:
                    cls_id = 255
                gt_occ[i][3] = cls_id

        if mask_filename is None:
            visible_occupancy_masks = [
                [] for i in range(len(results['images']))
            ]
        else:
            visible_occupancy_masks = []
            occ_masks = mmengine.load(mask_filename)
            for i in range(len(results['images'])):
                visible_occupancy_masks.append(
                    occ_masks[i]['visible_occupancy'])

        occupancy = torch.tensor(gt_occ)

        # to BEVDet format
        occupancy = occupancy.permute(2, 0, 1)
        occupancy = torch.rot90(occupancy, 1, [1, 2])
        occupancy = torch.flip(occupancy, [1])
        occupancy = occupancy.permute(1, 2, 0)

        for class_ in self.ignore_classes:
            occupancy[occupancy==class_] = 255

        results['gt_occupancy'] = occupancy

        return results


@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """Load a depth image from file.

    Required Keys:

    - depth_img_path

    Modified Keys:

    - depth_img
    - depth_img_shape

    Args:
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.imdecode_backend = imdecode_backend

        self.backend_args = None
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['depth_img_path']
        depth_shift = results['depth_shift']

        try:
            depth_img_bytes = mmengine.fileio.get(
                filename, backend_args=self.backend_args)
            depth_img = mmcv.imfrombytes(depth_img_bytes,
                                         flag='unchanged',
                                         backend=self.imdecode_backend).astype(
                                             np.float32) / depth_shift
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['depth_img'] = depth_img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.backend_args is not None:
            repr_str += f'backend_args={self.backend_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    """Multiview data processing pipeline.

    The transform steps are as follows:

        1. Select frames.
        2. Re-ororganize the selected data structure.
        3. Apply transforms for each selected frame.
        4. Concatenate data to form a batch.

    Args:
        transforms (list[dict | callable]):
            The transforms to be applied to each select frame.
        n_images (int): Number of frames selected per scene.
        ordered (bool): Whether to put these frames in order.
            Defaults to False.
    """

    def __init__(self, transforms, n_images, ordered=False):
        super().__init__()
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.ordered = ordered

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        imgs = []
        img_paths = []
        points = []
        intrinsics = []
        extrinsics = []
        ids = np.arange(len(results['img_path']))
        replace = True if self.n_images > len(ids) else False
        if self.ordered:
            step = (len(ids) - 1) // (self.n_images - 1
                                      )  # TODO: BUG, fix from branch fbocc
            if step > 0:
                ids = ids[::step]
                # sometimes can not get the accurate n_images in this way
                # then take the first n_images one
                ids = ids[:self.n_images]
            else:  # the number of images < pre-set n_images
                # randomly select n_images ids to enable batch-wise inference
                # In practice, can directly use the original ids to avoid
                # redundant computation
                ids = np.random.choice(ids, self.n_images, replace=replace)
        else:
            ids = np.random.choice(ids, self.n_images, replace=replace)
        for i in ids.tolist():
            _results = dict()
            _results['img_path'] = results['img_path'][i]
            if 'depth_img_path' in results:
                _results['depth_img_path'] = results['depth_img_path'][i]
                if isinstance(results['depth_cam2img'], list):
                    _results['depth_cam2img'] = results['depth_cam2img'][i]
                    _results['cam2img'] = results['depth2img']['intrinsic'][i]
                else:
                    _results['depth_cam2img'] = results['depth_cam2img']
                    _results['cam2img'] = results['cam2img']
                _results['depth_shift'] = results['depth_shift']
            _results = self.transforms(_results)
            if 'depth_shift' in _results:
                _results.pop('depth_shift')
            if 'img' in _results:
                imgs.append(_results['img'])
                img_paths.append(_results['img_path'])
            if 'points' in _results:
                points.append(_results['points'])
            if isinstance(results['depth2img']['intrinsic'], list):
                intrinsics.append(results['depth2img']['intrinsic'][i])
            else:
                intrinsics.append(results['depth2img']['intrinsic'])
            extrinsics.append(results['depth2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'points', 'img_path']:
                results[key] = _results[key]
        if len(imgs):
            results['img'] = imgs
            results['img_path'] = img_paths
        if len(points):
            results['points'] = points
        if 'visible_instance_masks' in results:
            results['visible_instance_masks'] = [
                results['visible_instance_masks'][i] for i in ids
            ]
        if 'visible_occupancy_masks' in results:
            results['visible_occupancy_masks'] = [
                results['visible_occupancy_masks'][i] for i in ids
            ]
        results['depth2img']['intrinsic'] = intrinsics
        results['depth2img']['extrinsic'] = extrinsics

        return results


@TRANSFORMS.register_module()
class AggregateMultiViewPoints(BaseTransform):
    """Aggregate points from each frame together.

    The transform steps are as follows:

        1. Collect points from each frame.
        2. Transform points from ego coordinate to global coordinate.
        3. Concatenate transformed points together.

    Args:
        coord_type (str): The type of output point coordinates.
            Defaults to 'DEPTH', corresponding to the global coordinate system
            in EmbodiedScan.
        save_slices (bool): Whether to save point index slices to convert all
            the points into the input for continuous 3D perception,
            corresponding to 1-N frames. Defaults to False.
    """

    def __init__(self,
                 coord_type: str = 'DEPTH',
                 save_slices: bool = False) -> None:
        super().__init__()
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.save_slices = save_slices

    def transform(self, results: dict) -> dict:
        # TODO: transforms should use numpy instead of torch
        points = results['points']
        global_points = []
        points_slice_indices = [0]
        for idx in range(len(points)):
            point = points[idx].tensor[..., :3]
            point = torch.cat([point, point.new_ones(point.shape[0], 1)],
                              dim=1)
            global2ego = torch.from_numpy(
                results['depth2img']['extrinsic'][idx]).to(point.device)
            global_point = (torch.linalg.solve(global2ego, point.transpose(
                0, 1))).transpose(0, 1)
            points[idx].tensor[:, :3] = global_point[:, :3]
            global_points.append(points[idx].tensor)
            if self.save_slices:
                points_slice_indices.append(points_slice_indices[-1] +
                                            len(points[idx].tensor))
        points = torch.cat(global_points)
        # a little hard code, to be improved
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points,
            points_dim=results['points'][0].points_dim,
            attribute_dims=results['points'][0].attribute_dims)
        results['points'] = points

        if self.save_slices:
            results['points_slice_indices'] = points_slice_indices

        return results


@TRANSFORMS.register_module()
class ConstructMultiSweeps(BaseTransform):
    """Construct N multi-view frames to 1-N continuous sweeps."""

    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        points = results['points']
        points_slice_indices = results['points_slice_indices']
        points_slice_indices = results['points_slice_indices']
        cumulated_points = points.tensor[
            points_slice_indices[0]:points_slice_indices[1]]
        batch_points = [cumulated_points]

        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        batch_gt_bboxes_3d = gt_bboxes_3d
        batch_gt_labels_3d = gt_labels_3d

        if 'visible_instance_masks' in results:
            visible_instance_masks = results['visible_instance_masks']
            visible_instance_ids = []
            for idx in range(len(visible_instance_masks)):
                visible_instance_ids.append(
                    set(
                        np.argwhere(np.array(
                            visible_instance_masks[idx])).flatten()))
            cumulated_ids = set(visible_instance_ids[0])
            indices = np.array(list(cumulated_ids), dtype=np.int32)
            batch_gt_bboxes_3d = [gt_bboxes_3d[indices]]
            batch_gt_labels_3d = [gt_labels_3d[indices]]

        if 'visible_occupancy_masks' in results:
            visible_occupancy_masks = results['visible_occupancy_masks']
            cumulated_masks = visible_occupancy_masks[0]
            batch_gt_occupancy_masks = [visible_occupancy_masks[0]]

        for idx in range(1, len(points_slice_indices) - 1):
            # construct sparse tensor and features
            start = points_slice_indices[idx]
            end = points_slice_indices[idx + 1]
            cumulated_points = torch.cat(
                [cumulated_points, points.tensor[start:end]])
            batch_points.append(cumulated_points)

            if 'visible_instance_masks' in results:
                cumulated_ids = cumulated_ids.union(visible_instance_ids[idx])
                indices = np.array(list(cumulated_ids), dtype=np.int32)
                batch_gt_bboxes_3d.append(gt_bboxes_3d[indices])
                batch_gt_labels_3d.append(gt_labels_3d[indices])

            if 'visible_occupancy_masks' in results:
                cumulated_masks = np.logical_or(cumulated_masks,
                                                visible_occupancy_masks[idx])
                batch_gt_occupancy_masks.append(cumulated_masks)

        results['points'] = batch_points

        if 'visible_instance_masks' in results:
            results['gt_bboxes_3d'] = batch_gt_bboxes_3d
            results['gt_labels_3d'] = batch_gt_labels_3d
            if 'eval_ann_info' in results:
                results['eval_ann_info']['gt_bboxes_3d'] = results[
                    'gt_bboxes_3d']
                results['eval_ann_info']['gt_labels_3d'] = results[
                    'gt_labels_3d']

        if 'visible_occupancy_masks' in results:
            results['gt_occupancy_masks'] = batch_gt_occupancy_masks
            if 'eval_ann_info' in results:
                results['eval_ann_info']['gt_occupancy_masks'] = results[
                    'gt_occupancy_masks']

        return results


@TRANSFORMS.register_module()
class ConstructMultiViewMasks:
    """Construct multi-view masks to only keep visible results.

    Only used for the occupancy prediction task temporarily.
    """

    def __call__(self, results):

        if 'visible_occupancy_masks' in results:
            visible_occupancy_masks = results['visible_occupancy_masks']
            cumulated_masks = visible_occupancy_masks[0]

        for idx in range(1, len(results['img']) - 1):
            if 'visible_occupancy_masks' in results:
                cumulated_masks = np.logical_or(cumulated_masks,
                                                visible_occupancy_masks[idx])

        if 'visible_occupancy_masks' in results:
            results['gt_occupancy_masks'] = cumulated_masks
            if 'eval_ann_info' in results:
                results['eval_ann_info']['gt_occupancy_masks'] = results[
                    'gt_occupancy_masks']

        return results
