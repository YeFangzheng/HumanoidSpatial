from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import math
from copy import deepcopy
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from . import OccMetric
from mmdet3d.registry import METRICS
from mmdet3d.ops.dvr import dvr_ext

@METRICS.register_module()
class RayMetric(OccMetric):
    def __init__(self, ignore_index: int = 255, **kwargs):
        super(RayMetric, self).__init__(**kwargs)
        self.ignore_index = int(ignore_index)

        self.lidar_rays = torch.from_numpy(self.generate_lidar_rays())
        self.pcd_pred_list = []
        self.pcd_gt_list = []

    def generate_lidar_rays(self):
        # prepare lidar ray angles
        pitch_angles = []
        for k in range(10):
            angle = math.pi / 2 - math.atan(k + 1)
            pitch_angles.append(-angle)
        
        # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
        while pitch_angles[-1] < 0.21:
            delta = pitch_angles[-1] - pitch_angles[-2]
            pitch_angles.append(pitch_angles[-1] + delta)

        lidar_rays = []
        for pitch_angle in pitch_angles:
            for azimuth_angle in np.arange(0, 360, 1):
                azimuth_angle = np.deg2rad(azimuth_angle)

                x = np.cos(pitch_angle) * np.cos(azimuth_angle)
                y = np.cos(pitch_angle) * np.sin(azimuth_angle)
                z = np.sin(pitch_angle)

                lidar_rays.append((x, y, z))

        return np.array(lidar_rays, dtype=np.float32)

    def get_rendered_pcds(self, origin, points, tindex, pred_dist):
        pcds = []
        
        for t in range(len(origin)):
            mask = (tindex == t)
            # skip the ones with no data
            if not mask.any():
                continue
            _pts = points[mask, :3]
            # use ground truth lidar points for the raycasting direction
            v = _pts - origin[t][None, :]
            d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
            pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
            pcds.append(torch.from_numpy(pred_pts))
            
        return pcds

    def process_one_sample(self, sem_pred, lidar_rays, output_origin, instance_pred=None, occ_class_names=None):
        # lidar origin in ego coordinate
        # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
        T = output_origin.shape[0]
        pred_pcds_t = []

        # free_id = len(occ_class_names) - 1
        # In this codebase, `free` is class 0.
        # IMPORTANT: ignore labels (default 255) must not be treated as occupied,
        # otherwise RayIoU will collapse when GT contains ignored voxels.
        free_id = 0
        occ_pred = deepcopy(sem_pred)
        # Map ignore label to free for ray casting / label sampling.
        # Only apply if ignore_index is present.
        if self.ignore_index is not None:
            occ_pred[sem_pred == self.ignore_index] = free_id
        occ_pred[sem_pred > free_id] = 1
        occ_pred[sem_pred == free_id] = 0
        occ_pred = occ_pred.permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()

        offset = torch.Tensor(self.point_cloud_range[:3])[None, None, :]
        scaler = torch.Tensor(self.occupancy_size)[None, None, :]

        lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
        
        for t in range(T): 
            lidar_origin = output_origin[None, t:t+1, :]  # [1, 1, 3]
            lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

            output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
            output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
            output_tindex_render = lidar_tindex  # [1, N], all zeros

            with torch.no_grad():
                pred_dist, _, coord_index = dvr_ext.render_forward(
                    occ_pred.cuda(),
                    output_origin_render.cuda(),
                    output_points_render.cuda(),
                    output_tindex_render.cuda(),
                    [1, 24, 200, 200],
                    "test"
                )
                pred_dist *= self.occupancy_size[0]

            pred_pcds = self.get_rendered_pcds(
                lidar_origin[0].cpu().numpy(),
                lidar_endpts[0].cpu().numpy(),
                lidar_tindex[0].cpu().numpy(),
                pred_dist[0].cpu().numpy()
            )
            coord_index = coord_index[0, :, :].int().cpu()  # [N, 3]

            pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]        
            pred_dist = pred_dist[0, :, None].cpu()

            if instance_pred is not None:
                pred_instance = instance_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
                pred_pcds = torch.cat([pred_label.float(), pred_instance.float(), pred_dist], dim=-1)
            else:
                pred_pcds = torch.cat([pred_label.float(), pred_dist], dim=-1)

            pred_pcds_t.append(pred_pcds)

        pred_pcds_t = torch.cat(pred_pcds_t, dim=0)
    
        return pred_pcds_t.numpy()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            semantics_pred = data_sample['pred_occupancy'].cpu()
            semantics_gt = data_sample['gt_occupancy'].cpu()
            lidar_origins = data_sample['lidar_origins']
            if self.use_image_mask:
                visible_mask = data_sample['visible_mask'].cpu().numpy()

        self.cnt += 1
        if len(semantics_pred.shape) == 4 or len(semantics_pred.shape) == 2:
            semantics_pred = semantics_pred.argmax(-1)

        # generate lidar rays
        lidar_rays = self.lidar_rays

        pcd_pred = self.process_one_sample(semantics_pred, lidar_rays, lidar_origins, occ_class_names=self.class_names)
        pcd_gt = self.process_one_sample(semantics_gt, lidar_rays, lidar_origins, occ_class_names=self.class_names)

        # evalute on non-free rays
        # drop free rays and ignore-label rays
        gt_label = pcd_gt[:, 0].astype(np.int32)
        if self.ignore_index is not None:
            valid_mask = (gt_label != 0) & (gt_label != self.ignore_index)
        else:
            valid_mask = (gt_label != 0)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        self.pcd_pred_list.append(pcd_pred)
        self.pcd_gt_list.append(pcd_gt)

        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[visible_mask]
            masked_semantics_pred = semantics_pred[visible_mask]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        _hist = self.compute_mIoU(masked_semantics_pred.numpy(), masked_semantics_gt.numpy(), self.num_classes)
        self.hist += _hist

    def calc_rayiou(self, pcd_pred_list, pcd_gt_list, occ_class_names):
        # thresholds = [1, 2, 4]
        thresholds = [0.1, 0.2, 0.5]

        gt_cnt = np.zeros([len(occ_class_names)])
        pred_cnt = np.zeros([len(occ_class_names)])
        tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

        for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
            for j, threshold in enumerate(thresholds):
                # L1
                depth_pred = pcd_pred[:, 1]
                depth_gt = pcd_gt[:, 1]
                l1_error = np.abs(depth_pred - depth_gt)
                tp_dist_mask = (l1_error < threshold)
                
                for i, cls in enumerate(occ_class_names):
                    cls_id = occ_class_names.index(cls)
                    cls_mask_pred = (pcd_pred[:, 0] == cls_id)
                    cls_mask_gt = (pcd_gt[:, 0] == cls_id)

                    gt_cnt_i = cls_mask_gt.sum()
                    pred_cnt_i = cls_mask_pred.sum()
                    if j == 0:
                        gt_cnt[i] += gt_cnt_i
                        pred_cnt[i] += pred_cnt_i

                    tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                    tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                    tp_cnt[j][i] += tp_mask.sum()
        
        iou_list = []
        for j, threshold in enumerate(thresholds):
            denom = (gt_cnt + pred_cnt - tp_cnt[j])
            # avoid RuntimeWarning when denom == 0 (class absent in both gt/pred)
            iou = np.divide(
                tp_cnt[j],
                denom,
                out=np.full_like(tp_cnt[j], np.nan, dtype=np.float64),
                where=denom != 0,
            )
            iou_list.append(iou[1:])

        return iou_list

    def compute_metrics(self):
        res = {}
        iou_list = self.calc_rayiou(self.pcd_pred_list, self.pcd_gt_list, self.class_names)
        rayiou = np.nanmean(iou_list)
        rayiou_0 = np.nanmean(iou_list[0])
        rayiou_1 = np.nanmean(iou_list[1])
        rayiou_2 = np.nanmean(iou_list[2])

        print(f'===> RayIoU: ' + str(round(rayiou * 100, 2)))
        print(f'===> RayIoU_0: ' + str(round(rayiou_0 * 100, 2)))
        print(f'===> RayIoU_1: ' + str(round(rayiou_1 * 100, 2)))
        print(f'===> RayIoU_2: ' + str(round(rayiou_2 * 100, 2)))
        res['rayiou'] = rayiou
        res['rayiou_0'] = rayiou_0
        res['rayiou_1'] = rayiou_1
        res['rayiou_2'] = rayiou_2

        mIoU = self.per_class_iu(self.hist)
        
        # 排除 IoU 为 0 的类后计算 Overall mIoU
        class_iou = mIoU[1:self.num_classes]
        # 保留非 NaN 且不为 0 的 IoU 值
        valid_iou = class_iou[(~np.isnan(class_iou)) & (class_iou != 0)]
        if len(valid_iou) > 0:
            overall_iou = round(np.mean(valid_iou) * 100, 2)
        else:
            overall_iou = 0.0
        
        print(f'===> mIoU of {self.cnt} samples: ' + str(overall_iou))
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(1, self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            res[self.class_names[ind_class]] = round(mIoU[ind_class] * 100, 2)

        res['Overall'] = overall_iou
        return res
    
    def evaluate(self, size):
        if is_main_process():
            _metrics = self.compute_metrics()  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]