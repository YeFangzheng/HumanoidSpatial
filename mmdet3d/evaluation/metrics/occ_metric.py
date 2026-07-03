from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from copy import deepcopy

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class OccMetric(BaseMetric):
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 class_names=None,
                 point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                 occupancy_size=[0.4, 0.4, 0.4],
                 use_lidar_mask=False,
                 use_image_mask=False,
                 min_d=-1,
                 max_d=100,
                 prefix=None,
                 collect_device='cpu',
                 ):
        self.default_prefix = 'mIoU'
        super(OccMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.class_names = class_names
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = point_cloud_range
        self.occupancy_size = occupancy_size
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        self.max_d = max_d
        self.min_d = min_d

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17
        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label
        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """

        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        denom = (hist.sum(1) + hist.sum(0) - np.diag(hist))
        # avoid RuntimeWarning when denom == 0 (class absent in both gt/pred)
        return np.divide(
            np.diag(hist),
            denom,
            out=np.full_like(np.diag(hist), np.nan, dtype=np.float64),
            where=denom != 0,
        )

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        # mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        # return round(np.nanmean(mIoUs) * 100, 2), hist
        return hist

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            semantics_pred = data_sample['pred_occupancy'].cpu().numpy()
            semantics_gt = data_sample['gt_occupancy'].cpu().numpy()
            if self.use_image_mask:
                visible_mask = data_sample['visible_mask'].cpu().numpy()

        self.cnt += 1
        if len(semantics_pred.shape) == 4:
            semantics_pred = semantics_pred.argmax(-1)

        # assert self.use_image_mask
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[visible_mask]
            masked_semantics_pred = semantics_pred[visible_mask]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
            
        _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

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


    def compute_metrics(self):
        res = {}
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[1:self.num_classes]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(1, self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            res[self.class_names[ind_class]] = round(mIoU[ind_class] * 100, 2)

        res['Overall'] =  round(np.nanmean(mIoU[1:self.num_classes]) * 100, 2)
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        return res