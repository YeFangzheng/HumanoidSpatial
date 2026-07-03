# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# VoxFormer-style occupancy prediction model.
#
# Architecture:
#   Backbone pipeline (inherited from COTR/BEVDetOcc):
#     ResNet50 → FPN → DepthNet → LSS View Transform → 3D ResNet → 3D FPN
#   Head (VoxFormer-style, new):
#     Binary Proposal → Proposal-Guided Enhancement → 3D Completion → Per-Voxel Classification

import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.cotr.cotr import COTR
from mmengine.runner import autocast


@MODELS.register_module()
class VoxFormerOcc(COTR):
    """VoxFormer-style occupancy prediction.

    Inherits COTR's feature extraction pipeline (backbone, depth, LSS, 3D encoder).
    Replaces COTR's MaskFormer head with VoxFormer-style per-voxel prediction:
      1. Binary occupancy proposal (which voxels are occupied?)
      2. Proposal-guided feature enhancement
      3. 3D CNN completion network (sparse → dense)
      4. Per-voxel classification with CE + scal losses
    """

    def __init__(self, occupancy_head=None, group_split=None, **kwargs):
        if group_split is None:
            group_split = []
        super().__init__(
            group_split=group_split,
            occupancy_head=occupancy_head,
            **kwargs
        )

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        device = batch_inputs_dict['imgs'].device
        img_metas = [item.metainfo for item in batch_data_samples]

        cam_params = []
        for key in ['cam2ego', 'intrinsic', 'distortion', 'post_trans', 'bda']:
            cam_params.append(
                torch.stack([meta[key] for meta in img_metas], dim=0).to(device)
            )

        img_feats, depth = self.extract_feat(batch_inputs_dict, img_metas=img_metas)

        losses = dict()

        # Depth loss (shared with COTR/BEVDet)
        gt_depth = torch.stack(
            [item.gt_pts_seg.depth for item in batch_data_samples], dim=0
        )
        loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
        if isinstance(loss_depth, dict):
            losses.update(loss_depth)
        elif isinstance(loss_depth, (list, tuple)):
            losses['loss_depth'] = loss_depth[0]
        else:
            losses['loss_depth'] = loss_depth

        # GT occupancy
        voxel_semantics = torch.stack(
            [item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0
        )

        # VoxFormer head forward + loss
        with autocast('cuda', enabled=False):
            outs = self.occupancy_head(img_feats, img_metas, cam_params)
            losses_occ = self.occupancy_head.loss(outs, voxel_semantics)
        losses.update(losses_occ)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        device = batch_inputs_dict['imgs'].device
        img_metas = [item.metainfo for item in batch_data_samples]

        cam_params = []
        for key in ['cam2ego', 'intrinsic', 'distortion', 'post_trans', 'bda']:
            cam_params.append(
                torch.stack([meta[key] for meta in img_metas], dim=0).to(device)
            )

        img_feats, depth = self.extract_feat(batch_inputs_dict, img_metas=img_metas)
        occ_pred = self.occupancy_head.predict(img_feats, img_metas, cam_params)

        gt_occ = torch.stack(
            [item.gt_pts_seg.occupancy for item in batch_data_samples], dim=0
        )
        lidar_origins = torch.stack(
            [meta['lidar_origins'] for meta in img_metas], dim=0
        )

        results = []
        for i in range(occ_pred.shape[0]):
            results.append(dict(
                pred_occupancy=occ_pred[i],
                gt_occupancy=gt_occ[i],
                lidar_origins=lidar_origins[i],
            ))
        return results