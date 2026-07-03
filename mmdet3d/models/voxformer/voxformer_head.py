# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# VoxFormer-style occupancy prediction head.
#
# Key ideas from VoxFormer (CVPR 2023):
#   1. Depth-based voxel proposal: predict which voxels are occupied
#   2. Proposal-guided feature weighting: focus on occupied regions
#   3. 3D CNN completion: dense prediction from sparse proposals (like LMSCNet)
#   4. Per-voxel classification with CE + geo_scal + sem_scal losses
#
# This implementation adapts VoxFormer's ideas to work with the BEVDet/COTR
# feature extraction pipeline, making it easy to train and get good results.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet3d.registry import MODELS


# ============================================================================
# Loss functions (adapted from VoxFormer's ssc_loss.py)
# ============================================================================
def CE_ssc_loss(pred, target, class_weights, ignore_index=255):
    """Cross-entropy loss for semantic scene completion."""
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    return criterion(pred, target.long())


def geo_scal_loss(pred, ssc_target, ignore_index=255):
    """Geometric scaling loss: precision + recall + specificity for occupancy."""
    pred = F.softmax(pred, dim=1)
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    mask = ssc_target != ignore_index
    nonempty_target = (ssc_target != 0)[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-6
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = (intersection / (nonempty_probs.sum() + eps)).clamp(eps, 1 - eps)
    recall = (intersection / (nonempty_target.sum() + eps)).clamp(eps, 1 - eps)
    spec = (((1 - nonempty_target) * empty_probs).sum() / ((1 - nonempty_target).sum() + eps)).clamp(eps, 1 - eps)

    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, ignore_index=255):
    """Semantic scaling loss: per-class precision + recall + specificity."""
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index
    n_classes = pred.shape[1]
    eps = 1e-6

    for i in range(n_classes):
        p = pred[:, i, :, :, :][mask]
        target = ssc_target[mask]
        completion_target = (target == i).float()

        if completion_target.sum() > 0:
            count += 1.0
            nominator = (p * completion_target).sum()
            loss_class = 0

            if p.sum() > 0:
                precision = (nominator / (p.sum() + eps)).clamp(eps, 1 - eps)
                loss_class += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if completion_target.sum() > 0:
                recall = (nominator / (completion_target.sum() + eps)).clamp(eps, 1 - eps)
                loss_class += F.binary_cross_entropy(recall, torch.ones_like(recall))
            if (1 - completion_target).sum() > 0:
                spec = (((1 - p) * (1 - completion_target)).sum() / ((1 - completion_target).sum() + eps)).clamp(eps, 1 - eps)
                loss_class += F.binary_cross_entropy(spec, torch.ones_like(spec))
            loss += loss_class

    return loss / max(count, 1)


# ============================================================================
# Network building blocks
# ============================================================================
class ConvBlock3D(nn.Module):
    """Double 3D convolution block with BN and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPBlock3D(nn.Module):
    """ASPP-like multi-scale 3D context aggregation (from LMSCNet/VoxFormer)."""
    def __init__(self, channels, dilations=[1, 2, 3]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm3d(channels),
            )
            for d in dilations
        ])

    def forward(self, x):
        out = sum(branch(x) for branch in self.branches)
        return F.relu(out + x)


class CompletionNet(nn.Module):
    """3D Completion Network (VoxFormer's core completion component).

    Upsamples features from encoder resolution to full occ resolution
    and produces dense per-voxel predictions.

    Architecture inspired by VoxFormer's Header + LMSCNet's SegmentationHead.
    """

    def __init__(self, in_channels, num_classes, mid_channels=64):
        super().__init__()

        # Process at encoder resolution (e.g. 50x50)
        self.enc = ConvBlock3D(in_channels, mid_channels)

        # Upsample 2x in XY (50→100)
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.dec1 = ConvBlock3D(mid_channels, mid_channels // 2)

        # Upsample 2x in XY (100→200)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(mid_channels // 2, mid_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(mid_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = ConvBlock3D(mid_channels // 2, mid_channels // 4)

        # Multi-scale context aggregation
        self.aspp = ASPPBlock3D(mid_channels // 4, dilations=[1, 2, 3])

        # Per-voxel classification
        self.classify = nn.Conv3d(mid_channels // 4, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, C, Z, X_enc, Y_enc] features at encoder resolution
        Returns:
            [B, num_classes, Z, X, Y] logits at full resolution
        """
        x = self.enc(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.aspp(x)
        return self.classify(x)


# ============================================================================
# VoxFormer Occupancy Head
# ============================================================================
@MODELS.register_module()
class VoxFormerOccHead(nn.Module):
    """VoxFormer-style occupancy prediction head.

    Pipeline:
        3D features → Proposal Prediction → Proposal-Guided Enhancement
                     → 3D Completion Network → Per-Voxel Classification

    This captures VoxFormer's key ideas:
    - Stage 1 (proposal): predict binary occupancy (occupied vs free)
    - Stage 2 (completion): dense semantic prediction guided by proposals

    Args:
        in_channels: Input feature channels from BEV encoder (default: 32)
        embed_dims: Intermediate feature dimensions (default: 64)
        num_classes: Number of semantic classes including free (default: 21)
        mid_channels: Channels in completion network (default: 64)
        CE_ssc_loss: Use cross-entropy loss (default: True)
        geo_scal_loss: Use geometric scaling loss (default: True)
        sem_scal_loss: Use semantic scaling loss (default: True)
        loss_ce_weight: Weight for CE loss (default: 10.0)
        loss_geo_weight: Weight for geo_scal loss (default: 1.0)
        loss_sem_weight: Weight for sem_scal loss (default: 1.0)
        loss_proposal_weight: Weight for proposal loss (default: 1.0)
        empty_class_weight: Class weight for empty/free class (default: 0.05)
    """

    def __init__(
        self,
        in_channels=32,
        embed_dims=64,
        num_classes=21,
        mid_channels=64,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        loss_ce_weight=10.0,
        loss_geo_weight=1.0,
        loss_sem_weight=1.0,
        loss_proposal_weight=1.0,
        empty_class_weight=0.05,
        train_cfg=None,
        test_cfg=None,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.CE_ssc_loss_flag = CE_ssc_loss
        self.geo_scal_loss_flag = geo_scal_loss
        self.sem_scal_loss_flag = sem_scal_loss
        self.loss_ce_weight = loss_ce_weight
        self.loss_geo_weight = loss_geo_weight
        self.loss_sem_weight = loss_sem_weight
        self.loss_proposal_weight = loss_proposal_weight

        # ---- Stage 1: Binary Occupancy Proposal ----
        # Predicts which voxels are occupied (VoxFormer's query proposal equivalent)
        self.proposal_net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 1, 1),
        )

        # ---- Proposal-Guided Feature Enhancement ----
        # Fuses proposal information into features (like VoxFormer's mask token mechanism)
        self.feat_enhance = nn.Sequential(
            nn.Conv3d(in_channels + 1, embed_dims, 3, padding=1, bias=False),
            nn.BatchNorm3d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dims, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        # ---- Stage 2: 3D Completion Network ----
        # Dense prediction from proposal-guided features (VoxFormer's Header/LMSCNet)
        self.completion = CompletionNet(in_channels, num_classes, mid_channels)

        # ---- Class weights for CE loss ----
        weights = torch.ones(num_classes)
        weights[0] = empty_class_weight  # Free/empty class gets lower weight
        self.register_buffer('class_weights', weights)

    def forward(self, img_feats, img_metas=None, cam_params=None):
        """
        Args:
            img_feats: [x, feats, mlvl_feats] from BEV encoder
                x: [B, C, Z, X_enc, Y_enc] main 3D feature
        Returns:
            dict with 'ssc_logit' at full resolution, 'proposal_logit' at encoder resolution
        """
        x = img_feats[0]  # [B, C, Z, X_enc, Y_enc]

        # Stage 1: Proposal prediction
        proposal_logit = self.proposal_net(x)  # [B, 1, Z, X_enc, Y_enc]
        proposal_mask = torch.sigmoid(proposal_logit)

        # Proposal-guided feature enhancement
        x_cat = torch.cat([x, proposal_mask], dim=1)  # [B, C+1, Z, X_enc, Y_enc]
        x_enhanced = self.feat_enhance(x_cat) + x  # Residual

        # Soft proposal weighting (occupied regions get boosted, free regions dampened)
        x_weighted = x_enhanced * (0.5 + proposal_mask)

        # Stage 2: 3D Completion (upsample + dense prediction)
        ssc_logit = self.completion(x_weighted)  # [B, num_classes, Z, X_full, Y_full]

        # Permute: [B, C, Z, X, Y] → [B, C, X, Y, Z] to match GT layout [B, X, Y, Z]
        ssc_logit = ssc_logit.permute(0, 1, 3, 4, 2).contiguous()

        return {
            'ssc_logit': ssc_logit,        # [B, num_classes, X, Y, Z]
            'proposal_logit': proposal_logit,  # [B, 1, Z, X_enc, Y_enc]
        }

    def loss(self, outs, voxel_semantics, **kwargs):
        """
        Args:
            outs: dict from forward()
            voxel_semantics: [B, X, Y, Z] GT labels (0=free, 1~20=classes, 255=ignore)
        Returns:
            dict of losses
        """
        ssc_pred = outs['ssc_logit']           # [B, num_classes, X, Y, Z]
        proposal_logit = outs['proposal_logit'] # [B, 1, Z, X_enc, Y_enc]
        target = voxel_semantics                # [B, X, Y, Z]

        losses = {}

        # ---- Semantic losses at full resolution ----
        if self.CE_ssc_loss_flag:
            losses['loss_ce'] = self.loss_ce_weight * CE_ssc_loss(
                ssc_pred, target, self.class_weights
            )

        if self.geo_scal_loss_flag:
            losses['loss_geo_scal'] = self.loss_geo_weight * geo_scal_loss(ssc_pred, target)

        if self.sem_scal_loss_flag:
            losses['loss_sem_scal'] = self.loss_sem_weight * sem_scal_loss(ssc_pred, target)

        # ---- Proposal loss at encoder resolution ----
        # Downsample GT to encoder resolution for proposal supervision
        enc_shape = proposal_logit.shape[2:]  # (Z, X_enc, Y_enc)
        # target [B, X, Y, Z] → [B, 1, Z, X, Y] for interpolation
        target_for_down = target.float().unsqueeze(1).permute(0, 1, 4, 2, 3)  # [B, 1, Z, X, Y]
        target_down = F.interpolate(target_for_down, size=enc_shape, mode='nearest')  # [B, 1, Z, Xe, Ye]
        target_down = target_down[:, 0]  # [B, Z, Xe, Ye]

        proposal_target = (target_down > 0) & (target_down < 255)  # Occupied = True
        valid_mask = target_down != 255

        if valid_mask.sum() > 0:
            losses['loss_proposal'] = self.loss_proposal_weight * F.binary_cross_entropy_with_logits(
                proposal_logit[:, 0][valid_mask],
                proposal_target[valid_mask].float(),
            )

        return losses

    def predict(self, img_feats, img_metas=None, cam_params=None):
        """
        Returns:
            occ_pred: [B, X, Y, Z] predicted class labels
        """
        outs = self.forward(img_feats, img_metas, cam_params)
        return outs['ssc_logit'].argmax(dim=1)  # [B, X, Y, Z]