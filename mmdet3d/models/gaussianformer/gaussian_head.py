import numpy as np
import torch, torch.nn as nn

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from .utils import get_rotation_matrix
from mmdet3d.models.fusionocc.losses import CE_ssc_loss, lovasz_softmax, humanoid_industry_frequencies


@MODELS.register_module()
class GaussianHead(BaseModule):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=21,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=0,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        combine_geosem=False,
        multi_loss_weights=None,
        balance_cls_weight=True,
        occ_sample_num: int | None = None,
        occ_sample_nonfree_ratio: float = 0.7,
        occ_sample_ignore_index: int | None = 255,
        occ_sample_free_id: int = 0,
        **kwargs,
    ):
        super().__init__(init_cfg)

        # local_aggregate*: use a tree where the ext was built (pip install -e), e.g.
        # mmdet3d/models/gaussianformer/localagg or .../GaussianFormer/model/head/localagg.
        # Do not prepend raw source-only paths: they shadow the install and break _C import.

        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
        if use_localaggprob:
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)
        
        self.combine_geosem = combine_geosem
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label
        self.occ_sample_num = None if occ_sample_num is None else int(occ_sample_num)
        self.occ_sample_nonfree_ratio = float(occ_sample_nonfree_ratio)
        self.occ_sample_ignore_index = None if occ_sample_ignore_index is None else int(occ_sample_ignore_index)
        self.occ_sample_free_id = int(occ_sample_free_id)

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

        self.loss_voxel_ce_weight = multi_loss_weights.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = multi_loss_weights.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = multi_loss_weights.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = multi_loss_weights.get('loss_voxel_lovasz_weight', 1.0)

        manual_class_weight = kwargs.get('manual_class_weight', None)
        if manual_class_weight is not None:
            self.class_weights = torch.tensor(manual_class_weight, dtype=torch.float)
        elif balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(humanoid_industry_frequencies[:num_classes] + 0.001))
        else:
            self.class_weights = torch.ones(num_classes)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)

        # Subsample occupancy supervision points only during training.
        # In val/test, `predict()` must reshape `final_occ` to full (H, W, D); subsampling
        # would shrink `final_occ` (e.g. 80k) and break reshape(-1, 200, 200, 24).
        sample_num = self.occ_sample_num if self.training else None
        if sample_num is None or sample_num <= 0 or sample_num >= gt_label.shape[1]:
            return gt_xyz, gt_label

        bs, n = gt_label.shape
        device = gt_label.device
        free_id = self.occ_sample_free_id
        ignore = self.occ_sample_ignore_index
        nonfree_ratio = float(self.occ_sample_nonfree_ratio)
        nonfree_ratio = max(0.0, min(1.0, nonfree_ratio))
        target_nonfree = int(round(sample_num * nonfree_ratio))

        out_xyz = []
        out_label = []
        for b in range(bs):
            label_b = gt_label[b]
            if ignore is None:
                valid_idx = torch.arange(n, device=device)
            else:
                valid_idx = torch.nonzero(label_b != ignore, as_tuple=False).squeeze(1)

            if valid_idx.numel() == 0:
                # 极端兜底：没有有效监督体素时，直接退回全量（避免训练崩溃）
                out_xyz.append(gt_xyz[b:b+1])
                out_label.append(gt_label[b:b+1])
                continue

            valid_label = label_b[valid_idx]
            nonfree_pool = valid_idx[valid_label != free_id]
            free_pool = valid_idx[valid_label == free_id]

            nonfree_cnt = min(target_nonfree, nonfree_pool.numel())
            free_cnt = sample_num - nonfree_cnt

            chosen = []
            if nonfree_cnt > 0:
                perm = torch.randperm(nonfree_pool.numel(), device=device)[:nonfree_cnt]
                chosen.append(nonfree_pool[perm])

            if free_cnt > 0:
                if free_pool.numel() > 0:
                    if free_pool.numel() >= free_cnt:
                        perm = torch.randperm(free_pool.numel(), device=device)[:free_cnt]
                        chosen.append(free_pool[perm])
                    else:
                        # free 不足时：先取完，再从有效体素中有放回补齐
                        chosen.append(free_pool)
                        remaining = free_cnt - free_pool.numel()
                        repl = valid_idx[torch.randint(0, valid_idx.numel(), (remaining,), device=device)]
                        chosen.append(repl)
                else:
                    # 没有 free 时：从 non-free/valid 中有放回补齐
                    repl = valid_idx[torch.randint(0, valid_idx.numel(), (free_cnt,), device=device)]
                    chosen.append(repl)

            sample_idx = torch.cat(chosen, dim=0)
            if sample_idx.numel() != sample_num:
                # 保险兜底：确保每个 batch 的采样点数一致，避免后续 bs>1 concat 失败
                repl = valid_idx[torch.randint(0, valid_idx.numel(), (sample_num - sample_idx.numel(),), device=device)]
                sample_idx = torch.cat([sample_idx, repl], dim=0)

            out_xyz.append(gt_xyz[b:b+1, sample_idx])
            out_label.append(gt_label[b:b+1, sample_idx])

        gt_xyz = torch.cat(out_xyz, dim=0)
        gt_label = torch.cat(out_label, dim=0)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            bs = means.shape[0]
            means = torch.cat([means, self.empty_mean.expand(bs, -1, -1)], dim=1)
            scales = torch.cat([scales, self.empty_scale.expand(bs, -1, -1)], dim=1)
            rotations = torch.cat([rotations, self.empty_rot.expand(bs, -1, -1)], dim=1)
            empty_sem = self.empty_sem.clone().expand(bs, -1, -1).clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, self.empty_opa.expand(bs, -1, -1)], dim=1)
        elif self.use_localaggprob:
            assert opacities.shape[-1] == self.num_classes - 1
            opacities = opacities.softmax(dim=-1)
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda()
        return means, origi_opa, opacities, scales, CovInv

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        prediction = []
        bin_logits = []
        density = []
        occ_xyz = torch.stack([meta['occ_xyz'] for meta in metas], dim=0).to(self.zero_tensor.device)
        occ_label = torch.stack([meta['occ_label'] for meta in metas], dim=0).to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)

        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]

            if bs == 1:
                semantics = self.aggregator(
                    sampled_xyz.clone().float(), 
                    means, 
                    origi_opa.reshape(bs, g),
                    opacities,
                    scales,
                    CovInv)
            else:
                sem_list = []
                for b in range(bs):
                    sem_b = self.aggregator(
                        sampled_xyz[b:b+1].clone().float(),
                        means[b:b+1],
                        origi_opa[b:b+1].reshape(1, g),
                        opacities[b:b+1],
                        scales[b:b+1],
                        CovInv[b:b+1])
                    sem_list.append(sem_b.unsqueeze(0))
                semantics = torch.cat(sem_list, dim=0)

            if self.use_localaggprob:
                if self.combine_geosem:
                    sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                    geo = 1 - semantics[1].unsqueeze(-1)
                    geosem = torch.cat([sem, geo], dim=-1)
                else:
                    geosem = semantics[0]
                prediction.append(geosem[None].transpose(1, 2))
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                prediction.append(semantics[None].transpose(1, 2))
        
        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)
        
        return {
            'pred_occ': prediction,
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation]
        }
    
    def loss(self, pred_occ, sampled_label):
        loss_dict = {}
        num_layers = len(pred_occ)

        for i, semantics in enumerate(pred_occ):
            pred_classes = semantics.shape[1]
            dynamic_weight = self.class_weights.to(dtype=semantics.dtype, device=semantics.device)

            valid_mask = sampled_label != 255
            if valid_mask.any():
                min_label = sampled_label[valid_mask].min().item()
                max_label = sampled_label[valid_mask].max().item()
                if min_label < 0 or max_label >= pred_classes:
                    print(f"\n[Warning] pred channels: {pred_classes}, label min: {min_label}, max: {max_label}")
                    sampled_label[valid_mask] = torch.clamp(sampled_label[valid_mask], 0, pred_classes - 1)

            ce = self.loss_voxel_ce_weight * \
                CE_ssc_loss(semantics, sampled_label, dynamic_weight, ignore_index=255)

            lovasz_input = torch.softmax(semantics, dim=1)
            lovasz = self.loss_voxel_lovasz_weight * lovasz_softmax(
                lovasz_input.transpose(1, 2).flatten(0, 1), sampled_label.flatten(), ignore=255)

            if num_layers > 1 and i < num_layers - 1:
                layer_weight = 0.5
            else:
                layer_weight = 1.0

            loss_dict[f'losses/loss_voxel_ce_layer{i}'] = ce * layer_weight
            loss_dict[f'losses/loss_voxel_lovasz_layer{i}'] = lovasz * layer_weight

        if self.training and hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
        
        if self._step_count % 500 == 0:
            with torch.no_grad():
                pred = pred_occ[-1].argmax(dim=1).flatten()
                gt = sampled_label.flatten()
                gt_valid = gt[gt != 255]
                pred_valid = pred[gt != 255]
                
                pred_unique, pred_counts = pred.unique(return_counts=True)
                gt_unique, gt_counts = gt_valid.unique(return_counts=True)
                
                total_pred = pred.numel()
                total_gt = gt_valid.numel()
                
                print(f"\n[Debug step {self._step_count}]")
                print(f"  GT distribution:   { {int(u): f'{100*c/total_gt:.2f}%' for u, c in zip(gt_unique, gt_counts)} }")
                print(f"  Pred distribution: { {int(u): f'{100*c/total_pred:.2f}%' for u, c in zip(pred_unique, pred_counts)} }")
                
                # 逐类accuracy
                correct = (pred_valid == gt_valid).float().mean().item()
                print(f"  Overall acc: {100*correct:.2f}%")
                
                free_mask = gt_valid == 0
                if free_mask.any():
                    free_acc = (pred_valid[free_mask] == 0).float().mean().item()
                    print(f"  Free acc: {100*free_acc:.2f}%")
                nonfree_mask = gt_valid != 0
                if nonfree_mask.any():
                    nonfree_acc = (pred_valid[nonfree_mask] == gt_valid[nonfree_mask]).float().mean().item()
                    print(f"  Non-free acc: {100*nonfree_acc:.2f}%")

        return loss_dict