# Copyright (c) OpenMMLab. All rights reserved.
#
# 室内外混合 GaussianFormer：官方 OpenOcc / mmseg ``BEVSegmentor`` 配置（vendored 源码在
# ``mmdet3d/models/GaussianFormer``）。勿用 ``tools/train.py`` 跑本文件。
#
# 官方权重 + RayMetric 评测（与 MMDet 室内外混合 RayIoU 对齐）。``torchrun --nproc_per_node>1`` 时
# 仅 ``LOCAL_RANK==0`` 跑评测，其余进程立即退出（本脚本不启 DDP；多卡并行推理需另做数据并行）::
#
#   cd /path/to/Occupancy_Giga-benchmark
#   export PYTHONPATH=$PWD:$PYTHONPATH
#   torchrun --nproc_per_node=4 tools/test.py configs/exp/gaussianformer_mix.py \\
#     output/gaussianformer_mix/epoch_10.pth --launcher pytorch --work-dir output_test/test
#
# 单卡::
#
#   python tools/test.py configs/exp/gaussianformer_mix.py \\
#     output/gaussianformer_mix/epoch_10.pth --work-dir output_test/test
#
# 单节点多卡（推荐，与 torch.distributed 一致）::
#
#   cd /path/to/Occupancy_Giga-benchmark
#   export PYTHONPATH=$PWD:$PYTHONPATH
#   torchrun --nproc_per_node=4 tools/train_gaussianformer_official.py \\
#     --py-config configs/exp/gaussianformer_mix.py \\
#     --work-dir output/gaussianformer_mix
#
# 或不用 torchrun（内部 spawn）::
#
#   python tools/train_gaussianformer_official.py \\
#     --py-config configs/exp/gaussianformer_mix.py \\
#     --work-dir output/gaussianformer_mix
#
# Python 依赖：官方栈需要 MMSegmentation::
#   pip install 'mmsegmentation>=1.2.0,<1.3.0'
#
# CUDA 扩展::
#   cd mmdet3d/models/GaussianFormer/model/head/localagg && pip install -e .
#   DeformableAggregation：通常与 MMDet 共用本仓库已编译的 mmdet3d.ops；若仍报错再装::
#   cd mmdet3d/models/GaussianFormer/model/encoder/gaussian_encoder/ops && pip install -e .
#
# mmengine Config 内勿用 os.environ；路径请按本机修改 ``_OCC_ROOT`` / ``data_root``。

# Occupancy_Giga-benchmark repo root (for ResNet pretrained path).
_OCC_ROOT = '$PATH_TO_OCCUPANCY_GIGA_BENCHMARK$'

class_names = [
    'free',
    'pedestrian',
    'irregular_pedestrian',
    'table',
    'chair',
    'objects',
    'plant',
    'equipment',
    'animal',
    'walkable area',
    'wall',
    'door',
    'other',
    'appliance',
    'furniture',
    'window',
    'bicycle',
    'cyclist',
    'tricycle',
    'vehicle',
    'drivable area',
]

point_cloud_range = [-10, -10, -1.5, 10, 10, 0.9]
data_root = '$PATH_TO_DATASET$/Data_mix'
occupancy_path = '$PATH_TO_DATASET$/Data_mix/annotation/occ'

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (768, 960),
    'src_size': (1536, 1920),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

remap_occ_labels = {
    2: 255,
    7: 255,
    8: 255,
    12: 255,
    17: 255,
    18: 255,
}

train_pipeline = [
    dict(
        type='XHumanoidPrepareImageInputs',
        data_config=data_config,
        is_train=True,
        undistort=True,
    ),
    dict(type='XHumanoidOpenOccProjection'),
    dict(
        type='XHumanoidOpenOccLoadOccupancy',
        occupancy_path=occupancy_path,
        class_names=class_names,
        remap_labels=remap_occ_labels,
        pc_range=point_cloud_range,
        grid_size=[200, 200, 24],
        voxel_size=0.1,
    ),
]

test_pipeline = [
    dict(
        type='XHumanoidOpenOccLidarOrigins',
        ann_file=f'{data_root}/val_frames.json',
    ),
    dict(
        type='XHumanoidPrepareImageInputs',
        data_config=data_config,
        is_train=False,
        undistort=True,
    ),
    dict(type='XHumanoidOpenOccProjection'),
    dict(
        type='XHumanoidOpenOccLoadOccupancy',
        occupancy_path=occupancy_path,
        class_names=class_names,
        remap_labels=remap_occ_labels,
        pc_range=point_cloud_range,
        grid_size=[200, 200, 24],
        voxel_size=0.1,
    ),
]

train_dataset_config = dict(
    type='XHumanoidOpenOccDataset',
    data_root=data_root,
    ann_file=f'{data_root}/train_frames.json',
    data_config=data_config,
    occupancy_path=occupancy_path,
    class_names=class_names,
    pipeline=train_pipeline,
    phase='train',
)

val_dataset_config = dict(
    type='XHumanoidOpenOccDataset',
    data_root=data_root,
    ann_file=f'{data_root}/val_frames.json',
    data_config=data_config,
    occupancy_path=occupancy_path,
    class_names=class_names,
    pipeline=test_pipeline,
    phase='val',
    return_keys=[
        'img',
        'projection_mat',
        'image_wh',
        'occ_label',
        'occ_xyz',
        'occ_cam_mask',
        'ori_img',
        'cam_positions',
        'focal_positions',
        'lidar_origins',
    ],
)

batch_size = 1
train_loader = dict(batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=2)

# ------------- misc (official train.py) -------------
print_freq = 50
max_epochs = 20
warmup_iters = 1000
min_lr_ratio = 0.05
# 官方 train.py 内置 MeanIoU 写死 nuScenes 类别，对室内外混合无意义；用 Occ-benchmark ``tools/test.py`` + RayMetric。
skip_builtin_eval = True
eval_every_epochs = 1
trainer_sleep_s = 0
syncBN = True
find_unused_parameters = False
amp = False
load_from = None

grad_max_norm = 35

optimizer = dict(
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

semantic_dim = 20
num_classes = len(class_names)

raymetric_eval = dict(
    type='RayMetric',
    num_classes=num_classes,
    class_names=class_names,
    point_cloud_range=point_cloud_range,
    occupancy_size=[0.1, 0.1, 0.1],
    use_image_mask=False,
)

embed_dims = 128
num_groups = 4
num_decoder = 4
num_single_frame_decoder = 1
use_deformable_func = True
num_levels = 3
pc_range = point_cloud_range
scale_range = [0.016, 0.128]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
semantics = True

manual_class_weight = [
    0.02, 4.07, 1.00, 0.67, 0.73, 1.03, 0.70, 1.34, 1.00, 0.34, 0.28, 1.12, 1.00,
    1.59, 1.25, 0.97, 1.00, 1.00, 1.00, 1.00, 1.00,
]

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=0,
            num_classes=num_classes,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0,
            ),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=255,
            manual_class_weight=manual_class_weight,
        )
    ],
)

loss_input_convertion = dict(
    pred_occ='pred_occ',
    sampled_xyz='sampled_xyz',
    sampled_label='sampled_label',
    occ_mask='occ_mask',
)

_backbone_ckpt = '$PATH_TO_CKPTS$/resnet50-0676ba61.pth'

model = dict(
    type='BEVSegmentor',
    # 与 ``out_indices=(1,2,3)`` 一致：backbone 返回 3 张图，在 tuple 中下标为 0,1,2（勿写 1,2,3）
    img_backbone_out_indices=[0, 1, 2],
    img_backbone=dict(
        type='ResNet',
        pretrained=_backbone_ckpt,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
    ),
    img_neck=dict(
        type='FPN',
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[512, 1024, 2048],
    ),
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim,
        ),
        norm_layer=dict(type='LN', normalized_shape=embed_dims),
        ffn=dict(
            type='AsymmetricFFN',
            in_channels=embed_dims * 2,
            pre_norm=dict(type='LN'),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode='cat',
            kps_generator=dict(
                type='SparseGaussian3DKeyPointsGenerator',
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range,
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[0.8, 0.8, 0.3],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            type='SparseConv3D',
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.1, 0.1, 0.1],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            'deformable', 'ffn', 'norm', 'refine',
        ] * num_single_frame_decoder + [
            'spconv', 'norm', 'deformable', 'ffn', 'norm', 'refine',
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            mean=[0, 0, -0.3],
            scale=[20, 20, 2.4],
        ),
        with_empty=True,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200,
            W=200,
            D=24,
            pc_min=[-10.0, -10.0, -1.5],
            grid_size=0.1,
        ),
        dataset_type='nusc',
        empty_label=0,
    ),
)
