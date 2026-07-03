# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


use_custom_eval_hook = True

# class_mapping = {
#     "free": "free",
#     "pedestrian": "pedestrian",
#     "robot": "objects",
#     "chair": "objects",
#     "table": "objects",
#     "floor": "floor",
#     "wall": "wall",
#     "window": "wall",
#     "door": "wall",
#     "plant": "objects",
#     "appliance": "objects",
#     "furniture": "objects",
#     "objects": "objects",
# }

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

_base_ = ['../_base_/default_runtime.py', '../_base_/schedules/cosine.py']

point_cloud_range = [-10, -10, -1.5, 10, 10, 0.9]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (768, 960),
    'src_size': (1536, 1920),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

sync_bn = "mmdet3d"

# Model
grid_config = {
    'x': [-10, 10, 0.1],
    'y': [-10, 10, 0.1],
    'z': [-1.5, 0.9, 0.1],
    'depth': [0.5, 10.5, 0.1]
}
depth_categories = 100
grid_config_bevformer={
    'x': [-10, 10, 0.4],
    'y': [-10, 10, 0.4],
    'z': [-1.5, 0.9, 0.1],
}


bev_h_ = 200
bev_w_ = 200
bev_z_ = 24
_dim_ = 128
numC_Trans = 32
_num_levels_= 1

empty_idx = 0
num_classes = len(class_names) # 0 free, 1-12 obj
img_norm_cfg = None

occ_size = [200, 200, 24]
voxel_out_channel = occ_size[2] * 16

# # group split
# group_split = [[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]] # empty, front, back
# group_detr = len(group_split) + 1
# group_classes = [12] + [group[-1] for group in group_split]

group_split = []          # 不做任何 remap
group_detr = 1
group_classes = [num_classes]

memory_len = 1
model = dict(
    type='COTR',
    group_split=group_split,
    memory_len=memory_len,
    bev_h=bev_h_,
    bev_w=bev_w_,
    bev_z=bev_z_,
    grid_config=grid_config,
    single_bev_dims=_dim_,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False),
    img_backbone=dict(
        type='ResNet',
        pretrained='$PATH_TO_CKPTS$/resnet50-0676ba61.pth',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet', # camera-aware depth net
        in_channels=_dim_,
        context_channels=_dim_,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        loss_depth_weight=3.0,
        use_dcn=False,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),
    bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=_dim_,
        stride=[1, 2, 2],
        num_channels=[numC_Trans, numC_Trans*2, numC_Trans*4]),
    bev_encoder_neck=dict(
        type='LSSFPN3D',
        in_channels=numC_Trans*7,
        out_channels=numC_Trans,
        reverse=True,
        size=(24, 50, 50)),
    occupancy_head=dict(
        type='COTRHead',
        in_channels=numC_Trans,
        embed_dims=_dim_,
        num_query=100,
        group_detr=group_detr,
        group_classes=group_classes,
        num_classes=num_classes,
        transformer=dict(
            type='TransformerMSOcc',
            embed_dims=_dim_,
            num_feature_levels=1,
            encoder=dict(
                type='OccEncoder',
                num_layers=1,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention3D',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_points=4),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=1),
                            embed_dims=_dim_,)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_dim_ * 4,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    feedforward_channels=_dim_ * 4,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')))),
        positional_encoding=dict(
            type='CustomLearnedPositionalEncoding3D',
            num_feats=[48, 48, 32],
            row_num_embed=int(50),
            col_num_embed=int(50),
            tub_num_embed=int(24)),
        transformer_decoder=dict(
            type='MaskOccDecoder',
            return_intermediate=True,
            num_layers=1,
            transformerlayers=dict(
                type='MaskOccDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiScaleDeformableAttention3D',
                        embed_dims=_dim_,
                        num_levels=1,
                        num_points=4,),
                    dict(
                        type='GroupMultiheadAttention',
                        group=group_detr,
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=2*_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                    'ffn', 'norm'))),
        predictor=dict(
            type='MaskPredictorHead_Group',
            nbr_classes=num_classes,
            group_detr=group_detr,
            group_classes=group_classes,
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            mask_dims=_dim_),
        use_camera_mask=False,
        use_lidar_mask=False,
        loss_occ=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=10.0,
            ignore_index=255,
            class_weight=[0.02, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.01],
        ),
        loss_cls= dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[0.01] + [1.0] * (num_classes - 1) + [0.01],
            ignore_index=255,
            ),
        loss_mask= dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice= dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0)),
    pts_bbox_head=None,
    train_cfg=dict(
        pts=dict(
            out_size_factor=4,
            # default cfg copy from MaskFormer
            assigner=dict(
                type='MaskHungarianAssigner3D',
                cls_cost=dict(type='MaskClassificationCost', weight=1.0),
                mask_cost=dict(type='MaskFocalLossCost', weight=20.0, binary_input=True),
                dice_cost=dict(type='MaskDiceLossCost', weight=1.0, pred_act=True, eps=1.0),
                use_camera_mask=False,
                use_lidar_mask=False),
            sampler=dict(
                type='MaskPseudoSampler',
                use_camera_mask=False,
                use_lidar_mask=False)
        )),
    test_cfg=dict(
        pts=dict(
            mask_threshold = 0.5,
            overlap_threshold = 0.6,
            occupy_threshold = 0.1,
            inf_merge=True,
            only_encoder=False,
        ))
)

# Data
dataset_type = 'XHumanoidDataset'
data_root = '$PATH_TO_DATASET$/Data_mix'
file_client_args = dict(backend='disk')
occupancy_path = '$PATH_TO_DATASET$/Data_mix/annotation/occ'

_remap = {
    2: 255,
    7: 255,
    8: 255,
    12: 255,
    17: 255,
    18: 255,
}

base_metas = ['ori_img', 'scene_token', 'scene_name', 'frame_idx', 'prev_exists', \
              'intrinsic', 'distortion', 'cam2ego', 'post_trans', 'bda', 'ego_pose', 'ego_pose_inv', 'timestamp']


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        undistort=True),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='LoadOccupancyXHumanoid',
        occupancy_path=occupancy_path,
        class_names=class_names,
    ),
    dict(
        type='RemapOccLabels',
        mapping=_remap,
        ignore_index=255,
        keys=['gt_occupancy'],
    ),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy', 'gt_depth'], meta_keys=[] + base_metas)
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, undistort=True),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(
        type='LoadOccupancyXHumanoid',
        occupancy_path=occupancy_path,
        class_names=class_names,
    ),
    dict(
        type='RemapOccLabels',
        mapping=_remap,
        ignore_index=255,
        keys=['gt_occupancy'],
    ),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=['lidar_origins', 'visible_mask'] + base_metas)
]

batch_size = 3
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_frames.json',
        serialize_data=False,
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True,
        seq_split_num=5,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_frames.json',
        serialize_data=False,
        pipeline=test_pipeline,
        test_mode=True,
        seq_split_num=1,
        box_type_3d='LiDAR'))
test_dataloader = val_dataloader

val_evaluator = dict(type='RayMetric',
                     num_classes=num_classes,
                     class_names=class_names,
                     point_cloud_range=point_cloud_range,
                     occupancy_size=[0.1, 0.1, 0.1],
                     use_image_mask=False)
test_evaluator = val_evaluator

# Optimizer
lr = 2e-4
num_epochs = 20

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=num_epochs,
        end=num_epochs,
        by_epoch=True,
        eta_min=1e-5)
]
optim_wrapper = dict(
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=num_epochs+1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2,
    ),
]
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = None

custom_imports = dict(
    imports=[
        'mmdet3d.models.cotr'
    ],
    allow_failed_imports=False
)
