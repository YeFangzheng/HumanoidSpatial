# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


use_custom_eval_hook = True

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
    'x': [-10, 10, 0.2],
    'y': [-10, 10, 0.2],
    'z': [-1.5, 0.9, 0.2],
    'depth': [0.5, 10.5, 0.2]
}
depth_categories = 50
grid_config_bevformer={
    'x': [-10, 10, 0.2],
    'y': [-10, 10, 0.2],
    'z': [-1.5, 0.9, 0.6],
}


bev_h_ = 100
bev_w_ = 100

empty_idx = 0
num_classes = len(class_names) # 0 free, 1-12 obj
img_norm_cfg = None

occ_size = [200, 200, 24]
_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [100, 50, 25]
volume_w_ = [100, 50, 25]
volume_z_ = [12, 6, 3]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

memory_len = 1
model = dict(
    type='SurroundOcc',
    bev_h=bev_h_,
    bev_w=bev_w_,
    grid_config=grid_config,
    single_bev_dims=_dim_,
    img_backbone=dict(
        type='ResNet',
        pretrained='$PATH_TO_CKPTS$/resnet50-0676ba61.pth',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        num_outs=3,
        start_level=0,
        out_ids=[0, 1, 2]),
    occupancy_head=dict(
        type='SurroundOccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=num_classes,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=True,
        transformer_template=dict(
            type='SurroundOccPerceptionTransformer',
            embed_dims=_dim_,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
    pts_bbox_head=None,
    train_cfg=dict()))

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
    dict(type='LoadOccupancyXHumanoid', occupancy_path=occupancy_path, 
        class_names=class_names),
    dict(
        type='RemapOccLabels',
        mapping=_remap,
        ignore_index=255,
        keys=['gt_occupancy'],
    ),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=[] + base_metas)
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, undistort=True),
    dict(type='LoadOccupancyXHumanoid', occupancy_path=occupancy_path, 
        class_names=class_names),
    dict(
        type='RemapOccLabels',
        mapping=_remap,
        ignore_index=255,
        keys=['gt_occupancy'],
    ),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=['lidar_origins', 'visible_mask'] + base_metas)
]

batch_size = 4
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
        'mmdet3d.models.surroundocc'
    ],
    allow_failed_imports=False
)