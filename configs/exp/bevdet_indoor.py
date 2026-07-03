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
_dim_ = 128
_num_levels_= 1

empty_idx = 0
num_classes = len(class_names) # 0 free, 1-12 obj
img_norm_cfg = None

occ_size = [200, 200, 24]
voxel_out_channel = occ_size[2] * 16

memory_len = 1
model = dict(
    type='BEVDetOcc',
    memory_len=memory_len,
    bev_h=bev_h_,
    bev_w=bev_w_,
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
        loss_depth_weight=1.,
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
        num_channels=[_dim_, _dim_, _dim_]),
    bev_encoder_neck=dict(
        type='FPN3D',
        in_channels=[_dim_, _dim_, _dim_],
        out_channels=voxel_out_channel),
    occupancy_head= dict(
        type='OccHead3D',
        use_focal_loss=True,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='BN3d', requires_grad=True),
        soft_weights=True,
        empty_idx=empty_idx,
        num_level=3,
        in_channels=[voxel_out_channel] * 3,
        out_channel=num_classes,
        point_cloud_range=point_cloud_range,
        occ_size=occ_size,
        balance_cls_weight=False,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    pts_bbox_head=None,
    train_cfg=dict())

# Data
dataset_type = 'XHumanoidDataset'
data_root = '$PATH_TO_DATASET$/Data_indoor'
file_client_args = dict(backend='disk')
occupancy_path = '$PATH_TO_DATASET$/Data_indoor/annotation/occ'

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
    # indoor
    dict(
        type='RemapOccLabels',
        mapping={2:255, 8:255, 12:255, 16:255, 17:255, 18:255, 19:255, 20:255},
        ignore_index=255,
        keys=['gt_occupancy'],
    ),
    # outdoor
    # dict(
    #     type='RemapOccLabels',
    #     mapping={2:255, 8:255, 12:255, 13:255, 14:255, 15:255},
    #     ignore_index=255,
    #     keys=['gt_occupancy'],
    # ),
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
    # indoor
    dict(
        type='RemapOccLabels',
        mapping={2:255, 8:255, 12:255, 16:255, 17:255, 18:255, 19:255, 20:255},
        ignore_index=255,
        keys=['gt_occupancy'],
    ),
    # outdoor
    # dict(
    #     type='RemapOccLabels',
    #     mapping={2:255, 8:255, 12:255, 13:255, 14:255, 15:255},
    #     ignore_index=255,
    #     keys=['gt_occupancy'],
    # ),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=['lidar_origins', 'visible_mask'] + base_metas)
]

batch_size = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', samples_per_gpu=batch_size, shuffle=True),
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
    sampler=dict(type='DistributedGroupSampler', shuffle=False),
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
    # type='OptimWrapper',
    type='AmpOptimWrapper',
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
