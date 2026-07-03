# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


use_custom_eval_hook = True

class_names = [
    "others", # 0
    "barrier", # 1
    "bicycle", # 2 
    "bus", # 3 
    "car", # 4
    "construction", # 5
    "motorcycle", # 6
    "pedestrian", # 7
    "trafficcone", # 8
    "trailer", # 9
    "truck", # 10
    "driveable_surface", # 11
    "other_flat", # 12
    "sidewalk", # 13
    "terrain", # 14
    "mannade", # 15 
    "vegetation", # 16
    "free", # 17
]

data_prefix = dict(pts='samples/LIDAR_TOP', 
                   CAM_BACK="samples/CAM_BACK",
                   CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
                   CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
                   CAM_FRONT="samples/CAM_FRONT",
                   CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
                   CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
                   img='', sweeps='sweeps/LIDAR_TOP')

# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/default_runtime.py', '../_base_/schedules/cosine.py']

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

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
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.8],
}      

bev_h_ = 200
bev_w_ = 200
numC_Trans = 256
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = numC_Trans * 4
_num_levels_= 1

empty_idx = 17
num_cls = len(class_names) # 0 others, 1-16 obj, 17 free
img_norm_cfg = None

occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
voxel_out_channel = 256
voxel_channels = [64, 64*2, 64*4]

voxel_size = [0.2, 0.2, 8]
model = dict(
    type='FusionOccV2',
    bev_h=bev_h_,
    bev_w=bev_w_,
    grid_config=grid_config,
    single_bev_dims=numC_Trans,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000),
            point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(400, 400)), # NOTE
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2], # [2, 2, 2]
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    img_backbone=dict(
        type='ResNet',
        pretrained='ckpts/resnet50-0676ba61.pth',
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
    backward_projection=dict(
        type='BackwardProjection',
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=numC_Trans,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                grid_config=grid_config,
                data_config=data_config,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=[2.0, 42.0, 0.5],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='MSDeformableAttention',
                                embed_dims=numC_Trans,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=numC_Trans,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
                    # operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                    # operation_order=('cross_attn', 'norm'))),
           ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
    ),
    bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        stride=[1, 2, 2],
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4]),
    bev_encoder_neck=dict(
        type='CustomFPN',
        in_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        out_channels=numC_Trans,
        num_outs=3,
        start_level=0,
        out_ids=[0, 1, 2]),
    occupancy_head= dict(
        type='OccHead',
        use_focal_loss=True,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_3d_cfg=dict(type='Conv3d'),
        norm_3d_cfg=dict(type='BN3d', requires_grad=True),
        soft_weights=True,
        empty_idx=empty_idx,
        num_level=3,
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        occ_size=occ_size,
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
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = '/media/datasets/nuscenes/occ3d'

base_metas = ['ori_img', 'scene_token', 'scene_name', 'frame_idx', 'prev_exists', \
              'intrinsic', 'cam2ego', 'post_trans', 'bda', 'ego_pose', 'ego_pose_inv']


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadOccupancy', occupancy_path=occupancy_path),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=[] + base_metas)
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadOccupancy', occupancy_path=occupancy_path),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=['lidar_origins'] + base_metas)
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

batch_size = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', samples_per_gpu=batch_size, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        serialize_data=False,
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        data_prefix=data_prefix,
        seq_split_num=2,
        # use_sequence_group_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        serialize_data=False,
        pipeline=test_pipeline,
        test_mode=True,
        modality=input_modality,
        data_prefix=data_prefix,
        seq_split_num=1,
        filter_empty_gt=False,
        box_type_3d='LiDAR'))
test_dataloader = val_dataloader

val_evaluator = dict(type='RayMetric')
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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=5)
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
