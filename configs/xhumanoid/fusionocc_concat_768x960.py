import json

use_custom_eval_hook = True

# 将不同场景数据类别映射成演示定义的几个类别
class_mapping_household = {
    "free": "free",
    "pedestrian": "pedestrian",
    "robot": "objects",
    "chair": "objects",
    "table": "objects",
    "floor": "floor",
    "wall": "wall",
    "window": "wall",
    "door": "wall",
    "plant": "objects",
    "appliance": "objects",
    "furniture": "objects",
    "objects": "objects",
}

class_mapping_industry = {
    "free": "free",
    "pedestrian": "pedestrian",
    "floor": "floor",
    "wall": "wall",
    "conveyor": "conveyor",
    "objects": "objects"
}

# 模型输出类别
class_names = [
    "free", # 0
    "pedestrian", # 1 
    "floor", # 2
    "wall", # 3
    "conveyor", # 4
    "objects", # 5
]

_base_ = ['../_base_/default_runtime.py', '../_base_/schedules/cosine.py']

point_cloud_range = [-8, -8, -1.5, 8, 8, 0.9]

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
    'x': [-8, 8, 0.1],
    'y': [-8, 8, 0.1],
    'z': [-1.5, 0.9, 0.6],
}      

bev_h_ = 160
bev_w_ = 160
_dim_ = 128
_num_levels_= 1

empty_idx = 0
num_classes = len(class_names) # 0 free, 1-12 obj
img_norm_cfg = None

occ_size = [160, 160, 24]
voxel_out_channel = occ_size[2] * 16
voxel_size = [0.1, 0.1, 4]

memory_len = 1
model = dict(
    type='FusionOcc',
    memory_len=memory_len,
    bev_h=bev_h_,
    bev_w=bev_w_,
    grid_config=grid_config,
    single_bev_dims=_dim_,
    pts_feat_dims=192,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=50,
            voxel_size=voxel_size,
            max_voxels=(5000, 5000),
            point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.1, 0.1, 4),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01), # default: eps=1e-5, momentum=0.1
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(160, 160)), # NOTE
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 64, 64],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 64, 64],
        out_channels=[64, 64, 64],
        upsample_strides=[1, 2, 4],
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
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                grid_config=grid_config,
                data_config=data_config,
                distortion=True,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dropout=0.0,
                            deformable_attention=dict(
                                type='MSDeformableAttention',
                                embed_dims=_dim_,
                                num_points=4,
                                num_Z_anchors=4,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_dim_ * 4,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    feedforward_channels=_dim_ * 4,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
           ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_dim_ // 2,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
    ),
    bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=_dim_,
        stride=[1, 2, 2],
        num_channels=[_dim_, _dim_ * 2, _dim_ * 4]),
    bev_encoder_neck=dict(
        type='CustomFPN',
        in_channels=[_dim_, _dim_ * 2, _dim_ * 4],
        out_channels=voxel_out_channel,
        num_outs=3,
        start_level=0,
        out_ids=[0, 1, 2]),
    occupancy_head=dict(
        type='OccHead',
        use_focal_loss=True,
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_3d_cfg=dict(type='Conv3d'),
        norm_3d_cfg=dict(type='BN3d', requires_grad=True),
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

base_metas = ['ori_img', 'scene_token', 'scene_name', 'frame_idx', 'prev_exists', \
              'intrinsic', 'distortion', 'cam2ego', 'post_trans', 'bda', 'ego_pose', 'ego_pose_inv', 'timestamp']

# 家居场景pipeline
train_pipeline_household = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        shoulder_aug=True, # 随机加入肩膀纹理
        undistort=False),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadOccupancyXHumanoid', occupancy_path='data/humanoid/household/annotation/occ', 
        class_mapping=class_mapping_household, class_names=class_names, range=[20, 180, 20, 180]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=[] + base_metas)
]

# 工业场景pipeline
train_pipeline_industry = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        shoulder_aug=True,
        undistort=False),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadOccupancyXHumanoid', occupancy_path='data/humanoid/industry/annotation/occ',
        class_mapping=class_mapping_industry, class_names=class_names, range=[20, 180, 20, 180]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=[] + base_metas)
]

# 展会场景pipeline
train_pipeline_exhibition = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        shoulder_aug=True,
        undistort=False),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadOccupancyXHumanoid', occupancy_path='data/humanoid/exhibition/annotation/occ',
        class_mapping=class_mapping_industry, class_names=class_names, range=[20, 180, 20, 180]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=[] + base_metas)
]

# 机器人场景pipeline
train_pipeline_robotic = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        shoulder_aug=True,
        undistort=False),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadOccupancyXHumanoid', occupancy_path='data/humanoid/robotic/annotation/occ',
         class_mapping=class_mapping_industry, class_names=class_names, range=[20, 180, 20, 180]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=[] + base_metas)
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, undistort=False, shoulder_aug=False),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadOccupancyXHumanoid', occupancy_path='data/humanoid/industry/annotation/occ',
         class_mapping=class_mapping_industry, class_names=class_names, range=[20, 180, 20, 180]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=['lidar_origins', 'visible_mask'] + base_metas)
]

batch_size = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', samples_per_gpu=batch_size, shuffle=True), # clip分组，连续帧sampler
    dataset=dict(
        type='ConcatDataset', # 多个场景数据合并
        datasets=[
            dict(
                type=dataset_type,
                data_root='data/humanoid/household/',
                ann_file='frames.json',
                serialize_data=False,
                pipeline=train_pipeline_household,
                test_mode=False,
                use_valid_flag=True,
                seq_split_num=5,
                box_type_3d='LiDAR'
            ),
            dict(
                type=dataset_type,
                data_root='data/humanoid/industry/',
                ann_file='frames.json',
                serialize_data=False,
                pipeline=train_pipeline_industry,
                test_mode=False,
                use_valid_flag=True,
                seq_split_num=5,
                box_type_3d='LiDAR'
            ),
            dict(
                type=dataset_type,
                data_root='data/humanoid/exhibition/',
                ann_file='frames.json',
                serialize_data=False,
                pipeline=train_pipeline_exhibition,
                test_mode=False,
                use_valid_flag=True,
                seq_split_num=5,
                box_type_3d='LiDAR'
            ),
            dict(
                type=dataset_type,
                data_root='data/humanoid/robotic/',
                ann_file='frames.json',
                serialize_data=False,
                pipeline=train_pipeline_robotic,
                test_mode=False,
                use_valid_flag=True,
                seq_split_num=5,
                box_type_3d='LiDAR'
            )
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='data/humanoid/industry/',
        ann_file='frames.json',
        serialize_data=False,
        pipeline=test_pipeline,
        test_mode=True,
        seq_split_num=1,
        box_type_3d='LiDAR'))
test_dataloader = val_dataloader

val_evaluator = dict(type='RayMetric', # 评价指标，mIoU和rayIoU
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
