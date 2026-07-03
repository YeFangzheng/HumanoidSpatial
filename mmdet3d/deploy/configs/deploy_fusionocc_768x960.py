# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


use_custom_eval_hook = True
default_scope = 'mmdet3d'

class_names = [
    "free", # 0
    "pedestrian", # 1
    "robot", # 2
    "chair", # 3 
    "table", # 4 
    "floor", # 5
    "wall", # 6
    "window", # 7
    "door", # 8
    "plant", # 9
    "appliance", # 10
    "furniture", # 11
    "objects", # 12
]

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
    'z': [-1.5, 0.9, 0.6],
}      

bev_h_ = 200
bev_w_ = 200
_dim_ = 128
_num_levels_= 1

empty_idx = 0
num_classes = len(class_names) # 0 free, 1-12 obj
img_norm_cfg = None

occ_size = [200, 200, 24]
voxel_out_channel = occ_size[2] * 16
voxel_size = [0.1, 0.1, 4]

memory_len = 1
model = dict(
    type='FusionOccDeploy',
    fp16=False,
    model=dict(
        type='FusionOccONNX',
        memory_len=memory_len,
        img_shape=(768, 960),
        bev_h=bev_h_,
        bev_w=bev_w_,
        grid_config=grid_config,
        single_bev_dims=_dim_,
        pts_feat_dims=192,
        voxel_layer=dict(
            max_num_points=50,
            voxel_size=voxel_size,
            max_voxels=5000,
            point_cloud_range=point_cloud_range,
            voxel_feature_num=4),
        pts_voxel_encoder=dict(
            type='PillarFeatureNet',
            in_channels=4,
            feat_channels=[64],
            with_distance=False,
            voxel_size=(0.1, 0.1, 4),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01), # default: eps=1e-5, momentum=0.1
            legacy=False),
        pts_middle_encoder=dict(
            type='PointPillarsScatterONNX', in_channels=64, output_shape=(200, 200)), # NOTE
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
                    type='BEVFormerEncoderONNX',
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
                                type='SpatialCrossAttentionONNX',
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
            num_channels=[_dim_, _dim_, _dim_]),
        bev_encoder_neck=dict(
            type='CustomFPN',
            in_channels=[_dim_, _dim_, _dim_],
            out_channels=voxel_out_channel,
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
)


# Data
dataset_type = 'XHumanoidDataset'
data_root = 'data/humanoid/household/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/humanoid/household/annotation/occ'

base_metas = ['ori_img', 'scene_token', 'scene_name', 'frame_idx', 'prev_exists', \
              'intrinsic', 'distortion', 'cam2ego', 'post_trans', 'bda', 'ego_pose', 'ego_pose_inv', 'timestamp']

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, undistort=False),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadOccupancyXHumanoid', occupancy_path=occupancy_path),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'points', 'gt_occupancy'], meta_keys=['lidar_origins', 'visible_mask'] + base_metas)
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='frames_mini.json',
        serialize_data=False,
        pipeline=test_pipeline,
        test_mode=True,
        seq_split_num=1,
        box_type_3d='LiDAR'))

test_evaluator = dict(type='RayMetric', 
                      num_classes=num_classes, 
                      class_names=class_names, 
                      point_cloud_range=point_cloud_range,
                      occupancy_size=[0.1, 0.1, 0.1],
                      use_image_mask=False)

test_cfg = dict(type='TestLoop')

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = None
