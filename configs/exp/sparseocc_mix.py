# SparseOcc Indoor - aligned with official MCG-NJU/SparseOcc parameters
#
# Key changes vs previous config:
#   embed_dims: 128 → 256
#   FPN: CustomFPN(2层) → FPN(4层)
#   out_indices: (2,3) → (0,1,2,3)
#   num_levels: 1 → 4
#   lr: 2e-4 → 5e-4 with backbone lr_mult=0.1


use_custom_eval_hook = True

class_names = [
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
    'free',
]

gt_class_names = [
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

# ============================================================
# Model - aligned with official SparseOcc
# ============================================================
_dim_ = 256          # 官方: 256 (之前是128)
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_queries_ = 100
_num_levels_ = 4     # 官方: 4 (之前是1)
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]

num_classes = len(class_names)

occ_size = [200, 200, 24]

model = dict(
    type='SparseOcc',
    bev_h=100,
    bev_w=100,
    grid_config={
        'x': [-10, 10, 0.2],
        'y': [-10, 10, 0.2],
        'z': [-1.5, 0.9, 0.2],
        'depth': [0.5, 10.5, 0.2]
    },
    single_bev_dims=_dim_,
    use_grid_mask=False,
    img_backbone=dict(
        type='ResNet',
        pretrained='$PATH_TO_CKPTS$/resnet50-0676ba61.pth',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),      # 官方: 4层输出 (之前是(2,3))
        frozen_stages=1,                # 官方: freeze stage1 (之前是-1)
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,                 # 官方: True (之前是False)
        style='pytorch',
        with_cp=True),                  # 官方: gradient checkpointing 节省显存
    img_neck=dict(
        type='mmdet.FPN',                     # 官方: 标准FPN (之前是CustomFPN)
        in_channels=[256, 512, 1024, 2048],  # ResNet50 4层输出通道
        out_channels=_dim_,             # 256
        num_outs=_num_levels_),         # 4层输出
    occupancy_head=dict(type='SparseOccHead',
        class_names=class_names,
        embed_dims=_dim_,               # 256
        occ_size=occ_size,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseOccTransformer',
            embed_dims=_dim_,           # 256
            num_layers=_num_layers_,
            num_frames=1,               # 无时序 (官方nuScenes用8)
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=_num_levels_,     # 4 (之前是1)
            num_classes=num_classes,
            pc_range=point_cloud_range,
            occ_size=occ_size,
            topk_training=_topk_training_,
            topk_testing=_topk_testing_),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2FormerLoss',
                num_classes=num_classes,
                no_class_weight=0.1,
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
            loss_geo_scal=dict(
                type='GeoScalLoss',
                num_classes=num_classes,
                loss_weight=1.0
            ),
            loss_sem_scal=dict(
                type='SemScalLoss',
                num_classes=num_classes,
                loss_weight=1.0
            ),
        ),
    ),
    pts_bbox_head=None,
    train_cfg=dict())

# ============================================================
# Data
# ============================================================
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
    dict(type='PointToMultiViewDepth', downsample=1, grid_config={
        'x': [-10, 10, 0.2],
        'y': [-10, 10, 0.2],
        'z': [-1.5, 0.9, 0.2],
        'depth': [0.5, 10.5, 0.2]
    }),
    dict(type='LoadSparseOcc', occupancy_path=occupancy_path,
        class_names=class_names, ignore_classes=[2, 7, 8, 12, 17, 18]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Pack3DDetInputs', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'], meta_keys=[] + base_metas)
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, undistort=True),
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='LoadSparseOcc', occupancy_path=occupancy_path,
        class_names=class_names, ignore_classes=[2, 7, 8, 12, 17, 18]),
    dict(type='BDA', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Pack3DDetInputs', keys=['img', 'voxel_semantics', 'voxel_instances'], meta_keys=['lidar_origins', 'visible_mask'] + base_metas)
]

batch_size = 8   # 256 dims + 4 levels 需要更多显存，降 batch_size
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
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
    num_workers=4,
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
                     class_names=gt_class_names,
                     point_cloud_range=point_cloud_range,
                     occupancy_size=[0.1, 0.1, 0.1],
                     use_image_mask=False)
test_evaluator = val_evaluator

# ============================================================
# Optimizer - aligned with official SparseOcc
# ============================================================
lr = 5e-4   # 官方: 5e-4 (之前是2e-4)
num_epochs = 20  # 官方: 24 (之前是20)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),    # 官方: backbone 学习率×0.1
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',       # 官方: step schedule (之前是cosine)
        begin=0,
        end=num_epochs,
        by_epoch=True,
        milestones=[22, 24],      # 官方: step at epoch 22, 24
        gamma=0.2)
]

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
        'mmdet3d.models.sparseocc'
    ],
    allow_failed_imports=False
)