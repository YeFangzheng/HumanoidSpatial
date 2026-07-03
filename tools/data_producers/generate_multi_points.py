import tqdm
import cv2 as cv
import torch
import torch.nn.functional as F
import numpy as np
from pyquaternion import Quaternion
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData
from mmengine import init_default_scope
init_default_scope('mmdet3d')

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

import supervision as sv
import open3d as o3d
import numba as nb

def _generate_nus_dataset_config():
    data_root = "/shared_disk/users/haoyu.wang/EmbodiedOcc/data/nuscenes"
    ann_file = "nuscenes_infos_train.pkl"
    classes = [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    ]

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
    
    if 'Identity' not in TRANSFORMS:

        @TRANSFORMS.register_module()
        class Identity(BaseTransform):

            def transform(self, results):
                results['data_samples'] = Det3DDataSample()
                if 'ann_info' in results:
                    results[
                        'data_samples'].gt_instances_3d = InstanceData()
                    results[
                        'data_samples'].gt_instances_3d.labels_3d = results[
                            'ann_info']['gt_labels_3d']
                return results

    pipeline = [
        dict(type='Identity'),
        # dict(
        #     type='PrepareImageInputs',
        #     is_train=False,
        #     data_config=data_config,
        #     sequential=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5
        ),
    ]
    modality = dict(use_lidar=True, use_camera=True)
    data_prefix = dict(
        pts="samples/LIDAR_TOP", 
        CAM_BACK="samples/CAM_BACK",
        CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
        CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
        CAM_FRONT="samples/CAM_FRONT", 
        CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
        CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
        img="", sweeps="sweeps/LIDAR_TOP"
    )
    return data_root, ann_file, classes, data_prefix, pipeline, modality

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            if counter[:-1].sum() == 0:
                processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = 255
            else:
                processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter[:-1])
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label


if __name__ == '__main__':
    data_root, ann_file, classes, data_prefix, pipeline, modality = (
        _generate_nus_dataset_config()
    )

    dataset = NuScenesDataset(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=data_prefix,
        pipeline=pipeline,
        metainfo=dict(classes=classes),
        modality=modality,
    )

    classes = ['car', 'pedestrian', 'large vehicle', 'cyclist',
               'tree', 'bush', 'pole', 'cone', 'traffic light', 'fence', 'building',
               'road', 'curb', 'lane', 'sidewalk', 'grassland']
    
    color_palette = sv.ColorPalette.from_hex([
        '#0000ff', '#191970', '#9370db', '#ffb6c1',
        '#008000', '#008000', '#ffff00', '#808080', '#d3d3d3', '#ff8c00', '#ffdead',
        '#ffffff', '#8b0000', '#ffffff', '#ffffff', '#90ee90'])
    color_array = np.array([(color.r, color.g, color.b) for color in color_palette.colors])

    self_range = [3.0, 3.0, 3.0]

    voxel_size = 0.4
    pc_range = [-40, -40, -1, 40, 40, 5.4]
    occ_size = [200, 200, 16]
    
    multi_frame_points = o3d.geometry.PointCloud()
    multi_frame_labels = []
    scene_token = 'cc8c0bf57f984915a77078b10eb33198'
    for i in tqdm.tqdm(range(len(dataset))):
        results = dataset[i]
        frame_idx = results['frame_idx']

        if results['scene_token'] != scene_token:
            multi_frame_labels = np.hstack(multi_frame_labels)
            np.savez_compressed(f'data/nuscenes_occ/stacks/MULTI_POINTS/{scene_token}.npz', points=np.asarray(multi_frame_points.points), labels=multi_frame_labels)

            multi_frame_points = o3d.geometry.PointCloud()
            multi_frame_labels = []
        scene_token = results['scene_token']

        cur_ego2global = np.array(results['ego2global'])
        cur_lidar2ego = np.array(results['lidar_points']['lidar2ego'])
        cur_lidar2global = cur_ego2global @ cur_lidar2ego

        cur_pc_path = results['lidar_path'].replace('nuscenes', 'nuscenes_occ').replace('samples', 'segmentations').replace('.pcd.bin', '.ply')
        cur_label_path = results['lidar_path'].replace('nuscenes', 'nuscenes_occ').replace('samples', 'masks').replace('.pcd.bin', '.npy')
        cur_points = o3d.io.read_point_cloud(cur_pc_path)
        cur_label = np.load(cur_label_path)

        # remove self_ego points
        # self_mask = (np.abs(np.asarray(cur_points.points)[:, 0]) < self_range[0]) & \
        #             (np.abs(np.asarray(cur_points.points)[:, 1]) < self_range[1]) & \
        #             (np.abs(np.asarray(cur_points.points)[:, 2]) < self_range[2])
        # cur_points = cur_points.select_by_index(np.where(~self_mask)[0].tolist())
        # cur_label = cur_label[~self_mask]

        # remove adj_points with dynamic class
        static_mask = cur_label > 3
        cur_points = cur_points.select_by_index(np.where(static_mask)[0].tolist())
        cur_label = cur_label[static_mask]
        
        cur_points_global = cur_points.transform(cur_lidar2global)
        multi_frame_points += cur_points_global
        multi_frame_labels.append(cur_label)




