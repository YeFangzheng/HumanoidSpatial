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
from grounded_sam2 import GroundedSAM2


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

    prompts = ['car', 'pedestrian', 'large vehicle', 'cyclist',
               'tree', 'bush', 'pole', 'cone', 'traffic light', 'fence', 'building',
               'road', 'curb', 'lane', 'sidewalk', 'grassland']
    
    class_map = {
        'car': 'vehicle',
        'pedestrian': 'pedestrian',
        'large vehicle': 'vehicle',
        'cyclist': 'cyclist',
        'tree': 'vegetation',
        'bush': 'vegetation',
        'pole': 'pole',
        'cone': 'cone',
        'traffic light': 'traffic light',
        'fence': 'fence',
        'building': 'building',
        'road': 'road',
        'curb': 'curb',
        'lane': 'lane',
        'sidewalk': 'road',
        'grassland': 'grassland'
    }

    color_palette = sv.ColorPalette.from_hex([
        '#0000ff', '#191970', '#9370db', '#ffb6c1',
        '#008000', '#008000', '#ffff00', '#808080', '#d3d3d3', '#ff8c00', '#ffdead',
        '#ffffff', '#8b0000', '#ffffff', '#ffffff', '#90ee90'])
    color_array = np.array([(color.r, color.g, color.b) for color in color_palette.colors])

    sam2 = GroundedSAM2(prompts, color_palette=color_palette)
    
    self_range = [3.0, 3.0, 3.0]

    for i in tqdm.tqdm(range(len(dataset))):
        results = dataset[i]

        if results['scene_token'] != 'cc8c0bf57f984915a77078b10eb33198':
            break
        images = results['images']
        points_lidar = results['points']
        
        points_labels = torch.ones(len(points_lidar), dtype=torch.int32) * 255
        points_visible = torch.zeros(len(points_lidar), dtype=torch.int32)

        point_images = dict()
        for cam_name, img_info in images.items():
            img_path = img_info['img_path']
            out_path = img_path.replace('nuscenes', 'nuscenes_occ').replace('samples', 'masks')

            labels = sam2.predict(img_path, out_path)

            labels = torch.tensor(labels)
            height, width = labels.shape

            cam2img = torch.eye(4)
            cam2img[:3, :3] = torch.tensor(img_info['cam2img'])
            lidar2cam = torch.tensor(img_info['lidar2cam'])
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)

            grid = points_img[:, :2].clone()
            grid[:, 0] = (grid[:, 0] / width) * 2 - 1
            grid[:, 1] = (grid[:, 1] / height) * 2 - 1
            img_points_labels = F.grid_sample(labels[None, None].float(), grid[None, None], mode='nearest')
            img_points_labels = img_points_labels.squeeze().int()

            coor = torch.round(points_img[:, :2]).int()
            depth = points_img[:, 2]
            kept = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (depth >= 0)

            valid_mask = torch.logical_and(img_points_labels >= 0, img_points_labels < 255)
            valid_mask = torch.logical_and(valid_mask, kept)
            points_labels[valid_mask] = img_points_labels[valid_mask]
            points_visible[kept] += 1

            points_colors = torch.zeros((len(img_points_labels), 3), dtype=torch.uint8)
            points_colors[valid_mask] = torch.tensor(color_array[img_points_labels[valid_mask]]).to(points_colors)
            
            img = cv.imread(img_path)
            coor, depth, points_colors = coor[kept], depth[kept], points_colors[kept]

            for i in range(len(coor)):
                cv.circle(img, tuple(coor[i].tolist()), 1, tuple(points_colors[i].tolist()[::-1]), -1)
            
            point_images[cam_name] = img
            cv.imwrite(out_path.replace('masks', 'segmentations'), img)
            
        # 多视角point_image
        # layout_front = np.hstack([point_images['CAM_FRONT_LEFT'], point_images['CAM_FRONT'], point_images['CAM_FRONT_RIGHT']])           
        # layout_back = np.hstack([point_images['CAM_BACK_LEFT'], point_images['CAM_BACK'], point_images['CAM_BACK_RIGHT']])
        # layout = np.vstack([layout_front, layout_back])
        # cv.imwrite('layout.jpg', layout)

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_lidar.tensor[:, :3]))
        points_colors = np.zeros((len(points_labels), 3), dtype=np.uint8)
        valid_mask = torch.logical_and(points_labels >= 0, points_labels < 255)
        points_colors[valid_mask] = color_array[points_labels[valid_mask]].astype(np.uint8)
        pc.colors = o3d.utility.Vector3dVector(points_colors / 255)

        # remove self_ego points
        self_mask = (np.abs(np.asarray(pc.points)[:, 0]) < self_range[0]) & \
                    (np.abs(np.asarray(pc.points)[:, 1]) < self_range[1]) & \
                    (np.abs(np.asarray(pc.points)[:, 2]) < self_range[2])
        pc = pc.select_by_index(np.where(~self_mask)[0].tolist())
        points_labels = points_labels[~self_mask]
        
        mask_out_path = results['lidar_path'].replace('nuscenes', 'nuscenes_occ').replace('samples', 'masks').replace('.pcd.bin', '.npy')
        pc_out_path = results['lidar_path'].replace('nuscenes', 'nuscenes_occ').replace('samples', 'segmentations').replace('.pcd.bin', '.ply')
        np.save(mask_out_path, points_labels.numpy())
        o3d.io.write_point_cloud(pc_out_path, pc)

        # points_colors = np.zeros((len(points_labels), 3), dtype=np.uint8)
        # points_colors[points_visible > 1] = 1
        # pc.colors = o3d.utility.Vector3dVector(points_colors)
        # o3d.io.write_point_cloud('visible.ply', pc)

        # break
