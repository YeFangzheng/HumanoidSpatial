import os
import json
import tqdm
import pickle
from pypcd import pypcd # pip install git+https://github.com/DanielPollithy/pypcd.git
import cv2 as cv
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp

import supervision as sv
import open3d as o3d
import numba as nb
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


classes = ['pedestrian', 'robot', 'chair', 'table',
            'floor', 'wall', 'window', 'door', 'plant', 
            'appliance','furniture', 'objects']

voxel_size = 0.1
pc_range = [-10, -10, -1.5, 10, 10, 0.9]
occ_size = [200, 200, 24]
blind_ranges = [(83.61, 96.1), (125.56, 135.97), (-96.1, -83.61), (-135.97, -125.56)] # lidar盲区角度区间

data_root = '$PATH_TO_DATASET$/Data_indoor' # 数据根目录
with open(f'{data_root}/clips.json') as f:
    clip_infos = json.load(f)
with open(f'{data_root}/frames.json') as f:
    frame_infos = json.load(f)

token2ind = dict()
for ind, frame in enumerate(frame_infos):
    token2ind[frame['token']] = ind


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
            if counter.sum() > 0: # denoise
                if counter[:-1].sum() == 0:
                    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = 255
                else:
                    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter[:-1])
            else:
                processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = 0
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

def process_clip(clip_info):
    try:
        frames = clip_info['frames']
        scene_token = clip_info['token']

        # 读取当前clip叠帧点云标签
        stack_pc = pypcd.PointCloud.from_path(f'{data_root}/annotation/stack/{scene_token}.pcd')
        stack_points = np.zeros([stack_pc.width, 3], dtype=np.float32)
        stack_points[:, 0] = stack_pc.pc_data['x'].copy()
        stack_points[:, 1] = stack_pc.pc_data['y'].copy()
        stack_points[:, 2] = stack_pc.pc_data['z'].copy()
        stack_labels = np.zeros([stack_pc.width], dtype=np.int32)
        stack_labels[:] = stack_pc.pc_data[['class']].copy()

        stack_points = stack_points[stack_labels != 1] # 去掉静态行人类别
        stack_labels = stack_labels[stack_labels != 1]
        
        # 提取动态目标点云
        dynamic_zoo = dict()
        dynamic_blind_flag = dict()
        for token in frames:
            frame = frame_infos[token2ind[token]]
            frame_id = frame['frame_id']
            timestamp = frame['timestamp']
            lidar_path = frame['lidars']['LIDAR_TOP']['lidar_path']
            lidar2ego = np.array(frame['lidars']['LIDAR_TOP']['lidar2ego'])
            ego2global = np.array(frame['ego2global'])
            bbox_path = frame['bbox_path']
            
            # 读取单帧点云标签
            seg_pc = pypcd.PointCloud.from_path(os.path.join(data_root, lidar_path))
            seg_points = np.zeros([seg_pc.width, 3], dtype=np.float32)
            seg_points[:, 0] = seg_pc.pc_data['x'].copy()
            seg_points[:, 1] = seg_pc.pc_data['y'].copy()
            seg_points[:, 2] = seg_pc.pc_data['z'].copy()
            seg_points = seg_points @ lidar2ego[:3, :3].T + lidar2ego[:3, 3] # lidar -> ego
            seg_labels = np.zeros([seg_pc.width], dtype=np.int32)
            seg_labels[:] = seg_pc.pc_data[['class']].copy()

            # 读取包围框标注，用于抠取动态目标点
            with open(f'{data_root}/{bbox_path}', 'r') as f:
                lines = f.readlines()
            kitti_bboxes = [line.strip().split() for line in lines]
            # 遍历包围框，逐个动态目标抠取点云
            for kitti_bbox in kitti_bboxes:
                # 提取obj包围框内的点
                obj = {
                    'type': kitti_bbox[0],
                    'dimensions': [float(x) for x in kitti_bbox[1:4][::-1]],
                    'location': [float(x) for x in kitti_bbox[4:7]],
                    'yaw': float(kitti_bbox[7]),
                    'track_id': int(kitti_bbox[8]),
                }
                bbox = LiDARInstance3DBoxes([obj['location'] + obj['dimensions'] + [obj['yaw']]], origin=(0.5, 0.5, 0.5))
                box_idx = bbox.points_in_boxes_part(torch.tensor(seg_points, device='cuda', dtype=torch.float32))
                in_box = box_idx.cpu().numpy() == 0
                if kitti_bbox[0] == 'pedestrian' or kitti_bbox[0] == 'irregular_pedestrian':
                    in_box = np.logical_and(in_box, seg_labels == 1) # pedestrian
                elif kitti_bbox[0] == 'dynamic_objects':
                    in_box = np.logical_and(in_box, seg_labels == 6) # dynamic_objects
                obj_points = seg_points[in_box]
                
                 # 将obj框内的点从ego系变换到obj坐标系
                obj2ego = np.eye(4)
                obj2ego[:3, :3] = R.from_euler('z', obj['yaw']).as_matrix()
                obj2ego[:3, 3] = obj['location']
                ego2obj = np.linalg.inv(obj2ego)
                obj_points = obj_points @ ego2obj[:3, :3].T + ego2obj[:3, 3]

                # 判断目标是否在lidar盲区内，如果在盲区，后续叠帧的帧数会更多一些
                azimuth = np.arctan2(obj['location'][1], obj['location'][0]) / np.pi * 180
                in_blind = False
                for r in blind_ranges:
                    if azimuth > r[0] and azimuth < r[1]:
                        in_blind = True
                        break
                if np.linalg.norm(obj['location'][:2]) < 1.2:
                    in_blind = True
                track_id = obj['track_id']

                # 按照track_id和frame_id保存抠取结果
                if kitti_bbox[0] == 'pedestrian' or kitti_bbox[0] == 'irregular_pedestrian' or kitti_bbox[0] == 'dynamic_objects':
                    if not track_id in dynamic_zoo:
                        dynamic_zoo[track_id] = dict()
                        dynamic_blind_flag[track_id] = dict()
                    dynamic_zoo[track_id][frame_id] = obj_points
                    dynamic_blind_flag[track_id][frame_id] = in_blind

        for token in frames:
            frame = frame_infos[token2ind[token]]
            frame_id = frame['frame_id']
            bbox_path = frame['bbox_path']
            with open(f'{data_root}/{bbox_path}', 'r') as f:
                lines = f.readlines()
            kitti_bboxes = [line.strip().split() for line in lines]

            dynamic_points = []
            dynamic_labels = []
            for kitti_bbox in kitti_bboxes:
                obj = {
                    'type': kitti_bbox[0],
                    'dimensions': [float(x) for x in kitti_bbox[1:4][::-1]],
                    'location': [float(x) for x in kitti_bbox[4:7]],
                    'yaw': float(kitti_bbox[7]),
                    'track_id': int(kitti_bbox[8]),
                }
                obj2ego = np.eye(4)
                obj2ego[:3, :3] = R.from_euler('z', obj['yaw']).as_matrix()
                obj2ego[:3, 3] = obj['location']
                track_id = obj['track_id']

                # 将前后帧动态目标点，根据包围框对齐，变换到当前帧（不同类别，叠帧数量不同）
                if kitti_bbox[0] == 'pedestrian':
                    seq = range(-8, 9)
                    if dynamic_blind_flag[track_id][frame_id]:
                        seq = range(-9, 10)
                    for offset in seq: # stack 9
                        # target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dynamic_zoo[track_id][frame_id]))
                        if frame_id + offset in dynamic_zoo[track_id].keys():
                            obj_points = dynamic_zoo[track_id][frame_id + offset] @ obj2ego[:3, :3].T + obj2ego[:3, 3]

                            # icp
                            # source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dynamic_zoo[track_id][frame_id + offset]))
                            # icp_results = o3d.pipelines.registration.registration_icp(
                            #     source, target, 0.1
                            # )
                            # obj_points = dynamic_zoo[track_id][frame_id + offset] @ icp_results.transformation[:3, :3].T + icp_results.transformation[:3, 3]
                            # obj_points = obj_points @ obj2ego[:3, :3].T + obj2ego[:3, 3]
                            dynamic_points.append(obj_points)
                            dynamic_labels.append(np.full(len(obj_points), 1, dtype=np.int32))
                elif kitti_bbox[0] == 'irregular_pedestrian':
                    seq = range(-3, 4)
                    if dynamic_blind_flag[track_id][frame_id]:
                        seq = range(-6, 7)
                    for offset in seq:
                        if frame_id + offset in dynamic_zoo[track_id].keys():
                            obj_points = dynamic_zoo[track_id][frame_id + offset] @ obj2ego[:3, :3].T + obj2ego[:3, 3]
                            dynamic_points.append(obj_points)
                            dynamic_labels.append(np.full(len(obj_points), 1, dtype=np.int32))

            if len(dynamic_points) > 0:
                dynamic_points = np.concatenate(dynamic_points, axis=0)
                dynamic_labels = np.concatenate(dynamic_labels, axis=0)
            else:
                dynamic_points = np.zeros((0, 3), dtype=np.float32)
                dynamic_labels = np.zeros((0), dtype=np.int32)
            
            # 静态点云（叠帧标注） + 动态点云（包围框对齐叠帧）
            ego2global = np.array(frame['ego2global'])
            global2ego = np.linalg.inv(ego2global)
            static_points = stack_points @ global2ego[:3, :3].T + global2ego[:3, 3]
            occ_points = np.concatenate([static_points, dynamic_points], axis=0)
            occ_labels = np.concatenate([stack_labels, dynamic_labels], axis=0)

            ### points & labels in range
            range_mask = (np.abs(occ_points[:, 0]) < pc_range[3]) & (np.abs(occ_points[:, 1]) < pc_range[4]) \
                & (occ_points[:, 2] > pc_range[2]) & (occ_points[:, 2] < pc_range[5])
            points = occ_points[range_mask]
            labels = occ_labels[range_mask]

            # 0 for unoccupied, 255 for unknown
            labels[labels == 13] = 255 # household

            ### convert points to voxels
            pcd_np_coor = points
            pcd_np_coor[:, 0] = (pcd_np_coor[:, 0] - pc_range[0]) / voxel_size
            pcd_np_coor[:, 1] = (pcd_np_coor[:, 1] - pc_range[1]) / voxel_size
            pcd_np_coor[:, 2] = (pcd_np_coor[:, 2] - pc_range[2]) / voxel_size
            pcd_np_coor = np.floor(pcd_np_coor).astype(np.int32)
            pcd_np = np.concatenate([pcd_np_coor, labels[..., None]], axis=-1)

            pcd_np = pcd_np[np.lexsort((pcd_np_coor[:, 0], pcd_np_coor[:, 1], pcd_np_coor[:, 2])), :]
            pcd_np = pcd_np.astype(np.int64)
            processed_label = np.zeros(occ_size, dtype=np.uint8)
            processed_label = nb_process_label(processed_label, pcd_np)

            ################# convert voxel coordinates to LiDAR system  ##############
            x = np.linspace(0, occ_size[0] - 1, occ_size[0])
            y = np.linspace(0, occ_size[1] - 1, occ_size[1])
            z = np.linspace(0, occ_size[2] - 1, occ_size[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            vv = np.stack([X, Y, Z], axis=-1)
            fov_voxels = vv[processed_label > 0]
            fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
            fov_voxels[:, 0] += pc_range[0]
            fov_voxels[:, 1] += pc_range[1]
            fov_voxels[:, 2] += pc_range[2]
            fov_labels = processed_label[processed_label > 0]

            lidar_path = frame['lidars']['LIDAR_TOP']['lidar_path']
            file_name = os.path.basename(lidar_path).replace('pcd', 'npz')
            occ_out_path = f'{data_root}/annotation/occ/{scene_token}/{file_name}'
            if not os.path.exists(os.path.dirname(occ_out_path)):
                os.makedirs(os.path.dirname(occ_out_path))
            np.savez(occ_out_path, occ=processed_label)
    except Exception as e:
        raise e

if __name__ == '__main__':
    mp.set_start_method('spawn')

    for clip_info in clip_infos:
        if clip_info['token'] == '68bfa5402fd0fcdf4ad7ad44':
            process_clip(clip_info)

    # with mp.Pool(4) as pool:
    #     results = list(tqdm.tqdm(pool.imap(process_clip, clip_infos), total=len(clip_infos)))
        