#!/usr/bin/env python3
import os
import argparse
import time
import json
import pickle
import trimesh
import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import supervision as sv
import open3d as o3d
import cv2 as cv
from pypcd import pypcd
from scipy.spatial.transform import Rotation as R

# currently need to calculate the color manually
# see https://github.com/rerun-io/rerun/issues/4409
cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(
    vmin=0.5,
    vmax=10.0,
)

pc_range = [-10, -10, -1.5, 10, 10, 0.9]
# pc_range = [-20, -20, -1.5, 20, 20, 0.9]

def open3d_to_rr_pointcloud(pointcloud: o3d.geometry.PointCloud):
    points = np.asarray(pointcloud.points)
    point_colors = np.asarray(pointcloud.colors)
    return rr.Points3D(points, colors=point_colors)

color_palette = sv.ColorPalette.from_hex([
    '#0000ff', '#9370db', '#f4a460', '#8b4513',
    '#90ee90', '#808080', '#87CEFF', '#8b0000', '#008000', 
    '#ffff00', '#ff8c00', '#ffdead', '#FFC0CB'])
# color_palette = sv.ColorPalette.from_hex([
#     '#0000ff', '#90ee90', '#808080', '#9370db', '#ffdead', '#87CEFF', '#8b0000', '#008000'])
# color_palette = sv.ColorPalette.from_hex([
    # '#0000ff', '#9370db', '#87CEFF', '#808080', '#8b0000', '#FFFFE0', '#ffff00', 
    # '#006400', '#008000', '#90ee90', '#ffdead', '#FFC0CB'])
color_array = np.array([(color.r, color.g, color.b) for color in color_palette.colors])

data_prefix = dict(
    CAM_FRONT_LEFT="CAM_FRONT_LEFT",
    CAM_FRONT="CAM_FRONT",
    CAM_FRONT_RIGHT="CAM_FRONT_RIGHT",
    CAM_BACK_LEFT="CAM_BACK_LEFT",
    CAM_BACK="CAM_BACK",
    CAM_BACK_RIGHT="CAM_BACK_RIGHT",
)

data_root = '$PATH_TO_DATASET$/Data_indoor'
# data_root = '/Users/0w0h0y/Desktop/media/datasets/humanoid/robotic'
with open(f'{data_root}/clips.json') as f:
    clip_infos = json.load(f)
with open(f'{data_root}/frames.json') as f:
    frame_infos = json.load(f)

token2ind = dict()
for ind, frame in enumerate(frame_infos):
    token2ind[frame['token']] = ind

def log_cameras(imgs, cam2lidar_list, cam2img_list, dist_list) -> None:
    """Log camera data."""
    for i, cam_name in enumerate(list(data_prefix.keys())):
        img = imgs[i]
        cam2lidar = cam2lidar_list[i]
        cam2img = cam2img_list[i]
        dist = dist_list[i]
        img = cv.undistort(img, cam2img, dist)
        rr.log(f"world/lidar/cam/{cam_name}", rr.Image(img[:,:,::-1]).compress(80))
        
        rr.log(
            f"world/lidar/cam/{cam_name}",
            rr.Transform3D(
                translation=cam2lidar[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(cam2lidar[:3, :3]).as_quat(False)),
                from_parent=False,
            ),
            static=True,
        )
        rr.log(
            f"world/lidar/cam/{cam_name}",
            rr.Pinhole(
                image_from_camera=cam2img,
                width=1920,
                height=1536,
            ),
            static=True,
        )

def log_lidar_and_lidar_pose(points, lidar2global=None, point_colors=None) -> None:
    """Log lidar data and vehicle pose."""
    if lidar2global is not None:
        rr.log(
            "world/lidar",
            rr.Transform3D(
                translation=lidar2global[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(lidar2global[:3, :3]).as_quat()),
                axis_length=1.0,  # The length of the visualized axis.
                from_parent=False,
            ),
        )

    if point_colors is None:
        point_distances = np.linalg.norm(points, axis=1)
        point_colors = cmap(norm(point_distances))
    rr.log(f"world/lidar/LIDAR_TOP", rr.Points3D(points, colors=point_colors))

def log_bbox(bboxes) -> None:
    sizes = [[float(x) for x in bbox[1:4][::-1]] for bbox in bboxes]
    centers = [[float(x) for x in bbox[4:7]] for bbox in bboxes]
    yaws = [float(bbox[7]) for bbox in bboxes]
    quaternions = [rr.Quaternion(xyzw=R.from_euler('z', yaw).as_quat()) for yaw in yaws]
    
    rr.log(
        "world/ego/bbox",
        rr.Boxes3D(
            sizes=sizes,
            centers=centers,
            quaternions=quaternions,
        ),
    )

def log_occ(gt_occ, ego2global=None) -> None:
    """Log occupancy."""
    if ego2global is not None:
        rr.log(
            "world/ego",
            rr.Transform3D(
                translation=ego2global[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(ego2global[:3, :3]).as_quat()),
                axis_length=1.0,  # The length of the visualized axis.
                from_parent=False,
            ),
        )

    ################# convert voxel coordinates to LiDAR system  ##############
    x = np.linspace(0, gt_occ.shape[0] - 1, gt_occ.shape[0])
    y = np.linspace(0, gt_occ.shape[1] - 1, gt_occ.shape[1])
    z = np.linspace(0, gt_occ.shape[2] - 1, gt_occ.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    valid_mask = np.logical_and(gt_occ > 0, gt_occ < 255)
    fov_voxels = vv[valid_mask]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * 0.1
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    fov_labels = gt_occ[valid_mask]

    num_points = len(fov_labels)
    sizes = [[0.1, 0.1, 0.1] for i in range(num_points)]
    centers = np.asarray(fov_voxels).tolist()
    quaternions = [rr.Quaternion.identity() for i in range(num_points)]
    colors = np.asarray(color_array[(fov_labels - 1) % len(color_array)]).tolist()
    
    rr.log(
        "world/ego/gt_occ",
        rr.Boxes3D(
            sizes=sizes,
            centers=centers,
            quaternions=quaternions,
            colors=colors,
        ),
    )

def main() -> None:

    sensor_views = [
        rrb.Spatial2DView(
            name=sensor_name,
            origin=f"world/lidar/cam/{sensor_name}",
            contents=["$origin/**", 
                      "world/lidar/LIDAR_TOP"
                      ],
        )
        for sensor_name in data_prefix.keys()
    ]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D",
                    origin="world",
                    # Set the image plane distance to 5m for all camera visualizations.
                    # defaults=[rr.Pinhole.from_fields(image_plane_distance=5.0)],
                    # overrides={"world/ego/pred_occ": rr.Boxes3D.from_fields(fill_mode="solid"),
                    #            "world/ego/gt_occ": rr.Boxes3D.from_fields(fill_mode="solid")},
                ),
                column_shares=[3, 1],
            ),
            rrb.Grid(*sensor_views, grid_columns=3),
            row_shares=[4, 2],
        ),
        rrb.TimePanel(state="collapsed"),
    )

    # rr.init("rerun_annotation", spawn=True, default_blueprint=blueprint)

    # 修改为 (去掉 spawn=True)
    rr.init("rerun_annotation", default_blueprint=blueprint)
    rr.serve(open_browser=False, web_port=9090)

    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    for k, clip_info in enumerate(clip_infos):
        if clip_info['token'] != '68c3c0a1a3d54bfb1dfe47eb':
            continue
        scene_token = clip_info['token']
        frames = clip_info['frames']
        for token in frames:
            frame = frame_infos[token2ind[token]]
            frame_id = frame['frame_id']
            if frame_id % 100 != 0:
                continue
            timestamp = frame['timestamp']
            ego2global = np.array(frame['ego2global'])
            lidar2ego = np.array(frame['lidars']['LIDAR_TOP']['lidar2ego'])
            lidar2global = ego2global @ lidar2ego

            lidar_path = frame['lidars']['LIDAR_TOP']['lidar_path']

            pc = pypcd.PointCloud.from_path(os.path.join(data_root, lidar_path))
            points = np.zeros([pc.width, 3], dtype=np.float32)
            points[:, 0] = pc.pc_data['x'].copy()
            points[:, 1] = pc.pc_data['y'].copy()
            points[:, 2] = pc.pc_data['z'].copy()
            labels = np.zeros([pc.width], dtype=np.int32)
            labels[:] = pc.pc_data[['class']].copy()
            point_colors = color_array[(labels - 1) % len(color_array)]

            imgs = []
            cam2lidar_list = []
            cam2img_list = []
            dist_list = []
            for cam_id in data_prefix.keys():
                img_path = frame['cameras'][cam_id]['img_path']
                cam2lidar = np.array(frame['cameras'][cam_id]['cam2lidar'])
                intrinsic = np.array(frame['cameras'][cam_id]['intrinsic'])
                cam2img = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
                distortion = np.array(list(frame['cameras'][cam_id]['distortion'].values()))
                cam2lidar_list.append(cam2lidar)
                cam2img_list.append(cam2img)
                dist_list.append(distortion)

                img = cv.imread(f'{data_root}/{img_path}')
                imgs.append(img)
            
            lidar_path = frame['lidars']['LIDAR_TOP']['lidar_path']
            file_name = os.path.basename(lidar_path).replace('pcd', 'npz')
            occ_path = f'{data_root}/annotation/occ/{scene_token}/{file_name}' 
            occ = np.load(occ_path)['occ'][:,:,:24]

            bbox_path = frame['bbox_path']
            with open(f'{data_root}/{bbox_path}', 'r') as f:
                lines = f.readlines()
            bboxes = [line.strip().split() for line in lines]

            rr.set_time_sequence('clip_id', sequence=k)
            rr.set_time_sequence('frame_id', sequence=frame['frame_id'] + k * 200)
            rr.set_time_seconds("timestamp", seconds=float(timestamp) / 1e3)
            log_lidar_and_lidar_pose(points, lidar2global)
            log_cameras(imgs, cam2lidar_list, cam2img_list, dist_list)
            log_bbox(bboxes)
            log_occ(occ, ego2global=ego2global)


if __name__ == "__main__":
    main()
