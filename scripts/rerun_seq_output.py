#!/usr/bin/env python3
import os
import argparse

import pickle
import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import supervision as sv
import open3d as o3d
import cv2 as cv
from scipy.spatial.transform import Rotation as R
    
color_palette = sv.ColorPalette.from_hex([
    '#0000ff', '#9370db', '#f4a460', '#8b4513',
    '#90ee90', '#808080', '#87CEFF', '#8b0000', '#008000', 
    '#ffff00', '#ff8c00', '#ffdead'])
# color_palette = sv.ColorPalette.from_hex([
#     '#0000ff', '#90ee90', '#808080', '#9370db', '#ffdead'])
color_array = np.array([(color.r, color.g, color.b) for color in color_palette.colors])

# currently need to calculate the color manually
# see https://github.com/rerun-io/rerun/issues/4409
cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(
    vmin=0.5,
    vmax=10.0,
)

pc_range = [-10, -10, -1.5, 10, 10, 0.9]
# pc_range = [-8, -8, -1.5, 8, 8, 0.9]

def open3d_to_rr_pointcloud(pointcloud: o3d.geometry.PointCloud):
    points = np.asarray(pointcloud.points)
    point_colors = np.asarray(pointcloud.colors)
    return rr.Points3D(points, colors=point_colors)

data_prefix = dict(
    CAM_FRONT_LEFT=0,
    CAM_FRONT=1,
    CAM_FRONT_RIGHT=2,
    CAM_BACK_LEFT=3,
    CAM_BACK=4,
    CAM_BACK_RIGHT=5,
)


def log_lidar_and_ego_pose(points) -> None:
    """Log lidar data and vehicle pose."""
    point_distances = np.linalg.norm(points, axis=1)
    point_colors = cmap(norm(point_distances))
    rr.log(f"world/ego/LIDAR_TOP", rr.Points3D(points, colors=point_colors))


def log_cameras(imgs, cam2ego_list, cam2img_list, dist_list) -> None:
    """Log camera data."""
    # for i, cam_name in enumerate(list(data_prefix.keys())):
    for cam_name, i in data_prefix.items():
        cam2ego = cam2ego_list[[1,0,5,2,3,4][i]]
        cam2img = cam2img_list[[1,0,5,2,3,4][i]]
        dist = dist_list[i]
        img = imgs[[1,0,5,2,3,4][i]][:,:,::-1]
        img = cv.resize(img, (1920, 1536))
        rr.log(f"world/ego/cam/{cam_name}", rr.Image(img).compress(80))
        
        rr.log(
            f"world/ego/cam/{cam_name}",
            rr.Transform3D(
                translation=cam2ego[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(cam2ego[:3, :3]).as_quat(False)),
                from_parent=False,
            ),
            static=True,
        )
        rr.log(
            f"world/ego/cam/{cam_name}",
            rr.Pinhole(
                image_from_camera=np.array(cam2img),
                width=1920,
                height=1536,
            ),
            static=True,
        )


def log_occ(pred_occ, gt_occ=None) -> None:
    """Log occupancy."""
    ################# convert voxel coordinates to LiDAR system  ##############
    x = np.linspace(0, pred_occ.shape[0] - 1, pred_occ.shape[0])
    y = np.linspace(0, pred_occ.shape[1] - 1, pred_occ.shape[1])
    z = np.linspace(0, pred_occ.shape[2] - 1, pred_occ.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    valid_mask = np.logical_and(pred_occ > 0, pred_occ < 13)
    fov_voxels = vv[valid_mask]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * 0.1
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    fov_labels = pred_occ[valid_mask]

    num_points = len(fov_labels)
    sizes = [[0.1, 0.1, 0.1] for i in range(num_points)]
    centers = np.asarray(fov_voxels).tolist()
    quaternions = [rr.Quaternion.identity() for i in range(num_points)]
    colors = np.asarray(color_array[fov_labels - 1]).tolist()
    
    rr.log(
        "world/ego/pred_occ",
        rr.Boxes3D(
            sizes=sizes,
            centers=centers,
            quaternions=quaternions,
            colors=colors,
        ),
    )

    if not gt_occ is None:
        valid_mask = np.logical_and(gt_occ > 0, gt_occ < 13)
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
        colors = np.asarray(color_array[fov_labels - 1]).tolist()
        
        rr.log(
            "world/ego/gt_occ",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                colors=colors,
            ),
        )


def log_front_occ(pred_occ, ego2front) -> None:
    """Log occupancy."""
    ################# convert voxel coordinates to LiDAR system  ##############
    x = np.linspace(0, pred_occ.shape[0] - 1, pred_occ.shape[0])
    y = np.linspace(0, pred_occ.shape[1] - 1, pred_occ.shape[1])
    z = np.linspace(0, pred_occ.shape[2] - 1, pred_occ.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    valid_mask = np.logical_and(pred_occ > 0, pred_occ < 13)
    fov_voxels = vv[valid_mask]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * 0.1
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    fov_labels = pred_occ[valid_mask]

    num_points = len(fov_labels)
    sizes = [[0.1, 0.1, 0.1] for i in range(num_points)]
    centers = np.asarray(fov_voxels).tolist()
    quaternions = [rr.Quaternion.identity() for i in range(num_points)]
    colors = np.asarray(color_array[fov_labels - 1]).tolist()
    
    rr.log(
        "front/ego/pred_occ",
        rr.Boxes3D(
            sizes=sizes,
            centers=centers,
            quaternions=quaternions,
            colors=colors,
        ),
    )

    rr.log(
        "front/ego",
        rr.Transform3D(
            translation=ego2front[:3, 3],
            rotation=rr.Quaternion(xyzw=R.from_matrix(ego2front[:3, :3]).as_quat(False)),
            axis_length=1.0,  # The length of the visualized axis.
            from_parent=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the nuScenes dataset using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    sensor_views = [
        rrb.Spatial2DView(
            name=sensor_name,
            origin=f"world/ego/cam/{sensor_name}",
            # contents=["$origin/**", "world/ego/pred_occ"],
            overrides={"world/ego/pred_occ": rr.Boxes3D.from_fields(half_sizes=[[0.05, 0.05, 0.05]], fill_mode="solid")},
        )
        for sensor_name in data_prefix.keys()]

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Occupancy Prediction",
                    origin="world",
                    # Set the image plane distance to 5m for all camera visualizations.
                    defaults=[rr.Pinhole.from_fields(image_plane_distance=5.0)],
                    overrides={"world/ego/pred_occ": rr.Boxes3D.from_fields(fill_mode="solid"),
                               "world/ego/gt_occ": rr.Boxes3D.from_fields(fill_mode="solid")},
                    background="SolidColor",
                    line_grid=rrb.archetypes.LineGrid3D(visible=False)
                ),
                column_shares=[1, 1],
            ),        
            rrb.Horizontal(
                rrb.Grid(*sensor_views, grid_columns=3),
            ),
            row_shares=[4, 2],
        ),
        rrb.TimePanel(state="collapsed"),
    )

    rr.script_setup(args, "rerun_xhumanoid_results", default_blueprint=blueprint)
    rr.send_blueprint(blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # with open('/Users/0w0h0y/Desktop/media/users/haoyu.wang/EmbodiedOcc/outputs.pkl', 'rb') as f:
    with open('/Users/0w0h0y/Downloads/outputs.pkl', 'rb') as f:
        seq_results = pickle.load(f)
    
    for i, results in enumerate(seq_results):
        points = results['points']
        ego2global = results['ego2global']
        cam2ego=results['cam2ego']
        cam2img=results['cam2img'][0]
        distortion=results['distortion'][0]
        imgs = results['imgs']
        pred_occ = results['pred_occ'].transpose(1, 0, 2)
        gt_occ = results['gt_occ'].transpose(1, 0, 2)

        rr.set_time_sequence('frame_id', sequence=i)
        # rr.set_time_seconds("timestamp", seconds=float(results["timestamp"]) / 1e3)

        rr.log(
            "world",
            rr.Transform3D(
                translation=np.eye(4)[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(np.eye(4)[:3, :3]).as_quat(False)),
                # axis_length=1.0,  # The length of the visualized axis.
                from_parent=False,
            ),
        )

        rr.log(
            "world/ego",
            rr.Transform3D(
                translation=ego2global[:3, 3],
                rotation=rr.Quaternion(xyzw=R.from_matrix(ego2global[:3, :3]).as_quat(False)),
                axis_length=1.0,  # The length of the visualized axis.
                from_parent=False,
            ),
        )                
        log_lidar_and_ego_pose(points)
        log_cameras(imgs, cam2ego, cam2img, distortion)
        log_occ(pred_occ)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
