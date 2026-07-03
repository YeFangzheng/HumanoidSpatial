import os
import time
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.ops.voxelize import Voxelization

raw_inputs = torch.load('raw_inputs.pth')
batch_inputs_dict = raw_inputs['batch_inputs_dict']
img_metas = raw_inputs['img_metas']
# memory_egopose_inv = raw_inputs['memory_egopose_inv']
# memory_bev = raw_inputs['memory_bev']

memory_len = 1

# 读取点云，调用算子体素化
points = batch_inputs_dict['points']

voxel_size = [0.1, 0.1, 4]
point_cloud_range = [-8, -8, -1.5, 8, 8, 0.9]
max_num_points = 50
max_voxels = 5000

voxels, coors, num_points, voxel_masks = [], [], [], []
for i, res in enumerate(points):
    res_voxels, res_coors, res_num_points, res_voxel_nums = Voxelization.apply(res, voxel_size, point_cloud_range,
                                                                        max_num_points, max_voxels, True)
    res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
    voxel_mask = torch.zeros_like(res_num_points)
    voxel_mask[:res_voxel_nums] = 1
    voxels.append(res_voxels)
    coors.append(res_coors)
    num_points.append(res_num_points)
    voxel_masks.append(voxel_mask)

voxels = torch.cat(voxels, dim=0)
coors = torch.cat(coors, dim=0)
num_points = torch.cat(num_points, dim=0)
voxel_masks = torch.cat(voxel_masks, dim=0)

imgs = []
for meta in img_metas:
    imgs_ = []
    ori_img = meta['ori_img'].numpy()
    for i in range(ori_img.shape[0]): 
        img = cv.resize(ori_img[i], (960, 768))
        imgs_.append(img)
    imgs.append(imgs_)
imgs = torch.tensor(np.array(imgs)).cuda().to(torch.uint8).permute(0,1,4,2,3)
fH, fW = 768, 960
B, N, _, H, W = imgs.shape

resize = float(fW) / float(1920)
resize_dims = (int(1536 * resize), int(1920 * resize))
newH, newW = resize_dims
crop_h = newH - fH
crop_w = int(max(0, newW - fW) / 2)
crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

post_trans = torch.eye(4, dtype=torch.float32).cuda()
post_trans[:2, :2] *= resize
post_trans[0, 3] -= crop[0]
post_trans[1, 3] -= crop[1]
post_trans = post_trans.reshape(1, 1, 4, 4).repeat(B, N, 1 ,1)

cam2ego = torch.stack([meta['cam2ego'] for meta in img_metas], dim=0).cuda()
ego2cam = torch.inverse(cam2ego)
distortion = torch.stack([meta['distortion'] for meta in img_metas], dim=0).cuda()
intrinsic = torch.stack([meta['intrinsic'] for meta in img_metas], dim=0).cuda()

ego_pose = torch.stack([meta['ego_pose'] for meta in img_metas], dim=0).cuda()
ego_pose_inv = torch.stack([meta['ego_pose_inv'] for meta in img_metas], dim=0).cuda()

# # init 
memory_bev = torch.zeros(1, memory_len, 128, 200, 200).cuda() # 初始帧memory_bev
memory_egopose_inv = torch.eye(4).reshape(1, 1, 4, 4).expand(1, memory_len, 4, 4).cuda() # 初始化历史帧ego_pose_inv

curr2prev = torch.zeros((B, memory_len, 4, 4), device='cuda')
for i in range(memory_len):
    memory_egopose_inv_i = memory_egopose_inv[:, i]
    curr2prev[i] = memory_egopose_inv_i @ ego_pose
curr2prev = curr2prev[:, :, [0, 1, 3], :][:, :, :, [0, 1, 3]]

memory_egopose_inv = torch.cat([memory_egopose_inv[:, 1:], ego_pose_inv.detach().unsqueeze(1)], dim=1) # 更新历史帧ego_pose_inv
inputs = (voxels, num_points, coors, voxel_masks, imgs, post_trans, ego2cam, distortion, intrinsic, curr2prev, memory_bev)

torch.save(inputs, 'inputs.pth')