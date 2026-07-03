import torch
from .utils import rotation_3d_in_axis, DUMP, decode_bbox
from mmdet3d.ops.msmv_sampling import msmv_sampling


def make_sample_points_from_bbox(query_bbox, offset, pc_range):
    '''
    query_bbox: [B, Q, 10]
    offset: [B, Q, num_points, 4], normalized by stride
    '''
    query_bbox = decode_bbox(query_bbox, pc_range)  # [B, Q, 9]

    xyz = query_bbox[..., 0:3]  # [B, Q, 3]
    wlh = query_bbox[..., 3:6]  # [B, Q, 3]

    # NOTE: different from SparseBEV
    xyz += wlh / 2  # conver to center
    
    delta_xyz = offset[..., 0:3]  # [B, Q, P, 3]
    delta_xyz = wlh[:, :, None, :] * delta_xyz  # [B, Q, P, 3]

    if query_bbox.shape[-1] > 6:
        ang = query_bbox[..., 6:7]  # [B, Q, 1]
        delta_xyz = rotation_3d_in_axis(delta_xyz, ang)  # [B, Q, P, 3]
    
    sample_xyz = xyz[:, :, None, :] + delta_xyz  # [B, Q, P, 3]

    return sample_xyz  # [B, Q, P, 3]


def make_sample_points_from_mask(valid_map, pc_range, occ_size, num_points, occ_loc=None, offset=None):
    '''
    valid_map: [B, Q, W, H, D] or [B, Q, N]
    occ_loc: [B, N, 3] if valid map is sparse
    Return: [B, Q, GP, 3] in pc_range
    '''
    B, Q = valid_map.shape[:2]
    occ_size = torch.tensor(occ_size).to(valid_map.device)
    
    sampling_pts = []
    for b in range(B):
        indices = torch.where(valid_map[b])
        if indices[0].shape[0] == 0:
            pts = torch.rand((Q, num_points, 3)).to(valid_map.device)
        else:
            if len(valid_map.shape) == 5:
                bin_count = valid_map[b].sum(dim=(1,2,3))
            else:
                bin_count = valid_map[b].sum(dim=1)
            sampling_rand = torch.rand((Q, num_points)).to(bin_count.device)
            sampling_index = (sampling_rand * bin_count[:, None]).floor().long()
            low_bound = torch.cumsum(bin_count, dim=0) - bin_count
            sampling_index = sampling_index + low_bound[:, None]
            sampling_index[sampling_index >= indices[0].shape[0]] = indices[0].shape[0] -1  # this can happen when zeros appear in the tail
            sampling_index = sampling_index.to(valid_map.device)
            
            if occ_loc is None: # dense occ points
                pts = torch.stack((indices[1][sampling_index], indices[2][sampling_index], indices[3][sampling_index]))
                pts = pts.permute(1, 2, 0)
            else:
                occ_idx = indices[1][sampling_index]
                pts = occ_loc[b][occ_idx]
        
            # pad queries with no valid occ
            pts = pts.float()
            rand_sampling_points = torch.rand(((bin_count==0).sum(), num_points, 3)).to(pts.device) * occ_size
            pts[bin_count==0] = rand_sampling_points
        sampling_pts.append(pts)
        
    sampling_pts = torch.stack(sampling_pts)
    if offset is not None:
        sampling_pts = sampling_pts + offset
    
    sampling_pts = sampling_pts / occ_size
    sampling_pts[..., 0] = sampling_pts[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    sampling_pts[..., 1] = sampling_pts[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    sampling_pts[..., 2] = sampling_pts[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    return sampling_pts


def sampling_4d(sample_points, mlvl_feats, scale_weights, img_metas, image_h, image_w, eps=1e-5):
    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 4]
    N = 6
    device = sample_points.device

    sample_points = sample_points.reshape(B, Q, T, G * P, 3)
    
    if DUMP.enabled:
        torch.save(sample_points,
                   '{}/sample_points_3d_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    cam2egos = torch.stack([meta['cam2ego'] for meta in img_metas], dim=0).to(device)
    intrinsics = torch.stack([meta['intrinsic'] for meta in img_metas], dim=0).to(device)
    post_trans = torch.stack([meta['post_trans'] for meta in img_metas], dim=0).to(device)
    bda = torch.stack([meta['bda'] for meta in img_metas], dim=0).to(device)

    sample_points = torch.inverse(bda).view(B, 1, 1, 1, 3,
                        3).matmul(sample_points.unsqueeze(-1)).squeeze(-1)

    sample_points = sample_points[:, :, None, ...]     # [B, Q, T, GP, 3]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 3)
    sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 3]

    ego2cams = torch.inverse(cam2egos)
    rots = ego2cams[..., :3, :3]
    trans = ego2cams[..., :3, 3]
    sample_points = rots.view(B, 1, N, 1, 1, 3, 3).matmul(sample_points.unsqueeze(-1)) + trans.view(B, 1, N, 1, 1, 3, 1)

    sample_points_cam = intrinsics.view(B, 1, N, 1, 1, 3, 3).matmul(sample_points).squeeze(-1)

    sample_points_cam = torch.cat([sample_points_cam[..., 0:2] / torch.maximum(
        sample_points_cam[..., 2:3], torch.ones_like(sample_points_cam[..., 2:3])*eps),  sample_points_cam[..., 2:3]], 5
        )
    sample_points_cam = post_trans[..., :3, :3].view(B, 1, N, 1, 1, 3, 3).matmul(sample_points_cam.unsqueeze(-1)).squeeze(-1)
    sample_points_cam += post_trans[..., :3, 3].view(B, 1, N, 1, 1, 3) 

    # # get the projection matrix
    # lidar2img = lidar2img[:, :(T*N), None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    # lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
    # lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

    # # expand the points
    # ones = torch.ones_like(sample_points[..., :1])
    # sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    # sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    # sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    # sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

    # # project 3d sampling points to image
    # sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    # sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
    sample_points_cam = sample_points_cam[..., 0:2]

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, T, N, Q, GP]

    if DUMP.enabled:
        torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1),
                   '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
        torch.save(valid_mask,
                   '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    # import cv2 as cv
    # import matplotlib
    # import numpy as np

    # cmap = matplotlib.colormaps["turbo_r"]
    # norm = matplotlib.colors.Normalize(
    #     vmin=0.5,
    #     vmax=5.0,
    # )
    # big_img = np.zeros((image_h * 2, image_w * 3, 3), dtype=np.uint8)
    # for i in range(2):
    #     for j in range(3):
    #         k = i * 3 + j
    #         img = np.zeros((image_h, image_w, 3))
    #         points_2d = sample_points_cam[0, :, :, :, k][valid_mask[0, :, :, :, k].bool()].detach().cpu().numpy()

    #         for p in points_2d:
    #             cv.circle(img, (int(p[0] * image_w), int(p[1] * image_h)), 3, (0, 0, 255))

    #         big_img[image_h * i:image_h * (i + 1), image_w * j:image_w * (j + 1)] = img
    # cv.imwrite(f'proj.jpg', big_img)

    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / 5], dim=-1)
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)

    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final
