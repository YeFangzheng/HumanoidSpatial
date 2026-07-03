import torch
import torch.nn as nn
import torch.nn.functional as F

def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''

    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).long() 
    for i in range(gt.shape[0]):
        coords = torch.where(gt_occ[i])
        new_coords = [coord // ratio for coord in coords]
        # coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, new_coords[0], new_coords[1], new_coords[2]] = gt_occ[i][coords].long()
    
    return gt