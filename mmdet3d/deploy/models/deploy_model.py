import os
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import onnx
import tensorrt as trt
from mmdet3d.registry import MODELS
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from tools.misc.fuse_conv_bn import fuse_module
from mmdet3d.ops.voxelize import Voxelization
import time


@MODELS.register_module()
class FusionOccDeploy(BaseModel):
    def __init__(self,
                 model=None, 
                 checkpoint=None, 
                 mode='torch', 
                 fp16=False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.model = MODELS.build(model)
        load_checkpoint(self.model, checkpoint)
        self.fuse()
        self.model.eval()

        self.mode = mode
        self.fp16 = fp16

        self.voxel_layer = self.model.voxel_layer
        self.memory_len = self.model.memory_len
        self.init_memory()

        self.input_names = ['voxels', 'num_points', 'coors', 'voxel_masks', 'imgs', 'post_trans', 'ego2cam', 'distortion', 'intrinsic', 'curr2prev', 'memory_bev']
        self.output_names = ['pred_occ', 'memory_bank']
        self.dynamic_axes = {}

        self.profile_shapes = {
            'voxels': {'min': [5000, 50, 4], 'opt': [5000, 50, 4], 'max': [5000, 50, 4]},
            'num_points': {'min': [5000], 'opt': [5000], 'max': [5000]},
            'coors': {'min': [5000, 4], 'opt': [5000, 4], 'max': [5000, 4]},
            'voxel_masks': {'min': [5000], 'opt': [5000], 'max': [5000]},
            'imgs': {'min': [1, 6, 3, 768, 960], 'opt': [1, 6, 3, 768, 960], 'max': [1, 6, 3, 768, 960]},
            'post_trans': {'min': [1, 6, 4, 4], 'opt': [1, 6, 4, 4], 'max': [1, 6, 4, 4]},
            'ego2cam': {'min': [1, 6, 4, 4], 'opt': [1, 6, 4, 4], 'max': [1, 6, 4, 4]},
            'distortion': {'min': [1, 6, 8], 'opt': [1, 6, 8], 'max': [1, 6, 8]},
            'intrinsic': {'min': [1, 6, 3, 3], 'opt': [1, 6, 3, 3], 'max': [1, 6, 3, 3]},
            'curr2prev': {'min': [1, self.memory_len, 3, 3], 'opt': [1, self.memory_len, 3, 3], 'max': [1, self.memory_len, 3, 3]},
            'memory_bev': {'min': [1, self.memory_len, 128, 160, 160], 'opt': [1, self.memory_len, 128, 160, 160], 'max': [1, self.memory_len, 128, 160, 160]},
        }

        if self.mode == 'trt':
            TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            
            # 不做量化，直接转模型
            # if not os.path.exists("fusionocc.engine"):
            #     builder = trt.Builder(TRT_LOGGER)
            #     network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            #     parser = trt.OnnxParser(network, TRT_LOGGER)
                
            #     # 解析 ONNX 模型
            #     with open("fusionocc.onnx", 'rb') as model:
            #         if not parser.parse(model.read()):
            #             for error in range(parser.num_errors):
            #                 print(parser.get_error(error))
            #             return None
                
            #     # 配置构建器
            #     config = builder.create_builder_config()
            #     if self.fp16:
            #         config.set_flag(trt.BuilderFlag.FP16)
                
            #     # 设置最大工作空间
            #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
                
            #     # 设置动态输入范围
            #     profile = builder.create_optimization_profile()
            #     for key, value in self.profile_shapes.items():
            #         profile.set_shape(key, min=value['min'], opt=value['opt'], max=value['max'])
            #     config.add_optimization_profile(profile)

            #     # 构建引擎
            #     serialized_engine = builder.build_serialized_network(network, config)
            #     with open("fusionocc.engine", "wb") as f:
            #         f.write(serialized_engine)

            with open("fusionocc_int8.engine", "rb") as f:
                serialized_engine = f.read()
            runtime = trt.Runtime(TRT_LOGGER) 
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.context = self.engine.create_execution_context()

    def init_memory(self):
        self.memory_bev = torch.zeros(1, self.memory_len, self.model.single_bev_dims, self.model.bev_h, self.model.bev_w).cuda()
        self.memory_egopose_inv = torch.eye(4).reshape(1, 1, 4, 4).expand(1, self.memory_len, 4, 4).cuda()

    def prepare_inputs(self, batch_inputs_dict, img_metas):
        B = 1
        imgs = []
        for meta in img_metas:
            imgs_ = []
            ori_img = meta['ori_img'].numpy()
            for i in range(ori_img.shape[0]): 
                img = cv.resize(ori_img[i], (self.model.img_shape[1], self.model.img_shape[0]))
                imgs_.append(img)
            imgs.append(imgs_)
        imgs = torch.tensor(np.array(imgs)).cuda().to(torch.uint8).permute(0,1,4,2,3)
        points = batch_inputs_dict['points']

        voxel_size = self.voxel_layer['voxel_size']
        point_cloud_range = self.voxel_layer['point_cloud_range']
        max_num_points = self.voxel_layer['max_num_points']
        max_voxels = self.voxel_layer['max_voxels']

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

        fH, fW = self.model.img_shape
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
        prev_exists = torch.tensor([meta['prev_exists'] for meta in img_metas]).cuda()

        if not prev_exists:
            self.init_memory()

        curr2prev = torch.zeros((B, self.memory_len, 4, 4), device='cuda')
        for i in range(self.memory_len):
            memory_egopose_inv = self.memory_egopose_inv[:, i]
            curr2prev[:, i] = memory_egopose_inv @ ego_pose
        curr2prev = curr2prev[:, :, [0, 1, 3], :][:, :, :, [0, 1, 3]]

        self.memory_egopose_inv = torch.cat([self.memory_egopose_inv[:, 1:], ego_pose_inv.detach().unsqueeze(1)], dim=1)        

        return voxels, num_points, coors, voxel_masks, imgs, post_trans, ego2cam, distortion, intrinsic, curr2prev, self.memory_bev

    def forward(self, inputs, data_samples, mode='predict'):
        img_metas = [item.metainfo for item in data_samples]
        inputs = self.prepare_inputs(inputs, img_metas)
                          
        if self.mode == 'torch':
            # 保存输入数据，用于量化标定
            # if img_metas[0]['frame_idx'] % 5 == 0:
            #     torch.save(inputs, f"inputs/{img_metas[0]['scene_token']}_{img_metas[0]['frame_idx']}.pth")
            pred_occ, memory_bank = self.forward_torch(inputs)
        elif self.mode == 'onnx':
            self.forward_onnx(inputs)
        elif self.mode == 'trt':
            pred_occ, memory_bank = self.forward_trt(inputs)

        self.memory_bev = memory_bank

        gt_occupancy = torch.stack([item.gt_pts_seg.occupancy for item in data_samples], dim=0)
        lidar_origins = torch.stack([item.lidar_origins for item in data_samples], dim=0)

        bbox_list = [dict() for _ in range(len(img_metas))]

        for i, result_dict in enumerate(bbox_list):
            result_dict['pred_occupancy'] = pred_occ.argmax(-1).permute(1, 0, 2).int()
            result_dict['gt_occupancy'] = gt_occupancy[i]
            result_dict['lidar_origins'] = lidar_origins[i]
        return bbox_list
    
    def forward_torch(self, inputs):
        pred_occ, memory_bev = self.model.forward(*inputs) 
        return pred_occ, memory_bev
    
    def forward_onnx(self, inputs):
        torch.onnx.export(
            self.model,
            inputs,
            'fusionocc.onnx',
            export_params=True,
            opset_version=16,  # 建议使用11或更高版本
            do_constant_folding=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            verbose=True,
            custom_opsets={"org.tensorrt": 1}
        )
        exit()
    
    def forward_trt(self, inputs):
        torch.cuda.synchronize()
        t0 = time.time()
        inputs_dict = dict()
        for input_name, input_data in zip(self.input_names, inputs):
            inputs_dict[input_name] = input_data

        for input_name, input_tensor in inputs_dict.items():
            input_tensor = input_tensor.contiguous()
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())

        # create output tensors
        shape = tuple(self.context.get_tensor_shape('pred_occ'))
        pred_occ = torch.empty(size=shape, dtype=torch.float32, device='cuda')
        self.context.set_tensor_address('pred_occ', pred_occ.data_ptr())

        shape = tuple(self.context.get_tensor_shape('memory_bank'))
        memory_bank = torch.empty(size=shape, dtype=torch.float32, device='cuda')
        self.context.set_tensor_address('memory_bank', memory_bank.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

        torch.cuda.synchronize()
        t1 = time.time()
        print(t1 - t0)
        return pred_occ, memory_bank

    def fuse(self):
        self.model = fuse_module(self.model)

