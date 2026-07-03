import torch
import torch.nn.functional as F
import tensorrt as trt
from mmengine.model import BaseModule
from mmdet3d.ops.voxelize import Voxelization, VoxelGenerator
import numpy as np
import cv2 as cv

class TorchModel(BaseModule):
    def forward(self, points):
        max_num_points = 20
        voxel_size = [0.2, 0.2, 8]
        max_voxels = 40000
        coors_range = [-40, -40, -1.0, 40, 40, 5.4]
        
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points, num_voxel = Voxelization.apply(
                res, voxel_size, coors_range, max_num_points, max_voxels, True)

            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        num_points = torch.cat(num_points, dim=0)
        
        return voxels, coors, num_points, num_voxel.int()
    
class TensorrtModel(BaseModule):
    def forward(self, points):
        max_num_points = 20
        voxel_size = [0.2, 0.2, 8]
        max_voxels = 40000
        coors_range = [-40, -40, -1.0, 40, 40, 5.4]
        voxel_feature_num = 5

        num_points = points.any(-1).sum(-1).int()
        voxels, coors, num_voxel = VoxelGenerator.apply(
            points, num_points, max_num_points, max_voxels, coors_range, voxel_feature_num, voxel_size)
        num_points = voxels.any(-1).sum(-1).int()
        return voxels, coors, num_points, num_voxel

def test_torch_trt_consistency():
    model = TorchModel().eval()
    
    points = torch.load('mmdet3d/ops/voxelize/input.pth')

    with torch.no_grad():
        torch_outputs = model(points)
    
    model_trt = TensorrtModel().eval()
    tensorrt_outputs = model_trt(points)

    input_names = ["points"]
    output_names = ["voxels", "coors", "num_points", "num_voxel"]
    torch.onnx.export(model_trt, (points,), "mmdet3d/ops/voxelize/test.onnx", 
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=16)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open("mmdet3d/ops/voxelize/test.onnx", 'rb') as model:
        if not parser.parse(model.read()):
            raise RuntimeError("ONNX解析失败")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 40)
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    context = engine.create_execution_context()

    input_tensor = points.contiguous()
    context.set_tensor_address("points", input_tensor.data_ptr())
    
    trt_outputs = []
    for tensorrt_output, output_name in zip(tensorrt_outputs, output_names):
        trt_output = torch.empty(size=tensorrt_output.shape, dtype=tensorrt_output.dtype, device='cuda').contiguous()
        trt_outputs.append(trt_output)
        context.set_tensor_address(output_name, trt_output.data_ptr())

    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()
    
    torch_pillar = torch.zeros((400, 400))
    torch_pillar[torch_outputs[1][:, 2], torch_outputs[1][:, 3]] = 1
    trt_pillar = torch.zeros((400, 400))
    trt_pillar[trt_outputs[1][0, :, 2], trt_outputs[1][0, :, 3]] = 1
    cv.imwrite('torch.png', torch_pillar.cpu().numpy().astype(np.uint8) * 255)
    cv.imwrite('trt.png', trt_pillar.cpu().numpy().astype(np.uint8) * 255)
    for torch_output, trt_output in zip(torch_outputs, trt_outputs):
        assert torch.allclose(torch_output, trt_output, atol=1e-5), "PyTorch和TensorRT结果不一致"

test_torch_trt_consistency()