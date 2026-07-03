import os
import time
import torch
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

input_names = ['voxels', 'num_points', 'coors', 'voxel_masks', 'imgs', 'post_trans', 'ego2cam', 'distortion', 'intrinsic', 'curr2prev', 'memory_bev']
output_names = ['pred_occ', 'memory_bank']

memory_len = 1
profile_shapes = {
    'voxels': {'min': [5000, 50, 4], 'opt': [5000, 50, 4], 'max': [5000, 50, 4]},
    'num_points': {'min': [5000], 'opt': [5000], 'max': [5000]},
    'coors': {'min': [5000, 4], 'opt': [5000, 4], 'max': [5000, 4]},
    'voxel_masks': {'min': [5000], 'opt': [5000], 'max': [5000]},
    'imgs': {'min': [1, 6, 3, 768, 960], 'opt': [1, 6, 3, 768, 960], 'max': [1, 6, 3, 768, 960]},
    'post_trans': {'min': [1, 6, 4, 4], 'opt': [1, 6, 4, 4], 'max': [1, 6, 4, 4]},
    'ego2cam': {'min': [1, 6, 4, 4], 'opt': [1, 6, 4, 4], 'max': [1, 6, 4, 4]},
    'distortion': {'min': [1, 6, 8], 'opt': [1, 6, 8], 'max': [1, 6, 8]},
    'intrinsic': {'min': [1, 6, 3, 3], 'opt': [1, 6, 3, 3], 'max': [1, 6, 3, 3]},
    'curr2prev': {'min': [1, memory_len, 3, 3], 'opt': [1, memory_len, 3, 3], 'max': [1, memory_len, 3, 3]},
    'memory_bev': {'min': [1, memory_len, 128, 160, 160], 'opt': [1, memory_len, 128, 160, 160], 'max': [1, memory_len, 128, 160, 160]},
}

profile_dtypes = {
    'voxels': np.float32,
    'num_points': np.int32,
    'coors': np.int32,
    'voxel_masks': np.int32,
    'imgs': np.uint8,
    'post_trans': np.float32,
    'ego2cam': np.float32,
    'distortion': np.float32,
    'intrinsic': np.float32,
    'curr2prev': np.float32,
    'memory_bev': np.float32,
}

def calibration_dataset():
    for data in os.listdir('inputs'):
        inputs = torch.load(os.path.join(f'inputs/{data}'))
        yield inputs

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, profile_shapes, profile_dtypes, cache_file="calib.cache"):
        super(Int8Calibrator, self).__init__()
        self.data_loader = data_loader
        self.profile_shapes = profile_shapes
        self.cache_file = cache_file
        
        cudart.cudaFree(0)
        self.device_inputs = {}
        for name, value in profile_shapes.items():
            err, ptr = cudart.cudaMalloc(int(np.prod(value['opt']) * np.dtype(profile_dtypes[name]).itemsize))
            self.device_inputs[name] = ptr
        
    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        try:
            sample = next(self.data_loader)
        except StopIteration:
            return None

        # 将数据拷贝到设备内存
        bindings = []
        for i, name in enumerate(names):
            host_input = sample[i].cpu().numpy().ravel()
            print(f"[Calibrator] Input: {name}, dtype: {host_input.dtype}, shape: {host_input.shape}")
            err, = cudart.cudaMemcpy(
                self.device_inputs[name],
                host_input.ctypes.data,
                host_input.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            
            bindings.append(int(self.device_inputs[name]))
        
        return bindings
    
    def read_calibration_cache(self):
        pass

    def write_calibration_cache(self, cache):
        pass


class MyProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.record = {}

    def report_layer_time(self, layer_name, time_ms):
        if layer_name not in self.record:
            self.record[layer_name] = 0.0
        self.record[layer_name] += time_ms

    def print_layer_times(self):
        cur_sum = 0
        total = sum(self.record.values())
        with open('layer_profile.txt', 'w') as f:
            f.write("Layer-wise execution time (ms): \n")
            for layer, time in self.record.items():
                cur_sum += time
                f.write(f"{layer}: {time:.3f} ms --- {(cur_sum / total) * 100:.2f}% \n")


profiler = MyProfiler()
data_iterator = calibration_dataset()
calibrator = Int8Calibrator(data_iterator, profile_shapes, profile_dtypes)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

if not os.path.exists("fusionocc_int8.engine"):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析 ONNX 模型
    with open("fusionocc.onnx", 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()

    # 配置构建器
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # int8量化
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # with open('layer_names.txt', 'w') as f:
    #     for i in range(network.num_layers):
    #         layer = network.get_layer(i)
    #         f.write(f"{layer.name} ===> type: {layer.type}, precision: {layer.precision}, outputs: ")
    #         for j in range(layer.num_outputs):
    #             output = layer.get_output(j)
    #             f.write(f"{output.dtype} ")
    #         f.write(' \n')

    # 逐层处理输出类型
    for i in range(network.num_layers):
        layer = network.get_layer(i)

        if "img_backbone" not in layer.name: # and 'myl' not in layer.name:
            unsupport_float = False
            for j in range(layer.num_outputs):
                output = layer.get_output(j)
                if output.dtype in (trt.DataType.INT64, trt.DataType.INT32, trt.DataType.BOOL):
                    unsupport_float = True
                    break
            if not unsupport_float and layer.precision == trt.DataType.FLOAT:
                layer.precision = trt.DataType.HALF

    # 设置最大工作空间
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # 设置动态输入范围
    profile = builder.create_optimization_profile()
    for key, value in profile_shapes.items():
        profile.set_shape(key, min=value['min'], opt=value['opt'], max=value['max'])
    config.add_optimization_profile(profile)

    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)
    with open("fusionocc_int8.engine", "wb") as f:
        f.write(serialized_engine)

with open("fusionocc_int8.engine", "rb") as f:
    serialized_engine = f.read()
runtime = trt.Runtime(TRT_LOGGER) 
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# 用于输入和输出精度对齐，验证不同设备上做量化的结果正确
# # inputs = torch.load('inputs.pth')
# # outputs = torch.load('outputs.pth')

# for i in range(10):
#     if i == 9:
#         context.profiler = profiler
#     torch.cuda.synchronize()
#     t0 = time.time()
#     inputs_dict = dict()
#     for input_name, input_data in zip(input_names, inputs):
#         inputs_dict[input_name] = input_data

#     for input_name, input_tensor in inputs_dict.items():
#         input_tensor = input_tensor.contiguous()
#         context.set_tensor_address(input_name, input_tensor.data_ptr())

#     # create output tensors
#     shape = tuple(context.get_tensor_shape('pred_occ'))
#     pred_occ = torch.empty(size=shape, dtype=torch.float32, device='cuda')
#     context.set_tensor_address('pred_occ', pred_occ.data_ptr())

#     shape = tuple(context.get_tensor_shape('memory_bank'))
#     memory_bank = torch.empty(size=shape, dtype=torch.float32, device='cuda')
#     context.set_tensor_address('memory_bank', memory_bank.data_ptr())

#     context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
#     torch.cuda.current_stream().synchronize()

#     torch.cuda.synchronize()
#     t1 = time.time()
#     print(t1 - t0)
    
#     output = (pred_occ, memory_bank)
#     torch.save(output, 'outputs.pth')
#     # torch.save(input, 'inputs.pth')
#     # assert torch.allclose(pred_occ, outputs[0], rtol=1e-2, atol=1e-1)
#     # assert torch.allclose(memory_bank, outputs[1], rtol=1e-2, atol=1e-1)

# profiler.print_layer_times()