import torch
import tensorrt as trt
from mmengine.model import BaseModule
from mmdet3d.ops.ms_deform_attn import MultiScaleDeformableAttnFunction

class TestModel(BaseModule):
    def forward(self, inputs):
        inputs = (*inputs, 64)
        output = MultiScaleDeformableAttnFunction.apply(*inputs)
        return output
    
def test_torch_trt_consistency():

    model = TestModel().eval()
    
    inputs = torch.load('mmdet3d/ops/ms_deform_attn/inputs.pth')[:5]

    with torch.no_grad():
        torch_output = model(inputs)
    
    input_names = ["value", "value_spatial_shapes", "value_level_start_index", "sampling_locations", "attention_weights"]
    output_names = ["output"]
    torch.onnx.export(model, (inputs,), "mmdet3d/ops/ms_deform_attn/test.onnx", 
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=16)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open("mmdet3d/ops/ms_deform_attn/test.onnx", 'rb') as model:
        if not parser.parse(model.read()):
            raise RuntimeError("ONNX解析失败")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    context = engine.create_execution_context()

    for input_tensor, input_name in zip(inputs, input_names):
        input_tensor = input_tensor.contiguous()
        context.set_tensor_address(input_name, input_tensor.data_ptr())
    
    shape = tuple(context.get_tensor_shape("output"))
    trt_output = torch.empty(size=shape, dtype=torch.float32, device='cuda')
    context.set_tensor_address("output", trt_output.data_ptr())

    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()
    
    assert torch.allclose(torch_output, trt_output, atol=1e-5, rtol=1e-5), "PyTorch和TensorRT结果不一致"

test_torch_trt_consistency()