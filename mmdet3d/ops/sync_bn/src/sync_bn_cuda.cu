// Copyright (c) OpenMMLab. All rights reserved.
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"

template <typename T>
__global__ void sync_bn_forward_mean_cuda_kernel(const T *input, float *mean,
                                                 int num, int channels,
                                                 int spatial) {
  __shared__ float buffer[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int c = blockIdx.x;
  buffer[tid] = 0;
  for (int i = tid; i < num * spatial; i += blockDim.x) {
    int index = (i / spatial) * channels * spatial + c * spatial + i % spatial;
    buffer[tid] += input[index];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }
  int total = num * spatial;
  if (tid == 0) {
    mean[c] = buffer[0] / total;
  }
}

template <>
__global__ void sync_bn_forward_mean_cuda_kernel(const phalf *input,
                                                 float *mean, int num,
                                                 int channels, int spatial) {
  __shared__ float buffer[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int c = blockIdx.x;
  buffer[tid] = 0;
  for (int i = tid; i < num * spatial; i += blockDim.x) {
    int index = (i / spatial) * channels * spatial + c * spatial + i % spatial;
    buffer[tid] += static_cast<float>(input[index]);
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }
  int total = num * spatial;
  if (tid == 0) {
    mean[c] = buffer[0] / total;
  }
}

template <typename T>
__global__ void sync_bn_forward_var_cuda_kernel(const T *input,
                                                const float *mean, float *var,
                                                int num, int channels,
                                                int spatial) {
  __shared__ float buffer[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int c = blockIdx.x;
  buffer[tid] = 0;
  for (int i = tid; i < num * spatial; i += blockDim.x) {
    int index = (i / spatial) * channels * spatial + c * spatial + i % spatial;
    float td = input[index] - mean[c];
    buffer[tid] += td * td;
  }
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }
  int total = num * spatial;
  if (tid == 0) {
    var[c] = buffer[0] / total;
  }
}

template <>
__global__ void sync_bn_forward_var_cuda_kernel(const phalf *input,
                                                const float *mean, float *var,
                                                int num, int channels,
                                                int spatial) {
  __shared__ float buffer[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int c = blockIdx.x;
  buffer[tid] = 0;
  for (int i = tid; i < num * spatial; i += blockDim.x) {
    int index = (i / spatial) * channels * spatial + c * spatial + i % spatial;
    float td = static_cast<float>(input[index]) - mean[c];
    buffer[tid] += td * td;
  }
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }
  int total = num * spatial;
  if (tid == 0) {
    var[c] = buffer[0] / total;
  }
}

template <typename T>
__global__ void sync_bn_forward_output_cuda_kernel(
    const T *input, const float *mean, const float *var, float *running_mean,
    float *running_var, const float *weight, const float *bias, float *norm,
    float *std, T *output, int num, int channels, int spatial, float eps,
    float momentum, int group_size) {
  int tid = threadIdx.x;
  int c = blockIdx.x;
  float mean_value = mean[c];
  float std_value = sqrt(var[c] + eps);

  if (weight != nullptr) {
    float weight_value = weight[c];
    float bias_value = bias[c];
    if (norm != nullptr) {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        norm[index] = (input[index] - mean_value) / std_value;
        output[index] = norm[index] * weight_value + bias_value;
      }
    } else {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        output[index] =
            (input[index] - mean_value) / std_value * weight_value + bias_value;
      }
    }
  } else {
    if (norm != nullptr) {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        output[index] = norm[index] = (input[index] - mean_value) / std_value;
      }
    } else {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        output[index] = (input[index] - mean_value) / std_value;
      }
    }
  }
  if (tid == 0) {
    if (std != nullptr) std[c] = std_value;
    if (running_mean != nullptr) {
      running_mean[c] =
          momentum * mean_value + (1 - momentum) * running_mean[c];
      int count = num * spatial * group_size;
      float var_unbias = count > 1 ? var[c] * count / (count - 1) : var[c];
      running_var[c] = momentum * var_unbias + (1 - momentum) * running_var[c];
    }
  }
}

template <>
__global__ void sync_bn_forward_output_cuda_kernel(
    const phalf *input, const float *mean, const float *var,
    float *running_mean, float *running_var, const float *weight,
    const float *bias, float *norm, float *std, phalf *output, int num,
    int channels, int spatial, float eps, float momentum, int group_size) {
  int tid = threadIdx.x;
  int c = blockIdx.x;
  float mean_value = mean[c];
  float std_value = sqrt(var[c] + eps);
  if (weight != nullptr) {
    float weight_value = weight[c];
    float bias_value = bias[c];
    if (norm != nullptr) {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        norm[index] =
            (static_cast<float>(input[index]) - mean_value) / std_value;
        output[index] =
            static_cast<phalf>(norm[index] * weight_value + bias_value);
      }
    } else {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        output[index] =
            static_cast<phalf>((static_cast<float>(input[index]) - mean_value) /
                                   std_value * weight_value +
                               bias_value);
      }
    }
  } else {
    if (norm != nullptr) {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        norm[index] =
            (static_cast<float>(input[index]) - mean_value) / std_value;
        output[index] = static_cast<phalf>(norm[index]);
      }
    } else {
      for (int i = tid; i < num * spatial; i += blockDim.x) {
        int index =
            (i / spatial) * channels * spatial + c * spatial + i % spatial;
        output[index] = static_cast<phalf>(
            (static_cast<float>(input[index]) - mean_value) / std_value);
      }
    }
  }
  if (tid == 0) {
    if (std != nullptr) std[c] = std_value;
    if (running_mean != nullptr) {
      running_mean[c] =
          momentum * mean_value + (1 - momentum) * running_mean[c];
      int count = num * spatial * group_size;
      float var_unbias = count > 1 ? var[c] * count / (count - 1) : var[c];
      running_var[c] = momentum * var_unbias + (1 - momentum) * running_var[c];
    }
  }
}

template <typename T>
__global__ void sync_bn_backward_param_cuda_kernel(const T *grad_output,
                                                   const float *norm,
                                                   float *grad_weight,
                                                   float *grad_bias, int num,
                                                   int channels, int spatial) {
  __shared__ float buffer1[THREADS_PER_BLOCK];
  __shared__ float buffer2[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int c = blockIdx.x;
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial; i += blockDim.x) {
    int index = (i / spatial) * channels * spatial + c * spatial + i % spatial;
    buffer1[tid] += grad_output[index] * norm[index];
    buffer2[tid] += grad_output[index];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_weight[c] = buffer1[0];
    grad_bias[c] = buffer2[0];
  }
}

template <>
__global__ void sync_bn_backward_param_cuda_kernel(const phalf *grad_output,
                                                   const float *norm,
                                                   float *grad_weight,
                                                   float *grad_bias, int num,
                                                   int channels, int spatial) {
  __shared__ float buffer1[THREADS_PER_BLOCK];
  __shared__ float buffer2[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int c = blockIdx.x;
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial; i += blockDim.x) {
    int index = (i / spatial) * channels * spatial + c * spatial + i % spatial;
    buffer1[tid] += static_cast<float>(grad_output[index]) * norm[index];
    buffer2[tid] += static_cast<float>(grad_output[index]);
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_weight[c] = buffer1[0];
    grad_bias[c] = buffer2[0];
  }
}

template <typename T>
__global__ void sync_bn_backward_data_cuda_kernel(
    int output_size, const T *grad_output, const float *weight,
    const float *grad_weight, const float *grad_bias, const float *norm,
    const float *std, T *grad_input, int num, int channels, int spatial) {
  int factor = num * spatial;
  CUDA_1D_KERNEL_LOOP(index, output_size) {
    int c = (index / spatial) % channels;
    grad_input[index] =
        weight[c] *
        (grad_output[index] -
         (grad_weight[c] * norm[index] + grad_bias[c]) / factor) /
        std[c];
  }
}

template <>
__global__ void sync_bn_backward_data_cuda_kernel(
    int output_size, const phalf *grad_output, const float *weight,
    const float *grad_weight, const float *grad_bias, const float *norm,
    const float *std, phalf *grad_input, int num, int channels, int spatial) {
  int factor = num * spatial;
  CUDA_1D_KERNEL_LOOP(index, output_size) {
    int c = (index / spatial) % channels;
    grad_input[index] = static_cast<phalf>(
        weight[c] *
        (static_cast<float>(grad_output[index]) -
         (grad_weight[c] * norm[index] + grad_bias[c]) / factor) /
        std[c]);
  }
}

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sync_bn_forward_mean_cuda_kernel", [&] {
        sync_bn_forward_mean_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(), num,
                channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sync_bn_forward_mean_cuda_kernel", [&] {
        sync_bn_forward_var_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(),
                var.data_ptr<float>(), num, channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void sync_bn_forward_output_cuda(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size) {
  int num = input.size(0);
  int channels = input.size(1);
  int spatial = input.size(2);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sync_bn_forward_mean_cuda_kernel", [&] {
        sync_bn_forward_output_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(), mean.data_ptr<float>(),
                var.data_ptr<float>(), running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(), weight.data_ptr<float>(),
                bias.data_ptr<float>(), norm.data_ptr<float>(),
                std.data_ptr<float>(), output.data_ptr<scalar_t>(), num,
                channels, spatial, eps, momentum, group_size);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void sync_bn_backward_param_cuda(const Tensor grad_output,
                                const Tensor norm,
                                Tensor grad_weight,
                                Tensor grad_bias) {
  int num = grad_output.size(0);
  int channels = grad_output.size(1);
  int spatial = grad_output.size(2);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "sync_bn_backward_param_cuda_kernel", [&] {
        sync_bn_backward_param_cuda_kernel<scalar_t>
            <<<channels, THREADS_PER_BLOCK, 0, stream>>>(
                grad_output.data_ptr<scalar_t>(), norm.data_ptr<float>(),
                grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(), num,
                channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void sync_bn_backward_data_cuda(const Tensor grad_output,
                                const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias,
                                const Tensor norm, const Tensor std,
                                Tensor grad_input) {
  int output_size = grad_input.numel();
  int num = grad_input.size(0);
  int channels = grad_input.size(1);
  int spatial = grad_input.size(2);

  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "sync_bn_backward_data_cuda_kernel", [&] {
        sync_bn_backward_data_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                weight.data_ptr<float>(), grad_weight.data_ptr<float>(),
                grad_bias.data_ptr<float>(), norm.data_ptr<float>(),
                std.data_ptr<float>(), grad_input.data_ptr<scalar_t>(), num,
                channels, spatial);
      });
  AT_CUDA_CHECK(cudaGetLastError());
}
