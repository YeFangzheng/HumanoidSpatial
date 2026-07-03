// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean);

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var);

void sync_bn_forward_output_cuda(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size);

void sync_bn_backward_param_cuda(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data_cuda(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input);

void sync_bn_forward_mean(const Tensor input, Tensor mean) {
  sync_bn_forward_mean_cuda(input, mean);
}

void sync_bn_forward_var(const Tensor input, const Tensor mean, Tensor var) {
  sync_bn_forward_var_cuda(input, mean, var);
}

void sync_bn_forward_output(const Tensor input, const Tensor mean,
                            const Tensor var, const Tensor weight,
                            const Tensor bias, Tensor running_mean,
                            Tensor running_var, Tensor norm, Tensor std,
                            Tensor output, float eps, float momentum,
                            int group_size) {
  sync_bn_forward_output_cuda(input, mean, var, running_mean, running_var,
                              weight, bias, norm, std, output, eps, momentum,
                              group_size);
}

void sync_bn_backward_param(const Tensor grad_output, const Tensor norm,
                            Tensor grad_weight, Tensor grad_bias) {
  sync_bn_backward_param_cuda(grad_output, norm, grad_weight, grad_bias);
}

void sync_bn_backward_data(const Tensor grad_output, const Tensor weight,
                           const Tensor grad_weight, const Tensor grad_bias,
                           const Tensor norm, const Tensor std,
                           Tensor grad_input) {
  sync_bn_backward_data_cuda(grad_output, weight, grad_weight, grad_bias, norm,
                             std, grad_input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sync_bn_forward_mean", &sync_bn_forward_mean, "sync_bn forward_mean",
        py::arg("input"), py::arg("mean"));
    m.def("sync_bn_forward_var", &sync_bn_forward_var, "sync_bn forward_var",
            py::arg("input"), py::arg("mean"), py::arg("var"));
    m.def("sync_bn_forward_output", &sync_bn_forward_output,
            "sync_bn forward_output", py::arg("input"), py::arg("mean"),
            py::arg("var"), py::arg("weight"), py::arg("bias"),
            py::arg("running_mean"), py::arg("running_var"), py::arg("norm"),
            py::arg("std"), py::arg("output"), py::arg("eps"), py::arg("momentum"),
            py::arg("group_size"));
    m.def("sync_bn_backward_param", &sync_bn_backward_param,
            "sync_bn backward_param", py::arg("grad_output"), py::arg("norm"),
            py::arg("grad_weight"), py::arg("grad_bias"));
    m.def("sync_bn_backward_data", &sync_bn_backward_data,
            "sync_bn backward_data", py::arg("grad_output"), py::arg("weight"),
            py::arg("grad_weight"), py::arg("grad_bias"), py::arg("norm"),
            py::arg("std"), py::arg("grad_input"));
    }