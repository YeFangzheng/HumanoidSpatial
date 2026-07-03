// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"


void sigmoid_focal_loss_forward_cuda(Tensor input, Tensor target,
  Tensor weight, Tensor output,
  const float gamma,
  const float alpha);

void sigmoid_focal_loss_backward_cuda(Tensor input, Tensor target,
  Tensor weight,
  Tensor grad_input,
  const float gamma,
  const float alpha);


void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  sigmoid_focal_loss_forward_cuda(input, target, weight, output, gamma, alpha);
}


void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha) {
  sigmoid_focal_loss_backward_cuda(input, target, weight, grad_input, gamma,
                                   alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward,
    "sigmoid_focal_loss_forward ", py::arg("input"), py::arg("target"),
    py::arg("weight"), py::arg("output"), py::arg("gamma"),
    py::arg("alpha"));
  m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward,
    "sigmoid_focal_loss_backward", py::arg("input"), py::arg("target"),
    py::arg("weight"), py::arg("grad_input"), py::arg("gamma"),
    py::arg("alpha"));  
}