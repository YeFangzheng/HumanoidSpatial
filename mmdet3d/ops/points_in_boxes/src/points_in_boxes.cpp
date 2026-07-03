#include <torch/extension.h>
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void points_in_boxes_part_forward_cuda(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points);

void points_in_boxes_all_forward_cuda(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points);

void points_in_boxes_part_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                  Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box params pts: (B, npoints, 3)
  // [x, y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints),
  // default -1
  int batch_size = boxes_tensor.size(0);
  int boxes_num = boxes_tensor.size(1);
  int pts_num = pts_tensor.size(1);
  points_in_boxes_part_forward_cuda(batch_size, boxes_num, pts_num,
                                    boxes_tensor, pts_tensor,
                                    box_idx_of_points_tensor);
}

void points_in_boxes_all_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center. params pts: (B, npoints, 3) [x, y, z]
  // in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default -1
  int batch_size = boxes_tensor.size(0);
  int boxes_num = boxes_tensor.size(1);
  int pts_num = pts_tensor.size(1);
  points_in_boxes_all_forward_cuda(batch_size, boxes_num, pts_num, boxes_tensor,
                                   pts_tensor, box_idx_of_points_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("points_in_boxes_part_forward", &points_in_boxes_part_forward,
            "points_in_boxes_part_forward", py::arg("boxes_tensor"),
            py::arg("pts_tensor"), py::arg("box_idx_of_points_tensor"));
    m.def("points_in_boxes_all_forward", &points_in_boxes_all_forward,
            "points_in_boxes_all_forward", py::arg("boxes_tensor"),
            py::arg("pts_tensor"), py::arg("box_idx_of_points_tensor"));
}   