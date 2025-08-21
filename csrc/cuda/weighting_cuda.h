#pragma once

#include "../extensions.h"

paddle::Tensor spline_weighting_fw_cuda(paddle::Tensor x,
                                        paddle::Tensor weight,
                                        paddle::Tensor basis,
                                        paddle::Tensor weight_index);

paddle::Tensor spline_weighting_bw_x_cuda(paddle::Tensor grad_out,
                                          paddle::Tensor weight,
                                          paddle::Tensor basis,
                                          paddle::Tensor weight_index);

paddle::Tensor spline_weighting_bw_weight_cuda(paddle::Tensor grad_out,
                                               paddle::Tensor x,
                                               paddle::Tensor basis,
                                               paddle::Tensor weight_index,
                                               int64_t kernel_size);

paddle::Tensor spline_weighting_bw_basis_cuda(paddle::Tensor grad_out,
                                              paddle::Tensor x,
                                              paddle::Tensor weight,
                                              paddle::Tensor weight_index);
