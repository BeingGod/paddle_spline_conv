#pragma once

#include "../extensions.h"

#if defined(__linux__) && defined(__x86_64__)
__asm__(".symver powf,powf@GLIBC_2.2.5");
#endif

std::tuple<paddle::Tensor, paddle::Tensor> spline_basis_fw_cuda(
    paddle::Tensor pseudo,
    paddle::Tensor kernel_size,
    paddle::Tensor is_open_spline,
    int64_t degree);

paddle::Tensor spline_basis_bw_cuda(paddle::Tensor grad_basis,
                                    paddle::Tensor pseudo,
                                    paddle::Tensor kernel_size,
                                    paddle::Tensor is_open_spline,
                                    int64_t degree);
