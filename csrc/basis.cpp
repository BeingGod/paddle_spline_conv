#include "cpu/basis_cpu.h"
#include "extensions.h"

#ifdef WITH_CUDA
#include "cuda/basis_cuda.h"
#endif


std::vector<paddle::Tensor> spline_basis_fw(paddle::Tensor &pseudo,
                                            paddle::Tensor &kernel_size,
                                            paddle::Tensor &is_open_spline,
                                            int64_t degree) {
  if (pseudo.is_gpu()) {
#ifdef WITH_CUDA
    auto ret =
        spline_basis_fw_cuda(pseudo, kernel_size, is_open_spline, degree);
    return {std::get<0>(ret), std::get<1>(ret)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    auto ret = spline_basis_fw_cpu(pseudo, kernel_size, is_open_spline, degree);
    return {std::get<0>(ret), std::get<1>(ret)};
  }
}


std::vector<paddle::DataType> spline_basis_fw_infer_dtype(
    paddle::DataType pseudo_dtype,
    paddle::DataType kernel_size_dtype,
    paddle::DataType is_open_spline_dtype) {
  return {pseudo_dtype, kernel_size_dtype};
}


std::vector<std::vector<int64_t>> spline_basis_fw_infer_shape(
    std::vector<int64_t> pseudo_shape,
    std::vector<int64_t> kernel_size_shape,
    std::vector<int64_t> is_open_spline_shape,
    int64_t degree) {
  auto E = pseudo_shape[0];
  auto D = pseudo_shape[1];
  auto S = (int64_t)(pow(degree + 1, D) + 0.5);

  return {{E, S}, {E, S}};
}

PD_BUILD_OP(spline_basis_fw)
    .Inputs({"pseudo", "kernel_size", "is_open_spline"})
    .Attrs({"degree:int64_t"})
    .Outputs({"basis", "weight_index"})
    .SetKernelFn(PD_KERNEL(spline_basis_fw))
    .SetInferShapeFn(PD_INFER_SHAPE(spline_basis_fw_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(spline_basis_fw_infer_dtype));


std::vector<paddle::Tensor> spline_basis_bw(paddle::Tensor &grad_basis,
                                            paddle::Tensor &pseudo,
                                            paddle::Tensor &kernel_size,
                                            paddle::Tensor &is_open_spline,
                                            int64_t degree) {
  if (grad_basis.is_gpu()) {
#ifdef WITH_CUDA
    return {spline_basis_bw_cuda(
        grad_basis, pseudo, kernel_size, is_open_spline, degree)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {spline_basis_bw_cpu(
        grad_basis, pseudo, kernel_size, is_open_spline, degree)};
  }
}

std::vector<paddle::DataType> spline_basis_bw_infer_dtype(
    paddle::DataType grad_basis_dtype,
    paddle::DataType pseudo_dtype,
    paddle::DataType kernel_size_dtype,
    paddle::DataType is_open_spline_dtype) {
  return {pseudo_dtype};
}


std::vector<std::vector<int64_t>> spline_basis_bw_infer_shape(
    std::vector<int64_t> grad_basis_shape,
    std::vector<int64_t> pseudo_shape,
    std::vector<int64_t> kernel_size_shape,
    std::vector<int64_t> is_open_spline_shape,
    int64_t degree) {
  auto E = pseudo_shape[0];
  auto D = pseudo_shape[1];

  return {{E, D}};
}

PD_BUILD_OP(spline_basis_bw)
    .Inputs({"grad_basis", "pseudo", "kernel_size", "is_open_spline"})
    .Attrs({"degree:int64_t"})
    .Outputs({"grad_pseudo"})
    .SetKernelFn(PD_KERNEL(spline_basis_bw))
    .SetInferShapeFn(PD_INFER_SHAPE(spline_basis_bw_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(spline_basis_bw_infer_dtype));
