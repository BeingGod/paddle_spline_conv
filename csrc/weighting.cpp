#include "cpu/weighting_cpu.h"
#include "extensions.h"

#ifdef WITH_CUDA
#include "cuda/weighting_cuda.h"
#endif

std::vector<paddle::Tensor> spline_weighting_fw(paddle::Tensor &x,
                                                paddle::Tensor &weight,
                                                paddle::Tensor &basis,
                                                paddle::Tensor &weight_index) {
  if (x.is_gpu()) {
#ifdef WITH_CUDA
    return {spline_weighting_fw_cuda(x, weight, basis, weight_index)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {spline_weighting_fw_cpu(x, weight, basis, weight_index)};
  }
}


std::vector<paddle::DataType> spline_weighting_fw_infer_dtype(
    paddle::DataType x_dtype,
    paddle::DataType weight_dtype,
    paddle::DataType basis_dtype,
    paddle::DataType weight_index_dtype) {
  return {x_dtype};
}


std::vector<std::vector<int64_t>> spline_weighting_fw_infer_shape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> weight_shape,
    std::vector<int64_t> basis_shape,
    std::vector<int64_t> weight_index_shape) {
  auto E = x_shape[0];
  auto M_out = weight_shape[2];

  return {{E, M_out}};
}

PD_BUILD_OP(spline_weighting_fw)
    .Inputs({"x", "weight", "basis", "weight_index"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(spline_weighting_fw))
    .SetInferShapeFn(PD_INFER_SHAPE(spline_weighting_fw_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(spline_weighting_fw_infer_dtype));


std::vector<paddle::Tensor> spline_weighting_bw_x(
    paddle::Tensor &grad_out,
    paddle::Tensor &weight,
    paddle::Tensor &basis,
    paddle::Tensor &weight_index) {
  if (grad_out.is_gpu()) {
#ifdef WITH_CUDA
    return {spline_weighting_bw_x_cuda(grad_out, weight, basis, weight_index)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {spline_weighting_bw_x_cpu(grad_out, weight, basis, weight_index)};
  }
}

std::vector<paddle::DataType> spline_weighting_bw_x_infer_dtype(
    paddle::DataType grad_out_dtype,
    paddle::DataType weight_dtype,
    paddle::DataType basis_dtype,
    paddle::DataType weight_index_dtype) {
  return {grad_out_dtype};
}


std::vector<std::vector<int64_t>> spline_weighting_bw_x_infer_shape(
    std::vector<int64_t> grad_out_shape,
    std::vector<int64_t> weight_shape,
    std::vector<int64_t> basis_shape,
    std::vector<int64_t> weight_index_shape) {
  auto E = grad_out_shape[0];
  auto M_in = weight_shape[1];

  return {{E, M_in}};
}

PD_BUILD_OP(spline_weighting_bw_x)
    .Inputs({"grad_out", "weight", "basis", "weight_index"})
    .Outputs({"grad_x"})
    .SetKernelFn(PD_KERNEL(spline_weighting_bw_x))
    .SetInferShapeFn(PD_INFER_SHAPE(spline_weighting_bw_x_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(spline_weighting_bw_x_infer_dtype));


std::vector<paddle::Tensor> spline_weighting_bw_weight(
    paddle::Tensor &grad_out,
    paddle::Tensor &x,
    paddle::Tensor &basis,
    paddle::Tensor &weight_index,
    int64_t kernel_size) {
  if (grad_out.is_gpu()) {
#ifdef WITH_CUDA
    return {spline_weighting_bw_weight_cuda(
        grad_out, x, basis, weight_index, kernel_size)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {spline_weighting_bw_weight_cpu(
        grad_out, x, basis, weight_index, kernel_size)};
  }
}

std::vector<paddle::DataType> spline_weighting_bw_weight_infer_dtype(
    paddle::DataType grad_out_dtype,
    paddle::DataType x_dtype,
    paddle::DataType basis_dtype,
    paddle::DataType weight_index_dtype) {
  return {grad_out_dtype};
}


std::vector<std::vector<int64_t>> spline_weighting_bw_weight_infer_shape(
    std::vector<int64_t> grad_out_shape,
    std::vector<int64_t> x_shape,
    std::vector<int64_t> basis_shape,
    std::vector<int64_t> weight_index_shape,
    int64_t kernel_size) {
  auto E = grad_out_shape[0];
  auto M_in = x_shape[1];
  auto M_out = grad_out_shape[1];

  return {{kernel_size, M_in, M_out}};
}

PD_BUILD_OP(spline_weighting_bw_weight)
    .Inputs({"grad_out", "x", "basis", "weight_index"})
    .Attrs({"kernel_size:int64_t"})
    .Outputs({"grad_weight"})
    .SetKernelFn(PD_KERNEL(spline_weighting_bw_weight))
    .SetInferShapeFn(PD_INFER_SHAPE(spline_weighting_bw_weight_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(spline_weighting_bw_weight_infer_dtype));


std::vector<paddle::Tensor> spline_weighting_bw_basis(
    paddle::Tensor &grad_out,
    paddle::Tensor &x,
    paddle::Tensor &weight,
    paddle::Tensor &weight_index) {
  if (grad_out.is_gpu()) {
#ifdef WITH_CUDA
    return {spline_weighting_bw_basis_cuda(grad_out, x, weight, weight_index)};
#else
    PD_THROW("Not compiled with CUDA support");
#endif
  } else {
    return {spline_weighting_bw_basis_cpu(grad_out, x, weight, weight_index)};
  }
}

std::vector<paddle::DataType> spline_weighting_bw_basis_infer_dtype(
    paddle::DataType grad_out_dtype,
    paddle::DataType x_dtype,
    paddle::DataType weight_dtype,
    paddle::DataType weight_index_dtype) {
  return {grad_out_dtype};
}


std::vector<std::vector<int64_t>> spline_weighting_bw_basis_infer_shape(
    std::vector<int64_t> grad_out_shape,
    std::vector<int64_t> x_shape,
    std::vector<int64_t> weight_shape,
    std::vector<int64_t> weight_index_shape,
    int64_t kernel_size) {
  auto E = grad_out_shape[0];
  auto S = weight_index_shape[1];

  return {{E, S}};
}

PD_BUILD_OP(spline_weighting_bw_basis)
    .Inputs({"grad_out", "x", "weight", "weight_index"})
    .Outputs({"grad_basis"})
    .SetKernelFn(PD_KERNEL(spline_weighting_bw_basis))
    .SetInferShapeFn(PD_INFER_SHAPE(spline_weighting_bw_basis_infer_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(spline_weighting_bw_basis_infer_dtype));
