#include "weighting_cpu.h"

#include "utils.h"

paddle::Tensor spline_weighting_fw_cpu(paddle::Tensor x,
                                       paddle::Tensor weight,
                                       paddle::Tensor basis,
                                       paddle::Tensor weight_index) {
  CHECK_CPU(x);
  CHECK_CPU(weight);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);

  CHECK_INPUT(x.shape()[1] == weight.shape()[1]);

  auto E = x.shape()[0];
  auto M_in = x.shape()[1];
  auto M_out = weight.shape()[2];
  auto S = basis.shape()[1];

  auto out = paddle::empty({E, M_out}, x.dtype(), x.place());

  auto weight_index_data = weight_index.data<int64_t>();

  PD_VISIT_FLOATING_TYPES(x.dtype(), "weighting_fw", [&] {
    auto x_data = x.data<data_t>();
    auto weight_data = weight.data<data_t>();
    auto basis_data = basis.data<data_t>();
    auto out_data = out.data<data_t>();

    data_t v;

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        v = 0;
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
            auto tmp = weight_data[wi * weight.strides()[0] +
                                   m_in * weight.strides()[1] +
                                   m_out * weight.strides()[2]];
            tmp *= b * x_data[e * x.strides()[0] + m_in * x.strides()[1]];
            v += tmp;
          }
        }
        out_data[e * M_out + m_out] = v;
      }
    }
  });

  return out;
}

paddle::Tensor spline_weighting_bw_x_cpu(paddle::Tensor grad_out,
                                         paddle::Tensor weight,
                                         paddle::Tensor basis,
                                         paddle::Tensor weight_index) {
  CHECK_CPU(grad_out);
  CHECK_CPU(weight);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);

  CHECK_INPUT(grad_out.shape()[1] == weight.shape()[2]);

  auto E = grad_out.shape()[0];
  auto M_in = weight.shape()[1];
  auto M_out = grad_out.shape()[1];
  auto S = basis.shape()[1];

  auto grad_x = paddle::zeros({E, M_in}, grad_out.dtype(), grad_out.place());

  auto weight_index_data = weight_index.data<int64_t>();

  PD_VISIT_FLOATING_TYPES(grad_out.dtype(), "weighting_bw_x", [&] {
    auto grad_out_data = grad_out.data<data_t>();
    auto weight_data = weight.data<data_t>();
    auto basis_data = basis.data<data_t>();
    auto grad_x_data = grad_x.data<data_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g = grad_out_data[e * grad_out.strides()[0] +
                               m_out * grad_out.strides()[1]];
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
            auto w = weight_data[wi * weight.strides()[0] +
                                 m_in * weight.strides()[1] +
                                 m_out * weight.strides()[2]];
            grad_x_data[e * M_in + m_in] += g * b * w;
          }
        }
      }
    }
  });

  return grad_x;
}

paddle::Tensor spline_weighting_bw_weight_cpu(paddle::Tensor grad_out,
                                              paddle::Tensor x,
                                              paddle::Tensor basis,
                                              paddle::Tensor weight_index,
                                              int64_t kernel_size) {
  CHECK_CPU(grad_out);
  CHECK_CPU(x);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);

  auto E = grad_out.shape()[0];
  auto M_in = x.shape()[1];
  auto M_out = grad_out.shape()[1];
  auto S = basis.shape()[1];

  auto grad_weight = paddle::zeros(
      {kernel_size, M_in, M_out}, grad_out.dtype(), grad_out.place());

  auto weight_index_data = weight_index.data<int64_t>();

  PD_VISIT_FLOATING_TYPES(x.dtype(), "weighting_bw_weight", [&] {
    auto grad_out_data = grad_out.data<data_t>();
    auto x_data = x.data<data_t>();
    auto basis_data = basis.data<data_t>();
    auto grad_weight_data = grad_weight.data<data_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g = grad_out_data[e * grad_out.strides()[0] +
                               m_out * grad_out.strides()[1]];
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
            auto v = g * b * x_data[e * x.strides()[0] + m_in * x.strides()[1]];
            grad_weight_data[wi * M_in * M_out + m_in * M_out + m_out] += v;
          }
        }
      }
    }
  });

  return grad_weight;
}

paddle::Tensor spline_weighting_bw_basis_cpu(paddle::Tensor grad_out,
                                             paddle::Tensor x,
                                             paddle::Tensor weight,
                                             paddle::Tensor weight_index) {
  CHECK_CPU(grad_out);
  CHECK_CPU(x);
  CHECK_CPU(weight);
  CHECK_CPU(weight_index);

  CHECK_INPUT(x.shape()[1] == weight.shape()[1]);
  CHECK_INPUT(grad_out.shape()[1] == weight.shape()[2]);

  auto E = grad_out.shape()[0];
  auto M_in = x.shape()[1];
  auto M_out = grad_out.shape()[1];
  auto S = weight_index.shape()[1];

  auto grad_basis = paddle::zeros({E, S}, grad_out.dtype(), grad_out.place());

  auto weight_index_data = weight_index.data<int64_t>();

  PD_VISIT_FLOATING_TYPES(x.dtype(), "weighting_bw_basis", [&] {
    auto grad_out_data = grad_out.data<data_t>();
    auto x_data = x.data<data_t>();
    auto weight_data = weight.data<data_t>();
    auto grad_basis_data = grad_basis.data<data_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g = grad_out_data[e * grad_out.strides()[0] +
                               m_out * grad_out.strides()[1]];
        for (int64_t s = 0; s < S; s++) {
          data_t b = 0;
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
            auto w = weight_data[wi * weight.strides()[0] +
                                 m_in * weight.strides()[1] +
                                 m_out * weight.strides()[2]];
            w *= x_data[e * x.strides()[0] + m_in * x.strides()[1]];
            b += w;
          }
          grad_basis_data[e * S + s] += g * b;
        }
      }
    }
  });

  return grad_basis;
}
