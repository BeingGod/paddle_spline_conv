from typing import Tuple

import paddle
import paddle_spline_conv_ops


class _SplineBasis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        pseudo: paddle.Tensor,
        kernel_size: paddle.Tensor,
        is_open_spline: paddle.Tensor,
        degree: int,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        basis, weight_index = paddle_spline_conv_ops.spline_basis_fw(
            pseudo, kernel_size, is_open_spline, degree
        )

        ctx.save_for_backward(pseudo, pseudo, kernel_size, is_open_spline)
        ctx.degree = degree

        return basis, weight_index

    @staticmethod
    def backward(ctx, grad_basis: paddle.Tensor, grad_weight_index: paddle.Tensor):
        pseudo, pseudo, kernel_size, is_open_spline = ctx.saved_tensor()

        grad_pseudo = paddle_spline_conv_ops.spline_basis_bw(
            grad_basis, pseudo, kernel_size, is_open_spline, ctx.degree
        )

        return grad_pseudo, None, None


def spline_basis(
    pseudo: paddle.Tensor,
    kernel_size: paddle.Tensor,
    is_open_spline: paddle.Tensor,
    degree: int,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    return _SplineBasis.apply(pseudo, kernel_size, is_open_spline, degree)
