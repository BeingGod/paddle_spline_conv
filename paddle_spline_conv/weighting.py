import paddle
import paddle_spline_conv_ops


class _SplineWeighting(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        x: paddle.Tensor,
        weight: paddle.Tensor,
        basis: paddle.Tensor,
        weight_index: paddle.Tensor,
    ) -> paddle.Tensor:
        out = paddle_spline_conv_ops.spline_weighting_fw(x, weight, basis, weight_index)

        ctx.save_for_backward(x, weight, basis, weight_index)

        return out

    @staticmethod
    def backward(ctx, grad_out: paddle.Tensor):
        x, weight, basis, weight_index = ctx.saved_tensor()

        grad_x = None
        if not x.stop_gradient:
            grad_x = paddle_spline_conv_ops.spline_weighting_bw_x(
                grad_out, weight, basis, weight_index
            )

        grad_weight = None
        if not weight.stop_gradient:
            grad_weight = paddle_spline_conv_ops.spline_weighting_bw_weight(
                grad_out, x, basis, weight_index, weight.shape[0]
            )

        grad_basis = None
        if not basis.stop_gradient:
            grad_basis = paddle_spline_conv_ops.spline_weighting_bw_basis(
                grad_out, x, weight, weight_index
            )

        return grad_x, grad_weight, grad_basis, None


def spline_weighting(
    x: paddle.Tensor,
    weight: paddle.Tensor,
    basis: paddle.Tensor,
    weight_index: paddle.Tensor,
) -> paddle.Tensor:
    return _SplineWeighting.apply(x, weight, basis, weight_index)
