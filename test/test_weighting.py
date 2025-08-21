from itertools import product

import numpy as np
import paddle
import pytest

from paddle_spline_conv import spline_basis
from paddle_spline_conv import spline_weighting
from paddle_spline_conv.testing import devices
from paddle_spline_conv.testing import dtypes
from paddle_spline_conv.testing import maybe_skip_testing
from paddle_spline_conv.testing import set_testing_device
from paddle_spline_conv.testing import tensor

try:
    import torch
    import torch_spline_conv

    HAVE_TORCH_SPLINE_CONV = True
except ImportError:
    HAVE_TORCH_SPLINE_CONV = False


tests = [
    {
        "x": [[1, 2], [3, 4]],
        "weight": [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]],
        "basis": [[0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5]],
        "weight_index": [[0, 1, 2, 3], [0, 1, 2, 3]],
        "expected": [
            [0.5 * ((1 * (1 + 5)) + (2 * (2 + 6)))],
            [0.5 * ((3 * (5 + 7)) + (4 * (6 + 8)))],
        ],
    }
]


@pytest.mark.parametrize("test,dtype,device", product(tests, dtypes, devices))
def test_spline_weighting_forward(test, dtype, device):
    maybe_skip_testing(dtype, device)
    set_testing_device(device)

    if dtype == paddle.bfloat16 and str(device)[6:-1] == "gpu:0":
        return

    x = tensor(test["x"], dtype, device)
    weight = tensor(test["weight"], dtype, device)
    basis = tensor(test["basis"], dtype, device)
    weight_index = tensor(test["weight_index"], paddle.int64, device)
    expected = tensor(test["expected"], dtype, device)

    out = spline_weighting(x, weight, basis, weight_index)
    assert paddle.allclose(out, expected)


def spline_weighting_backward_paddle(device):
    set_testing_device(device)

    np.random.seed(42)
    pseudo_np = np.random.rand(4, 2).astype(np.float64)
    x_np = np.random.rand(4, 2).astype(np.float64)
    weight_np = np.random.rand(25, 2, 4).astype(np.float64)

    pseudo = paddle.to_tensor(pseudo_np)
    kernel_size = paddle.to_tensor([5, 5], paddle.int64)
    is_open_spline = paddle.to_tensor([1, 1], paddle.uint8)
    degree = 1

    basis, weight_index = spline_basis(pseudo, kernel_size, is_open_spline, degree)
    basis.stop_gradient = False

    x = paddle.to_tensor(x_np)
    x.stop_gradient = False
    weight = paddle.to_tensor(weight_np)
    weight.stop_gradient = False

    data = (x, weight, basis, weight_index)
    out = spline_weighting(*data)
    out.backward(grad_tensor=paddle.ones_like(out))

    return out.detach(), x.grad.detach(), weight.grad.detach(), basis.grad.detach()


def spline_weighting_backward_torch(device):
    device = "cuda:0" if str(device)[6:-1] == "gpu:0" else "cpu"

    np.random.seed(42)
    pseudo_np = np.random.rand(4, 2).astype(np.float64)
    x_np = np.random.rand(4, 2).astype(np.float64)
    weight_np = np.random.rand(25, 2, 4).astype(np.float64)

    pseudo = torch.from_numpy(pseudo_np).to(device)
    kernel_size = torch.tensor([5, 5], dtype=torch.long).to(device)
    is_open_spline = torch.tensor([1, 1], dtype=torch.uint8).to(device)
    degree = 1

    basis, weight_index = torch_spline_conv.spline_basis(
        pseudo, kernel_size, is_open_spline, degree
    )
    basis.requires_grad_()

    x = torch.from_numpy(x_np).to(device)
    x.requires_grad_()
    weight = torch.from_numpy(weight_np).to(device)
    weight.requires_grad_()

    data = (x, weight, basis, weight_index)
    out = torch_spline_conv.spline_weighting(*data)
    out.backward(gradient=torch.ones_like(out))

    return out.detach(), x.grad.detach(), weight.grad.detach(), basis.grad.detach()


@pytest.mark.parametrize("device", devices)
def test_spline_weighting_backward(device):
    if not HAVE_TORCH_SPLINE_CONV:
        pytest.skip("torch-spiline-conv is not be installed.")

    # testing paddle
    out, x_grad, weight_grad, basis_grad = spline_weighting_backward_paddle(device)
    # testing torch
    (
        out_ref,
        x_grad_ref,
        weight_grad_ref,
        basis_grad_ref,
    ) = spline_weighting_backward_torch(device)

    # check forward
    np.testing.assert_array_equal(out.numpy(), out_ref.cpu())

    # check backward
    np.testing.assert_array_equal(x_grad.numpy(), x_grad_ref.cpu())
    np.testing.assert_array_equal(weight_grad.numpy(), weight_grad_ref.cpu())
    np.testing.assert_array_equal(basis_grad.numpy(), basis_grad_ref.cpu())
