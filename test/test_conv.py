from itertools import product

import numpy as np
import paddle
import pytest

from paddle_spline_conv import spline_conv
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


degrees = [1, 2, 3]

tests = [
    {
        "x": [[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]],
        "edge_index": [[0, 0, 0, 0], [1, 2, 3, 4]],
        "pseudo": [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]],
        "weight": [
            [[0.5], [1]],
            [[1.5], [2]],
            [[2.5], [3]],
            [[3.5], [4]],
            [[4.5], [5]],
            [[5.5], [6]],
            [[6.5], [7]],
            [[7.5], [8]],
            [[8.5], [9]],
            [[9.5], [10]],
            [[10.5], [11]],
            [[11.5], [12]],
        ],
        "kernel_size": [3, 4],
        "is_open_spline": [1, 0],
        "root_weight": [[12.5], [13]],
        "bias": [1],
        "expected": [
            [1 + 12.5 * 9 + 13 * 10 + (8.5 + 40.5 + 107.5 + 101.5) / 4],
            [1 + 12.5 * 1 + 13 * 2],
            [1 + 12.5 * 3 + 13 * 4],
            [1 + 12.5 * 5 + 13 * 6],
            [1 + 12.5 * 7 + 13 * 8],
        ],
    }
]


@pytest.mark.parametrize("test,dtype,device", product(tests, dtypes, devices))
def test_spline_conv_forward(test, dtype, device):
    maybe_skip_testing(dtype, device)
    set_testing_device(device)

    if dtype == paddle.bfloat16 and str(device)[6:-1] == "gpu:0":
        return

    x = tensor(test["x"], dtype, device)
    edge_index = tensor(test["edge_index"], paddle.int64, device)
    pseudo = tensor(test["pseudo"], dtype, device)
    weight = tensor(test["weight"], dtype, device)
    kernel_size = tensor(test["kernel_size"], paddle.int64, device)
    is_open_spline = tensor(test["is_open_spline"], paddle.uint8, device)
    root_weight = tensor(test["root_weight"], dtype, device)
    bias = tensor(test["bias"], dtype, device)
    expected = tensor(test["expected"], dtype, device)

    out = spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        1,
        True,
        root_weight,
        bias,
    )

    error = 1e-2 if dtype == paddle.bfloat16 else 1e-7
    assert paddle.allclose(out, expected, rtol=error, atol=error)

    jit = paddle.jit.to_static(spline_conv)
    jit_out = jit(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        1,
        True,
        root_weight,
        bias,
    )
    assert paddle.allclose(jit_out, expected, rtol=error, atol=error)


def spline_conv_backward_paddle(degree, device):
    set_testing_device(device)

    np.random.seed(42)
    x_np = np.random.rand(3, 2).astype(np.float64)
    pseudo_np = np.random.rand(4, 3).astype(np.float64)
    weight_np = np.random.rand(125, 2, 4).astype(np.float64)
    root_weight_np = np.random.rand(2, 4).astype(np.float64)
    basis_np = np.random.rand(4).astype(np.float64)

    x = paddle.to_tensor(x_np)
    x.stop_gradient = False
    edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]], paddle.int64, device)
    pseudo = paddle.to_tensor(pseudo_np)
    pseudo.stop_gradient = False
    weight = paddle.to_tensor(weight_np)
    weight.stop_gradient = False
    kernel_size = tensor([5, 5, 5], paddle.int64, device)
    is_open_spline = tensor([1, 0, 1], paddle.uint8, device)
    root_weight = paddle.to_tensor(root_weight_np)
    root_weight.stop_gradient = False
    bias = paddle.to_tensor(basis_np)
    bias.stop_gradient = False

    data = (
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        degree,
        True,
        root_weight,
        bias,
    )
    out = spline_conv(*data)
    out.backward(grad_tensor=paddle.ones_like(out))

    return (
        out.detach(),
        x.grad.detach(),
        pseudo.grad.detach(),
        weight.grad.detach(),
        root_weight.grad.detach(),
        bias.grad.detach(),
    )


def spline_conv_backward_torch(degree, device):
    device = "cuda:0" if str(device)[6:-1] == "gpu:0" else "cpu"

    np.random.seed(42)

    np.random.seed(42)
    x_np = np.random.rand(3, 2).astype(np.float64)
    pseudo_np = np.random.rand(4, 3).astype(np.float64)
    weight_np = np.random.rand(125, 2, 4).astype(np.float64)
    root_weight_np = np.random.rand(2, 4).astype(np.float64)
    basis_np = np.random.rand(4).astype(np.float64)

    x = torch.from_numpy(x_np).to(device)
    x.requires_grad_()
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).to(device)
    pseudo = torch.from_numpy(pseudo_np).to(device)
    pseudo.requires_grad_()
    weight = torch.from_numpy(weight_np).to(device)
    weight.requires_grad_()
    kernel_size = torch.tensor([5, 5, 5], dtype=torch.long).to(device)
    is_open_spline = torch.tensor([1, 0, 1], dtype=torch.uint8).to(device)
    root_weight = torch.from_numpy(root_weight_np).to(device)
    root_weight.requires_grad_()
    bias = torch.from_numpy(basis_np).to(device)
    bias.requires_grad_()

    data = (
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        degree,
        True,
        root_weight,
        bias,
    )
    out = torch_spline_conv.spline_conv(*data)
    out.backward(gradient=torch.ones_like(out))

    return (
        out.detach(),
        x.grad.detach(),
        pseudo.grad.detach(),
        weight.grad.detach(),
        root_weight.grad.detach(),
        bias.grad.detach(),
    )


@pytest.mark.parametrize("degree,device", product(degrees, devices))
def test_spline_conv_backward(degree, device):
    if not HAVE_TORCH_SPLINE_CONV:
        pytest.skip("torch-spiline-conv is not be installed.")

    # testing paddle
    results = spline_conv_backward_paddle(degree, device)
    # testing torch
    results_ref = spline_conv_backward_torch(degree, device)

    # check forward
    np.testing.assert_array_equal(results[0].numpy(), results_ref[0].cpu())

    # check backward
    for grad, grad_ref in zip(results[1:], results_ref[1:]):
        np.testing.assert_array_equal(grad.numpy(), grad_ref.cpu())
