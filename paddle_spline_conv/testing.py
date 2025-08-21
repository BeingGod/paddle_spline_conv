from typing import Any

import paddle
import pytest

dtypes = [paddle.float32, paddle.float64]


devices = [paddle.CPUPlace()]
if paddle.device.cuda.device_count() > 0:
    devices += [paddle.CUDAPlace(0)]
    # NOTE: paddle cpu not support bfloat16 dtype
    dtypes += [paddle.bfloat16]


def tensor(x: Any, dtype: paddle.dtype, device: paddle.device):
    return None if x is None else paddle.to_tensor(x, dtype=dtype, place=device)


def maybe_skip_testing(dtype: paddle.dtype, device: paddle.base.libpaddle.Place):
    device = str(device)[6:-1]
    if device == "cpu" and dtype in [paddle.float16, paddle.bfloat16]:
        pytest.skip()


def set_testing_device(device: paddle.base.libpaddle.Place):
    device = str(device)[6:-1]
    paddle.device.set_device(device)
