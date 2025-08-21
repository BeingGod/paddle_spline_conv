import os.path as osp

import paddle

__version__ = "1.2.2"


try:
    import paddle_spline_conv_ops  # noqa
except ImportError:
    raise ImportError(
        f"Could not import `paddle_spline_conv_ops` in {osp.dirname(__file__)}."
        f"Please run `python setup_ops.py install` manually."
    )


cuda_version = paddle_spline_conv_ops.cuda_version().item()
if paddle.version.cuda_version is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in paddle.version.cuda_version.split(".")]

    if t_major != major:
        raise RuntimeError(
            f"Detected that Paddle and paddle_spline_conv were compiled with "
            f"different CUDA versions. Paddle has CUDA version "
            f"{t_major}.{t_minor} and paddle_spline_conv has CUDA version "
            f"{major}.{minor}. Please reinstall the paddle_spline_conv that "
            f"matches your Paddle install."
        )

from .basis import spline_basis  # noqa
from .conv import spline_conv  # noqa
from .weighting import spline_weighting  # noqa

__all__ = [
    "spline_basis",
    "spline_weighting",
    "spline_conv",
    "__version__",
]
