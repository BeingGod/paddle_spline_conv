import glob
import os
import os.path as osp
import platform
import re
from itertools import product

import paddle
from paddle.utils.cpp_extension import CppExtension
from paddle.utils.cpp_extension import CUDAExtension
from paddle.utils.cpp_extension import setup
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME


def get_version():
    current_dir = osp.dirname(osp.abspath(__file__))
    with open(osp.join(current_dir, "paddle_spline_conv/__init__.py")) as f:
        content = f.read()
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Cannot find __version__ in paddle_spline_conv/__init__.py")


__version__ = get_version()


WITH_CUDA = False
if paddle.device.cuda.device_count() > 0:
    WITH_CUDA = CUDA_HOME is not None

suffices = ["cpu", "cuda"] if WITH_CUDA else ["cpu"]
if os.getenv("FORCE_CUDA", "0") == "1":
    suffices = ["cuda", "cpu"]
if os.getenv("FORCE_ONLY_CUDA", "0") == "1":
    suffices = ["cuda"]
if os.getenv("FORCE_ONLY_CPU", "0") == "1":
    suffices = ["cpu"]

BUILD_DOCS = os.getenv("BUILD_DOCS", "0") == "1"

assert platform.system() == "Linux", "Only support build on linux now."


def set_cuda_archs():
    major, _ = paddle.version.cuda_version.split(".")
    if int(major) >= 12:
        paddle_known_gpu_archs = [50, 60, 61, 70, 75, 80, 90]
    elif int(major) >= 11:
        paddle_known_gpu_archs = [50, 60, 61, 70, 75, 80]
    elif int(major) >= 10:
        paddle_known_gpu_archs = [50, 52, 60, 61, 70, 75]
    else:
        raise ValueError("Not support cuda version.")

    os.environ["PADDLE_CUDA_ARCH_LIST"] = ",".join(
        [str(arch) for arch in paddle_known_gpu_archs]
    )


def get_extensions():
    extensions = []

    extensions_dir = osp.join("csrc")
    main_files = glob.glob(osp.join(extensions_dir, "*.cpp"))
    # remove generated 'hip' files, in case of rebuilds
    main_files = [path for path in main_files]

    define_macros = [("WITH_PYTHON", None)]
    undef_macros = []

    extra_compile_args = {"cxx": ["-O2"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = ["-s"]

    extra_compile_args["cxx"] += ["-fopenmp"]

    if "cuda" in suffices:
        set_cuda_archs()
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O2"]
        extra_compile_args["nvcc"] = nvcc_flags

        nvcc_flags += ["--expt-relaxed-constexpr"]

    sources = set()
    for main, suffix in product(main_files, suffices):
        name = main.split(os.sep)[-1][:-4]
        sources.add(main)

        path = osp.join(extensions_dir, "cpu", f"{name}_cpu.cpp")
        if osp.exists(path):
            sources.add(path)

        path = osp.join(extensions_dir, "cuda", f"{name}_cuda.cu")
        if suffix == "cuda" and osp.exists(path):
            sources.add(path)

    Extension = CUDAExtension if "cuda" in suffices else CppExtension
    extension = Extension(
        list(sources),
        include_dirs=[extensions_dir],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    extensions += [extension]

    return extensions


if __name__ == "__main__":
    setup(
        name="paddle_spline_conv_ops",
        version=__version__,
        description=(
            "Paddle Custom Extension Library of the Spline-Based Convolution Operator of SplineCNN"
        ),
        author="Ruibin Cheung",
        author_email="beinggod@foxmail.com",
        python_requires=">=3.8",
        ext_modules=get_extensions(),
    )
