import subprocess
import sys

import paddle
from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

from setup_ops import get_version

__version__ = get_version()


install_requires = [
    "scipy<=1.15.3",
]

test_requires = [
    "pytest",
    "pytest-cov",
]

support_commands = ("install", "develop")
assert (
    sys.argv[1] in support_commands
), f"paddle-spline-conv setup.py only support {support_commands} commands."


class CustomCommand:
    """Common functionality for install and develop commands"""

    def run_setup_ops_install(self):
        try:
            subprocess.check_call([sys.executable, "setup_ops.py", "install"])
        except subprocess.CalledProcessError as e:
            print(f"Error running setup_ops.py: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


class CustomInstallCommand(install, CustomCommand):
    def run(self):
        self.run_setup_ops_install()
        install.run(self)


class CustomDevelopCommand(develop, CustomCommand):
    def run(self):
        self.run_setup_ops_install()
        develop.run(self)


include_package_data = True
if paddle.device.cuda.device_count() > 0:
    include_package_data = False


if __name__ == "__main__":
    setup(
        name="paddle_spline_conv",
        version=__version__,
        description=(
            "Implementation of the Spline-Based Convolution Operator of "
            "SplineCNN in Paddle. Originally from https://github.com/rusty1s/pytorch_spline_conv."
        ),
        author="Ruibin Cheung",
        author_email="beinggod@foxmail.com",
        keywords=[
            "paddlepaddle",
            "geometric-deep-learning",
            "graph-neural-networks",
            "spline-cnn",
        ],
        python_requires=">=3.8",
        install_requires=install_requires,
        extras_require={
            "test": test_requires,
        },
        cmdclass={"install": CustomInstallCommand, "develop": CustomDevelopCommand},
        packages=find_packages(),
        include_package_data=include_package_data,
    )
