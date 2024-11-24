from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="happypose",
    version="1.0.0",
    description="HappyPose",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
