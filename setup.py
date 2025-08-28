from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_name = "modules._rmsnorm_cuda"

sources = [
    "src/cuda_kernels/rmsnorm_kernels.cu",
]

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
    ],
}

setup(
    name="rmsnorm_custom_cuda",
    description="Custom CUDA RMSNorm kernels (forward/backward) for PyTorch",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name=ext_name,
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)