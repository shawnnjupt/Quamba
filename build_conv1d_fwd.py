import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cc_flag = []

def append_nvcc_threads(nvcc_extra_args):
    max_jobs = os.getenv("MAX_JOBS", str(os.cpu_count()))
    return nvcc_extra_args + ["--threads", max_jobs]

# 定义你只想编译的这一个模块
ext_modules = [
    CUDAExtension(
        name="quamba2_conv1d_cuda",
        sources=[
            "csrc/causal_conv1d/quamba2_conv1d.cpp",
            "csrc/causal_conv1d/quamba2_conv1d_fwd.cu",
            "csrc/causal_conv1d/quamba2_conv1d_update.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            os.path.abspath("csrc"),
            os.path.abspath("csrc/causal_conv1d"),
        ],
    )
]

setup(
    name="quamba_single",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)