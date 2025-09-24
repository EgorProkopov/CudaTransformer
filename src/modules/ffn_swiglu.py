from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load


def load_cuda_extension():
    path = Path(__file__).resolve()

    src_dir = path.parents[1]
    cu_path = src_dir / "cuda_kernels" / "ffn_swiglu_kernels.cu"
    if not cu_path.exists(): raise FileNotFoundError(f"CUDA file not found: {cu_path}")

    extra_nvcc_flags = ["-O3", "-std=c++17"]
    extra_cuda_flags = ["-O3", "--use_fast_math"]

    return load(
        name="ffn_swiglu_cuda",
        sources=[str(cu_path)],
        extra_cflags=extra_nvcc_flags,
        extra_cuda_cflags=extra_cuda_flags,
        verbose=False
    )


_ffn_swiglu_extension = load_cuda_extension()


class FFNSwiGLUv1Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor
    ):
        pass

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        pass



class FFNSwiGLUv2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor
    ):
        pass

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        pass


class FFNSwiGLUv1(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class FFNSwiGLUv1(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

        