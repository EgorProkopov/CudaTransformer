from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load


def load_cuda_extension():
    path = Path(__file__).resolve()

    src_dir = path.parents[1]
    cu_path = src_dir / "cuda_kernels" / "softmax_kernels.cu"
    if not cu_path.exists(): raise FileNotFoundError(f"CUDA file not found: {cu_path}")

    extra_nvcc_flags = ["-O3", "-std=c++17"]
    extra_cuda_flags = ["-O3", "--use_fast_math"]

    return load(
        name="safe_softmax_cuda",
        sources=[str(cu_path)],
        extra_cflags=extra_nvcc_flags,
        extra_cuda_cflags=extra_cuda_flags,
        verbose=False
    )


_safe_softmax_extension = load_cuda_extension()


class SafeSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("Safe-Softmax: CUDA tensor was expected")
        if x.dtype != torch.float32:
            raise RuntimeError("Safe-Softmax: Current softmax kernel supports only fp32 data")
        if x.dim() != 2:
            raise RuntimeError(f"Safe-Softmax: expected input data to has shape [batch_size, seq_len], but got {tuple(x.shape)} with dim {x.dim()}")

        x_2d = x.contiguous()
        y_2d = _safe_softmax_extension.forward(x_2d, float(eps))  # runs forward pass

        ctx.save_for_backward(y_2d)
        return y_2d

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple[torch.Tensor, Optional[None], Optional[None]]:
        (grad_out,) = grad_outputs
        (y_2d, ) = ctx.saved_tensors
        dx_2d =  _safe_softmax_extension.backward(y_2d, grad_out.contiguous())  # runs backward pass
        return dx_2d, None, None


class SafeSoftmax(nn.Module):
    """
    Safe softmax module with custom CUDA kernel

    safe_softmax(x_i) = exp(x_i - m) / sum_j(exp(x_j - m)), where m = max_j x_j
    Args:
        dim: softmax compute dimension
        eps: numerical stabilization to avoid division by 0
    """
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SafeSoftmaxFunction.apply(x, self.dim, self.eps)
