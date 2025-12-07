from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load


def load_rmsnorm_cuda_extension():
    path = Path(__file__).resolve()
    src_dir = path.parents[1]
    cu_path = src_dir / "cuda_kernels" / "rmsnorm_kernels.cu"
    if not cu_path.exists():
        raise FileNotFoundError(f"CUDA file not found: {cu_path}")

    extra_nvcc_flags = ["-O3", "-std=c++17"]
    extra_cuda_flags = ["-O3", "--use_fast_math"]

    return load(
        name="rmsnorm_cuda",
        sources=[str(cu_path)],
        extra_cflags=extra_nvcc_flags,
        extra_cuda_cflags=extra_cuda_flags,
        verbose=False
    )


_rmsnorm_extension = load_rmsnorm_cuda_extension()


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("RMSNorm expects CUDA tensors")
        if x.dtype not in (torch.float32, ):
            raise RuntimeError("RMSNorm kernel supports float32 inputs")
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        if x.size(-1) != weight.numel():
            raise RuntimeError(
                f"Last dimension of input ({x.size(-1)}) must match weight ({weight.numel()})"
            )
        x_2d = x.contiguous().view(-1, x.size(-1))
        weight_c = weight.contiguous()
        y = _rmsnorm_extension.rmsnorm_forward(x_2d, weight_c, float(eps))
        ctx.save_for_backward(x_2d, weight_c)
        ctx.eps = float(eps)
        ctx.input_shape = x.shape
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if grad_out is None:
            return None, None, None
        if not grad_out.is_cuda:
            raise RuntimeError("RMSNorm backward expects CUDA tensor")
        if grad_out.dtype not in (torch.float32, ):
            raise RuntimeError("RMSNorm backward expects float32 grads")
        needs_dx = ctx.needs_input_grad[0]
        needs_dgamma = ctx.needs_input_grad[1]
        if not (needs_dx or needs_dgamma):
            return None, None, None

        (x_2d, weight_c) = ctx.saved_tensors
        grad_out_2d = grad_out.contiguous().view(-1, grad_out.size(-1))
        dx_full, dgamma_full = _rmsnorm_extension.rmsnorm_backward(
            x_2d, grad_out_2d, weight_c, float(ctx.eps)
        )

        dx = dx_full.view(ctx.input_shape) if needs_dx else None
        dgamma = dgamma_full if needs_dgamma else None
        return dx, dgamma, None


class RMSNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
        *,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if dtype not in (torch.float32, ):
            raise ValueError("RMSNorm supports float32 dtype (only)")
        self.embedding_dim = int(embedding_dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.empty(self.embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("RMSNorm module currently supports only CUDA tensors")
        return RMSNormFunction.apply(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, eps={self.eps}"


if __name__ == "__main__":
    pass
