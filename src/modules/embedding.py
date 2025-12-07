import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load


def load_embedding_cuda_extension():
    path = Path(__file__).resolve()
    src_dir = path.parents[1]
    cu_path = src_dir / "cuda_kernels" / "embedding_kernels.cu"
    if not cu_path.exists():
        raise FileNotFoundError(f"CUDA file not found: {cu_path}")

    extra_nvcc_flags = ["-O3", "-std=c++17"]
    extra_cuda_flags = ["-O3", "--use_fast_math"]

    return load(
        name="embedding_cuda",
        sources=[str(cu_path)],
        extra_cflags=extra_nvcc_flags,
        extra_cuda_cflags=extra_cuda_flags,
        verbose=False
    )


_embedding_extension = load_embedding_cuda_extension()


class EmbeddingsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if not weight.is_cuda:
            raise RuntimeError("Embedding weight must be a CUDA tensor")
        if weight.dtype != torch.float32:
            raise RuntimeError("Embedding kernel currently supports only float32 weights")
        if not indices.is_cuda:
            raise RuntimeError("Embedding indices must be CUDA tensors")
        if indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("Embedding indices must be int32 or int64")
        if indices.device != weight.device:
            raise RuntimeError("Embedding weights and indices must be on the same CUDA device")

        original_indices_shape = indices.shape
        indices_flat = indices.contiguous().view(-1)
        if indices_flat.dtype == torch.int32:
            indices_int32 = indices_flat
        else:
            indices_int32 = indices_flat.to(torch.int32)

        num_indices = indices_int32.numel()
        embedding_dim = weight.size(1)
        out_flat = torch.empty(
            (num_indices, embedding_dim), device=weight.device, dtype=weight.dtype
        )
        out_flat = _embedding_extension.embedding_forward_fp32(weight, indices_int32, out_flat)

        ctx.save_for_backward(indices_int32)
        ctx.embedding_dim = embedding_dim
        ctx.weight_shape = tuple(weight.shape)

        out = out_flat.view(*original_indices_shape, embedding_dim)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_out,) = grad_outputs
        if grad_out is None:
            return None, None

        if grad_out.dtype != torch.float32:
            raise RuntimeError("Embedding grad_output must be float32")

        (indices_int32,) = ctx.saved_tensors
        grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_out_flat = grad_out.contiguous().view(-1, ctx.embedding_dim)
            grad_weight_tensor = torch.empty(
                ctx.weight_shape, device=grad_out.device, dtype=torch.float32
            )
            grad_weight = _embedding_extension.embedding_backward_fp32(
                indices_int32, grad_out_flat, grad_weight_tensor
            )
        return grad_weight, None


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if dtype != torch.float32:
            raise ValueError("Embedding kernel currently supports only float32 dtype")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.num_embeddings))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if not indices.is_cuda:
            raise RuntimeError("Embedding indices must be CUDA tensors")
        return EmbeddingsFunction.apply(self.weight, indices)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
