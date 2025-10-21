import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load


os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;8.9"


def load_cuda_extension():
    path = Path(__file__).resolve()

    src_dir = path.parents[1]
    cu_path = src_dir / "cuda_kernels" / "ffn_swiglu_kernels.cu"
    if not cu_path.exists(): raise FileNotFoundError(f"CUDA file not found: {cu_path}")

    extra_nvcc_flags = ["-O3", "-std=c++17"]
    extra_cuda_flags = ["-O3", "--use_fast_math", "-Xptxas=-v", "-lineinfo"]

    return load(
        name="ffn_swiglu_cuda",
        sources=[str(cu_path)],
        extra_cflags=extra_nvcc_flags,
        extra_cuda_cflags=extra_cuda_flags,
        verbose=True
    )


_ffn_swiglu_extension = load_cuda_extension()


class FFNSwiGLUv1Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor,
        p: float, seed: int
    ):
        # Use the appropriate forward version based on dtype
        if x.dtype == torch.float16:
            y = _ffn_swiglu_extension.forward_v5_fp16(
                x, W_gate, b_gate, W_in, b_in, W_out, b_out, float(p), int(seed)
            )
        elif x.dtype == torch.bfloat16:
            y = _ffn_swiglu_extension.forward_v5_bf16(
                x, W_gate, b_gate, W_in, b_in, W_out, b_out, float(p), int(seed)
            )
        else:
            raise ValueError("FFNSwiGLUv5 only supports FP16 and BF16 dtypes")
        
        ctx.save_for_backward(x, W_gate, b_gate, W_in, b_in, W_out, b_out)
        ctx.p = p
        ctx.seed = seed
        return y

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        # FFNSwiGLUv5 backward pass not implemented yet
        raise NotImplementedError("FFNSwiGLUv5 backward pass is not implemented yet")



class FFNSwiGLUv2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor,
        p: float, seed: int
    ):
        y = _ffn_swiglu_extension.forward_v2(
            x, W_gate, b_gate, W_in, b_in, W_out, b_out, float(p), int(seed)
        )
        ctx.save_for_backward()
        ctx.p = p
        ctx.seed = seed
        return y

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        pass


class FFNSwiGLUv3Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor,
        p: float, seed: int
    ):
        y = _ffn_swiglu_extension.forward_v3(
            x, W_gate, b_gate, W_in, b_in, W_out, b_out, float(p), int(seed)
        )
        ctx.save_for_backward()
        ctx.p = p
        ctx.seed = seed
        return y

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        pass


class FFNSwiGLUv1(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, p: float = 0.0, seed: int = 239):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.p = float(p)
        self.seed = int(seed)

        self.W_gate = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_in = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_out = nn.Parameter(torch.empty(self.hidden_dim, self.embedding_dim))
        self.b_out = nn.Parameter(torch.zeros(self.embedding_dim))

        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0: p = self.p
        else: p = 0.0
        return FFNSwiGLUv1Function.apply(
            x, self.W_gate, self.b_gate, self.W_in, self.b_in, self.W_out, self.b_out,
            float(p), int(self.seed)
        )


class FFNSwiGLUv2(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, p: float = 0.0, seed: int = 239):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.p = float(p)
        self.seed = int(seed)

        self.W_gate = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_in = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_out = nn.Parameter(torch.empty(self.hidden_dim, self.embedding_dim))
        self.b_out = nn.Parameter(torch.zeros(self.embedding_dim))

        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0: p = self.p
        else: p = 0.0
        return FFNSwiGLUv2Function.apply(
            x, self.W_gate, self.b_gate, self.W_in, self.b_in, self.W_out, self.b_out,
            float(p), int(self.seed)
        )


class FFNSwiGLUv3(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, p: float = 0.0, seed: int = 239):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.p = float(p)
        self.seed = int(seed)

        self.W_gate = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_in = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_out = nn.Parameter(torch.empty(self.hidden_dim, self.embedding_dim))
        self.b_out = nn.Parameter(torch.zeros(self.embedding_dim))

        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0: p = self.p
        else: p = 0.0
        return FFNSwiGLUv3Function.apply(
            x, self.W_gate, self.b_gate, self.W_in, self.b_in, self.W_out, self.b_out,
            float(p), int(self.seed)
        )


class FFNSwiGLUv4Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor,
        p: float, seed: int
    ):
        y = _ffn_swiglu_extension.forward_v4(
            x, W_gate, b_gate, W_in, b_in, W_out, b_out, float(p), int(seed)
        )
        ctx.save_for_backward()
        ctx.p = p
        ctx.seed = seed
        return y

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        pass


class FFNSwiGLUv4(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, p: float = 0.0, seed: int = 239):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.p = float(p)
        self.seed = int(seed)

        self.W_gate = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_in = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_out = nn.Parameter(torch.empty(self.hidden_dim, self.embedding_dim))
        self.b_out = nn.Parameter(torch.zeros(self.embedding_dim))

        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0: p = self.p
        else: p = 0.0
        return FFNSwiGLUv4Function.apply(
            x, self.W_gate, self.b_gate, self.W_in, self.b_in, self.W_out, self.b_out,
            float(p), int(self.seed)
        )


class FFNSwiGLUv5Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        W_gate: torch.Tensor, b_gate: torch.Tensor,
        W_in: torch.Tensor, b_in: torch.Tensor,
        W_out: torch.Tensor, b_out: torch.Tensor,
        p: float, seed: int
    ):
        y = _ffn_swiglu_extension.forward_v5(
            x, W_gate, b_gate, W_in, b_in, W_out, b_out, float(p), int(seed)
        )
        ctx.save_for_backward()
        ctx.p = p
        ctx.seed = seed
        return y

    @staticmethod
    def backward(
        ctx, *grad_outputs
    ) -> tuple[torch.Tensor, Optional[None], Optional[None], Optional[None], Optional[None], Optional[None], Optional[None]]:
        pass


class FFNSwiGLUv5(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, p: float = 0.0, seed: int = 239, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.p = float(p)
        self.seed = int(seed)

        # Use specified reduced precision dtype (fp16 or bf16)
        self.W_gate = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim, dtype=dtype))
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_dim, dtype=dtype))
        self.W_in = nn.Parameter(torch.empty(self.embedding_dim, self.hidden_dim, dtype=dtype))
        self.b_in = nn.Parameter(torch.zeros(self.hidden_dim, dtype=dtype))
        self.W_out = nn.Parameter(torch.empty(self.hidden_dim, self.embedding_dim, dtype=dtype))
        self.b_out = nn.Parameter(torch.zeros(self.embedding_dim, dtype=dtype))

        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0: p = self.p
        else: p = 0.0
        # Convert input to same dtype as weights if needed
        if x.dtype != self.W_gate.dtype:
            x = x.to(self.W_gate.dtype)
        return FFNSwiGLUv5Function.apply(
            x, self.W_gate, self.b_gate, self.W_in, self.b_in, self.W_out, self.b_out,
            float(p), int(self.seed)
        )

