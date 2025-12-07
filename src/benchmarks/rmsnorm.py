import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.modules.rmsnorm import RMSNorm


WARM_UP_RUNS = 5
ITERATIONS = 10


def benchmark_gpu(function: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    for _ in range(WARM_UP_RUNS):
        result = function()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result: Optional[torch.Tensor] = None
    for _ in range(ITERATIONS):
        result = function()
    end.record()
    torch.cuda.synchronize()
    assert result is not None
    ms = start.elapsed_time(end) / ITERATIONS
    return ms, result


def bf16_supported() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        return bool(checker())
    return False


class TorchRMSNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + self.eps)
        return x * inv_rms * self.weight


def tie_weights(src: TorchRMSNorm, dst: RMSNorm) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight.to(dtype=dst.weight.dtype, device=dst.weight.device))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("RMSNorm benchmark requires CUDA")

    torch.manual_seed(239)
    device = torch.device("cuda")
    embedding_dim = 4096
    batch_size = 64
    seq_len = 256
    eps = 1e-6

    dtypes = [torch.float32, ]

    for dtype in dtypes:
        if dtype == torch.bfloat16 and not bf16_supported():
            continue

        print(f"\n[RMSNorm Benchmark] dtype={dtype}")
        x = torch.randn(batch_size, seq_len, embedding_dim, device=device, dtype=dtype)
        grad_out = torch.randn_like(x)

        torch_mod = TorchRMSNorm(embedding_dim, eps=eps).to(device=device, dtype=dtype)
        custom_mod = RMSNorm(embedding_dim, eps=eps, device=device, dtype=dtype).to(device)
        tie_weights(torch_mod, custom_mod)

        def torch_forward():
            return torch_mod(x)

        def custom_forward():
            return custom_mod(x)

        torch_ms, torch_out = benchmark_gpu(torch_forward)
        custom_ms, custom_out = benchmark_gpu(custom_forward)
        max_diff = (torch_out - custom_out).abs().max().item()

        print(f"  forward: torch={torch_ms:.3f} ms | custom={custom_ms:.3f} ms | max|diff|={max_diff:.3e}")

        def torch_backward():
            torch_mod.weight.grad = None
            out = torch_mod(x)
            loss = (out * grad_out).sum()
            loss.backward()
            return torch_mod.weight.grad.detach().clone()

        def custom_backward():
            custom_mod.weight.grad = None
            out = custom_mod(x)
            loss = (out * grad_out).sum()
            loss.backward()
            return custom_mod.weight.grad.detach().clone()

        torch_bwd_ms, torch_grad = benchmark_gpu(torch_backward)
        custom_bwd_ms, custom_grad = benchmark_gpu(custom_backward)
        grad_diff = (torch_grad - custom_grad).abs().max().item()

        print(
            f"  backward: torch={torch_bwd_ms:.3f} ms | custom={custom_bwd_ms:.3f} ms | max|diff|={grad_diff:.3e}"
        )
