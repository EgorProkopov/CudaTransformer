import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from src.modules.embedding import Embedding


WARM_UP_RUNS = 5
ITERATIONS = 10


def benchmark_gpu(function: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
    for _ in range(WARM_UP_RUNS):
        out = function()
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


def benchmark_cpu(function: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
    for _ in range(WARM_UP_RUNS):
        result = function()

    start_time = time.perf_counter()
    result = None
    for _ in range(ITERATIONS):
        result = function()
    end_time = time.perf_counter()
    assert result is not None
    ms = (end_time - start_time) * 1000.0 / ITERATIONS
    return ms, result


def tie_weights(src: nn.Embedding, dst: Embedding) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Embedding benchmark requires CUDA")

    torch.manual_seed(239)
    device = torch.device("cuda")

    num_embeddings = 65536
    embedding_dim = 1024
    batch_size = 64
    seq_len = 256

    indices = torch.randint(
        0, num_embeddings, (batch_size, seq_len), device=device, dtype=torch.int64
    )
    grad_out = torch.randn(batch_size, seq_len, embedding_dim, device=device, dtype=torch.float32)

    torch_mod = nn.Embedding(num_embeddings, embedding_dim, device=device, dtype=torch.float32)
    custom_mod = Embedding(num_embeddings, embedding_dim, device=device, dtype=torch.float32)
    tie_weights(torch_mod, custom_mod)

    def torch_forward():
        return torch_mod(indices)

    def custom_forward():
        return custom_mod(indices)

    torch_ms, torch_out = benchmark_gpu(torch_forward)
    custom_ms, custom_out = benchmark_gpu(custom_forward)
    max_diff = (torch_out - custom_out).abs().max().item()

    print("[Embedding] forward (CUDA)")
    print(f"  torch : {torch_ms:.3f} ms")
    print(f"  custom: {custom_ms:.3f} ms | max|diff|={max_diff:.3e}")

    grad_out = grad_out.contiguous()

    def torch_backward():
        torch_mod.weight.grad = None
        out = torch_mod(indices)
        loss = (out * grad_out).sum()
        loss.backward()
        return torch_mod.weight.grad.detach().clone()

    def custom_backward():
        custom_mod.weight.grad = None
        out = custom_mod(indices)
        loss = (out * grad_out).sum()
        loss.backward()
        return custom_mod.weight.grad.detach().clone()

    torch_bwd_ms, torch_grad = benchmark_gpu(torch_backward)
    custom_bwd_ms, custom_grad = benchmark_gpu(custom_backward)
    grad_max_diff = (torch_grad - custom_grad).abs().max().item()

    print("\n[Embedding] backward (CUDA)")
    print(f"  torch : {torch_bwd_ms:.3f} ms")
    print(f"  custom: {custom_bwd_ms:.3f} ms | max|diff|={grad_max_diff:.3e}")
