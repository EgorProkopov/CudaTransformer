import time
from typing import Optional

import torch

from src.modules.softmax import SafeSoftmax


WARM_UP_RUNS = 50
ITERATIONS = 200


def benchmark_gpu(function):
    for _ in range(WARM_UP_RUNS):
        function()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(ITERATIONS):
        out = function()

    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERATIONS
    return ms, out


def benchmark_cpu(function):
    for _ in range(WARM_UP_RUNS):
        function()

    start_time = time.perf_counter()
    out = None
    for _ in range(ITERATIONS):
        out = function()

    end_time = time.perf_counter()

    ms = (end_time - start_time) * 1000.0 / ITERATIONS
    return ms, out


def torch_softmax_forward_benchmarks(x: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    def function():
        return torch.nn.functional.softmax(x, dim=-1)

    if x.is_cuda: ms, out = benchmark_gpu(function)
    else: ms, out = benchmark_cpu(function)

    if return_res: return ms, out
    else: return ms


def torch_softmax_backward_benchmarks(dy: torch.Tensor, y: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    def function():
        s = (dy * y).sum(dim=-1, keepdim=True)
        return y * (dy - s)

    if dy.is_cuda: ms, out = benchmark_gpu(function)
    else: ms, out = benchmark_cpu(function)

    if return_res: return ms, out
    else: return ms


def custom_softmax_forward_benchmarks(x: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    custom_safe_softmax = SafeSoftmax(dim=-1, eps=1e-12)

    def function():
        return custom_safe_softmax(x)

    ms, out = benchmark_gpu(function)
    if return_res: return ms, out
    else: return ms


def custom_softmax_backward_benchmarks(dy: torch.Tensor, x: torch.Tensor) -> float:
    x_req = x.detach().clone().requires_grad_(True)
    dy = dy.contiguous()

    custom_safe_softmax = SafeSoftmax(dim=-1, eps=1e-12)
    y = custom_safe_softmax(x_req)
    loss = (y * dy).sum()

    for _ in range(WARM_UP_RUNS):
        x_req.grad = None
        loss.backward(retain_graph=True)

    # замер
    def function():
        x_req.grad = None
        loss.backward(retain_graph=True)
        return x_req.grad

    ms, _ = benchmark_gpu(function)
    return ms


if __name__ == "__main__":
    torch.manual_seed(239)
    B, D = 128, 4096

    x  = torch.randn(B, D, device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)

    x_cpu = x.clone().detach()
    ms_t_fwd_cpu, y_t = torch_softmax_forward_benchmarks(x_cpu)
    print(f"torch cpu: {ms_t_fwd_cpu:.3f} ms")

    ms_t_fwd, y_t = torch_softmax_forward_benchmarks(x)
    ms_c_fwd, y_c = custom_softmax_forward_benchmarks(x)
    print(f"torch  forward: {ms_t_fwd:.3f} ms")
    print(f"custom forward: {ms_c_fwd:.3f} ms, max|diff|={ (y_t - y_c).abs().max().item():.3e}")

    ms_t_bwd, dx_t = torch_softmax_backward_benchmarks(dy, y_t)
    ms_c_bwd = custom_softmax_backward_benchmarks(dy, x)
    print(f"torch  backward: {ms_t_bwd:.3f} ms")
    print(f"custom backward: {ms_c_bwd:.3f} ms")
    