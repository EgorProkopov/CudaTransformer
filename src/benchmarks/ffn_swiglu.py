import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.ffn_swiglu import (
    FFNSwiGLUv1, FFNSwiGLUv2, FFNSwiGLUv3, FFNSwiGLUv4, FFNSwiGLUv5
)

WARM_UP_RUNS = 5
ITERATIONS = 10


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


class TorchFFNSwiGLU(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, p: float = 0.0):
        super().__init__()
        H, D = embedding_dim, hidden_dim
        self.lin_gate = nn.Linear(H, D, bias=True)
        self.lin_in   = nn.Linear(H, D, bias=True)
        self.lin_out  = nn.Linear(D, H, bias=True)
        self.p = float(p)

        nn.init.xavier_uniform_(self.lin_gate.weight)
        nn.init.xavier_uniform_(self.lin_in.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.lin_gate(x)
        u = self.lin_in(x)
        a = F.silu(g) * u
        if self.training and self.p > 0:
            a = F.dropout(a, p=self.p, training=True)
        y = self.lin_out(a)
        if self.training and self.p > 0:
            y = F.dropout(y, p=self.p, training=True)
        return y + x


def tie_weights_torch_to_custom(torch_mod: TorchFFNSwiGLU, custom_mod: nn.Module) -> None:
    with torch.no_grad():
        custom_mod.W_gate.copy_(torch_mod.lin_gate.weight.t().contiguous())
        custom_mod.b_gate.copy_(torch_mod.lin_gate.bias.contiguous())
        custom_mod.W_in.copy_(torch_mod.lin_in.weight.t().contiguous())
        custom_mod.b_in.copy_(torch_mod.lin_in.bias.contiguous())
        custom_mod.W_out.copy_(torch_mod.lin_out.weight.t().contiguous())
        custom_mod.b_out.copy_(torch_mod.lin_out.bias.contiguous())


def torch_ffn_forward_benchmarks(
    x: torch.Tensor, mod: TorchFFNSwiGLU, return_res=True
) -> Tuple[float, Optional[torch.Tensor]]:
    def function():
        return mod(x)
    if x.is_cuda: ms, out = benchmark_gpu(function)
    else:         ms, out = benchmark_cpu(function)
    return (ms, out) if return_res else (ms, None)


def custom_ffn_forward_benchmarks(
        x: torch.Tensor, custom_mod, return_res=True
) -> Tuple[float, Optional[torch.Tensor]]:
    def function():
        return custom_mod(x)
    ms, out = benchmark_gpu(function) if x.is_cuda else benchmark_cpu(function)
    return (ms, out) if return_res else (ms, None)


def torch_ffn_backward_benchmarks(
    dy: torch.Tensor, mod: TorchFFNSwiGLU, x: torch.Tensor
) -> float:
    was_training = mod.training
    old_p = mod.p
    mod.train(True)
    mod.p = 0.0

    x_req = x.detach().clone().requires_grad_(True)
    y = mod(x_req)
    loss = (y * dy).sum()

    for _ in range(WARM_UP_RUNS):
        x_req.grad = None
        loss.backward(retain_graph=True)

    def function():
        x_req.grad = None
        loss.backward(retain_graph=True)
        return x_req.grad

    ms, _ = benchmark_gpu(function) if x.is_cuda else benchmark_cpu(function)

    mod.p = old_p
    mod.train(was_training)
    return ms


if __name__ == "__main__":
    torch.manual_seed(239)
    device = "cuda"
    batch_size, S, H, D = 64, 256, 2048, 1024
    p = 0.0

    x = torch.randn(batch_size, S, H, device=device, dtype=torch.float32)
    dy = torch.randn_like(x)

    ref = TorchFFNSwiGLU(H, D, p=p).to(device).train()

    v1 = FFNSwiGLUv1(H, D, p=p).to(device).train()
    v2 = FFNSwiGLUv2(H, D, p=p).to(device).train()
    v3 = FFNSwiGLUv3(H, D, p=p).to(device).train()
    v4 = FFNSwiGLUv4(H, D, p=p).to(device).train()
    v5_fp16 = FFNSwiGLUv5(H, D, p=p, dtype=torch.float16).to(device).train()
    v5_bf16 = FFNSwiGLUv5(H, D, p=p, dtype=torch.bfloat16).to(device).train()

    tie_weights_torch_to_custom(ref, v1)
    tie_weights_torch_to_custom(ref, v2)
    tie_weights_torch_to_custom(ref, v3)
    tie_weights_torch_to_custom(ref, v4)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        ref_fp16 = ref.to(torch.float16)
        tie_weights_torch_to_custom(ref_fp16, v5_fp16)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        ref_bf16 = ref.to(torch.bfloat16)
        tie_weights_torch_to_custom(ref_bf16, v5_bf16)

    # FP32 benchmarks
    ms_ref_fwd, y_ref = torch_ffn_forward_benchmarks(x, ref)
    ms_ref_fwd_cpu, y_ref_cpu = torch_ffn_forward_benchmarks(x.clone().to("cpu"), ref.to("cpu"))
    ms_v1_fwd,  y_v1  = custom_ffn_forward_benchmarks(x, v1)
    ms_v2_fwd,  y_v2  = custom_ffn_forward_benchmarks(x, v2)
    ms_v3_fwd,  y_v3  = custom_ffn_forward_benchmarks(x, v3)
    ms_v4_fwd,  y_v4  = custom_ffn_forward_benchmarks(x, v4)

    # FP16/BF16 benchmarks
    x_fp16 = x.to(torch.float16)
    x_bf16 = x.to(torch.bfloat16)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        ms_ref_fp16_fwd, y_ref_fp16 = torch_ffn_forward_benchmarks(x_fp16, ref_fp16)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        ms_ref_bf16_fwd, y_ref_bf16 = torch_ffn_forward_benchmarks(x_bf16, ref_bf16)
    ms_v5_fp16_fwd, y_v5_fp16 = custom_ffn_forward_benchmarks(x_fp16, v5_fp16)
    ms_v5_bf16_fwd, y_v5_bf16 = custom_ffn_forward_benchmarks(x_bf16, v5_bf16)

    # FP32 diffs
    maxdiff_v1 = (y_ref - y_v1).abs().max().item()
    maxdiff_v2 = (y_ref - y_v2).abs().max().item()
    maxdiff_v3 = (y_ref - y_v3).abs().max().item()
    maxdiff_v4 = (y_ref - y_v4).abs().max().item()

    # FP16/BF16 diffs (comparing to corresponding torch implementations)
    maxdiff_v5_fp16 = (y_ref_fp16 - y_v5_fp16.to(y_ref_fp16.dtype)).abs().max().item()
    maxdiff_v5_bf16 = (y_ref_bf16 - y_v5_bf16.to(y_ref_bf16.dtype)).abs().max().item()

    print(f"[FFN+SwiGLU] forward (FP32):")
    print(f"  torch (cpu) : {ms_ref_fwd_cpu:.3f} ms")
    print(f"  torch (cuda): {ms_ref_fwd:.3f} ms")
    print(f"  v1          : {ms_v1_fwd:.3f} ms | max|diff|={maxdiff_v1:.3e}")
    print(f"  v2          : {ms_v2_fwd:.3f} ms | max|diff|={maxdiff_v2:.3e}")
    print(f"  v3          : {ms_v3_fwd:.3f} ms | max|diff|={maxdiff_v3:.3e}")
    print(f"  v4          : {ms_v4_fwd:.3f} ms | max|diff|={maxdiff_v4:.3e}")

    print(f"\n[FFN+SwiGLU] forward (FP16/BF16):")
    print(f"  torch (fp16): {ms_ref_fp16_fwd:.3f} ms")
    print(f"  torch (bf16): {ms_ref_bf16_fwd:.3f} ms")
    print(f"  v5 (fp16)   : {ms_v5_fp16_fwd:.3f} ms | max|diff|={maxdiff_v5_fp16:.3e}")
    print(f"  v5 (bf16)   : {ms_v5_bf16_fwd:.3f} ms | max|diff|={maxdiff_v5_bf16:.3e}")

    ms_ref_bwd = torch_ffn_backward_benchmarks(dy, ref.to("cuda"), x)
    print(f"\n[FFN+SwiGLU] backward (torch only, p=0): {ms_ref_bwd:.3f} ms")
