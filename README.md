# CudaTransformer

Implementation of Transformer-Decoder with MLA, RoPE, SwiGLU and RMSNorm with custom CUDA Kernels

## Structure 
```
src/
  benchmarks/       # benchmarking scripts
  cuda_kernels/     # CUDA kernels implementations, torch wrappers and python bindings
  modules/          # python wrappers and modules
```

## Realization

- RMSNorm: rmsnorm implemented with shared memory buffer and double warp-reductions for squared root calculation in both forward and backward pass kernels and for dot product sum in backward pass kernel.
- Safe Softmax: softmax implemented with max calculation and substraction for numerical stability and to prevent
  - Realized only for fp32
  - Realized via warp-on-row calculation
  - TODO: realize for fp16/bf16 with vectorized memory access and block-on-row calculation  
- Embeddings layer: work in progress
- FFN with SwiGLU: step-by-step optimizations for 2 kernels (gemm-swiglu-dropout and gemm-residual-dropout)
  - Realized naive matmul version
  - Realized coalesced memory access version
  - SMEM tiling
  - Registers 2D tiling
  - fp16/bf16 support with vector access to GMEM/SMEM/registers (work in progreess)
  - tensor cores support (work in progreess)
  - double buffer and async tiles loading (work in progreess)

- MLA with RoPE: work in progress


## Installation

This project requires a working CUDA toolkit and a PyTorch installation with matching CUDA version (I was working with CUDA 12.9 version).
