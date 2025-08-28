# CudaTransformer

Implementation of Transformer-Decoder with MLA, RoPE, SwiGLU and RMSNorm with custom CUDA Kernels

## Structure 
```
src/
  benchmarks/       # benchmarking scripts
  cuda_kernels/     # CUDA kernels implementations, torch wrappers and python bindings
  modules/          # python wrappers and modules
setup.py            # Setup file
```

## Realization

- RMSNorm: rmsnorm implemented with shared memory buffer and double warp-reductions for squared root calculation in both forward and backward pass kernels and for dot product sum in backward pass kernel.
- Safe Softmax: work in progress
- Embeddings layer: work in progress
- FFN with SwiGLU: work in progress
- MLA with RoPE: work in progress


## Installation

This project requires a working CUDA toolkit and a PyTorch installation with matching CUDA version (I was working with CUDA 12.9 version).

To build the package run:

```bash
python setup.py build_ext --inplace
```