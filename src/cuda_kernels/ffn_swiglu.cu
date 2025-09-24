// #include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


// device functions



// 1st fwd-kernel: GEMM + SwiGLU + Dropout
__global__ void ffn_swiglu_dropout_fp32_kernel(
    const float* __restrict__ x,    // input data to ffn block
    float* __restrict__ z,          // output data

    const float* __restrict__ W_1,  // [W_gate | W_in]
    const float* __restrict__ b_1,  // [b_gate | b_in]

    const uint32_t num_vectors,     // batch_size * seq_len      

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){

}


// 2st fwd-kernel: GEMM + residual + dropout
__global__ void ffn_residual_dropout_fp32_kernel(
    const float* __restrict__ x,    // input data to ffn block
    const float* __restrict__ z,    // input data from prev kernel
    float* y,                       // output data

    const float* __restrict__ W_2,  // second ffn layer weights
    const float* __restrict__ b_2,  // second ffn layer bias

    const uint32_t num_vectors,     // batch_size * seq_len

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){

}


// backward-pass kernel



// host functions
// forward



// backward


// torch forward wrapper



// torch backward wrapper



// bindings
