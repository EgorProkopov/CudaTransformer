// #include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <math_constants.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


// device functions




// ===========================================================================
// =============================== v1 ========================================
// ===========================================================================
// 1st fwd-kernel v1: GEMM + SwiGLU + Dropout naive
__global__ void ffn_swiglu_dropout_fp32_kernel_v1(
    const float* __restrict__ x,           // input data to ffn block
    float* __restrict__ z,                 // output data

    const float* __restrict__ W_gate,      // weights for gating layer
    const float* __restrict__ b_gate,      // bias for gating layer

    const float* __restrict__ W_in,        // weights for input layer
    const float* __restrict__ b_in,        // bias for input layer

    uint8_t* __restrict__ mask,            // dropout mask
    const float p,                         // dropout rate
    const uint32_t seed,                   // dropout seed
    const uint32_t offset,                 // rnd offset

    const uint32_t num_vectors,            // batch_size * seq_len      

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){
    // row and col are calculated for result matrix with shape (num_vectors, hidden_dim)
    const uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    const uint64_t x_offset = (uint64_t)row * (uint64_t)embedding_dim;
    const uint64_t z_index = (uint64_t)row * (uint64_t)hidden_dim + col;

    if (row < num_vectors && col < hidden_dim){
        float sum_gate = 0.0f;
        float sum_in = 0.0f;
        #pragma unroll 1
        for (uint32_t k = 0; k < embedding_dim; k++){
            float x_value = x[x_offset + k];
            uint64_t w_index = (uint64_t)k * hidden_dim + col;
            sum_gate = fmaf(x_value, W_gate[w_index], sum_gate);
            sum_in = fmaf(x_value, W_in[w_index], sum_in);
        }
        sum_gate += b_gate[col];
        sum_in += b_in[col];
        
        // SwiGLU:
        // y = (x @ W_gate + b_gate) * sigmoid(x @ W_gate + b_gate) * (x @ W_in + b_in)
            
        // if z_gate = x @ W_gate + b_gate
        // if z_in = x @ W_in + b_in
        // and z = [z_gate | z_in]
        // then:
        // y = z_gate * sigmoid(z_gate) * z_in

        // sigmoid = 1 / (1 + exp(-x))
        float sigmoid = __fdividef(1, 1 + __expf(-sum_gate));
        float value = (sum_gate * sigmoid) * sum_in;

        if (mask && p > 0.0f){
            curandStatePhilox4_32_10_t st;  // unique state for random number generator
            curand_init(
                (uint64_t)seed,
                z_index,
                (uint64_t)offset,
                &st
            );                              // rng initalization
            float r = curand_uniform(&st);
            uint8_t m = (r > p);
            mask[z_index] = m;

            float inv_dropout = 1.0f / (1.0f - p);  // to avoid math expectation change
            if (m){
                z[z_index] = value * inv_dropout;
            } else {
                z[z_index] = 0.0f;
            }
        } else {
            z[z_index] = value;
        }
    }
}


// 2st fwd-kernel v2: GEMM + Residual + Dropout naive
__global__ void ffn_residual_dropout_fp32_kernel_v1(
    const float* __restrict__ residual,    // input data to ffn block
    const float* __restrict__ z,           // input data from prev kernel
    float* y,                              // output data

    const float* __restrict__ W_out,       // second ffn layer weights
    const float* __restrict__ b_out,       // second ffn layer bias

    uint8_t* __restrict__ mask,            // dropout mask
    const float p,                         // dropout rate
    const uint32_t seed,                   // dropout seed
    const uint32_t offset,                 // rnd offset

    const uint32_t num_vectors,            // batch_size * seq_len

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){
    const uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    const uint64_t z_offset = (uint64_t)row * (uint64_t)hidden_dim;
    const uint64_t y_index = (uint64_t)row * (uint64_t)embedding_dim + col;

    if (row < num_vectors && col < embedding_dim){
        float sum = 0.0f;

        #pragma unroll 1
        for (uint32_t k = 0; k < hidden_dim; k++){
            sum = fmaf(z[z_offset + k], W_out[k * embedding_dim + col], sum);
        }

        float value = sum + b_out[col];
        float res = residual[y_index];
        if (mask && p > 0.0f){
            curandStatePhilox4_32_10_t st;
            curand_init(
                (uint64_t)seed,
                y_index,
                (uint64_t)offset,
                &st
            );
            float r = curand_uniform(&st);
            uint8_t m = (r > p);
            mask[y_index] = m;

            float inv_dropout = 1.0f / (1.0f - p);
            if (m){
                value = value * inv_dropout;
            } else {
                value = 0.0f;
            }
        }
        y[y_index] = value + res;
    }
}

// ===========================================================================
// =============================== v2 ========================================
// ===========================================================================

// 1st fwd-kernel v2: GEMM + SwiGLU + Dropout with coalesced access
__global__ void ffn_swiglu_dropout_fp32_kernel_v2(
    const float* __restrict__ x,           // input data to ffn block
    float* __restrict__ z,                 // output data

    const float* __restrict__ W_1,         // [W_gate | W_in]
    const float* __restrict__ b_1,         // [b_gate | b_in]

    uint8_t* __restrict__ mask,            // dropout mask
    const float p,                         // dropout rate
    const uint32_t seed,                   // dropout seed
    const uint32_t offset,                 // rnd offset

    const uint32_t num_vectors,            // batch_size * seq_len      

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){

}


// 2st fwd-kernel v2: GEMM + Residual + Dropout with coalesced access
__global__ void ffn_residual_dropout_fp32_kernel_v2(
    const float* __restrict__ residual,    // input data to ffn block
    const float* __restrict__ z,           // input data from prev kernel
    float* y,                              // output data

    const float* __restrict__ W_2,         // second ffn layer weights
    const float* __restrict__ b_2,         // second ffn layer bias

    uint8_t* __restrict__ mask,            // dropout mask
    const float p,                         // dropout rate
    const uint32_t seed,                   // dropout seed
    const uint32_t offset,                 // rnd offset

    const uint32_t num_vectors,            // batch_size * seq_len

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){

}


// ===========================================================================
// =============================== v3 ========================================
// ===========================================================================

// 1st fwd-kernel v3: GEMM + SwiGLU + Dropout with SMEM tiling
__global__ void ffn_swiglu_dropout_fp32_kernel_v3(
    const float* __restrict__ x,           // input data to ffn block
    float* __restrict__ z,                 // output data

    const float* __restrict__ W_1,         // [W_gate | W_in]
    const float* __restrict__ b_1,         // [b_gate | b_in]

    uint8_t* __restrict__ mask,            // dropout mask
    const float p,                         // dropout rate
    const uint32_t seed,                   // dropout seed
    const uint32_t offset,                 // rnd offset

    const uint32_t num_vectors,            // batch_size * seq_len      

    // vector shapes
    const uint32_t embedding_dim,
    const uint32_t hidden_dim
){

}


// 2st fwd-kernel v3: GEMM + Residual + Dropout with SMEM tiling
__global__ void ffn_residual_dropout_fp32_kernel_v3(
    const float* __restrict__ residual,    // input data to ffn block
    const float* __restrict__ z,           // input data from prev kernel
    float* y,                              // output data

    const float* __restrict__ W_2,         // second ffn layer weights
    const float* __restrict__ b_2,         // second ffn layer bias

    uint8_t* __restrict__ mask,            // dropout mask
    const float p,                         // dropout rate
    const uint32_t seed,                   // dropout seed
    const uint32_t offset,                 // rnd offset

    const uint32_t num_vectors,            // batch_size * seq_len

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
