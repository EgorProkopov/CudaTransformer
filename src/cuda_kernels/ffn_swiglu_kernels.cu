#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <math_constants.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#ifndef BLOCK_SIZE_N
#define BLOCK_SIZE_N 32
#endif

#ifndef BLOCK_SIZE_M
#define BLOCK_SIZE_M 8
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

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


// 2st fwd-kernel v1: GEMM + Residual + Dropout naive
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
// Заметки по кернелу:
// Необходимо, чтобы blockDim.x == BLOCK_SIZE * BLOCK_SIZE, а blockDim == 1,
// иначе сломается коалесцированный доступ
__global__ void ffn_swiglu_dropout_fp32_kernel_v2(
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
    const uint32_t row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
    const uint32_t col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;

    const uint64_t x_offset = (uint64_t)row * (uint64_t)embedding_dim;
    const uint64_t z_index = (uint64_t)row * (uint64_t)hidden_dim + col;

    if (!(row < num_vectors && col < hidden_dim)) return;
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


// 2st fwd-kernel v2: GEMM + Residual + Dropout with coalesced access
__global__ void ffn_residual_dropout_fp32_kernel_v2(
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
    const uint32_t row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
    const uint32_t col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;

    const uint64_t z_offset = (uint64_t)row * (uint64_t)hidden_dim;
    const uint64_t y_index = (uint64_t)row * (uint64_t)embedding_dim + col;

    if (row >= num_vectors || col >= embedding_dim) return;

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


// ===========================================================================
// =============================== v3 ========================================
// ===========================================================================

// 1st fwd-kernel v3: GEMM + SwiGLU + Dropout with SMEM tiling
__global__ void ffn_swiglu_dropout_fp32_kernel_v3(
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
    __shared__ float x_s[TILE_SIZE][TILE_SIZE + 1];      // +1 to avoid bank conflicts
    __shared__ float W_gate_s[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float W_in_s[TILE_SIZE][TILE_SIZE + 1];

    const uint32_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const uint32_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const uint64_t x_offset = (uint64_t)row * (uint64_t)embedding_dim;
    const uint64_t z_index = (uint64_t)row * (uint64_t)hidden_dim + col;

    float sum_gate = 0.0f;
    float sum_in = 0.0f;

    // load tiles to shared memory and compute partial sums
    for (uint32_t tile = 0; tile < (embedding_dim + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < num_vectors && (tile * TILE_SIZE + threadIdx.x) < embedding_dim) {
            x_s[threadIdx.y][threadIdx.x] = x[x_offset + tile * TILE_SIZE + threadIdx.x];
        } else {
            x_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((tile * TILE_SIZE + threadIdx.y) < embedding_dim && col < hidden_dim) {
            const uint64_t w_idx = (uint64_t)(tile * TILE_SIZE + threadIdx.y) * (uint64_t)hidden_dim + col;
            W_gate_s[threadIdx.y][threadIdx.x] = W_gate[w_idx];
            W_in_s[threadIdx.y][threadIdx.x] = W_in[w_idx];
        } else {
            W_gate_s[threadIdx.y][threadIdx.x] = 0.0f;
            W_in_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (row < num_vectors && col < hidden_dim) {
            #pragma unroll
            for (uint32_t k = 0; k < TILE_SIZE; ++k) {
                sum_gate = fmaf(x_s[threadIdx.y][k], W_gate_s[k][threadIdx.x], sum_gate);
                sum_in = fmaf(x_s[threadIdx.y][k], W_in_s[k][threadIdx.x], sum_in);
            }
        }

        __syncthreads();
    }

    // compute swiglu and dropout
    if (row < num_vectors && col < hidden_dim) {
        sum_gate += b_gate[col];
        sum_in += b_in[col];
        
        // SwiGLU activation
        float sigmoid = __fdividef(1.0f, 1.0f + __expf(-sum_gate));
        float value = (sum_gate * sigmoid) * sum_in;

        if (mask && p > 0.0f) {
            curandStatePhilox4_32_10_t st;
            curand_init(
                (uint64_t)seed,
                z_index,
                (uint64_t)offset,
                &st
            );
            float r = curand_uniform(&st);
            uint8_t m = (r > p);
            mask[z_index] = m;

            float inv_dropout = 1.0f / (1.0f - p);
            if (m) {
                z[z_index] = value * inv_dropout;
            } else {
                z[z_index] = 0.0f;
            }
        } else {
            z[z_index] = value;
        }
    }
}


// 2st fwd-kernel v3: GEMM + Residual + Dropout with SMEM tiling
__global__ void ffn_residual_dropout_fp32_kernel_v3(
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
    __shared__ float z_s[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float W_out_s[TILE_SIZE][TILE_SIZE + 1];

    const uint32_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const uint32_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const uint64_t z_offset = (uint64_t)row * (uint64_t)hidden_dim;
    const uint64_t y_index = (uint64_t)row * (uint64_t)embedding_dim + col;

    float sum = 0.0f;

    // load tiles to shared memory and compute partial sums
    for (uint32_t tile = 0; tile < (hidden_dim + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < num_vectors && (tile * TILE_SIZE + threadIdx.x) < hidden_dim) {
            z_s[threadIdx.y][threadIdx.x] = z[z_offset + tile * TILE_SIZE + threadIdx.x];
        } else {
            z_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((tile * TILE_SIZE + threadIdx.y) < hidden_dim && col < embedding_dim) {
            const uint64_t w_idx = (uint64_t)(tile * TILE_SIZE + threadIdx.y) * (uint64_t)embedding_dim + col;
            W_out_s[threadIdx.y][threadIdx.x] = W_out[w_idx];
        } else {
            W_out_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (row < num_vectors && col < embedding_dim) {
            #pragma unroll
            for (uint32_t k = 0; k < TILE_SIZE; ++k) {
                sum = fmaf(z_s[threadIdx.y][k], W_out_s[k][threadIdx.x], sum);
            }
        }

        __syncthreads();
    }

    // compute residual and dropout
    if (row < num_vectors && col < embedding_dim) {
        float value = sum + b_out[col];
        float res = residual[y_index];

        if (mask && p > 0.0f) {
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
            if (m) {
                value = value * inv_dropout;
            } else {
                value = 0.0f;
            }
        }
        y[y_index] = value + res;
    }
}

// ===========================================================================
// =============================== v4 ========================================
// ===========================================================================

// 1st fwd-kernel v4: GEMM + SwiGLU + Dropout with RMEM tiling
__global__ void ffn_swiglu_dropout_fp32_kernel_v4(
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

}


// 2st fwd-kernel v4: GEMM + Residual + Dropout with RMEM tiling
__global__ void ffn_residual_dropout_fp32_kernel_v4(
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

}

// ===========================================================================
// =============================== v5 ========================================
// ===========================================================================

// 1st fwd-kernel v5: GEMM + SwiGLU + Dropout with fp16/bf16 support
__global__ void ffn_swiglu_dropout_fp32_kernel_v5(
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

}


// 2st fwd-kernel v5: GEMM + Residual + Dropout with fp16/bf16 support
__global__ void ffn_residual_dropout_fp32_kernel_v5(
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

}


// ===========================================================================
// =============================== v6 ========================================
// ===========================================================================

// 1st fwd-kernel v6: GEMM + SwiGLU + Dropout with tensor cores support
__global__ void ffn_swiglu_dropout_fp32_kernel_v6(
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

}


// 2st fwd-kernel v6: GEMM + Residual + Dropout with tensor cores support
__global__ void ffn_residual_dropout_fp32_kernel_v6(
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

}



// ===========================================================================
// =========================== backward-pass =================================
// ===========================================================================
// backward-pass kernel



// ===========================================================================
// =========================== host-functions ================================
// ===========================================================================
// forward
static inline void check_device_forward(
    const torch::Tensor& x,
    const torch::Tensor& W_gate,
    const torch::Tensor& b_gate,
    const torch::Tensor& W_in,
    const torch::Tensor& b_in,
    const torch::Tensor& W_out,
    const torch::Tensor& b_out
){
    TORCH_CHECK(x.is_cuda(), "Input data (x) must be CUDA");

    TORCH_CHECK(W_gate.is_cuda(), "Weights (W_gate) must be CUDA");
    TORCH_CHECK(b_gate.is_cuda(), "Weights (b_gate) must be CUDA");

    TORCH_CHECK(W_in.is_cuda(), "Weights (W_in) must be CUDA");
    TORCH_CHECK(b_in.is_cuda(), "Weights (b_in) must be CUDA");

    TORCH_CHECK(W_out.is_cuda(), "Weights (W_out) must be CUDA");
    TORCH_CHECK(b_out.is_cuda(), "Weights (b_out) must be CUDA");

    TORCH_CHECK(
        (
            W_gate.get_device() == b_gate.get_device() && 
            W_in.get_device() == b_in.get_device() && 
            W_out.get_device() == b_out.get_device() && 
            W_gate.get_device() == W_in.get_device() && 
            W_in.get_device() == W_out.get_device() && 
            b_gate.get_device() == b_in.get_device() && 
            b_in.get_device() == b_out.get_device()
        ),
        "All weights must be on the same device"
    );

    TORCH_CHECK(
        x.get_device() == W_in.get_device(), "Weights and data must be on the same device"
    );
}

static inline void check_shape_forward(
    const torch::Tensor& x,
    const torch::Tensor& W_gate,
    const torch::Tensor& b_gate,
    const torch::Tensor& W_in,
    const torch::Tensor& b_in,
    const torch::Tensor& W_out,
    const torch::Tensor& b_out
){
    // check data shape
    TORCH_CHECK(
        x.dim() == 3, 
        "Input data (x) must be 3 dim: [batch_size, seq_len, embedding_dim], but dim=", x.dim()
    );
    
    // check weights shapes
    TORCH_CHECK(
        W_gate.dim() == 2, 
        "Weights (W_gate) must be 2 dim: [embedding_dim, hidden_dim], but dim=", W_gate.dim()
    );
    TORCH_CHECK(
        b_gate.dim() == 1, 
        "Weights (b_gate) must be 1 dim: [hidden_dim], but dim=", b_gate.dim()
    );

    TORCH_CHECK(
        W_in.dim() == 2, 
        "Weights (W_in) must be 2 dim: [embedding_dim, hidden_dim], but dim=", W_in.dim()
    );
    TORCH_CHECK(
        b_in.dim() == 1, 
        "Weights (b_in) must be 1 dim: [hidden_dim], but dim=", b_in.dim()
    );

    TORCH_CHECK(
        W_out.dim() == 2, 
        "Weights (W_out) must be 2 dim: [hidden_dim, embedding_dim], but dim=", W_out.dim()
    );
    TORCH_CHECK(
        b_out.dim() == 1, 
        "Weights (b_out) must be 1 dim: [embedding_dim], but dim=", b_out.dim()
    );

    // check weights dimensions
    TORCH_CHECK(
        W_in.size(1) == W_gate.size(1), "W_in.shape[1] must be hidden_size=", W_gate.size(1), ", got ", W_in.size(1)
    );
    TORCH_CHECK(
        b_gate.size(0) == W_gate.size(1), "b_gate.shape must be hidden_size=", W_gate.size(1), ", got ", b_gate.sizes()
    );
    TORCH_CHECK(
        b_in.size(0) == W_gate.size(1), "b_in.shape must be hidden_size=", W_gate.size(1), ", got ", b_in.sizes()
    );

    TORCH_CHECK(
        W_out.size(0) == W_gate.size(1),
        "W_out.shape[0] must equal hidden_dim=", W_gate.size(1), ", got ", W_out.size(0)
    );
    TORCH_CHECK(
        W_out.size(1) == W_in.size(0),
        "W_out.shape[1] must equal embedding_dim=", W_in.size(0), ", got ", W_out.size(1)
    );
    TORCH_CHECK(
        b_out.size(0) == W_in.size(0),
        "b_out.shape must equal embedding_dim=", W_in.size(0), ", got ", b_out.sizes()
    );
}

static inline void check_is_float_forward(
    const torch::Tensor& x,
    const torch::Tensor& W_gate,
    const torch::Tensor& b_gate,
    const torch::Tensor& W_in,
    const torch::Tensor& b_in,
    const torch::Tensor& W_out,
    const torch::Tensor& b_out
){
    TORCH_CHECK(
        x.scalar_type() == at::kFloat, "Input (x) must be float32, but got=", x.scalar_type()
    );

    TORCH_CHECK(
        W_gate.scalar_type() == at::kFloat, "Weights (W_gate) must be float32, but got=", W_gate.scalar_type()
    );
    TORCH_CHECK(
        b_gate.scalar_type() == at::kFloat, "Weights (b_gate) must be float32, but got=", b_gate.scalar_type()
    );

    TORCH_CHECK(
        W_in.scalar_type() == at::kFloat, "Weights (W_in) must be float32, but got=", W_in.scalar_type()
    );
    TORCH_CHECK(
        b_in.scalar_type() == at::kFloat, "Weights (b_in) must be float32, but got=", b_in.scalar_type()
    );

    TORCH_CHECK(
        W_out.scalar_type() == at::kFloat, "Weights (W_out) must be float32, but got=", W_out.scalar_type()
    );
    TORCH_CHECK(
        b_out.scalar_type() == at::kFloat, "Weights (b_out) must be float32, but got=", b_out.scalar_type()
    );
}


// backward



// ===========================================================================
// =========================== torch wrappers ================================
// ===========================================================================
// fwd v1 wrapper
torch::Tensor ffn_forward_fp32_kernel_v1(
    torch::Tensor x, 
    
    torch::Tensor W_gate,
    torch::Tensor b_gate,
    torch::Tensor W_in,
    torch::Tensor b_in,
    torch::Tensor W_out,
    torch::Tensor b_out,

    const float p,
    const uint32_t seed
){
    c10::cuda::CUDAGuard device_guard(x.device());
    check_device_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );
    check_shape_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );
    check_is_float_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );

    TORCH_CHECK(p >= 0.f && p < 1.f, "dropout p must be in [0,1), got ", p);

    auto x_c = x.contiguous();

    auto W_gate_c = W_gate.contiguous();
    auto b_gate_c = b_gate.contiguous();

    auto W_in_c = W_in.contiguous();
    auto b_in_c = b_in.contiguous();

    auto W_out_c = W_out.contiguous();
    auto b_out_c = b_out.contiguous();

    const uint32_t batch_size = x_c.size(0);
    const uint32_t seq_len = x_c.size(1);
    const uint32_t embedding_dim = x_c.size(2);
    const uint32_t hidden_dim = W_gate_c.size(1);

    const uint32_t num_vectors = batch_size * seq_len;

    // x_c_reshaped = torch.reshape(x_c, (num_vectors, embedding_dim));
    auto x2d = x_c.view({num_vectors, embedding_dim}); 
    auto z = torch::empty({num_vectors, hidden_dim}, x_c.options());
    auto y = torch::empty({num_vectors, embedding_dim}, x_c.options());

    // dropout masks
    torch::Tensor mask1, mask2;
    uint8_t* mask1_ptr = nullptr;
    uint8_t* mask2_ptr = nullptr;

    if (p > 0.f) {
        mask1 = torch::empty({num_vectors, hidden_dim}, x_c.options().dtype(torch::kUInt8));
        mask2 = torch::empty({num_vectors, embedding_dim}, x_c.options().dtype(torch::kUInt8));
        mask1_ptr = mask1.data_ptr<uint8_t>();
        mask2_ptr = mask2.data_ptr<uint8_t>();
    }

    // launch configs
    dim3 block1(BLOCK_SIZE_N, BLOCK_SIZE_N);
    dim3 grid1(
        (hidden_dim + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (num_vectors + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N
    );
    dim3 block2(BLOCK_SIZE_N, BLOCK_SIZE_N);
    dim3 grid2(
        (embedding_dim + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (num_vectors + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N
    );

    // Philox offsets
    // first kernel uses [0 .. ceil(num_vectors*hidden_dim/4)-1], second one starts from offset2
    const uint64_t n1 = static_cast<uint64_t>(num_vectors) * static_cast<uint64_t>(hidden_dim);
    const uint32_t offset1 = 0;
    const uint32_t offset2 = static_cast<uint32_t>((n1 + 4 - 1) / 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ffn_swiglu_dropout_fp32_kernel_v1<<<grid1, block1, 0, stream>>>(
        x2d.data_ptr<float>(),
        z.data_ptr<float>(),
        W_gate_c.data_ptr<float>(), b_gate_c.data_ptr<float>(),
        W_in_c.data_ptr<float>(),   b_in_c.data_ptr<float>(),
        mask1_ptr, p, seed, offset1,
        num_vectors, embedding_dim, hidden_dim
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FFN forward: 1st kernel launch failed");

    ffn_residual_dropout_fp32_kernel_v1<<<grid2, block2, 0, stream>>>(
        x2d.data_ptr<float>(),
        z.data_ptr<float>(),
        y.data_ptr<float>(),

        W_out_c.data_ptr<float>(),   b_out_c.data_ptr<float>(),
        mask2_ptr, p, seed, offset2,
        num_vectors, embedding_dim, hidden_dim
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FFN forward: 2nd kernel launch failed");

    return y.view({batch_size, seq_len, embedding_dim});
}

// fwd v2 wrapper
torch::Tensor ffn_forward_fp32_kernel_v2(
    torch::Tensor x, 
    
    torch::Tensor W_gate,
    torch::Tensor b_gate,
    torch::Tensor W_in,
    torch::Tensor b_in,
    torch::Tensor W_out,
    torch::Tensor b_out,

    const float p,
    const uint32_t seed
){
    c10::cuda::CUDAGuard device_guard(x.device());
    check_device_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );
    check_shape_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );
    check_is_float_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );

    TORCH_CHECK(p >= 0.f && p < 1.f, "dropout p must be in [0,1), got ", p);

    auto x_c = x.contiguous();

    auto W_gate_c = W_gate.contiguous();
    auto b_gate_c = b_gate.contiguous();

    auto W_in_c = W_in.contiguous();
    auto b_in_c = b_in.contiguous();

    auto W_out_c = W_out.contiguous();
    auto b_out_c = b_out.contiguous();

    const uint32_t batch_size = x_c.size(0);
    const uint32_t seq_len = x_c.size(1);
    const uint32_t embedding_dim = x_c.size(2);
    const uint32_t hidden_dim = W_gate_c.size(1);

    const uint32_t num_vectors = batch_size * seq_len;

    // x_c_reshaped = torch.reshape(x_c, (num_vectors, embedding_dim));
    auto x2d = x_c.view({num_vectors, embedding_dim}); 
    auto z = torch::empty({num_vectors, hidden_dim}, x_c.options());
    auto y = torch::empty({num_vectors, embedding_dim}, x_c.options());

    // dropout masks
    torch::Tensor mask1, mask2;
    uint8_t* mask1_ptr = nullptr;
    uint8_t* mask2_ptr = nullptr;

    if (p > 0.f) {
        mask1 = torch::empty({num_vectors, hidden_dim}, x_c.options().dtype(torch::kUInt8));
        mask2 = torch::empty({num_vectors, embedding_dim}, x_c.options().dtype(torch::kUInt8));
        mask1_ptr = mask1.data_ptr<uint8_t>();
        mask2_ptr = mask2.data_ptr<uint8_t>();
    }

    // launch configs
    dim3 block1(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid1(
        (hidden_dim + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (num_vectors + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );
    dim3 block2(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 grid2(
        (embedding_dim + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (num_vectors + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );

    // Philox offsets
    // first kernel uses [0 .. ceil(num_vectors*hidden_dim/4)-1], second one starts from offset2
    const uint64_t n1 = static_cast<uint64_t>(num_vectors) * static_cast<uint64_t>(hidden_dim);
    const uint32_t offset1 = 0;
    const uint32_t offset2 = static_cast<uint32_t>((n1 + 4 - 1) / 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ffn_swiglu_dropout_fp32_kernel_v2<<<grid1, block1, 0, stream>>>(
        x2d.data_ptr<float>(),
        z.data_ptr<float>(),
        W_gate_c.data_ptr<float>(), b_gate_c.data_ptr<float>(),
        W_in_c.data_ptr<float>(),   b_in_c.data_ptr<float>(),
        mask1_ptr, p, seed, offset1,
        num_vectors, embedding_dim, hidden_dim
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FFN forward: 1st kernel launch failed");

    ffn_residual_dropout_fp32_kernel_v2<<<grid2, block2, 0, stream>>>(
        x2d.data_ptr<float>(),
        z.data_ptr<float>(),
        y.data_ptr<float>(),

        W_out_c.data_ptr<float>(),   b_out_c.data_ptr<float>(),
        mask2_ptr, p, seed, offset2,
        num_vectors, embedding_dim, hidden_dim
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FFN forward: 2nd kernel launch failed");

    return y.view({batch_size, seq_len, embedding_dim});
}

// backward wrapper

// fwd v3 wrapper
torch::Tensor ffn_forward_fp32_kernel_v3(
    torch::Tensor x, 
    
    torch::Tensor W_gate,
    torch::Tensor b_gate,
    torch::Tensor W_in,
    torch::Tensor b_in,
    torch::Tensor W_out,
    torch::Tensor b_out,

    const float p,
    const uint32_t seed
){
    c10::cuda::CUDAGuard device_guard(x.device());
    check_device_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );
    check_shape_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );
    check_is_float_forward(
        x, W_gate, b_gate, W_in, b_in, W_out, b_out
    );

    TORCH_CHECK(p >= 0.f && p < 1.f, "dropout p must be in [0,1), got ", p);

    auto x_c = x.contiguous();

    auto W_gate_c = W_gate.contiguous();
    auto b_gate_c = b_gate.contiguous();

    auto W_in_c = W_in.contiguous();
    auto b_in_c = b_in.contiguous();

    auto W_out_c = W_out.contiguous();
    auto b_out_c = b_out.contiguous();

    const uint32_t batch_size = x_c.size(0);
    const uint32_t seq_len = x_c.size(1);
    const uint32_t embedding_dim = x_c.size(2);
    const uint32_t hidden_dim = W_gate_c.size(1);

    const uint32_t num_vectors = batch_size * seq_len;

    auto x2d = x_c.view({num_vectors, embedding_dim}); 
    auto z = torch::empty({num_vectors, hidden_dim}, x_c.options());
    auto y = torch::empty({num_vectors, embedding_dim}, x_c.options());

    // dropout masks
    torch::Tensor mask1, mask2;
    uint8_t* mask1_ptr = nullptr;
    uint8_t* mask2_ptr = nullptr;

    if (p > 0.f) {
        mask1 = torch::empty({num_vectors, hidden_dim}, x_c.options().dtype(torch::kUInt8));
        mask2 = torch::empty({num_vectors, embedding_dim}, x_c.options().dtype(torch::kUInt8));
        mask1_ptr = mask1.data_ptr<uint8_t>();
        mask2_ptr = mask2.data_ptr<uint8_t>();
    }

    // launch configs
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1(
        (hidden_dim + TILE_SIZE - 1) / TILE_SIZE,
        (num_vectors + TILE_SIZE - 1) / TILE_SIZE
    );
    dim3 block2(TILE_SIZE, TILE_SIZE);
    dim3 grid2(
        (embedding_dim + TILE_SIZE - 1) / TILE_SIZE,
        (num_vectors + TILE_SIZE - 1) / TILE_SIZE
    );

    // Philox offsets
    const uint64_t n1 = static_cast<uint64_t>(num_vectors) * static_cast<uint64_t>(hidden_dim);
    const uint32_t offset1 = 0;
    const uint32_t offset2 = static_cast<uint32_t>((n1 + 4 - 1) / 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ffn_swiglu_dropout_fp32_kernel_v3<<<grid1, block1, 0, stream>>>(
        x2d.data_ptr<float>(),
        z.data_ptr<float>(),
        W_gate_c.data_ptr<float>(), b_gate_c.data_ptr<float>(),
        W_in_c.data_ptr<float>(),   b_in_c.data_ptr<float>(),
        mask1_ptr, p, seed, offset1,
        num_vectors, embedding_dim, hidden_dim
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FFN forward: 1st kernel launch failed");

    ffn_residual_dropout_fp32_kernel_v3<<<grid2, block2, 0, stream>>>(
        x2d.data_ptr<float>(),
        z.data_ptr<float>(),
        y.data_ptr<float>(),
        W_out_c.data_ptr<float>(),   b_out_c.data_ptr<float>(),
        mask2_ptr, p, seed, offset2,
        num_vectors, embedding_dim, hidden_dim
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "FFN forward: 2nd kernel launch failed");

    return y.view({batch_size, seq_len, embedding_dim});
}


// bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_v1",
        &ffn_forward_fp32_kernel_v1,
        py::arg("x"),
        py::arg("W_gate"), py::arg("b_gate"),
        py::arg("W_in"), py::arg("b_in"),
        py::arg("W_out"), py::arg("b_out"),
        py::arg("p") = 0.3, py::arg("seed") = 239,
        "FFN+SwiGLU with naive GEMM realization"
    );

    m.def(
        "forward_v2",
        &ffn_forward_fp32_kernel_v2,
        py::arg("x"),
        py::arg("W_gate"), py::arg("b_gate"),
        py::arg("W_in"), py::arg("b_in"),
        py::arg("W_out"), py::arg("b_out"),
        py::arg("p") = 0.3, py::arg("seed") = 239,
        "FFN+SwiGLU with coalesced memory access GEMM realization"
    );

    m.def(
        "forward_v3",
        &ffn_forward_fp32_kernel_v3,
        py::arg("x"),
        py::arg("W_gate"), py::arg("b_gate"),
        py::arg("W_in"), py::arg("b_in"),
        py::arg("W_out"), py::arg("b_out"),
        py::arg("p") = 0.3, py::arg("seed") = 239,
        "FFN+SwiGLU with shared memory tiling GEMM realization"
    );
}
