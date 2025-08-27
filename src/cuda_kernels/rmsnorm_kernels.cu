/// RMSNorm kernels
/// RMSNorm(x) = gamma * x / sqrt(||x||_2^2 + epsilon)
/// There is reduction op


// includes
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>


// -------------------------------------------//
// device kernels and funcitons
__device__ __forceinline__ float to_float(float v){ return v;}
__device__ __forceinline__ float to_float(__half v){ return __half2float(v);}
__device__ __forceinline__ float to_float(__nv_bfloat16 v){ return __bfloat162float(v);}


// честно сгенерировано гпт
template <typename T>
__host__ __device__ __forceinline__ T from_float(float v) {
    if constexpr (std::is_same_v<T, float>)            return v;
    else if constexpr (std::is_same_v<T, __half>)       return __float2half_rn(v);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)return __float2bfloat16(v);
    else {
        static_assert(!std::is_same_v<T, T>, "from_float: unsupported T");
    }
}


__device__ __forceinline__ float warp_sum_fp32(float sum, unsigned mask){
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2){
        sum += __shfl_down_sync(mask, sum, offset);
    }
    return sum;
}


// forward-pass kernel
template <typename scalar_t>
__global__ void rmsnorm_fwd_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ gamma,
    int64_t seq_len, int64_t embedding_dim, float eps
){
    // smem allocation
    extern __shared__ unsigned char shared_row[];
    scalar_t* row = reinterpret_cast<scalar_t*>(shared_row);

    // reduction buffer
    float* warp_sums = reinterpret_cast<float*>(row + embedding_dim);

    const int64_t warp_id = threadIdx.x / warpSize;
    const int64_t lane_id = threadIdx.x % warpSize;
    const int64_t num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // grid-stride, for the case when seq_len > gridDim.x
    for (int64_t row_idx = blockIdx.x; row_idx < seq_len; row_idx+=gridDim.x){
        // block-stride load, for the case when embedding_dim > blockDim.x
        // in this case every thread in the block loads embedding_dim / blockDim.x values
        float local_squared_sum = 0.0f; // partial squares sum 
        for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x){
            // load to smem
            scalar_t value = x[row_idx * embedding_dim + i];
            row[i] = value;
            // partial squares sum
            float value_fp32 = to_float(value);
            local_squared_sum += value_fp32 * value_fp32;
        }

        // warp partial sums reduction by shfl
        unsigned mask = __activemask();
        float partial_sum = warp_sum_fp32(local_squared_sum, mask);

        // warp leaders loads partial sums to shared mem buffer
        if (lane_id == 0){
            warp_sums[warp_id] = partial_sum;
        }
        __syncthreads();

        // final reduction in 1st warp
        float block_sum = 0.0f;
        if (threadIdx.x < num_warps){
            block_sum = warp_sums[threadIdx.x];
        }

        if (warp_id == 0){
            unsigned mask0 = __ballot_sync(0xffffffffu, lane_id < num_warps);
            block_sum = warp_sum_fp32(block_sum, mask0);

            if (lane_id == 0) {
                warp_sums[0] = block_sum; // сохранить общий результат
            }
        }
        __syncthreads();

        // rms calculation
        float inv_rms = 0.0f;
        if (threadIdx.x == 0){
            float mean_sq = warp_sums[0] / to_float(embedding_dim);
            inv_rms = rsqrtf(mean_sq + eps);
            warp_sums[0] = inv_rms;  // threads broadcast
        }
        __syncthreads();
        inv_rms = warp_sums[0]; // finish broadcast

        for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x){
            float value_fp32 = to_float(row[i]);
            float gamma_value_fp32 = to_float(gamma[i]);
            float output_fp32 = value_fp32 * gamma_value_fp32 * inv_rms;
            y[row_idx * embedding_dim + i] = from_float<scalar_t>(output_fp32);
        }
        __syncthreads();
    }
}