// includes 
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>


// device functions
__device__ __forceinline__ float warp_reduce_max_fp32(float v, unsigned mask) {
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v,  8));
    v = fmaxf(v, __shfl_down_sync(mask, v,  4));
    v = fmaxf(v, __shfl_down_sync(mask, v,  2));
    v = fmaxf(v, __shfl_down_sync(mask, v,  1));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum_fp32(float sum, unsigned mask){
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);

    return sum;
}

// forward-pass kernel

// Заметки по кернелу:
// 1. softmax --- крайне mem-bound алгоритм, перенос в SMEM не окупается
// 2. Лучше использовать warp-per-row, поскольку:
// - линии варпа идут по соседним элементам строки, коалесцированный доступ
// - внутри варпа не нужны синхронизации, warp-shuffle редукции - быстрые регистровые операции
// - можно не использовать межблоковые редукции и атомики, поскольку максимум и сумма считаются локально в варпе
 
// План кернела:
// 1. Чтение должно быть коалесцированно:
// - lane_id = 0 читает c=0, 32, 64 и тд
// варп берет данные пачками по 32
// 2. Вычисление локального максимума по лейну через fmaxf
// 3. Вычисление глобального максимума строки через варп-редукцию
// 4. Броадкаст полученного максимума по лейнам варпа
// 5. Вычисление t_i = exp(x_i - row_max)
// 6. Вычисление суммы (снова редукция через варп) Z = sum_i(t_i)
// 7. Вычисление результата y_i = t_i / Z

__global__ void safe_softmax_fwd_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t batch_size, int64_t seq_len, float eps 
){
    const int64_t lane_id = threadIdx.x % warpSize;
    const int64_t warp_in_block_id = threadIdx.x / warpSize;
    const int64_t num_warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    const int64_t row_idx = blockIdx.x * num_warps_per_block + warp_in_block_id;
    
    const unsigned mask = __activemask();

    // no need in grid-stride, batch_size = 2^16 is too big to be possible
    if (row_idx >= batch_size) {
        return;
    }

    const int64_t stride = row_idx * seq_len;

    float lane_max = -CUDART_INF_F;
    #pragma unroll 1
    for (int64_t col_id = lane_id; col_id < seq_len; col_id += warpSize){
        float value = x[stride + col_id];
        lane_max = fmaxf(lane_max, value);
    }

    // warp-reduction for row_max
    float row_max = warp_reduce_max_fp32(lane_max, mask);

    // lane 0 broadcast
    row_max = __shfl_sync(mask, row_max, 0);

    float lane_exp_sum = 0;
    #pragma unroll 1
    for (int64_t col_id = lane_id; col_id < seq_len; col_id += warpSize){
        float value = x[stride + col_id];
        float exp_value = __expf(value - row_max);
        lane_exp_sum += exp_value;
    }

    // warp-reduction for sum
    float row_exp_sum = warp_reduce_sum_fp32(lane_exp_sum, mask);
    // lane 0 broadcast
    row_exp_sum = __shfl_sync(mask, row_exp_sum, 0);
    row_exp_sum += eps;
    float inv_exp_sum = __fdividef(1.f, row_exp_sum);

    #pragma unroll 1
    for (int64_t col_id = lane_id; col_id < seq_len; col_id += warpSize){
        // выгоднее 1 лишний раз прочесть x из GMEM и пересчитать exp, 
        // чем записать y = exp(x-m) (запись в GMEM) и прочесть y из GMEM, 
        // чтобы разделить на Z
        float value = x[stride + col_id];
        float exp_value = __expf(value - row_max); 

        y[stride + col_id] = exp_value * inv_exp_sum;
    }
}


// backward-pass kernel


// host functions
// utils



// torch forward wrapper



// torch backward wrapper



// bindings
