// includes 
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <math_constants.h>
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
    const int64_t batch_size,
    const int64_t seq_len,
    const float eps 
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
    #pragma unroll 1  // to avoid register overflosw
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
// Заметки по кернелу:
// 1. Формула бэк пропа:
// dx_j = y_j (dy_j - sum_i(dy_i * y_i))
// 2. Нужен gradient checkpointing, так как считать y заново очень дорого

// План кернела:
// 1. Одну строку в батче обрабатывает один варп
// 2. С помощью варп-редукции вычисляется суммма sum_i(dy_i * y_i)
// 3. Сумма броадкастится на все лейны варпа
// 4. Вычисляется dx, зная эту сумму
// итого 4 чтения и 1 запись. 
// Дорого, но без gradient-checkpointing было бы еще дороже

__global__ void safe_softmax_bwd_fp32_kernel(
    const float* __restrict__ y,          // fwd kernel output (grad checkpointing)
    const float* __restrict__ dy,         // dL/dy
    float* __restrict__ dx,               // dL/dx, output
    const int64_t batch_size,             // batch size, obviously
    const int64_t seq_len                 // sequence length
){
    const int64_t lane_id = threadIdx.x % warpSize;
    const int64_t warp_in_block_id = threadIdx.x / warpSize;
    const int64_t num_warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    const int64_t row_idx = blockIdx.x * num_warps_per_block + warp_in_block_id;

    const unsigned mask = __activemask();

    if (row_idx >= batch_size){
        return;
    }

    const int64_t stride = row_idx * seq_len;

    float lane_sum_ydy = 0;
    #pragma unroll 1
    for(int64_t col_id = lane_id; col_id < seq_len; col_id += warpSize){
        float y_value = y[stride + col_id];
        float dy_value = dy[stride + col_id];
        lane_sum_ydy = fmaf(y_value, dy_value, lane_sum_ydy);
    }

    // warp reduction sum
    float row_sum_ydy = warp_reduce_sum_fp32(lane_sum_ydy, mask);
    // lane 0 broadcasting
    row_sum_ydy = __shfl_sync(mask, row_sum_ydy, 0);

    // dx_j = y_j (dy_j - sum_i(dy_i * y_i))
    #pragma unroll 1
    for (int64_t col_id = lane_id; col_id < seq_len; col_id += warpSize){
        dx[stride + col_id] = y[stride + col_id] * (dy[stride + col_id] - row_sum_ydy);
    }
}

// host functions
// forward
static inline void check_device_forward(const torch::Tensor& x){
    TORCH_CHECK(x.is_cuda(), "Input data (x) must be CUDA");
}

static inline void check_shape_forward(const torch::Tensor& x){
    TORCH_CHECK(x.dim() == 2, "Input data (x) must be [batch_size, seq_len], but dim=", x.dim());
}
static inline void check_dtype_forward(const torch::Tensor& x){
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input (x) must be float32, but got=", x.scalar_type());
}


// backward
static inline void check_device_backward(
    const torch::Tensor& y, const torch::Tensor& dy
){
    TORCH_CHECK(y.is_cuda(), "Input data (y) must be CUDA");
    TORCH_CHECK(dy.is_cuda(), "Input data (dy) must be CUDA");

    TORCH_CHECK(y.get_device() == dy.get_device(), "Input elements must be on the same device");
}

static inline void check_shape_backward(
    const torch::Tensor& y, const torch::Tensor& dy
){
    TORCH_CHECK(y.dim() == 2, "Input data (y) must be [batch_size, seq_len]");
    TORCH_CHECK(dy.dim() == 2, "Input data (dy) must be [batch_size, seq_len]");

    TORCH_CHECK(y.sizes() == dy.sizes(), "Input data (y and dy) sizes mismatch");
}

static inline void check_dtype_backward(
    const torch::Tensor& y,
    const torch::Tensor& dy
){
    TORCH_CHECK(
        y.scalar_type() == at::kFloat, 
        "Input (y) must be float32, but got=", y.scalar_type()
    );
    TORCH_CHECK(
        dy.scalar_type() == at::kFloat, 
        "Input (dy) must be float32, but got=", dy.scalar_type()
    );
}


// torch forward wrapper
torch::Tensor safe_softmax_forward_fp32(torch::Tensor x, float eps){
    c10::cuda::CUDAGuard device_guard(x.device());
    check_device_forward(x);
    check_shape_forward(x);
    check_dtype_forward(x);

    auto x_c = x.contiguous();
    
    const auto batch_size = x_c.size(0);
    const auto seq_len = x_c.size(1);

    auto y = torch::empty_like(x_c);

    const int warp_size = 32;
    const int warps_per_block = 4;
    const int threads = warp_size * warps_per_block;
    const int num_blocks = (batch_size + warps_per_block -1) / warps_per_block;

    auto stream = at::cuda::getCurrentCUDAStream();

    safe_softmax_fwd_fp32_kernel<<<num_blocks, threads, 0, stream.stream()>>>(
        x_c.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, seq_len, eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}


// torch backward wrapper
torch::Tensor safe_softmax_backward_fp32(torch::Tensor y, torch::Tensor dy){
    c10::cuda::CUDAGuard device_guard(y.device());
    check_device_backward(y, dy);
    check_shape_backward(y, dy);
    check_dtype_backward(y, dy);

    auto y_c = y.contiguous();
    auto dy_c = dy.contiguous();

    const auto batch_size = y_c.size(0);
    const auto seq_len = y_c.size(1);
    
    auto dx = torch::empty_like(y_c);

    const int warp_size = 32;
    const int warps_per_block = 4;
    const int threads = warp_size * warps_per_block;
    const int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    auto stream = at::cuda::getCurrentCUDAStream();

    safe_softmax_bwd_fp32_kernel<<<num_blocks, threads, 0, stream.stream()>>>(
        y_c.data_ptr<float>(),
        dy_c.data_ptr<float>(),
        dx.data_ptr<float>(),
        batch_size, seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}


// bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &safe_softmax_forward_fp32,
          py::arg("x"),
          py::arg("eps") = 1e-12f,
          "Safe softmax forward (fp32, CUDA). Expects x[B, D]; returns y[B, D].");

    m.def("backward",
          &safe_softmax_backward_fp32,
          py::arg("y"),
          py::arg("dy"),
          "Safe softmax backward (fp32, CUDA). Given y[B, D], dy[B, D] -> dx[B, D].");
}