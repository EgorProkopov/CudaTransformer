/// RMSNorm kernels
/// RMSNorm(x) = gamma * x / sqrt(||x||_2^2 + epsilon)
/// There is reduction op


// includes 
#include <torch/extension.h>
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
                warp_sums[0] = block_sum; // 
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
    }
}

// backward-pass kernel
template <typename scalar_t>
__global__ void rmsnorm_bwd_kernel(
    const scalar_t* __restrict__ x,             // data
    const scalar_t* __restrict__ dy,            // dL/dy grad
    scalar_t* __restrict__ dx,                  // output dL/dx grad
    const float* __restrict__ gamma,            // gamma weights 
    float* __restrict__ dgamma,                 // output dL/dgamma grad
    int64_t seq_len, int64_t embedding_dim,     // input data sizes
    float eps                                   // epsilon
){
    // smem allocation
    extern __shared__ unsigned char smem[];

    scalar_t* row_x = reinterpret_cast<scalar_t*>(smem);
    float* warp_sums = reinterpret_cast<float*>(row_x + embedding_dim); 
    float* warp_sums_dot = warp_sums + (blockDim.x + warpSize - 1)/ warpSize;

    const int64_t warp_id = threadIdx.x / warpSize;
    const int64_t lane_id = threadIdx.x % warpSize;
    const int64_t num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // grid-stride, for the case when seq_len > gridDim.x
    for (int64_t row_idx = blockIdx.x; row_idx < seq_len; row_idx+=gridDim.x){
        float local_squares_sum = 0.0f;
        float local_dot = 0.0f;
        
        for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x){
            scalar_t value = x[row_idx * embedding_dim + i];
            row_x[i] = value;
            
            // x**2 local sums
            float value_fp32 = to_float(value);
            local_squares_sum += value_fp32*value_fp32;

            // dy * gamma * x local sums
            float dy_fp32 = to_float(dy[row_idx * embedding_dim + i]);
            float gamma_fp32 = to_float(gamma[i]);
            local_dot += dy_fp32 * gamma_fp32 * value_fp32;
        }

        // in-warp reduction
        unsigned mask = __activemask();
        float warp_squares_sum = warp_sum_fp32(local_squares_sum, mask);
        float warp_dot = warp_sum_fp32(local_dot, mask);

        if (lane_id == 0){
            warp_sums[warp_id] = warp_squares_sum;
            warp_sums_dot[warp_id] = warp_dot;
        }
        __syncthreads();

        // final reduction in 1st warp
        float squares_sum = 0.0f;
        float dot_sum = 0.0f;

        if (threadIdx.x < num_warps){
            squares_sum = warp_sums[threadIdx.x];
            dot_sum = warp_sums_dot[threadIdx.x];
        }
        if (warp_id == 0){
            unsigned mask0 = __ballot_sync(0xffffffffu, lane_id < num_warps);
            squares_sum = warp_sum_fp32(squares_sum, mask0);
            dot_sum = warp_sum_fp32(dot_sum, mask0);
            if (lane_id == 0){
                warp_sums[0] = squares_sum;
                warp_sums_dot[0] = dot_sum;
            }
        }
        __syncthreads();

        // inv_rms calc
        float inv_rms = 0.0f; // r

        if (threadIdx.x == 0){
            float mean_sq = warp_sums[0] / static_cast<float>(embedding_dim);
            inv_rms = rsqrtf(mean_sq + eps);

            warp_sums[0] = inv_rms;  // threads broadcast
        }
        __syncthreads();

        inv_rms = warp_sums[0];

        // dx and dgamma calc
        // dx_i = r * gamma_i * dy_i - coeff * x_i _ sum(dy * gamma * x)
        const float sum_dy_gx = warp_sums_dot[0];
        // coeff = r^3 / D
        float coeff = inv_rms * inv_rms * inv_rms / static_cast<float>(embedding_dim);
        for (int64_t i = threadIdx.x; i < embedding_dim; i+=blockDim.x){
            float value_fp32 = to_float(row_x[i]);
            float gamma_fp32 = to_float(gamma[i]);
            float dy_fp32 = to_float(dy[row_idx * embedding_dim + i]);

            // dx_i = r * gamma_i * dy_i - coeff * x_i _ sum(dy * gamma * x)
            float dx_fp32 = inv_rms * gamma_fp32 * dy_fp32 - coeff * value_fp32 * sum_dy_gx;
            dx[row_idx * embedding_dim + i] = from_float<scalar_t>(dx_fp32);

            // dgamma_i += r * x_i * dy_i
            float dg_i = inv_rms * value_fp32 * dy_fp32;
            atomicAdd(dgamma + i, dg_i);
        }
    }
}


// host functions
// utils
static inline void check_shapes_forward(const torch::Tensor& x, const torch::Tensor& gamma){
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    
    TORCH_CHECK(x.dim() == 2, "x must be [seq_len, embedding_dim]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [embedding_dum]");

    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma.shape[0] must be equal to x.shape[1]");
}

static inline void check_shapes_backward(const torch::Tensor& x, const torch::Tensor& dy, const torch::Tensor& gamma){
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(dy.is_cuda(), "dy must be CUDA");

    TORCH_CHECK(x.dim() == 2, "x must be [seq_len, embedding_dim]");
    TORCH_CHECK(dy.dim() == 2, "dy must be [seq_len, embedding_dim]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [embedding_dum]");

    TORCH_CHECK(x.sizes() == dy.sizes(), "x and dy sizes mismatch");
    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma.shape[0] must be equal to x.shape[1]");
}

static inline int choose_threads(int embedding_size){
    if (embedding_size >= 512) return 512;
    if (embedding_size >= 256) return 256;
    if (embedding_size >= 128) return 128;
    return 64;
}

static inline size_t smem_bytes_fwd(int num_warps, int64_t embedding_dim, size_t sizeofT) {
    return embedding_dim * sizeofT + num_warps * sizeof(float);
}

// torch host wrapper
// forward wrapper
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor gamma, float eps){
    check_shapes_forward(x, gamma);
    
    if (gamma.scalar_type() != x.scalar_type()) {
        gamma = gamma.to(x.scalar_type());
    }

    auto x_c = x.contiguous();
    auto g_c = gamma.contiguous();

    const auto seq_len = x_c.size(0);
    const auto embedding_dim = x_c.size(1);

    auto y = torch::empty_like(x_c);

    const int threads = choose_threads(embedding_dim);
    const int blocks = std::min<int64_t>(seq_len, sizeof(uint16_t));
    const int num_warps = (threads + 31) / 32;
    const size_t smem = embedding_dim * x_c.element_size() + num_warps * sizeof(float);

    auto stream = torch::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, x_c.scalar_type(), "rmsnorm_fwd_launch", [&]{
        rmsnorm_fwd_kernel<scalar_t>
            <<<blocks, threads, smem, stream>>>(
                x_c.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                seq_len, embedding_dim, static_cast<float>(eps)
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// backward wrapper
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward(
    torch::Tensor x,
    torch::Tensor dy,
    torch::Tensor gamma,
    float eps
){
    check_shapes_backward(x, dy, gamma);

    auto x_c  = x.contiguous();
    auto dy_c = dy.contiguous();

    const auto seq_len = x_c.size(0);
    const auto embedding_dim = x_c.size(1);

    auto gamma_f32 = gamma.to(torch::kFloat).contiguous();

    auto dx = torch::empty_like(x_c);
    auto dgamma = torch::empty_like({embedding_dim}, x.options().dtype(torch::kFloat));

    const int threads = choose_threads(embedding_dim);
    const int blocks = std::min<int64_t>(seq_len, sizeof(uint16_t));
    const int num_warps = (threads + 31) / 32;
    const size_t smem = embedding_dim * x_c.element_size() + 2* num_warps * sizeof(float);

    auto stream = torch::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, x_c.scalar_type(), "rmsnorm_bwd_launch", [&]{
        rmsnorm_bwd_kernel<scalar_t>
            <<<blocks, threads, smem, stream>>>(
                x_c.data_ptr<scalar_t>(),
                dy_c.data_ptr<scalar_t>(),
                gamma_f32.data_ptr<float>(),
                dx.data_ptr<scalar_t>(),
                dgamma.data_ptr<float>(),
                seq_len, embedding_dim, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dgamma};
}


// python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (Custom CUDA kernel)");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward (Custom CUDA kernel)");
}
