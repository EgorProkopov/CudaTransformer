// Embedding Layer kernels
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>


#include <cuda_fp16.h>
#include <cuda_bf16.h>


// device kernels and funcitons
__global__ void embedding_fwd_kernel(
    const float* __restrict__ weights,                               // [vocab_size, embedding_dim]
    const uint32_t* __restrict__ indices,                             // [num_indices]
    float* __restrict__ out,                                         // [num_indices, embedding_dim]
    const uint32_t vocab_size, const uint32_t embedding_dim, 
    const uint32_t num_indices
){
    // Тут можно не беспокоиться о регистровом давлении, поскольку это не GEMM
    // поэтому индексы, длины и прочее можно кастить к uint64_t ради безопасности  
    // и почти без ущерба для производительности

    // В свою очередь, чтобы не сохранить общий вид интерфейса с другими ядрами,
    // uint32_t остается в аргументах функции, поэтому дальше они кастятся к uint64_t
    const uint64_t total_elements = static_cast<uint64_t>(num_indices) * embedding_dim;
    const uint64_t linear_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t idx = linear_index; idx < total_elements; idx += stride){
        const uint32_t row = idx / embedding_dim;
        const uint32_t col = idx % embedding_dim;

        const uint32_t token_id = indices[row];
        const uint64_t weight_index = static_cast<uint64_t>(token_id) * embedding_dim + col;
        out[idx] = weights[weight_index];
    }
}


__global__ void embedding_bwd_kernel(
    const uint32_t* __restrict__ indices,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    const uint32_t vocab_size, const uint32_t embedding_dim, const uint32_t num_indices
){
    const uint64_t total_elements = static_cast<uint64_t>(num_indices) * embedding_dim;
    const uint64_t linear_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t idx = linear_index; idx < total_elements; idx += stride){
        const uint32_t row = idx / embedding_dim;
        const uint32_t col = idx % embedding_dim;

        const uint32_t token_id = indices[row];
        const uint64_t weight_index = static_cast<uint64_t>(token_id) * embedding_dim + col;
        atomicAdd(&grad_weight[weight_index], grad_out[idx]);
    }
}


// ===========================================================================
// =========================== host-functions ================================
// ===========================================================================
// forward util functions
static inline void check_device_forward(
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    torch::Tensor& out
){
    TORCH_CHECK(weights.defined(), "Weights tensor must be defined");
    TORCH_CHECK(indices.defined(), "Indices tensor must be defined");
    TORCH_CHECK(out.defined(), "Output tensor (out) must be defined");

    TORCH_CHECK(weights.is_cuda(), "Weights must be CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "Indices must be CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "Output tensor (out) must be CUDA tensor");

    const int64_t device_index = weights.get_device();
    TORCH_CHECK(
        indices.get_device() == device_index && out.get_device() == device_index,
        "Weights, indices and output tensors must be located on the same CUDA device"
    );
}

static inline void check_shape_forward(
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    torch::Tensor& out
){
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D [vocab_size, embedding_dim], got dim=", weights.dim());
    TORCH_CHECK(indices.dim() == 1, "Indices must be 1D [num_indices], got dim=", indices.dim());
    TORCH_CHECK(out.dim() == 2, "Output tensor must be 2D [num_indices, embedding_dim], got dim=", out.dim());
    TORCH_CHECK(out.is_contiguous(), "Output tensor (out) must be contiguous");

    const int64_t vocab_size = weights.size(0);
    const int64_t embedding_dim = weights.size(1);
    const int64_t num_indices = indices.size(0);

    TORCH_CHECK(out.size(0) == num_indices, "Output rows must equal number of indices. Expected ", num_indices, ", got ", out.size(0));
    TORCH_CHECK(out.size(1) == embedding_dim, "Output embedding dim must equal weights embedding dim=", embedding_dim, ", got ", out.size(1));

    TORCH_CHECK(
        vocab_size <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Vocab size exceeds kernel limit (", std::numeric_limits<uint32_t>::max(), ")"
    );
    TORCH_CHECK(
        embedding_dim <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Embedding dim exceeds kernel limit (", std::numeric_limits<uint32_t>::max(), ")"
    );
    TORCH_CHECK(
        num_indices <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Number of indices exceeds kernel limit (", std::numeric_limits<uint32_t>::max(), ")"
    );
}

static inline void check_dtype_forward(
    const torch::Tensor& weights,
    const torch::Tensor& indices,
    torch::Tensor& out
){
    TORCH_CHECK(weights.scalar_type() == at::kFloat, "Weights must be float32, got ", weights.scalar_type());
    TORCH_CHECK(out.scalar_type() == at::kFloat, "Output tensor must be float32, got ", out.scalar_type());
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
        "Indices tensor must be int32 or int64, got ", indices.scalar_type()
    );
}

static inline void check_indices_bounds(
    const torch::Tensor& indices,
    const int64_t vocab_size
){
    if (indices.numel() == 0){
        return;
    }

    const auto min_idx = indices.min().item<int64_t>();
    const auto max_idx = indices.max().item<int64_t>();

    TORCH_CHECK(min_idx >= 0, "Embedding indices must be non-negative, got min=", min_idx);
    TORCH_CHECK(
        max_idx < vocab_size,
        "Embedding indices must be less than vocab_size=", vocab_size, ", got max=", max_idx
    );
    TORCH_CHECK(
        max_idx <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "Embedding kernel supports up to INT32_MAX indices, got max index=", max_idx
    );
}

// backward util functions
static inline void check_device_backward(
    const torch::Tensor& indices,
    const torch::Tensor& grad_out,
    torch::Tensor& grad_weight
){
    TORCH_CHECK(indices.defined(), "Indices tensor must be defined");
    TORCH_CHECK(grad_out.defined(), "grad_out tensor must be defined");
    TORCH_CHECK(grad_weight.defined(), "grad_weight tensor must be defined");

    TORCH_CHECK(indices.is_cuda(), "Indices must be CUDA tensor");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA tensor");
    TORCH_CHECK(grad_weight.is_cuda(), "grad_weight must be CUDA tensor");

    const int64_t device_index = grad_out.get_device();
    TORCH_CHECK(
        indices.get_device() == device_index && grad_weight.get_device() == device_index,
        "Indices, grad_out and grad_weight tensors must be located on the same CUDA device"
    );
}

static inline void check_shape_backward(
    const torch::Tensor& indices,
    const torch::Tensor& grad_out,
    torch::Tensor& grad_weight
){
    TORCH_CHECK(
        indices.dim() == 1, "Indices must be 1D [num_indices], got dim=", indices.dim()
    );
    TORCH_CHECK(
        grad_out.dim() == 2, "grad_out must be 2D [num_indices, embedding_dim], got dim=", 
        grad_out.dim()
    );
    TORCH_CHECK(
        grad_weight.dim() == 2, "grad_weight must be 2D [vocab_size, embedding_dim], got dim=", 
        grad_weight.dim()
    );
    TORCH_CHECK(
        grad_weight.is_contiguous(), "grad_weight tensor must be contiguous"
    );

    const int64_t num_indices = indices.size(0);
    const int64_t embedding_dim = grad_out.size(1);

    TORCH_CHECK(
        grad_out.size(0) == num_indices, "grad_out rows must match number of indices, expected ", 
        num_indices, ", got ", grad_out.size(0)
    );
    TORCH_CHECK(
        grad_weight.size(1) == embedding_dim, "grad_weight embedding dim must match grad_out dim=", 
        embedding_dim, ", got ", grad_weight.size(1)
    );

    TORCH_CHECK(
        grad_weight.size(0) <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Vocab size exceeds kernel limit (", std::numeric_limits<uint32_t>::max(), ")"
    );
    TORCH_CHECK(
        embedding_dim <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Embedding dim exceeds kernel limit (", std::numeric_limits<uint32_t>::max(), ")"
    );
    TORCH_CHECK(
        num_indices <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "Number of indices exceeds kernel limit (", std::numeric_limits<uint32_t>::max(), ")"
    );
}

static inline void check_dtype_backward(
    const torch::Tensor& indices,
    const torch::Tensor& grad_out,
    torch::Tensor& grad_weight
){   
    TORCH_CHECK(
        grad_out.scalar_type() == at::kFloat, 
        "grad_out must be float32, got ", 
        grad_out.scalar_type()
    );
    TORCH_CHECK(
        grad_weight.scalar_type() == at::kFloat, 
        "grad_weight must be float32, got ", 
        grad_weight.scalar_type()
    );
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
        "Indices tensor must be int32 or int64, got ", indices.scalar_type()
    );
}

// ===========================================================================
// =========================== wrapper-functions =============================
// ===========================================================================
// forward wrapper 
torch::Tensor embedding_forward_fp32_kernel(
    torch::Tensor weights,
    torch::Tensor indices,

    torch::Tensor out
) {
    TORCH_CHECK(out.defined(), "Output tensor (out) must be provided");
    check_device_forward(weights, indices, out);
    check_shape_forward(weights, indices, out);
    check_dtype_forward(weights, indices, out);

    c10::cuda::CUDAGuard device_guard(weights.device());
    check_indices_bounds(indices, weights.size(0));

    auto weights_c = weights.contiguous();
    torch::Tensor indices_int32;
    if (indices.scalar_type() == at::kInt){
        indices_int32 = indices.contiguous();
    } else {
        indices_int32 = indices.to(at::kInt).contiguous();
    }
    auto out_tensor = out;

    const uint32_t vocab_size = static_cast<uint32_t>(weights_c.size(0));
    const uint32_t embedding_dim = static_cast<uint32_t>(weights_c.size(1));
    const uint32_t num_indices = static_cast<uint32_t>(indices_int32.size(0));

    const uint64_t total_elements = static_cast<uint64_t>(num_indices) * embedding_dim;
    if (total_elements == 0){
        out_tensor.zero_();
        return out_tensor;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    embedding_fwd_kernel<<<blocks, threads, 0, stream.stream()>>>(
        weights_c.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(indices_int32.data_ptr<int32_t>()),
        out_tensor.data_ptr<float>(),
        vocab_size, embedding_dim, num_indices
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_tensor;
}

// backward wrapper
torch::Tensor embedding_backward_fp32_kernel(
    torch::Tensor indices,
    torch::Tensor grad_out,
    torch::Tensor grad_weight
) {
    check_device_backward(indices, grad_out, grad_weight);
    check_shape_backward(indices, grad_out, grad_weight);
    check_dtype_backward(indices, grad_out, grad_weight);

    c10::cuda::CUDAGuard device_guard(grad_out.device());
    check_indices_bounds(indices, grad_weight.size(0));

    torch::Tensor indices_int32;
    if (indices.scalar_type() == at::kInt){
        indices_int32 = indices.contiguous();
    } else {
        indices_int32 = indices.to(at::kInt).contiguous();
    }

    auto grad_out_c = grad_out.contiguous();
    auto grad_weight_tensor = grad_weight;
    grad_weight_tensor.zero_();

    const uint32_t vocab_size = static_cast<uint32_t>(grad_weight_tensor.size(0));
    const uint32_t embedding_dim = static_cast<uint32_t>(grad_weight_tensor.size(1));
    const uint32_t num_indices = static_cast<uint32_t>(indices_int32.size(0));

    const uint64_t total_elements = static_cast<uint64_t>(num_indices) * embedding_dim;
    if (total_elements == 0){
        return grad_weight_tensor;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    embedding_bwd_kernel<<<blocks, threads, 0, stream.stream()>>>(
        reinterpret_cast<const uint32_t*>(indices_int32.data_ptr<int32_t>()),
        grad_out_c.data_ptr<float>(),
        grad_weight_tensor.data_ptr<float>(),
        vocab_size, embedding_dim, num_indices
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_weight_tensor;
}

// ===========================================================================
// =========================== pybind ========================================
// ===========================================================================
// bingings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "embedding_forward_fp32", &embedding_forward_fp32_kernel, "Embedding forward fp32 kernel"
    );
    m.def(
        "embedding_backward_fp32", &embedding_backward_fp32_kernel, "Embedding backward fp32 kernel"
    );
}
