// Embedding Layer kernels
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>


// device kernels and funcitons
__global__ void embedding_fwd_kernel(
    const float* __restrict__ weights,
    const int32_t* __restrict__ indices,
    float* __restrict__ out,
    const int32_t vocab_size, const int32_t embedding_dim, 
    const int32_t num_indices
){
    
}


__global__ void embedding_bwd_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ dy,
    float* __restrict__ dweights,
    const int32_t vocab_size, const int32_t embedding_dim
){

}