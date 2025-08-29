// Embedding Layer kernels
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>


// device kernels and funcitons
template <typename scalar_t>
__global__ void embedding_fwd_kernel(
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ out,
    const int64_t vocab_size, const int64_t embedding_dim, 
    const int64_t num_indices
){
    
}


template <typename scalar_t>
__global__ void embedding_bwd_kernel(
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ dy,
    scalar_t* __restrict__ dweights,
    const int64_t vocab_size, const int64_t embedding_dim
){

}