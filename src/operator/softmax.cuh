#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"
namespace train_llm {

template<const int NUM_THREADS = 128>
__global__ void softmax_forward_kernel(float* x, float* y, float* total, int N) {
    const int tx = threadIdx.x;
    const int tid = blockIdx.x * blockDim.x + tx;
    auto exp_tmp = tid < N? expf(x[tid]): 0.0f;
    float exp_sum = blockReduceSum<float, NUM_THREADS, 32>(exp_tmp, 0.0f);
    if (tx == 0) {
        atomicAdd(total, exp_sum);
    }            
    __threadfence();
    if (tid < N) {
        y[tid] = exp_tmp / *total;
    }
}
}  // namespace train_llm
