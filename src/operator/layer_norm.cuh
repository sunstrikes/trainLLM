#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"
namespace train_llm {

//适用于 128 <= dim1< 1024的矩阵. N: batch_size, K: tensor_dim
__global__ void layer_norm_forward_kernel(float* x, float* y, float g, float bias, int N, int K) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const float epsilon = 1e-5;
    __shared__ float shm_mean = 0.0;
    __shared__ float shm_var = 0.0;

    auto idx = bx * K + tx;
    float sum = BlockReduceSum<float>(x[idx]);
    shm_mean = sum / K;
    __syncthreads();
    float var_sum = BlockReduceSum<float>(pow(x[idx] - shm_mean, 2));
    if (tx == 0) {
        shm_var = sqrtf(var_sum / K + epsilon);
    }
    __syncthreads();
    if (idx < N * K) {
        y[idx] = g / shm_var * (x[idx] - shm_mean) + bias;
    }
}
}  // namespace train_llm
