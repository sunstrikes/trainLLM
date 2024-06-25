#pragma once
#include <cuda_runtime.h>
#include "util.h"
namespace train_llm {

template<const int NUM_THREADS = 128, int WARP_SIZE=32>
__global__ void dot_forward_kernel(float* a, float* b, float* y, int N) {
    const int tx = threadIdx.x;
    const int laneid = tx % WARP_SIZE;
    const int warpid = tx / WARP_SIZE;
    const int warp_num = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    assert(warp_num <= 32);
    __shared__ float shm[warp_num];
    if (tx > N) {
        return;
    }
    auto sum = WarpReduceSum<float, WARP_SIZE>(sum);
    if (laneid == 0) {
        shm[warp_id] = sum;
    }
    __syncthreads();
    sum = tx < warp_num? shm[tx]: 0.0f;
    if (warp_id == 0) {
        float block_sum = WarpReduceSum<float, WARP_SIZE>(sum);
    }
    if (tx == 0) {
        atomicAdd(y, block_sum);
    }
}
}  // namespace train_llm
