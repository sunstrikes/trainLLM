#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"
namespace train_llm {

__global__ void relu_vec4_forward_kernel(float* x, float* y, int N) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx = tid << 2;
    if (idx < N) {
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = fmaxf(tmp_x.x, 0);
        tmp_y.y = fmaxf(tmp_x.y, 0);
        tmp_y.z = fmaxf(tmp_x.z, 0);
        tmp_y.w = fmaxf(tmp_x.x, 0);
        FLOAT4(y[idx]) = tmp_y;
    }
}
}  // namespace train_llm
