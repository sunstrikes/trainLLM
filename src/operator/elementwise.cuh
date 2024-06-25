#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"
namespace train_llm {

__global__ void elementwise_add_forward_kernel(float* a, float* b, float* c, int N) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void elementwise_add_forward_vec4_kernel(float* a, float* b, float* c, int N) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid * 4 < N) {
        float ta = FLOAT4(a[tid]);
        float tb = FLOAT4(b[tid]);
        float tc = FLOAT4(c[tid]);
        tc.x = ta.x + tb.x;
        tc.y = ta.y + tb.y;
        tc.z = ta.z + tb.z;
        tc.w = ta.w + tb.w;
        FLOAT4(c) = tc;
    }
}
}  // namespace train_llm
