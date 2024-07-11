#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#include "time_recorder.h"
#include "util.h"

namespace train_llm {

/********************* prefix_sum *********************/
void cpu_prefix_sum(const int32_t* input, size_t n, int32_t* output) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}

template <int WARP_SIZE = 32>
__device__ void scan_warp(int32_t* shm_data, int32_t lane) {
    if (lane == 0) {  // naive implemention
        int32_t acc = 0;
        for (int32_t i = 0; i < WARP_SIZE; ++i) {
            acc += shm_data[i];
            shm_data[i] = acc;
        }
    }
}

__device__ void scan_warp2(int32_t* shm_data) {
    volatile int32_t* vshm_data = shm_data;
    vshm_data[0] += vshm_data[-1];
    vshm_data[0] += vshm_data[-2];
    vshm_data[0] += vshm_data[-4];
    vshm_data[0] += vshm_data[-8];
    vshm_data[0] += vshm_data[-16];
}

template <int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__global__ void block_prefix_sum_kernel1(const int32_t* input, size_t n, int32_t* output) {
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    auto tx = threadIdx.x;
    __shared__ int shm[BLOCK_SIZE];
    __shared__ int shm2[BLOCK_SIZE / WARP_SIZE];
    if (tid < n) {
        shm[threadIdx.x] = input[tid];
    }
    __syncthreads();
    const int laneid = threadIdx.x % WARP_SIZE;
    scan_warp<WARP_SIZE>(shm + tx, laneid);
    __syncthreads();
    auto warp_id = tx / WARP_SIZE;
    if (laneid == 0) {
        int warp_sum = 0;
        for (auto i = 0; i < warp_id; ++i) {
            warp_sum += shm[(i + 1) * WARP_SIZE - 1];
        }
        printf("warp_id: %d, warp_sum: %d\n", warp_id, warp_sum);
        shm2[warp_id] = warp_sum;
    }
    __syncthreads();
    output[tid] = shm[tx] + shm2[warp_id];
}

template <int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__global__ void block_prefix_sum_kernel2(const int32_t* input, size_t n, int32_t* output) {
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    auto tx = threadIdx.x;
    __shared__ int shm[BLOCK_SIZE];
    __shared__ int shm2[BLOCK_SIZE / WARP_SIZE];
    if (tid < n) {
        shm[threadIdx.x] = input[tid];
    }
    __syncthreads();
    const int laneid = threadIdx.x % WARP_SIZE;
    scan_warp<WARP_SIZE>(shm + tx, laneid);
    __syncthreads();
    auto warp_id = tx / WARP_SIZE;
    if (laneid == 0) {
        int warp_sum = 0;
        for (auto i = 0; i < warp_id; ++i) {
            warp_sum += shm[(i + 1) * WARP_SIZE - 1];
        }
        printf("warp_id: %d, warp_sum: %d\n", warp_id, warp_sum);
        shm2[warp_id] = warp_sum;
    }
    __syncthreads();
    output[tid] = shm[tx] + shm2[warp_id];
}

void prefix_sum(const int32_t* input, size_t n, int32_t* output) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    block_prefix_sum_kernel1<<<grid_size, block_size>>>(input, n, output);
}
}  // namespace train_llm

using namespace train_llm;
void test_prefix_sum() {
    int* d_values;
    int* d_output;
    const int NUM_VALS = 64;
    size_t size = NUM_VALS * sizeof(int);
    std::vector<int> values(NUM_VALS);
    std::vector<int> h_output(NUM_VALS, 0);
    random_int_tensor(values.data(), NUM_VALS, NUM_VALS);
    cpu_prefix_sum(values.data(), NUM_VALS, h_output.data());
    cudaMalloc((void**)&d_values, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_values, values.data(), size, cudaMemcpyHostToDevice);
    prefix_sum(d_values, NUM_VALS, d_output);
    debug_display_mat("gpu_prefix_sum: ", d_output, 1, NUM_VALS);
    debug_display_mat<int, false>("cpu_prefix_sum: ", h_output.data(), 1, NUM_VALS);
    cudaFree(d_values);
    cudaFree(d_output);
}
int main() {
    test_prefix_sum();
}