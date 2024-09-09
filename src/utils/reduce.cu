#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#include "time_recorder.h"
#include "util.h"
#include "reduce.cuh"
namespace train_llm {

//参考cub::DeviceReduce
//https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html?highlight=devicereduce#_CPPv4N3cub12DeviceReduceE
template <typename T, int numThreads = 256, int warpSize = 32>
__global__ void DeviceReduceKernel(T* in, T* out, int N) {
    int sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = BlockReduceSum<T>(sum);
    printf("blockIdx.x = %d, sum = %d\n", blockIdx.x, sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

template<typename T>
void device_reduce_sum(T* input, T* output, int num) {
    auto grid_size  = (num + 256 - 1) / 256;
    DeviceReduceKernel<T><<<grid_size, 256>>>(input, output, num);
    // 使用atomic
    DeviceReduceKernel<T><<<1, 1024>>>(output, output, grid_size);
}

void cpu_reduce_sum(int* input, int* output, int num) {
    int sum = 0;
    for (auto i = 0; i < num; ++i) {
        sum += input[i];
    }
    output[0] = sum;
}
}

using namespace train_llm;
void test_reduce_sum() {
    int* d_values;
    int* d_output;
    const int NUM_VALS = 64;
    size_t size = NUM_VALS * sizeof(int);
    std::vector<int> values(NUM_VALS);
    random_int_tensor(values.data(), NUM_VALS, 10);
    debug_display_mat<int, false>("cpu_ori: ", values.data(), 1, NUM_VALS);
    std::vector<int> h_output(NUM_VALS, 0);
    cudaMalloc((void**)&d_values, size);
    cudaMalloc((void**)&d_output, size);
    auto res = cudaMemcpy(d_values, values.data(), size, cudaMemcpyHostToDevice);
    std::cout << res << std::endl;
    debug_display_mat("gpu_ori_key: ", d_values, 1, NUM_VALS);
    //device_reduce_sum(d_values, d_output, NUM_VALS);
    debug_display_mat("gpu_reduce_sum: ", d_output, 1, 1);

    cpu_reduce_sum(values.data(), h_output.data(), NUM_VALS);
    debug_display_mat<int, false>("cpu_reduce_sum: ", h_output.data(), 1, 1);
    cudaFree(d_values);
    cudaFree(d_output);
}
int main() {
    test_reduce_sum();
}