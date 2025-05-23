#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>

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

// https://zhuanlan.zhihu.com/p/564325738
// 1. inclusive_scan
template <int BLOCK_SIZE>
__global__ void inclusive_block(const int* input, int* output, size_t n) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int shm[BLOCK_SIZE];
  if (tid < n) {
    shm[threadIdx.x] = input[tid];
  } else {
    shm[threadIdx.x] = 0;
  }
  __syncthreads();
  // hillis-steele algo
  for (int i = 1; i < BLOCK_SIZE; i <<= 1) {
    if (threadIdx.x >= i) {
      shm[threadIdx.x] += shm[threadIdx.x - i];
    }
    __syncthreads();
  }
  if (tid < n) {
    output[tid] = shm[threadIdx.x];
  }
}

// 2. exclusive_scan
template <int BLOCK_SIZE>
__global__ void exclusive_block(const int* input, int* output, size_t n) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ int shm[BLOCK_SIZE];
  if (tid < n) {
    shm[threadIdx.x] = input[tid];
  } else {
    shm[threadIdx.x] = 0;
  }
  __syncthreads();
  // Blelloch upsweep
  for (int i = 1; i < BLOCK_SIZE; i *= 2) {
    auto idx = (threadIdx.x + 1) * (i << 1) - 1;
    if (idx < BLOCK_SIZE) {
      shm[idx] += shm[idx - i];
    }
    __syncthreads();
  }
  int blockSum = 0;
  if (threadIdx.x == 0) {
    blockSum = shm[BLOCK_SIZE - 1];
    shm[BLOCK_SIZE - 1] = 0;
  }
  __syncthreads();
  // downsweep

  for (int i = (BLOCK_SIZE >> 1); i >= 1; i >>= 1) {
    auto idx = (threadIdx.x + 1) * (i << 1) - 1;
    if (idx < BLOCK_SIZE) {
      auto tmp = shm[idx - i];
      shm[idx - i] = shm[idx];
      shm[idx] += tmp;
    }
    __syncthreads();
  }
  if (tid < n) {
    output[tid] = shm[threadIdx.x];
  }
}

const int BLOCK_SIZE = 256;

void inclusive_scan(int* input, int* output, size_t n) {
  auto grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  inclusive_block<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(input, output, n);
}

void exclusive_scan(int* input, int* output, size_t n) {
  auto grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  exclusive_block<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(input, output, n);
}

//==============================recrusive_scan==============

template <int BLOCK_SIZE>
__global__ void inclusive_block_inplace(int* input, int* block_sums, size_t n) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int shm[BLOCK_SIZE];
  if (tid < n) {
    shm[threadIdx.x] = input[tid];
  } else {
    shm[threadIdx.x] = 0;
  }
  __syncthreads();
  // hillis-steele algo
  for (int i = 1; i < BLOCK_SIZE; i <<= 1) {
    if (threadIdx.x >= i) {
      shm[threadIdx.x] += shm[threadIdx.x - i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = shm[BLOCK_SIZE - 1];
  }
  if (tid < n) {
    input[tid] = shm[threadIdx.x];
  }
}

template <int BLOCK_SIZE>
__global__ void add_block_sums(int* input, int* block_sums, size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = tid; i < n; i += stride) {
    int block_id = i / BLOCK_SIZE;
    if (block_id >= 1) {
      input[i] = input[i] + block_sums[block_id - 1];
    }
  }
}

bool cal_power(int a, int b) {
  if (a == 1) return (b == 1);
  if (b == 1) return (a == 1);
  int power = 1;
  int count = 0;
  while (power < a) {
    power *= b;
    count += 1;
  }
  return count + 1;
}

template <int BLOCK_SIZE>
void recursive_inclusive_scan(int* input, int* block_sums, int n) {
  if (block_sums == nullptr) {
    // 根据n递归算出实际需要分配的显存大小, 等比数列求和: (BLOCK_SIZE)^N - 1
    auto sum_size = BLOCK_SIZE ^ (cal_power(n, BLOCK_SIZE));
    cudaMalloc((void**)&block_sums, sum_size * sizeof(int));
  }
  auto grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  inclusive_block_inplace<BLOCK_SIZE>
      <<<grid_size, BLOCK_SIZE>>>(input, block_sums, n);
  debug_display_mat("input: ", input, 1, n);
  if (n < BLOCK_SIZE) {
    return;
  }
  auto block_sum_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  recursive_inclusive_scan<BLOCK_SIZE>(block_sums, block_sums + block_sum_size,
                                       block_sum_size);

  add_block_sums<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(input, block_sums, n);
}
//===================================================================

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
  debug_display_mat("input: ", d_values, 1, NUM_VALS);
  inclusive_scan(d_values, d_output, NUM_VALS);
  debug_display_mat("inclusive_block_scan: ", d_output, 1, NUM_VALS);
  exclusive_scan(d_values, d_output, NUM_VALS);
  debug_display_mat("exclusive_block_scan: ", d_output, 1, NUM_VALS);

  debug_display_mat<int, false>("cpu_prefix_sum: ", h_output.data(), 1,
                                NUM_VALS);
  recursive_inclusive_scan<8>(d_values, nullptr, NUM_VALS);
  debug_display_mat("recursive_inclusive_scan: ", d_values, 1, NUM_VALS);
  cudaFree(d_values);
  cudaFree(d_output);
}
int main() { test_prefix_sum(); }