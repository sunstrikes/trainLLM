#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "time_recorder.h"
#include "util.h"

namespace train_llm {

// naive
__global__ void transpose_kernel1(float* input, float* output, int M, int N) {
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (col_idx < N && row_idx < M) {
    int idx = row_idx * N + col_idx;
    int trans_idx = col_idx * M + row_idx;
    output[trans_idx] = input[idx];
  }
}

// float4
__global__ void transpose_kernel_float4(float* input, float* output, int M,
                                        int N) {
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  int float4_N = N >> 2;
  int float4_M = M >> 2;
  if (col_idx < float4_N && row_idx < float4_M) {
    auto float4_off = (row_idx << 2) * N + col_idx << 2;
    // 4 * 1
    const float4* float4_input =
        reinterpret_cast<const float4*>(input + float4_off);

    // 4 * 4
    float4 src_row0 = float4_input[0];
    float4 src_row1 = float4_input[float4_N];
    float4 src_row2 = float4_input[float4_N << 1];
    float4 src_row3 = float4_input[float4_N * 3];

    float4 dst_row0 =
        make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
    float4 dst_row1 =
        make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
    float4 dst_row2 =
        make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
    float4 dst_row3 =
        make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

    int out_off = (col_idx << 2) * M + row_idx << 2;
    float4* float4_out = reinterpret_cast<float4*>(output + out_off);
    float4_out[0] = dst_row0;
    float4_out[float4_M] = dst_row1;
    float4_out[float4_M << 1] = dst_row2;
    float4_out[float4_M * 3] = dst_row3;
  }
}

// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
//  对于计算强度比较低的 kernel, 增加线程处理的元素个数即计算强度,
//  一定程度上能增大 GPU 中计算与访存的掩盖, 并配合循环展开提高指令级并行;
// 此外, 由于线程块数量的减少, 减少 GPU 的线程块调度上可能也会带来性能的收益.
template <int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void __launch_bounds__(1024)
    transpose_kernel_shm(float* input, float* output, int M, int N) {
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  __shared__ float shm[BLOCK_SIZE][BLOCK_SIZE + 1];
  int xoff = bx * BLOCK_SIZE + tx;
  int yoff = by * BLOCK_SIZE + ty;
  if (xoff < N) {
#pragma unroll
    for (auto y = 0; y < BLOCK_SIZE; y += blockDim.y) {
      if (yoff + y < M) {
        shm[ty + y][tx] = input[(yoff + y) * N + xoff];
      }
    }
  }
  __syncthreads();
  xoff = by * BLOCK_SIZE + tx;
  yoff = bx * BLOCK_SIZE + ty;
  if (xoff < M) {
#pragma unroll
    for (auto y = 0; y < BLOCK_SIZE; y += blockDim.y) {
      if (yoff + y < N) {
        output[(yoff + y) * M + xoff] = shm[tx][ty + y];
      }
    }
  }
}

// share memory transpose
template <typename T>
__global__ void __launch_bounds__(1024)
    transpose_kernel(const T* src, T* dst, int dstM, int dstN) {
  __shared__ T share_arrary[32][33];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  for (int block_offest_y = blockIdx.y * blockDim.y; block_offest_y < dstM;
       block_offest_y += blockDim.y * gridDim.y) {
    for (int block_offest_x = blockIdx.x * blockDim.x; block_offest_x < dstN;
         block_offest_x += blockDim.x * gridDim.x) {
      // src coordinate
      int src_col = block_offest_y + tx;
      int src_row = block_offest_x + ty;

      if (src_col < dstM && src_row < dstN) {
        share_arrary[ty][tx] = src[src_row * dstM + src_col];  // 合并访存
      }
      __syncthreads();
      // dst coordinate
      // Block thread的坐标映射是根据 dst来着
      int dst_row = block_offest_y + ty;
      int dst_col = block_offest_x + tx;
      if (dst_row < dstM && dst_col < dstN) {
        dst[dst_row * dstN + dst_col] = share_arrary[tx][ty];
      }
    }
  }
}

void transpose(float* input, float* output, int M, int N) {
  {
    TimeRecorder("transpose1");
    dim3 block(4, 64);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_kernel1<<<grid, block>>>(input, output, M, N);
  }
  {
    TimeRecorder("transpose2");
    dim3 block(16, 16);
    dim3 grid((N >> 2 + block.x - 1) / block.x,
              (M >> 2 + block.y - 1) / block.y);
    transpose_kernel_float4<<<grid, block>>>(input, output, M, N);
  }
  {
    TimeRecorder("transpose_shm");
    const int NUM_PER_THREAD = 4;
    dim3 block(32, 32 / NUM_PER_THREAD);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_kernel_shm<32, NUM_PER_THREAD>
        <<<grid, block>>>(input, output, M, N);
  }
  {
    TimeRecorder("transpose_shm2");
    const dim3 block(32, 32);
    const dim3 grid((N + 31) / 32, (M + 31) / 32);
    transpose_kernel<<<grid, block>>>(input, output, N, M);
  }
}
}  // namespace train_llm

using namespace train_llm;
// test
int main() {
  int M = 8196;
  int N = 8196;
  size_t len = M * N;
  std::vector<float> data(len, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      data[i * N + j] = dis(gen);
    }
  }
  float* d_mat = nullptr;
  float* d_output = nullptr;
  cudaMalloc((void**)&d_mat, len * sizeof(float));
  cudaMalloc((void**)&d_output, len * sizeof(float));
  cudaMemcpy(d_mat, data.data(), len * sizeof(float), cudaMemcpyHostToDevice);
  train_llm::transpose(d_mat, d_output, M, N);
  // debug_display_mat("origin mat", d_mat, M, N);
  // debug_display_mat("transpose mat", d_output, N, M);
  return 0;
}