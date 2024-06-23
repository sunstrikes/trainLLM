#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cub/cub.cuh>
#include "time_recorder.h"
#include "util.h"

namespace train_llm {

/********************************transpose**********************************/

}  // namespace train_llm

using namespace train_llm;
//warp操作集合: https://zhuanlan.zhihu.com/p/572820783
template <typename T, int warpSize=32>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    //表示被 mask 指定的线程返回向后偏移为 delta 的线程中的变量 var 的值，其余线程返回0；
    //T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
    val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
}

template <typename T, int warpSize=32>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int laneid = threadIdx.x % warpSize;
  const int warpid = threadIdx.x / warpSize;
  val = WarpReduceSum(val);
  __syncthreads();
  if (laneid == 0) { //只有laneid是0的线程上是正确结果
    shared[warpid] = val;
  }
  __syncthreads();
  //用第一个warp再做一次reduce, 因为最大线程数1024 = 32 * 32, 不会存在1个warp存不下的问题
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : T(0);
  if (warpid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}
