#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#include "time_recorder.h"
#include "util.h"
#include "reduce.cuh"

namespace train_llm {

/********************************transpose**********************************/
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
__global__ void transpose_kernel_float4(float* input, float* output, int M, int N) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int float4_N = N >> 2;
    int float4_M = M >> 2;
    if (col_idx < float4_N && row_idx < float4_M) {
        auto float4_off = (row_idx << 2) * N + col_idx << 2;
        // 4 * 1
        const float4* float4_input = reinterpret_cast<const float4*>(input + float4_off);

        // 4 * 4
        float4 src_row0 = float4_input[0];
        float4 src_row1 = float4_input[float4_N];
        float4 src_row2 = float4_input[float4_N << 1];
        float4 src_row3 = float4_input[float4_N * 3];

        float4 dst_row0 = make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
        float4 dst_row1 = make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
        float4 dst_row2 = make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
        float4 dst_row3 = make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

        int out_off = (col_idx << 2) * M + row_idx << 2;
        float4* float4_out = reinterpret_cast<float4*>(output + out_off);
        float4_out[0] = dst_row0;
        float4_out[float4_M] = dst_row1;
        float4_out[float4_M << 1] = dst_row2;
        float4_out[float4_M * 3] = dst_row3;
    }
}

// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// 对于计算强度比较低的 kernel, 增加线程处理的元素个数即计算强度, 一定程度上能增大 GPU 中计算与访存的掩盖,
// 并配合循环展开提高指令级并行;
//此外, 由于线程块数量的减少, 减少 GPU 的线程块调度上可能也会带来性能的收益.
template <int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void __launch_bounds__(1024) transpose_kernel_shm(float* input, float* output, int M, int N) {
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
__global__ void __launch_bounds__(1024) transpose_kernel(const T* src, T* dst, int dstM, int dstN) {
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
        dim3 grid((N >> 2 + block.x - 1) / block.x, (M >> 2 + block.y - 1) / block.y);
        transpose_kernel_float4<<<grid, block>>>(input, output, M, N);
    }
    {
        TimeRecorder("transpose_shm");
        const int NUM_PER_THREAD = 4;
        dim3 block(32, 32 / NUM_PER_THREAD);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        transpose_kernel_shm<32, NUM_PER_THREAD><<<grid, block>>>(input, output, M, N);
    }
    {
        TimeRecorder("transpose_shm2");
        const dim3 block(32, 32);
        const dim3 grid((N + 31) / 32, (M + 31) / 32);
        transpose_kernel<<<grid, block>>>(input, output, N, M);
    }
}

/*---------------------------matmul--------------------------*/
void cpu_matmul(float* a, float* b, float* c, int a_rows, int a_cols, int b_cols) {
    // auto b_rows = a_cols;
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            double tmp = 0;
            for (int k = 0; k < a_cols; ++k) {
                tmp += a[i * a_cols + k] * b[k * b_cols + j];
            }
            c[i * b_cols + j] = tmp;
        }
    }
}

// A: M*K, B: K*N, C: M*N
__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int M, int K, int N) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;
    if (x < M && y < N) {
        for (auto i = 0; i < K; ++i) {
            sum += a[x * M + i] * b[i * N + y];
        }
        c[x * N + y] = sum;
    }
}

// 列主序
template <int BK = 16>
__global__ void matmul_kernel_shm(const float* a, const float* b, float* c, int M, int K, int N) {
    __shared__ float shm_a[BK][BK];
    __shared__ float shm_b[BK][BK];
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_a = bx * BK;
    const int col_b = by * BK;
    if (row_a + tx < M && col_b + ty < N) {
        float sum = 0.0;
        for (int i = 0; i < (K + BK - 1) / BK; ++i) {
            if (i * BK > K) {
                break;
            }
            const int col_a = i * BK;
            const int row_b = col_a;
            shm_a[tx][ty] = a[(row_a + tx) * K + col_a + ty];
            shm_b[ty][tx] = b[(row_b + tx) * N + col_b + ty];
            __syncthreads();
#pragma unroll
            for (int k = 0; k < BK; ++k) {
                sum += shm_a[tx][k] * shm_b[ty][k];
                __syncthreads();
            }
        }
        c[(row_a + tx) * M + col_b + ty] = sum;
    }
}

// 行主序, 这个是正确的顺序, cuda 2维block是按照先x后y的顺序访问
// 对比列主序的性能提升10%
template <int BK = 16>
__global__ void matmul_kernel_shm_row(const float* a, const float* b, float* c, int M, int K, int N) {
    __shared__ float shm_a[BK][BK];
    __shared__ float shm_b[BK][BK];
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_a = by * BK;
    const int col_b = bx * BK;
    if (row_a + ty < M && col_b + tx < N) {
        float sum = 0.0;
        for (int i = 0; i < (K + BK - 1) / BK; ++i) {
            if (i * BK > K) {
                break;
            }
            const int col_a = i * BK;
            const int row_b = col_a;
            shm_a[ty][tx] = a[(row_a + ty) * K + col_a + tx];
            shm_b[tx][ty] = b[(row_b + ty) * N + col_b + tx];
            __syncthreads();
#pragma unroll
            for (int k = 0; k < BK; ++k) {
                sum += shm_a[ty][k] * shm_b[tx][k];
                __syncthreads();
            }
        }
        c[(row_a + ty) * M + col_b + tx] = sum;
    }
}

#define FLOAT4(x) *(reinterpret_cast<float4*>(&(x)))
#define CONST_FLOAT4(x) *(reinterpret_cast<const float4*>(&(x)))
// float4 & shm
template <int BK = 8, int BM = 128, int BN = 128>
__global__ void matmul_kernel_float4_shm(const float* a, const float* b, float* c, int M, int K, int N) {
    // 1. 从gloabl->shm
    __shared__ float shm_a[BM][BK];
    __shared__ float shm_b[BK][BN];

    const int TM = 8;
    const int TN = 8;
    float reg_c[TM][TN] = {0.0};

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 每个block要从global取128*8的数据, 共256线程, 每个线程取一个float4
    const int row_a_m = tid >> 1;        // 0, 0, 1, 1, 2, 2, ...
    const int col_a_k = (tid & 1) << 2;  // 0, 4, 0, 4, 0, 4, ...
    const int row_b_k = tid >> 5;        // 128/4 = 32
    const int col_b_n = (tid % 32) << 2;

    int gmem_a_off = by * BM + row_a_m;
    int gmem_b_off = bx * BN + col_b_n;

    if (gmem_a_off < M && gmem_b_off < N) {
        for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
            if (bk * BK >= K) {
                break;
            }
            
            int gmem_a_k_off = bk * BK + col_a_k;
            //int load_a_gmem_addr = gem_a_off * K + gmem_a_k_off;
            // a需要转置
            FLOAT4(shm_a[row_a_m][col_a_k]) = CONST_FLOAT4(a[gmem_a_off * K + gmem_a_k_off]);
            //printf("shm_a[%d][%d] = %f\n", row_a_m, col_a_k, shm_a[row_a_m][col_a_k]);
            auto gmem_b_k_off = bk * BK + row_b_k;
            FLOAT4(shm_b[row_b_k][col_b_n]) = CONST_FLOAT4(b[gmem_b_off * K + gmem_b_k_off]);
            __syncthreads();
// 2. 从shm->reg, 计算matmul
#pragma unroll
            for (int k = 0; k < BK; ++k) {
#pragma unroll
                for (int m = 0; m < TM; ++m) {
#pragma unroll
                    for (int n = 0; n < TN; ++n) {
                        reg_c[m][n] = shm_a[ty * TM + m][k] * shm_b[k][tx * TN + n];
                        //printf("reg_c[%d][%d] = %f shm_a[%d, %d]: %f,  shm_b[%d, %d]: %f\n", m, n, reg_c[m][n], ty * TM + m, k, shm_a[ty * TM + m][k], k, tx * TN + n, shm_b[k][tx * TN + n]);
                    }
                }
            }
            __syncthreads();
        }
// 4. 写回global
#pragma unroll
        for (auto i = 0; i < TM; ++i) {
            int gmem_m_c_off = by * BM + ty * TM + i;
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int gmem_n_c_off = bx * BN + tx * TN + j;
                FLOAT4(c[gmem_m_c_off * N + gmem_n_c_off]) = FLOAT4(reg_c[i][j]);
            }
        }
    }
}

void matmul(const float* a, const float* b, float* c, int M, int K, int N) {
    {
        dim3 block(16, 16);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        matmul_naive_kernel<<<grid, block>>>(a, b, c, M, K, N);
    }
    {
        const int BK = 16;
        dim3 blockDim(BK, BK);
        dim3 gridDim((M + BK - 1) / BK, (N + BK - 1) / BK);
        matmul_kernel_shm<<<gridDim, blockDim>>>(a, b, c, M, K, N);
    }
    {
        const int BK = 16;
        dim3 blockDim(BK, BK);
        dim3 gridDim((N + BK - 1) / BK, (M + BK - 1) / BK);
        matmul_kernel_shm_row<<<gridDim, blockDim>>>(a, b, c, M, K, N);
    }
    {
        // BN / TN = 16
        const int BN = 128;
        const int BM = 128;
        dim3 blockDim(16, 16);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_kernel_float4_shm<<<gridDim, blockDim>>>(a, b, c, M, K, N);
    }
}

/*---------------------------sgemv(mat * vec)--------------------------*/
// mat: M * N, x: N * 1, out: M * 1
template<int warpSize=128>
void sgemv(float* __restrict__ mat, float* __restrict__ x, float* out, const int M, const int N) {
    const int bx = blockIdx.x;
    const tx = threadIdx.x;
    const ty = threadIdx.y;

    int laneid = tx % wrapSize;

    //grid_size: M / 4
    int m = bx * blockDim.y + ty;
    if (m < M) {
        float sum = 0.0;
        //这里只适合N < blockDim.x的场景
        const int warp_num = (N + warpSize - 1) / warpSize;
        #pragma unroll
        for (int i = 0; i < warp_num; ++i) {
            auto n = i * warp_size + laneid;
            sum += mat[m * N + n] * x[n];
        }
        sum = WarpReduceSum<float, warpSize>(sum);
        if (laneid == 0) {
            out[m] = sum;
        }
    }
}

/*---------------------------dot(vec * vec)--------------------------*/

}  // namespace train_llm

using namespace train_llm;
// test
int test_transpose() {
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

// test
int test_matmul() {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    //int M = 2;
    //int N = 2;
    //int K = 3;
    std::vector<float> h_mat1(M * K, 0);
    std::vector<float> h_mat2(N * K, 0);
    std::vector<float> h_out(N * M, 0);
    random_mat(h_mat1.data(), M, K);
    random_mat(h_mat2.data(), K, N);

    float* d_mat1 = nullptr;
    float* d_mat2 = nullptr;
    float* d_output = nullptr;
    cudaMalloc((void**)&d_mat1, M * K * sizeof(float));
    cudaMalloc((void**)&d_mat2, N * K * sizeof(float));
    cudaMalloc((void**)&d_output, M * N * sizeof(float));
    cudaMemcpy(d_mat1, h_mat1.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    matmul(d_mat1, d_mat2, d_output, M, K, N);
    // debug_display_mat("mat1", d_mat1, M, K);
    // debug_display_mat("mat2", d_mat2, K, N);
    cpu_matmul(h_mat1.data(), h_mat2.data(), h_out.data(), M, K, N);
    //debug_display_mat("output", d_output, M, N);
    //debug_display_mat<float, false>("cpu_out", h_out.data(), M, N);
    return 0;
}

int main() {
    // test_transpose();
    test_matmul();
}