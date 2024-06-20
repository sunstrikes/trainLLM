#include "attention.h"
#include "common.h"
#include <cub/cub.cuh>
namespace train_llm {

/* B: batch_size, T: token length, C: channel
   NH: multihead size
   output shape: (B, T, C)
   preatt, att shape: (B, NH, T, T), 把这俩中间结果存下来是为了方便backward
   input shape: (B, T, 3C) C0: Q, C1: K, C2: V
*/
void attention_forward_cpu(float* out, float* preatt, float* att, const float* inp, int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;                // head size
    float scale = 1.0 / sqrtf(hs);  // 这里算scale缓存, 减少除法导致的性能变差

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                // pass1 遍历所有的token, 当前Q对Q前面的所有token的点积 dot(Q, K)/hs
                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;  // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // 把Q之后的token内存清空, 避免有随机值影响结果对比
                for (int t2 = t + 1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: 计算softmax exp(t2) / sum(exp(t))
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: softmax结果 * V
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;  // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// inp/dinp are (B, T, 3C) Q,K,V
// att/datt/dpreatt are (B, NH, T, T)
// dout is (B, T, C)
void attention_backward_cpu(
        float* dinp,
        float* dpreatt,
        float* datt,
        float* dout,
        float* inp,
        float* att,
        int B,
        int T,
        int C,
        int NH) {
    int C3 = C * 3;
    int hs = C / NH;  // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;
                float* datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 < T; t2++) {  // ADJUSTED! this was t2 <= t (see note on function)
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;  // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    for (int i = 0; i < hs; i++) {
                        // forward: out_bth[i] += att_bth[t2] * value_t2[i];
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // softmax求导推导见readme.md
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;    // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C;  // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // forward: preatt_bth[t2] += query_t[i] * key_t2[i]
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
                    }
                }
            }
        }
    }
}

//求Q/K内积 kernel
template <typename T1>
__global__ void attention_query_kernel(T1* preatt, const T1* inp, int B, int T, int C, int NH) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto threads_num = B * NH * T * T;  //这里的for循环嵌套关系和cpu不一致, 要注意
    if (tid < threads_num) {
        auto t2 = tid % T;
        auto t = (tid / T) % T;  // t是在for循环倒数第二层, 可以使一部分超过t的线程提前退出
        if (t2 > t) {
            preatt[idx] = -INFINITY;
            return;
        }
        auto h = (tid / (T * T)) % NH;
        auto b = (tid / (T * T * NH));
        int C3 = C * 3;
        int hs = C / NH;
        auto query_t = inp + b * T * C3 + t * C3 + h * hs;
        auto key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
        T1 val = 0.0;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val = val / sqrtf(hs);
        preatt[idx] = val;
    }
}

template <typename T1>
__global__ void attention_softmax_kernel(T1* att, const T1* preatt, int B, int T, int NH) {
    // t2按y连续存储
    const size_t i = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    extern __shared__ T1 smem[];
    typedef cub::WarpReduce<int> WarpReduce;
    auto max_val = smem;
    auto expsum = smem + blockDim.y;
    if (idx < B * NH * T) {
        if (i == 0) {
            maxval[threadIdx.y] = -10000.0f;
            expsum[threadIdx.y] = 0;
        }
        auto tid = idx * T + i;
        // TODO: 可以优化
        __syncthreads();
        atomicMax(&maxval[], preatt[tid]);
        __syncthreads();
        // softmax
        auto t = idx % T;
        if (i <= t) {  // i即为t2
            T1 expv = expf(preatt[tid] - maxval[threadIdx.y]);
            att[tid] = expv;
            atomicAdd(expsum[threadIdx.y], expf(preatt[tid] - maxval[i]));
        }
        __syncthreads();
        if (i <= t) {
            att[tid] /= expsum[threadIdx.y];
        } else {
            att[tid] = 0.0;
        }
    }
}

template <typename T1>
__global__ void attention_value_kernel(T1* out, const T1* att, const T1* inp, int B, int T, int C, int NH) {
    
}

template <typename T1>
__global__ void attention_forward_gpu(
        T1* out,
        T1* preatt,
        T1* att,
        const float* inp,
        int B,
        int T,
        int C,
        int NH,
        cudaStream_t stream) {
    auto threads_num = B * NH * T * T;
    auto block_num = (threads_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    attention_query_kernel<<<block_num, BLOCK_SIZE, 0, stream>>>(preatt, inp, B, T, C, NH);
    threads_num = B * NH * T;
    assert(T <= MAX_BLOCK_SIZE);
    auto block_t_dim = kernel_block_size(T);
    dim3 block(T, block_t_dim);
    block_num = (threads_num + block_t_dim - 1) / block_t_dim;
    attention_softmax_kernel<<<block_num, block, sizeof(T1) * block_t_dim * 2, stream>>>(att, preatt, B, T, NH);

    block_num = (threads_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    attention_value_kernel<<<block_num, BLOCK_SIZE, 0, stream>>>(out, att, inp, B, T, C, NH);
}

template <typename T>
void attention_backward_gpu() {}

}  // namespace train_llm