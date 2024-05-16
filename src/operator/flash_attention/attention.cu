#include "attention.h"
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

                // 遍历所有的token, 当前Q对Q前面的所有token的点积 dot(Q, K)/hs
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

}  // namespace train_llm