#pragma once
namespace train_llm {
// cpu baseline
void attention_forward_cpu(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH);

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
        int NH);

// 标准 attention kernel实现
template <typename T1>
void attention_forward_gpu(
        T1* out,
        T1* preatt,
        T1* att,
        const float* inp,
        int B,
        int T,
        int C,
        int NH,
        cudaStream_t stream);

void attention_backward_gpu();

}  // namespace train_llm