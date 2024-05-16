#pragma once
namespace train_llm {
// cpu baseline
void attention_forward_cpu(floatX* out, floatX* qkvr, floatX* att, floatX* inp, int B, int T, int C, int NH);
}  // namespace train_llm