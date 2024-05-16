---
typora-copy-images-to: ./assets
---

## attention 原理

<img src="/Users/sunminqi/code/github/trainLLM/src/operator/flash_attention/assets/image-20240516170029031.png" alt="image-20240516170029031" style="zoom: 33%;" />

以self-attention为例:

1. `A = malmul(Q, K)`. Q, K, V都是来自同一个token的输入. 假设我们要翻译"I am a student"这句话, 注意力机制的逻辑是当我们在翻译student这个词的时候, 他作为搜索query(Q), 去和句子里的"I/am/a/student"的K进行匹配, 看相关性有多高, 相关性越高的词说明在上下文翻译中对这个词的影响越大.  

2. `B = scale(A, len(Q))`, 通过缩放一个Q/K的维度, 避免点乘结果过大
3. `C=softmax(B)` , 通过归一化形成每个词的概率分布
4. `D=matmul(C, V)`, 最后将概率分布*V, 得到注意力的加权结果

那么Q, K, V是怎么得到的呢?

<img src="/Users/sunminqi/code/github/trainLLM/src/operator/flash_attention/assets/image-20240516172037022.png" alt="image-20240516172037022" style="zoom:33%;" />

X是词向量矩阵, 这里2代表batch_size, 4代表词向量长度. 与dense WQ,WK,WV分别做一次矩阵乘法得到Q,K,V

Multihead_attention其实就是指定了 X通过和多个WQ相乘得到多个Q/K/V, 最后再通过再合一个W0权重相乘得到最终合并后的Z

<img src="/Users/sunminqi/code/github/trainLLM/src/operator/flash_attention/assets/image-20240516173008961.png" alt="image-20240516173008961" style="zoom:33%;" />

## attention backward

### out bp

以`attention_backward_cpu`中的`dvalue_t2`为例, 根据链式求导法则<img src="/Users/sunminqi/code/github/trainLLM/src/operator/flash_attention/assets/image-20240516194532989.png" alt="image-20240516194532989" style="zoom:33%;" />

dout/dv 是 `out_bth += att_bth * value_t2 `对V求导, 得到的就是att_bth的前向值, 所以就有了

`dvalue_t2[i] += att_bth[t2] * dout_bth[i];`

### softmax bp:

<img src="/Users/sunminqi/code/github/trainLLM/src/operator/flash_attention/assets/image-20240516195713607.png" alt="image-20240516195713607" style="zoom:33%;" />

<img src="/Users/sunminqi/code/github/trainLLM/src/operator/flash_attention/assets/image-20240516195725744.png" alt="image-20240516195725744" style="zoom:33%;" />

这里为了简化逻辑 使用了indicator来区分i=j和i!=j的情况