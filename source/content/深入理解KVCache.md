---
title: 深入理解KVCache
tags:
  - transformer
  - attention
  - llm
  - AI
  - 学习笔记
date: 2025-06-07
---
# 前言

上一篇 [[3b1b 直观解释注意力机制笔记]] 里重新学习了Transformer里的注意力机制，其实我主要是想弄懂现在很火的 KV Cache 到底是什么原理，以及由此引申的一些新的技术比如 PD分离，LMCache 是什么。这篇文章就来系统地梳理下。

# KVCache

我们重新回顾下 Attention 的那个公式：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$
我们说过表达式中的 Q、K、V 并不是三个固定的向量或者矩阵，他们是 $W_Q$ 、$W_K$ 和 $W_V$ 三个矩阵和输入token的 embedding $\overrightarrow{E}$ 构成的序列 $X$ 的相乘的结果。这里面包含的计算步骤可以拆分成：
1. $Q = W_Q*\overrightarrow{X}$
2. $K = W_K*\overrightarrow{X}$
3. ${Attention Pattern} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)$
4. $V = W_V*\overrightarrow{X}$
5. ${AttentionPattern * V}$

Transformer 的 decoder 过程就是对 $X_t = \{\overrightarrow{E_1}, \overrightarrow{E_2} ... \overrightarrow{E_t}\}$  不断执行上述过程以生成下一个 token ，再将新生成的 token的 embedding 加入到 $X_{t+1}$ 重新执行。
我们可以看到，对于步骤 1、2、4，完全可以省略掉 矩阵和 $\{\overrightarrow{E_1}, \overrightarrow{E_2} ... \overrightarrow{E_t}\}$ 相乘的过程，而只计算 $W * \overrightarrow{E_{t+1}}$ 即可。对于 3 也只需要把 $Q_{t+1}$ 和 $\{K_1, K_2... K_{t+1}\}$ 逐个求点积即可，步骤5也类似。

[Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249) 有个非常生动的动画解释这一过程：

![[kvcache-explained.gif]]

那最后一个问题，为什么叫 KVCache而不是 QKVCache 呢？ 

因为 Q 前面计算的结果连缓存都不用了，直接丢弃也不影响计算。
# 参考

- [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)