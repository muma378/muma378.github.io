---
title: LLM 推理技术总结
tags:
  - transformer
  - attention
  - llm
  - AI
  - 学习笔记
date: 2025-10-28
---

# Batching 策略

提升GPU使用率最简单的办法就是批处理（Batching），即将多个请求拼接成一个 Q、K、V 矩阵一次性地处理。
传统的Batching（static batching）由于LLM对于不同的请求生成的token数量不一样，可能会出现同一个批次里不同请求间相互等待直到全部完成后才一起返回的问题。

1. _continuous batching_ 
	1. https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/
	2. https://www.anyscale.com/blog/continuous-batching-llm-inference
2. chunked prefill 

# 并行策略

## 数据并行

data parallelism 主要是用于训练时，权重在多个设备间拷贝，通过增大batch来降低训练时间。

## 流水线并行

最大的问题是下一个设备等待上一个计算的结果 (activations, gradients) 的时候会空闲，被称为“气泡”。Microbatcing （图c）可以减小但不能消除气泡。

![[Pasted image 20250628120154.png]]

## 张量并行

Tensor parallelism involves sharding (horizontally) individual layers of the model into smaller, independent blocks of computation that can be executed on different devices. Attention blocks and multi-layer perceptron (MLP) layers are major components of transformers that can take advantage of tensor parallelism.

![[Pasted image 20250628120554.png]]

可以看到 MLP 和 self-attention 天生就很适合并行，但是像 LayerNorm 和 Dropout 函数导致需要在不同设备间**复制**（聚合不同block计算的结果），这导致内存要求更高。
## 序列并行

将一段输入序列拆分成不同段，让每个GPU处理序列的一部分，然后通过跨 GPU 的通信（比如 all-to-all）实现 attention 和 residual 连接等操作。（可以看作batching的反向操作？）


# 基于注意力机制的优化

## multi-query attention (MQA)

所有头共用一组 K/V，**缓存减少约 8x~16x** ，但可能会 **削弱注意力的表达能力**。然而实验表明，在大多数实际任务中，**损失的表达能力并不会显著影响效果**；

https://arxiv.org/abs/1911.02150

## [Grouped-query attention](https://arxiv.org/pdf/2305.13245v2.pdf) (GQA)

MQA 和 MHA 的折中版， by projecting key and values to a few groups of query heads。Models originally trained with MHA, can be “uptrained” with GQA。

![[Pasted image 20250629092617.png]]

## Multi head Latent Attention

![[Pasted image 20250717103007.png]]

# 基于 KV Cache 管理的优化

## Flash Attention


![[Pasted image 20250716125252.png]]
from: https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention
## Paged Attention

因为输入序列的不可预测，内存总是需要保留模型支持的最大序列长度。基于 Paging 机制允许将连续的序列存储到不连续的空间中，而且支持按需申请block。

## RadixAttention

使用树（radix）来维护 kv cache，使用LRU的策略来决定移除哪些cache。

![[Pasted image 20250716133210.png]]
Figure 4. Examples of RadixAttention operations with an LRU eviction policy, illustrated across nine steps.

可以结合 Paged Attention 和 Continous Batching 来使用。
# 模型优化技术

## 量化 Quantization

量化又分为 reduced precision on either the activations, the weights, or both。

> activations （激活值）是每一层计算后的输出值，输入 x → 乘以权重 W → 加偏置 b → 激活函数 → 输出 a
> **Weights 是模型的“记忆”，Activations 是模型的“思考过程”**

对 weights 量化是很显而易见的，重点是对 activations 如何处理。一种方案是在将 weights 和 activations 操作时重新转换成高精度的。（因为没有对  INT8 和 FP16 相乘优化的硬件）

另一种方案是对 activations 也量化，但是由于它经常会包含超过边界的值（outliers），

策略 1（LLM.int8() 的方法）：

1. **用一组典型输入数据**跑一遍模型（称为 calibration）；
2. 找出哪些层或哪些 token 的 **activations 经常出现离群值**；    
3. 对这些部分用更高精度（如 FP16）保存，而其它部分仍用 INT8;
4. 这种方法就是著名的 **LLM.int8() 论文提出的 mixed-precision 量化**。

第二种方法是：    
1. **weights（权重）通常更稳定、分布更可控**，所以可以先对权重进行量化，得到其 **最小/最大值（动态范围）**。 然后 **用同样的范围** 来对 activations 进行量化；
2. 这个方案本质是 **放弃捕捉激活中的离群极值，转而专注保留主干信息精度**的策略，通过使用稳定的参考尺度，避免 outlier 拉宽量化区间、造成全面精度下降。

### 🧠 总结一下两种策略：

|策略|描述|优点|方法代表|
|---|---|---|---|
|1. 动态挑选|找出哪些 activations 需要高精度表示，只对那部分提高精度|精度高，适应性强|LLM.int8()|
|2. 借权重范围|用 weight 的量化范围套用在 activation 上|简单，不需要额外数据跑模型|有些推理引擎的静态量化|


![[Pasted image 20250629150220.png]]

## 稀疏矩阵 

GPUs in particular have hardware acceleration for a certain kind of _structured sparsity_, where two out of every four values are represented by zeros.

## 蒸馏

![[Pasted image 20250629152011.png]]


# 推理优化

## Continous batching

This takes advantage of the fact that the overall text generation process for an LLM can be broken down into multiple iterations of execution on the model. 推理服务器可以提前将已经完成的batch从计算中驱逐而保持未完成的继续计算。算是工程上的优化。

## Speculative inference

 a draft model temporarily predicts multiple future steps that are verified or rejected in parallel

![[Pasted image 20250629153344.png]]

draft model是串行的，但是 verify model可以并行，通常用一个便宜的作为draft，用大的作为verify

## Auto Prefix Caching

KV Cache 主要是用decode阶段加速，避免了每次生成token都要重新计算K、Q、V。 vLLM的 [automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) 可以跳过部分 Prefill 的阶段。当请求的文本前缀相同时，会直接复用之前的 KV Cache，不用 Prefill 阶段再重新计算了。这种跨请求级别的缓存机制使得服务在应对高并发的场景下提高了吞吐量。想像两种场景：
1. 用户重复对一段长文本进行提问（RAG）；
2. 多轮对话；

这两种场景下每次请求的前缀是相同的，因此可以直接复用上一轮的缓存。


# 参考
- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- 