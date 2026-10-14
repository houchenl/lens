---
title: 注意力机制-13 Lightning Attention
date: 2026-10-14
description: 从带衰减的外积记忆解释 Lightning Attention，推导跨块状态与块内计算的关系，再对照 MiniMax-Text-01/M1 的 7:1 混合结构，分析输出门控、检索能力、状态容量与整体复杂度。
---

# 注意力机制-13 Lightning Attention

[基础线性注意力](/ai-infra/attention-12-linear-attention)通过累计外积，将逐条历史转成固定形状的关联状态。MiniMax-Text-01 与 M1 使用 Lightning Attention 相关设计，在多数层维护递归状态，周期性保留 softmax 注意力层，以兼顾持续处理历史的成本和显式历史检索能力。

本文先解释带时间衰减的线性记忆，再说明分块形式与递归形式怎样表达同一个模型，最后对照实际混合层型。依据为 [MiniMax-01 报告](https://arxiv.org/abs/2501.08313)、[M1 报告](https://arxiv.org/abs/2506.13585)、[Lightning Attention-2](https://arxiv.org/abs/2401.04658)与官方模型代码，查阅于 **2026-10-14**。

## 00 先区分名称、模型公式和执行方法

Lightning Attention 是面向线性注意力的高效执行方法，其模型基础与 TransNormer/NormAttention 路线相关。MiniMax 的实际层还包含激活、衰减、输出归一化与门控，不能只凭一句“交换矩阵乘法顺序”描述完整结构。

DSA 中的 Lightning Indexer 是另一种组件：它给历史位置打分、选择候选，仍保留可检索的历史条目。本篇 Lightning Attention 维护的是递归关联矩阵，不存在这一步 Top-K 历史路由。

后文讨论分块，是为了证明递归记忆可以批量处理；线程调度、显存分页和 FlashAttention 内核不在范围内。

## 01 从归一化核注意力，走到输出归一化的关联记忆

上一篇的基础形式维护 $S_t,z_t$，输出为 $S_t\widehat q_t/(z_t^\top\widehat q_t)$。MiniMax 的相关线性层采用另一种尺度控制方式：先计算未做 token 概率归一化的关联读取，再对输出做归一化与门控。

设单头 $q_t,k_t\in\mathbb R^{d_k}$，$v_t\in\mathbb R^{d_v}$，记忆矩阵 $S_t\in\mathbb R^{d_v\times d_k}$。暂时省略激活和输出处理，最简单的核心是：

$$
S_t=S_{t-1}+v_tk_t^\top,\qquad \widetilde o_t=S_tq_t
$$

这里没有基础核注意力中的 $z_t$ 分母。点积权重不必是非负概率，不能说它仍在对历史 token 做一组和为 1 的加权平均。归一化位置的改变属于模型定义，不是执行时可任意省略的细节。

## 02 时间衰减怎样进入状态更新

### 2.1 旧状态先衰减，再写入新关联

引入某个头的衰减系数 $\lambda\in(0,1]$：

$$
S_t=\lambda S_{t-1}+v_tk_t^\top,\qquad \widetilde o_t=S_tq_t
$$

展开后得到：

$$
S_t=\sum_{j\le t}\lambda^{t-j}v_jk_j^\top,\qquad
\widetilde o_t=\sum_{j\le t}\lambda^{t-j}(k_j^\top q_t)v_j
$$

衰减给时间距离引入指数偏好。$\lambda=1$ 时退化成普通累加；越接近 0，旧记忆消退越快。不同头可以具有不同时间尺度。

官方 Text-01 参考实现使用 `ratio = exp(-slope_rate)` 更新状态，并在层级上调整 slope。它不等于由每个 token 内容动态产生的遗忘门；后续 Gated DeltaNet 与 KDA 将引入更强的数据依赖控制。[参考 `MiniMaxText01LightningAttention`](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py)。

### 2.2 半衰期比抽象系数更直观

若希望旧贡献经过 $n_{1/2}$ 步后只剩一半，则：

$$
\lambda^{n_{1/2}}=\frac12,\qquad
n_{1/2}=\frac{\log(1/2)}{\log\lambda}
$$

例如 $\lambda=0.5$ 的半衰期是一 token，$\lambda=0.99$ 约为 69 token。它表示单个贡献的相对衰减，不是有效上下文长度的硬截断；更远历史仍可能通过非零状态影响输出。

衰减还能抑制无限累加的尺度，但它不保证消除记忆冲突。若两个相似 key 写入不同值，状态仍混合两者，只是更近的贡献权重较大。

## 03 为什么分块处理不改变递归定义

设进入一块前的状态为 $S_s$，块内位置记为 $i=1,\ldots,m$。则块内第 $i$ 个输出可分解为：

$$
\widetilde o_{s+i}=\lambda^i S_sq_{s+i}
+\sum_{j=1}^{i}\lambda^{i-j}(k_{s+j}^\top q_{s+i})v_{s+j}
$$

第一项读取之前所有块的合并状态，第二项只在当前块内部计算因果关联。块尾状态为：

$$
S_{s+m}=\lambda^m S_s+
\sum_{j=1}^{m}\lambda^{m-j}v_{s+j}k_{s+j}^\top
$$

因此，可以在块内批量计算短程匹配，用一个矩阵概括块外历史，再按块推进状态。块内长度固定时，不需要为全部 $T$ 个 token 形成 $T\times T$ 的全局匹配矩阵。

这类块内/块间分工是 [Lightning Attention-2 算法](https://arxiv.org/html/2401.04658#S3)的核心；它是同一个线性记忆的计算重组，不是把每块永久压成一条供未来 Top-K 检索的 KV。

## 04 实际层还有哪些决定输出的组件

Text-01 参考实现从同一个输入投影得到 Q/K/V，并施加 SiLU（Sigmoid Linear Unit）激活。各头计算上述状态读取后，拼接输出，经过 RMSNorm（均方根归一化）、输入相关输出门控和输出投影：

$$
y_t=W_o\left[\sigma(W_gh_t)\odot
\operatorname{RMSNorm}\bigl(\operatorname{Concat}_h\widetilde o_{t,h}\bigr)\right]
$$

输出门控决定当前输入应暴露多少读取结果；状态衰减决定过去信息保留多少，两者职责不同。将输出乘小，并不意味着状态中的旧关联已经被删除。

线性层也不能直接套用 softmax 层的整套位置处理。Text-01 的 softmax 层有 RoPE（旋转位置编码），线性路径则通过衰减等机制引入位置偏好；配置中的 `rope_theta` 并不意味着所有类型的层都执行相同旋转。

## 05 验证递归、显式展开与分块边界

下面的 Python 3 示例使用单头二维 key/query、标量 value，省略激活、归一化与门控。它检验递归展开相同，并确认两段连续处理不会丢掉前一段状态。

```python
from math import isclose

keys = [[1, 0], [0, 1], [1, 1], [1, 0]]
queries = [[1, 0], [1, 1], [0, 1], [1, 0]]
values = [10, 20, 30, 40]
decay = 0.5


def run(start, end, state):
    outputs = []
    state = state[:]
    for t in range(start, end):
        state = [decay * state[c] + values[t] * keys[t][c]
                 for c in range(2)]
        outputs.append(sum(state[c] * queries[t][c] for c in range(2)))
    return outputs, state


whole, final = run(0, 4, [0, 0])
first, middle = run(0, 2, [0, 0])
second, split_final = run(2, 4, middle)
assert whole == first + second and final == split_final
for t, output in enumerate(whole):
    explicit = sum(decay**(t-j) * values[j]
                   * sum(keys[j][c] * queries[t][c] for c in range(2))
                   for j in range(t + 1))
    assert isclose(output, explicit)
```

该检查对应模型代数的一致性；实际低精度分块路径还要比较误差累积和长期状态稳定性。参考代码使用浮点累加状态，也说明“缓存占几个字节”需要核对具体精度，不能只看模型权重 dtype。

## 06 MiniMax-01/M1 为什么保留每八层一次 softmax

Text-01 与 M1 的公开 `attn_type_list` 都采用七个线性层、一个 softmax 层的周期。80 个主干层对应 70 个线性层、10 个 softmax 层；后者在 0 起始编号的 7、15、23、…、79 层。

| 状态 | 线性层 | softmax 层 |
|---|---|---|
| 记录历史 | 固定形状关联矩阵 | 每个位置的独立 KV |
| 当前读取 | 与矩阵状态相乘 | query 重新匹配历史位置 |
| 写入冲突 | 相似 key 的贡献会叠加 | 位置独立保存，可重新比较 |
| 时间偏好 | 状态衰减等机制 | 位置编码与 query-key 匹配 |

线性层可以持续构造上下文表示，softmax 层保留直接重新匹配历史条目的机会。两种层经隐藏状态串联，不是把同一层的两个输出简单平均。

M1 基于 Text-01 继续训练、发展推理能力，不能因为名称更换就认定注意力公式彻底改变。M2 的 GQA 与 M3 的 MSA 属于其他已发布分支，已在前篇单独讨论。[Text-01 配置](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json)、[M1 配置](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/blob/main/config.json)。

## 07 混合模型仍有哪些随长度增长的项

固定维度下，70 个线性层各自的状态不随 $T$ 增长；10 个完整 softmax 层仍保存长度为 $T$ 的历史。简化表达为：

$$
M\approx L_{\mathrm{lin}}H_{\mathrm{lin}}d_kd_v s_S
+L_{\mathrm{soft}}TH_{kv}(d_k+d_v)s_{KV}
$$

$s_S,s_{KV}$ 分别为状态和 KV 每元素字节数。以 70 个线性层、每层 64 头、维度 128、状态 FP32 为假设，线性状态约为 280 MiB；另设 10 个 softmax 层、8 个 KV 头、K/V 各 128 维、BF16，在 $T=2^{20}$ 时这部分约为 40 GiB。

这些是指定布局的逻辑容量，不包括批量、权重、临时激活与对齐。例子说明：多数层用固定状态后，剩余显式历史仍可能支配长上下文缓存。

计算量同理：混合模型包含线性层的 $O(T)$ 项，也包含完整 softmax 层的 $O(T^2)$ prefill 项。7:1 会降低后者的系数，却不能将整个模型的渐近复杂度改称严格线性。

配置中的 `max_position_embeddings` 也不是质量保证。支持输入一个长度、训练达到一个长度、在该长度完成可靠检索，是三个不同判断，应分别查阅模型报告和实际评测。

## 08 遗忘解决尺度，定向修正解决写入冲突

指数衰减让旧状态逐渐退场，输出门控控制当前读出的贡献，两者都改善基础线性记忆的灵活性。但新关联仍按外积直接相加，没有先检查“这个 key 目前已经记住什么”。

当同一个变量不断更新时，简单叠加会混合新旧值；整体衰减又会同时削弱其他仍然有效的变量。下一篇的 **DeltaNet 与 Gated DeltaNet** 将引入读取误差：先沿 key 读取旧值，再只写入需要修正的部分，进一步加入遗忘门。

继续阅读：[注意力机制-14 DeltaNet 与 Gated DeltaNet](/ai-infra/attention-14-gated-deltanet)。

## 参考资料

1. MiniMax，[MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313)，2025。重点见 §2.2 的模型、分块形式与混合结构。
2. MiniMax，[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)，2025。用于确认 M1 与 Text-01 的关系及训练背景。
3. Qin et al.，[Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658)，2024。重点见块内/块间分解，不将标题里的 unlimited 解释为无限精确记忆。
4. MiniMax，[Text-01 配置](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json)、[参考模型代码](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py)与[M1-80k 配置](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/blob/main/config.json)。滚动分支，查阅于 2026-10-14。
