---
title: 注意力机制-10 MoBA
date: 2026-10-14
description: 推导 MoBA 的无新增参数块路由、当前块因果约束和选中 token 的统一 softmax，结合最小程序分析均值路由的能力边界、块大小取舍，以及它与 NSA、DSA 的区别。
---

# 注意力机制-10 MoBA

处理一部书或一整个代码仓库时，相关证据常常集中在若干连续段落。MoBA（Mixture of Block Attention，混合块注意力）将历史切成连续 KV 块，用 query 与块 key 均值的匹配决定访问哪些块，再在选中块内部做逐 token 注意力。它把混合专家的路由思想用于历史位置，但不需要另外增加一套专家前馈网络。

本文沿“块怎样表示、query 怎样路由、怎样保持因果性、选中后怎样归一化”展开，并与[NSA](/ai-infra/attention-8-nsa)和[DSA](/ai-infra/attention-4-dsa)比较。依据为[MoBA 论文](https://arxiv.org/abs/2502.13189)和[Moonshot 官方代码](https://github.com/MoonshotAI/MoBA)，查阅于 **2026-10-14**。

## 00 从逐位置选择，到连续区域的内容路由

DSA 用辅助索引器筛选历史 token；NSA 从压缩注意力中推导块重要性。MoBA 选择另一种较直接的路由：利用已有主 key 为每个块生成一个均值代表，不额外训练独立路由投影，再让当前 query 选择最相关的块。

如果把每个块看作一个可访问的“历史专家”，query 就是在决定本次调用哪些专家。但每个专家保存的是该段历史的 K/V，计算仍是注意力；并不存在按块复制一套前馈参数。

需要保留的区别是：块均值只用于找区域，输出仍来自块内 token 的独立 value。均值不是最终替代原始 KV 的摘要缓存。

## 01 必要记号：块预算与 token 预算不同

以一个 query 头为例，历史 key 为 $k_j\in\mathbb R^{d_k}$，value 为 $v_j\in\mathbb R^{d_v}$，query 为 $q_t\in\mathbb R^{d_k}$。序列长度为 $T$，块大小为 $b$，位置从 0 开始。

第 $a$ 个块为 $B_a=\{ab,\ldots,\min((a+1)b,T)-1\}$，块数为 $N=\lceil T/b\rceil$。$k$ 表示每次最多使用的块数，包含当前 query 所在块。

若设 $k=3$，最多是两个历史块加当前块，而不是三个远程块再额外加当前块。最终读取的 token 数至多为 $kb$，当前块的未来部分还要遮蔽。

## 02 用块均值路由，不增加路由参数

每个块的 key 代表为：

$$
\mu_a=\frac1{|B_a|}\sum_{j\in B_a}k_j
$$

query 与代表的路由分数为：

$$
r_{t,a}=q_t^\top\mu_a
$$

这个分数不需要独立可学习的路由权重矩阵，但 query、key 本身仍由模型训练获得。“无新增路由参数”不等于“路由与训练无关”，也不意味着离散 Top-K 的决策天然无损。

路由通常按 query 头计算；不能直接套用 NSA 在 GQA 组内先求和、整组共享块的规则。若具体实现要共享路由，必须另外明确其共享约束。

从已有 key 生成代表使计算便宜：一个 query 只需与约 $T/b$ 个代表匹配，而不是先与全部 $T$ 个原始 key 匹配。[定义见论文 §2.2](https://arxiv.org/html/2502.13189#S2.SS2)。

## 03 当前块为何不能参与普通均值竞争

设 query 位于 $t$，其当前块号为 $a_t=\lfloor t/b\rfloor$。过去完整块可以正常参与路由；未来块必须排除。当前块则强制选中，并在块内施加逐 token 的因果遮罩：

$$
\mathcal R_t=\{a_t\}\cup
\operatorname{Top}_{k-1}\{r_{t,a}:a<a_t\}
$$

当历史不足时使用更少块。实际可访问位置为：

$$
\mathcal J_t=\{j\le t:j\in B_a,\ a\in\mathcal R_t\}
$$

为什么不让当前块普通地竞争？在训练或并行前向中，完整当前块的均值可能包含 $t$ 之后的 key。即使最终注意力屏蔽未来 token，如果这个均值影响是否选中当前块或挤掉某个历史块，未来信息仍通过路由泄漏。

强制当前块、只让过去完整块竞争，消除了这条路由依赖；随后因果遮罩再限制当前块内读取。官方[朴素实现](https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_naive.py)分别处理块级路由与 token 级遮罩。

例如 $b=2,t=4$，当前块为位置 4、5。query 4 必须选中它，却只能读取位置 4，不能读取 5；位置 5 的 key 如何变化，也不应改变 query 4 的历史路由。

当前块保留了块内局部依赖，但它不是固定宽度滑动窗口：query 位于块首时，块内可见位置很少；位于块尾时则更多。历史块的动态选择可以补充其他近期区域，不能把这一规则直接等同于固定窗口 $w$。

## 04 选中之后：仍然对 token 计算同一个 softmax

路由分数只决定块是否进入候选，不作为输出的额外专家权重。最终注意力在所有选中且可见的 token 上统一归一化：

$$
p_{t,j}=\frac{\exp(q_t^\top k_j/\sqrt{d_k})}
{\sum_{u\in\mathcal J_t}\exp(q_t^\top k_u/\sqrt{d_k})},\qquad
o_t=\sum_{j\in\mathcal J_t}p_{t,j}v_j
$$

即使不同块在实现中分开计算，也必须合并归一化统计量。设某块分母为 $Z_a=\sum_{j\in B_a\cap\mathcal J_t}\exp\ell_j$，块内归一化输出为 $o_a$，则合并结果是：

$$
o_t=\frac{\sum_a Z_a o_a}{\sum_a Z_a}
$$

例如两块分别有 $Z_1=1,o_1=0$ 与 $Z_2=9,o_2=10$，总输出是 9；直接平均两个输出会得到 5。二者的差别来自 softmax 概率质量，不能用均匀相加代替。

因此，MoBA 与 NSA 三路独立注意力后再门控的公式不同。前者是在稀疏候选集上保留标准注意力的归一化；后者显式组合不同归一化路径。[论文算法 1](https://arxiv.org/html/2502.13189#S2.SS3)说明了分块计算后的合并。

## 05 一个可运行的路由与因果示例

以下 Python 3 代码只依赖标准库，使用单头二维 query/key 和标量 value。它验证当前块、过去块竞争、最终统一归一化，以及未来 token 不影响早期输出。它不是高效模型实现。

```python
from math import exp, isclose, sqrt


def moba_one(q, keys, values, t, block_size, topk):
    current = t // block_size
    scores = {}
    for a in range(current):
        chunk = keys[a * block_size:(a + 1) * block_size]
        mean = [sum(row[c] for row in chunk) / len(chunk)
                for c in range(len(q))]
        scores[a] = sum(x * y for x, y in zip(q, mean))
    past = sorted(scores, key=scores.get, reverse=True)[:topk - 1]
    blocks = set(past + [current])
    positions = [j for j in range(t + 1) if j // block_size in blocks]
    logits = [sum(x * y for x, y in zip(q, keys[j])) / sqrt(len(q))
              for j in positions]
    peak = max(logits)
    weights = [exp(x - peak) for x in logits]
    output = sum(w * values[j] for w, j in zip(weights, positions)) / sum(weights)
    return positions, output


keys = [[1, 0], [3, 0], [0, 1], [0, 3], [1, 1], [100, 100]]
values = [1, 2, 3, 4, 5, 6]
positions, output = moba_one([1, 0], keys, values, 4, 2, 2)
assert positions == [0, 1, 4]
keys[5] = [-999, 999]  # 改变当前块里的未来 token
again_positions, again_output = moba_one([1, 0], keys, values, 4, 2, 2)
assert again_positions == positions and isclose(again_output, output)
```

示例中，第 0 块均值与 query 更匹配，所以历史路由选择位置 0、1；当前块只开放位置 4。没有选中的位置 2、3 虽不影响本次输出，却仍应保留，未来 query 可能需要它们。

## 06 块均值能代表相关性，但不能保证找回最高分 token

均值路由相当于按块内平均点积分数排名：$q^\top\mu_a=\operatorname{mean}_{j\in B_a}(q^\top k_j)$。但真正的注意力概率质量与 $\sum_j\exp(q^\top k_j/\sqrt{d_k})$ 有关，平均与指数求和一般不等价。

设两个块的缩放后逐 token logits 分别为 $(-8,10)$ 和 $(2,2)$。第一块均值为 1，第二块为 2，均值路由更偏好第二块；但第一块含有分数 10 的突出位置，其指数质量远大于第二块。

这个反例揭示“孤立关键证据被块平均稀释”的风险。训练可以让表示更适合块级路由，较小块也能减少稀释，但都不是对任意输入的精确 Top-K 保证。

| 块变小 | 块变大 |
|---|---|
| 路由代表更多，选择开销上升 | 代表更少，路由更便宜 |
| 选区更精细，少带无关邻居 | 一个关键位置可能带入更多邻居 |
| 固定 token 预算下可覆盖更多分离区域 | 局部连续区域更容易整块保留 |

比较块大小时，还必须说明固定的是块数 $k$ 还是 token 预算 $kb$。固定 $k$ 时缩小块，既增加精细度，也减少最终可读取的 token 数，无法把质量变化只归因于粒度。

## 07 复杂度、训练和模型归属

单 query 的路由匹配约为 $O(d_kT/b)$，选中后的主注意力约为 $O((d_k+d_v)kb)$。于是固定 $b,k$ 的长序列 prefill 仍含有 $O(d_kT^2/b)$ 的路由项，不能直接称整个机制严格线性。

主 KV 仍为逐 token 保存，均值代表只增加一份较小的块级状态。稀疏读取不会自动让全局缓存容量只剩 $kb$；如果丢弃未选块，就额外引入了缓存淘汰机制。

MoBA 不增加模型参数，便于在全注意力与稀疏注意力之间切换，但结构兼容不等于免训练等价。官方仓库明确要求继续训练以获得相应收益。论文还研究训练阶段切换，以及保留最后若干完整注意力层，以缓解稀疏连接下的训练与任务适配问题。

论文和官方仓库报告 MoBA 已用于 Kimi 长上下文请求，这证明其服务应用，不能自动证明 Kimi-K2、K2.5 或 K3 的每个公开 checkpoint 都使用 MoBA。具体开源模型仍需独立核对配置与官方技术说明；后续 KDA 篇会讨论另一条公开的线性混合分支。

验证应同时检查路由因果性、Top-K 是否包含当前块、分块归一化是否正确，以及受稀疏训练适配的质量。单纯在完整注意力分数上加 mask 能验证候选语义，却不证明计算已经跳过被屏蔽位置。

MoBA 把“找区域”和“读细节”分开，代价是粗粒度路由误差。下一篇的 **MiniMax MSA** 同样选择块，却用专门索引分支与 GQA 组级选择处理相关性，适合继续比较路由分数与共享范围。

继续阅读：[注意力机制-11 MiniMax MSA](/ai-infra/attention-11-minimax-msa)。

## 参考资料

1. Lu et al.，[MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)，2025。重点见 §2 的路由和因果性、§3.2 的混合训练。
2. Moonshot AI，[MoBA 官方仓库](https://github.com/MoonshotAI/MoBA)与[朴素实现](https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_naive.py)。滚动分支，资料查阅于 2026-10-14。
