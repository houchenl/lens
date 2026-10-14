---
title: 注意力机制-11 MiniMax MSA
date: 2026-10-14
description: 深入解析 MiniMax Sparse Attention 的组级索引 query、共享索引 key、块内最大值选择和精确块内注意力，推导 KL 训练与梯度隔离，并以 M3 配置分析读取预算、FLOPs 与缓存边界。
---

# 注意力机制-11 MiniMax MSA

[MoBA](/ai-infra/attention-10-moba)用块均值路由历史，便宜却可能稀释孤立的关键位置。MiniMax Sparse Attention（MSA，MiniMax 稀疏注意力）为 GQA 增加一个专门的低维索引分支：每个 KV 组独立给历史打分，以块内最高分确定相关块，再让组内 query 头共享选区、独立细读。

本文解释分数、选择、主注意力和训练信号如何连起来，并用 MiniMax-M3 的公开配置计算预算。依据为[MSA 技术报告](https://arxiv.org/abs/2606.13392)、[官方代码仓库](https://github.com/MiniMax-AI/MSA)与[M3 配置](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)，查阅于 **2026-10-14**。内核只作为可执行证据入口，本文不展开其调度细节。

## 00 从 GQA 出发：共享 KV，不必共享所有组的选区

GQA（Grouped-Query Attention，分组查询注意力）使用 $H_q$ 个 query 头、$H_{kv}$ 个 KV 头，每个 KV 组服务 $G=H_q/H_{kv}$ 个 query 头。组内头共享一份历史 K/V，仍具有各自的 query 和注意力概率。

如果所有组都使用同一个稀疏选区，选择开销较低，却可能限制各组不同的检索需求；若每个 query 头都独立选择，则会增加路由和重复读取。MSA 在两者之间选择：每个 GQA 组一个选区。

MiniMax 的模型演进也需要按分支理解。M2 系列存在公开的 GQA 配置；M3 引入 MSA。01/M1 的线性混合注意力是另一条路线，将在第 13 篇回顾，不能把它们串成所有模型统一从线性转向稀疏的单线历史。

## 01 两条分支，三种共享范围

设当前层隐藏矩阵 $X\in\mathbb R^{T\times d}$，主头维度为 $d_h$，索引维度为 $d_I$，采用整段输入的行向量约定。索引分支产生：

$$
Q^I=XW_q^I\in\mathbb R^{T\times H_{kv}\times d_I},\qquad
K^I=XW_k^I\in\mathbb R^{T\times1\times d_I}
$$

主分支则产生 $H_q$ 个 query 头和 $H_{kv}$ 个 K/V 头。各对象的共享范围为：

| 对象 | 共享方式 |
|---|---|
| 索引 key | 所有 GQA 组共享一个索引 key 头 |
| 索引 query | 每个 GQA 组独立一个 query 头 |
| 主 K/V | 同组 query 头共享，对应不同 KV 组 |
| 选中块地址 | 组内共享，组间可以不同 |

索引 key 共享不等于主 KV 只剩一个头；主 KV 仍按 GQA 分组保存。也不能把它写成 DSA 的“多个索引头聚合成全层一个选区”。[架构定义见论文 §3.1](https://arxiv.org/html/2606.13392#S3.SS1)。

## 02 先逐 token 打分，再用块最大值选择

### 2.1 最大值是在什么空间计算

令位置为 $t$、组号为 $g$，每个块包含 $b$ 个连续 token。索引分数为：

$$
s^I_{t,g,j}=\frac{(q^I_{t,g})^\top k_j^I}{\sqrt{d_I}},\qquad
R_{t,g,a}=\max_{j\in B_a,\ j\le t}s^I_{t,g,j}
$$

完全不可见的块分数为负无穷。随后选择分数较高的块，并强制包含当前位置所属块；本地块占用一个选择名额。

MSA 没有先把一块 key 平均，再用 query 匹配均值。它先比较各个位置，再取块内最高分，因此一个强相关位置不会被同块许多低分位置平均掉。代价是索引器仍扫描逐 token 的低维 key，而不是只扫描 $T/b$ 个代表。

### 2.2 因果性必须在块归约前生效

当前块可能包含未来 token。若先对完整块取最大值，再在主注意力中遮蔽未来，未来 key 仍可能改变块的选择。因此，索引器的 max 只能作用于当前可见位置。

对当前块强制保留能保证近期信息的可达性，但它不意味着索引阶段可以忽略因果性。特别是用分数训练索引器、构造辅助目标或比较不同块时，未来信息不应进入当前 query 的有效分布。

### 2.3 为什么推理选择可以不做 softmax

在同一组、同一 query 下，softmax 的分母是公共项，指数函数严格单调，故块最大值排序可直接由原始 logits 得到。计算指数再取最大值、再排名，不会改变选中地址。

这就是 exp-free selection（不计算指数的选择）在这里的含义。它省去索引选择中的归一化操作，不表示主注意力也删除 softmax，更不表示训练 KL 时不需要概率分布。[依据：论文 §4.1](https://arxiv.org/html/2606.13392#S4.SS1)。

## 03 组选区如何转化为各头输出

设组 $g$ 选中的块为 $\mathcal B_{t,g}$，可见 token 集合为：

$$
\mathcal J_{t,g}=\left(\bigcup_{a\in\mathcal B_{t,g}}B_a\right)\cap\{0,\ldots,t\}
$$

对组内主头 $h$，读取该组的主 key 和 value：

$$
p_{t,h,j}=\operatorname{softmax}_{j\in\mathcal J_{t,g}}
\left(\frac{q_{t,h}^\top k_{g,j}}{\sqrt{d_h}}\right),\qquad
o_{t,h}=\sum_{j\in\mathcal J_{t,g}}p_{t,h,j}v_{g,j}
$$

主注意力在选中支持集上是标准 softmax 注意力，但这个支持集由近似索引器决定。“精确块稀疏注意力”指给定选区后的计算精确，不表示结果等于完整历史注意力。

索引分数不会直接作为最终主注意力权重。共享一个选区的头可以在块内关注不同位置，也可以对同一个位置分配不同概率。

## 04 一个小例子：两个组，两个远程偏好

设位置 0—7、块长 2，query 位于 7；每组预算两块，其中一块固定为当前块 3。两组索引分数如下：

| 组 | 块 0：0—1 | 块 1：2—3 | 块 2：4—5 | 块 3：6—7 |
|---|---|---|---|---|
| A | $(-8,10)$ | $(2,2)$ | $(0,1)$ | $(1,0)$ |
| B | $(0,1)$ | $(3,9)$ | $(5,4)$ | $(1,0)$ |

A 选择块 0 与本地块，B 选择块 1 与本地块。A 的块 0 均值只有 1，却包含最高分 10；max 路由保留了它，而均值路由可能更偏向块 1。

```python
scores = [[-8, 10, 2, 2, 0, 1, 1, 0],
          [0, 1, 3, 9, 5, 4, 1, 0]]


def choose_blocks(row, t, block_size, budget):
    current = t // block_size
    candidates = list(range(current))
    ranked = sorted(candidates,
                    key=lambda a: max(row[a * block_size:(a + 1) * block_size]),
                    reverse=True)
    return sorted(ranked[:budget - 1] + [current])


assert choose_blocks(scores[0], 7, 2, 2) == [0, 3]
assert choose_blocks(scores[1], 7, 2, 2) == [1, 3]
```

此标准库例子只验证组间独立选块，未模拟模型投影和注意力输出。最大值也有边界：一个偶然高分可以带入整块噪声，多个中等相关证据的总价值也未必由块内最大值准确表示。

## 05 离散选区不能直接训练索引器，信号从哪里来

### 5.1 用主注意力概率作为 teacher

硬 Top-K 地址不可微，普通语言建模损失不会直接教会索引 Q/K 如何选择。MSA 用 KL 散度对齐索引分布与主分布。对某个 query 和组，记其已选 token 集为 $\mathcal J$：

$$
\pi^I(j)=\operatorname{softmax}_{j\in\mathcal J}(s^I_j),\qquad
p(j)=\frac1G\sum_{h\in\mathcal H_g}
\operatorname{softmax}_{j\in\mathcal J}(s^{\mathrm{main}}_{h,j})
$$

teacher 在概率层面平均组内头，不能用“先平均 logits 再 softmax”替代。辅助损失为：

$$
\mathcal L_{\mathrm{KL}}=\operatorname{mean}_{t,g}
D_{\mathrm{KL}}(\operatorname{stopgrad}(p)\Vert\pi^I)
$$

这里对齐的是选区诱导的可见 token 分布，不是直接将 Top-K 块编号当作可微概率。暖启动阶段则使用完整可见支持集进行对齐。

### 5.2 两次停止梯度解决不同问题

teacher 停止梯度，避免 KL 直接驱动主注意力改变目标；索引分支输入也采用 $\operatorname{stopgrad}(X)$，避免 KL 经隐藏状态反向改变主干。

后者尤其容易漏掉：即使 teacher 已 detach，索引 Q/K 仍从 $X$ 投影，KL 依然可能通过学生路径更新主干。隔离输入之后，辅助损失主要训练索引投影，让索引器适应主干，而不是让主干迎合索引器的简化偏好。

### 5.3 暖启动与本地块提供稳定起点

随机索引器若一开始就控制主路径，可能持续错过有用历史。论文先用短暂完整注意力阶段训练索引器，再切换稀疏选择；强制本地块则防止局部上下文被异常路由完全遗漏。

附录还比较了给索引分支增加 value 和额外输出的方案，但最终暖启动配方不要求索引分支参与 value 聚合。不能把消融中的附加输出误写成所有 MSA 层的必备结构。[完整依据：§3.2 与附录 B/C](https://arxiv.org/html/2606.13392#S3.SS2)。

## 06 M3 的实际预算与成本边界

M3 文本主干配置为 64 个 query 头、4 个 KV 头、头维度 128；索引维度 128、索引 query 头数 4；块长 128、最多 16 块，本地块占一块。`sparse_attention_freq` 的前三项为 0，其余主干项为 1，因此不能把全部 60 层都当作相同稀疏层。

对于一个稀疏层，每组最多读取 $16\times128=2048$ 个主 token。不同组可以读取不同位置；组内 16 个主 query 头共享选区。

按论文的近似 FLOPs 口径，一次乘加计 2 FLOPs，忽略 softmax、投影、短前缀未满预算等细节：

$$
F_{\mathrm{GQA}}\approx2H_qd_hT^2
$$

$$
F_{\mathrm{MSA}}\approx H_{kv}d_IT^2+4H_qd_hTkb
$$

第一项是因果索引 QK，第二项是选中主 KV 上的匹配与 value 聚合。在上述维度、$T=2^{20}$ 时，二者比值约为 28.44。它是注意力算术估算，不是全模型延迟倍数；论文的设备实验还需保留模型、硬件与负载条件。

```python
T = 2**20
Hq, Hkv, dh, di, k, b = 64, 4, 128, 128, 16, 128
gqa = 2 * Hq * dh * T * T
msa = Hkv * di * T * T + 4 * Hq * dh * T * k * b
assert abs(gqa / msa - 28.444444444444443) < 1e-10
assert k * b == 2048
```

固定块预算下，主注意力的 prefill 近似线性，但索引器仍有二次项。MSA 减少低维索引与高维细读之间的成本比例，没有消除所有全历史搜索。

容量也要单独计算。若主 K/V 和索引 key 均以每元素 2 字节存储，一个稀疏层每 token 的逻辑缓存为 $2[H_{kv}(d_h+d_h)+d_I]=2304$ 字节，百万量级 $2^{20}$ token 约为 2.25 GiB。不含权重、临时激活和元数据；实际精度与布局另行核对。它比只保存同配置 GQA 主 KV 还多一份索引 key，不能把读取下降直接解释成 KV 缓存同比下降。

## 07 与已学稀疏方案比较

| 机制 | 路由依据 | 选区共享范围 | 最终读取 |
|---|---|---|---|
| MoBA | query 与主 key 的块均值匹配 | 基础形式按 query 头 | 选中块内 token |
| NSA | 压缩注意力概率映射并组内聚合 | GQA 组内 | 三路独立路径 |
| DSA | 多头索引分数聚合，选 token | 基础形式主头共享 token 集 | 选中 token 表示 |
| MSA | 每组索引 query，逐 token 打分后块 max | GQA 组内，组间独立 | 选中块内 token |

MSA 的设计重点是让块级执行与组级需求一致。验证应检查可见域内的 max、本地块预算、组内共享、概率层面的 teacher 平均，以及 KL 对主干的梯度隔离。配置里有稀疏字段，只能说明结构意图；还需确认执行路径真正跳过未选块。

到这里，本系列已经比较了多种“保留历史、选择一部分读取”的方案。下一篇转向 **线性注意力**：不再逐条保存所有历史关联，而是将历史写入可递归更新的矩阵状态。

## 参考资料

1. Lai et al.，[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)，2026。重点见 §3 的结构、训练、复杂度与附录消融。
2. MiniMax，[MSA 官方仓库](https://github.com/MiniMax-AI/MSA)、[MiniMax-M3 模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3)和[模型配置](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)。
3. MiniMax，[M2.5 配置](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json)。用于核对 GQA 分支。资料查阅于 2026-10-14，代码与配置 `main` 会滚动更新。
