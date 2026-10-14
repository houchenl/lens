---
title: 注意力机制-12 线性注意力
date: 2026-10-14
description: 从核相似度推导线性注意力的矩阵结合与因果递归状态，解释归一化分母、外积写入、记忆干扰、容量和训练边界，并用标准库程序验证逐步更新与显式历史计算的一致性。
---

# 注意力机制-12 线性注意力

前几篇稀疏机制都保留某种可检索历史，再决定当前 query 读取哪些条目。线性注意力改变了历史的组织方式：将 key-value 关联累积进固定形状的状态，每来一个 token 更新一次，再用 query 从状态中读取。

本篇从标准注意力的分子、分母出发，推导为什么某些相似度可以递归计算，以及这种状态会怎样合并和干扰记忆。它是后续 Lightning Attention、DeltaNet 和 KDA 的基础。主要依据为[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html)，公式采用适合本系列的列向量方向；资料核对于 **2026-10-14**。

## 00 从“保存所有条目”到“保存可更新统计量”

假设模型连续读取很多组“名字—属性”。标准注意力为每个历史位置保存独立 key 和 value，未来 query 可以重新比较全部位置。稀疏注意力仍保存这些条目或其压缩表示，只是少读一部分。

另一种思路是维护一个关联矩阵：把遇到的 key-value 写进去，查询时只读取矩阵。这样状态大小可以不随历史长度增长，但历史会被叠加到有限表示中。我们需要先找到允许“先汇总、后查询”的相似度函数。

本篇讨论基础的归一化核线性注意力。后续模型可能采用不带该分母的形式、额外归一化、遗忘门或误差修正，不能将一个公式当作所有线性注意力的统一实现。

## 01 为什么标准 softmax 不能直接交换乘法顺序

设 $q_t,k_j\in\mathbb R^{d_k}$、$v_j\in\mathbb R^{d_v}$，因果注意力为：

$$
o_t=\frac{\sum_{j\le t}\exp(q_t^\top k_j/\sqrt{d_k})v_j}
{\sum_{j\le t}\exp(q_t^\top k_j/\sqrt{d_k})}
$$

矩阵乘法满足 $(QK^\top)V=Q(K^\top V)$，但标准注意力中间还有逐行 softmax。一般不能写成 $\operatorname{softmax}(QK^\top)V=Q(K^\top V)$；右侧既没有指数匹配，也没有随 query 改变的归一化。

因此，线性注意力的起点是定义适合重排的相似度，而不是跳过 softmax 后仍声称完全等价。不同特征映射可以定义新的核，也可以用来近似特定核；两者的误差与训练需求必须区分。

## 02 用核特征分解相似度

引入特征映射 $\phi:\mathbb R^{d_k}\rightarrow\mathbb R^r$，定义：

$$
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k)
$$

$r$ 是特征维度，不必等于原始头维度。若希望形成非负归一化权重，应保证相似度非负，且可见历史中的总相似度大于零。

原始 Linear Transformer 的一个选择是逐元素 $\phi(x)=\operatorname{ELU}(x)+1$。在参数为 1 的 ELU（Exponential Linear Unit，指数线性单元）下：

$$
\phi(x)=
\begin{cases}
x+1,&x>0\\
e^x,&x\le0
\end{cases}
$$

它给出正特征，便于稳定归一化，但不构成对 softmax 核的精确恒等分解。[论文 §3.2](https://arxiv.org/html/2006.16236#S3.SS2)区分了核形式与具体映射选择。

为简化后续推导，记 $\widehat q_t=\phi(q_t)$、$\widehat k_j=\phi(k_j)$，二者均为 $r$ 维列向量。输出变为：

$$
o_t=\frac{\sum_{j\le t}(\widehat q_t^\top\widehat k_j)v_j}
{\sum_{j\le t}\widehat q_t^\top\widehat k_j}
$$

对这个已经定义好的核，接下来的代数改写是精确的；它与标准 softmax 是否接近，是另一个问题。

## 03 外积写入：把 query 移到求和之外

观察分子中的一项，标量相似度可以重新组合为：

$$
(\widehat q_t^\top\widehat k_j)v_j
=(v_j\widehat k_j^\top)\widehat q_t
$$

外积 $v_j\widehat k_j^\top\in\mathbb R^{d_v\times r}$ 将 key 的每个特征与 value 的每个维度关联起来。定义历史状态：

$$
S_t=\sum_{j\le t}v_j\widehat k_j^\top\in\mathbb R^{d_v\times r},\qquad
z_t=\sum_{j\le t}\widehat k_j\in\mathbb R^r
$$

则读取为：

$$
o_t=\frac{S_t\widehat q_t}{z_t^\top\widehat q_t}
$$

分子矩阵保存加权 value 关联，分母向量保存相似度总量需要的统计量。只存 $S_t$、忘掉 $z_t$，就不再是前面定义的归一化注意力。

具体实现常给分母加一个很小的 $\varepsilon$，防止极端数值下除零；这会轻微改变理想公式，应视为数值稳定处理。它不能补救任意有符号相似度导致的严重抵消问题。

## 04 因果递归：只更新当前 token 的贡献

空历史记为 $S_{-1}=0,z_{-1}=0$。每个 token 执行：

$$
S_t=S_{t-1}+v_t\widehat k_t^\top,\qquad
z_t=z_{t-1}+\widehat k_t
$$

再用当前 query 读取更新后的状态。因为当前位置可以参与自身注意力，所以是先写入、后读取；若任务使用严格只读过去的约定，需要相应改变时序。

```text
旧状态 S,z ──加上当前 key-value 外积与 key 特征──> 新状态 S,z
当前 query ──特征映射─────────────────────────> 读取新状态 → 输出
```

这组状态不依赖未来 query，多个未来 token 都能复用它。因果性也由更新顺序自然保证：计算位置 $t$ 时，状态仅包含 $j\le t$ 的贡献。[论文 §3.3—3.4](https://arxiv.org/html/2006.16236#S3.SS3)给出了对应递归解释。

对于非因果整段计算，结合律可以直接将 $T\times T$ 的两两匹配改写成特征维度上的矩阵乘法；因果场景则必须使用前缀和，不能拿包含未来的全序列总状态回答过去 query。

## 05 一个二维关联记忆：读取、重复写入与平均

令 $r=2,d_v=1$，使用已经完成特征映射的非负 key。写入 $\widehat k_0=(1,0),v_0=10$ 后，再写入 $\widehat k_1=(0,1),v_1=20$，状态为：

$$
S_1=\begin{bmatrix}10&20\end{bmatrix},\qquad
z_1=\begin{bmatrix}1\\1\end{bmatrix}
$$

query $(1,0)$ 读到 10，query $(0,1)$ 读到 20，query $(1,1)$ 则读到 15。最后一种查询同时匹配两个关联，因此产生平均。

再写入相同 key $(1,0)$、新 value 30，得到 $S_2=[40,20]$、$z_2=(2,1)$。此时 query $(1,0)$ 读取 20，是 10 与 30 的平均，并非“最新值覆盖旧值”。

该例使用正交特征便于展示。一般 key 并不正交，读取展开为 $\sum_j v_j(\widehat k_j^\top\widehat q_t)$，许多旧关联会贡献交叉项。这种累积干扰，是后面引入遗忘和定向更新的重要动机。

## 06 验证递归形式与显式历史形式一致

下面用 Python 3 标准库计算两条路径：一条显式遍历可见历史，另一条递归更新 $S,z$。输入直接给出非负特征，不复现 ELU 投影；所有分母均为正，未加入 $\varepsilon$。

```python
from math import isclose

keys = [[1, 0], [0, 1], [1, 0], [1, 1]]
queries = [[1, 0], [1, 1], [1, 0], [0.5, 1]]
values = [[10, 1], [20, 2], [30, 4], [40, 8]]
r, dv = 2, 2
S = [[0.0] * r for _ in range(dv)]
z = [0.0] * r

for t, (key, query, value) in enumerate(zip(keys, queries, values)):
    for a in range(dv):
        for b in range(r):
            S[a][b] += value[a] * key[b]
    for b in range(r):
        z[b] += key[b]
    denominator = sum(z[b] * query[b] for b in range(r))
    recurrent = [sum(S[a][b] * query[b] for b in range(r)) / denominator
                 for a in range(dv)]
    weights = [sum(query[b] * keys[j][b] for b in range(r))
               for j in range(t + 1)]
    explicit = [sum(weights[j] * values[j][a] for j in range(t + 1))
                / sum(weights) for a in range(dv)]
    assert all(isclose(a, b) for a, b in zip(recurrent, explicit))
    if t == 2:
        assert recurrent[0] == 20
```

这个检查证明两种执行形式计算同一个核注意力，没有证明它们等于 softmax 注意力，也没有评价训练后的模型能力。验证范围应与结论对应。

## 07 线性复杂度与固定状态，分别以什么为前提

单头每个 token 更新和读取状态需要约 $O(rd_v)$ 运算，$T$ 个 token 的主要注意力工作为 $O(Trd_v)$。持久状态元素数为 $rd_v+r$，在 $r,d_v$ 固定时不随 $T$ 增长。

| 对象 | 逐 token KV 注意力 | 本篇核递归形式 |
|---|---|---|
| 解码持久历史 | $O(T(d_k+d_v))$ | $O(rd_v+r)$ |
| 当前 query 的主要历史计算 | $O(T(d_k+d_v))$ | $O(rd_v)$ |
| 历史读取方式 | 重新匹配独立条目 | 读取已累积的关联状态 |
| 精确历史位置访问 | 可以保留独立地址 | 状态本身没有逐 token 地址 |

“固定状态”描述这部分递归记忆，不等于训练总内存恒定。输入激活、反向传播保存与重计算、模型权重和输出序列仍有自己的开销；朴素地保存每个 $S_t$ 甚至会浪费大量内存。

若 $r=d_v=128$，以每元素 2 字节计，$S,z$ 共 33,024 字节，约 32.25 KiB。这个状态在极短序列中可能比少量 KV 更大。线性长度依赖的优势也需结合维度与实现常数判断，不能仅凭大 O 断言所有长度都更快。

## 08 训练可以并行，递归状态却不是无限记忆

基础更新是外积贡献的前缀和，因此训练或 prefill 可以按块构造局部贡献，再合并前缀状态，不必像 Python 示例那样完全串行。后续遗忘门和 Delta 更新会改变组合规则，但仍可设计相应分块计算；这是执行形式问题，不改变模型定义中的因果依赖。

固定形状、有限精度的状态必须合并历史信息。特征相似的 key 会相互干扰，重复写入会累积；缺少位置或衰减机制时，还可能无法表达任务需要的时间偏好。分母归一化控制聚合尺度，不能让冲突记忆自动分开。

历史无法按原 token 地址任意回读，也是混合设计保留部分 softmax 层的重要原因。线性层可以高效持续处理历史，显式 KV 层则保留重新检索某些独立位置的机会。混合后整个模型是否仍是线性，必须重新统计那些保留层的长度项。

本篇建立了最基础的“累加写入—归一化读取”。下一篇讨论 **Lightning Attention 与 MiniMax-01/M1**，引入时间衰减和线性/softmax 混合；随后 DeltaNet 将把累加写入改为基于读取误差的定向修正。

## 参考资料

1. Katharopoulos et al.，[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)，ICML 2020；[论文 HTML](https://arxiv.org/html/2006.16236)。重点见 §3.2 的核映射、§3.3 的因果前缀与 §3.4 的递归形式。
2. 同文[补充材料](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a-supp.pdf)。用于进一步阅读前向与反向累计计算，资料查阅于 2026-10-14。
