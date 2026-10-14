---
title: 注意力机制-15 KDA 与 Kimi 的混合注意力
date: 2026-10-14
description: 从 Gated DeltaNet 的标量遗忘推导 KDA 的按 key 通道遗忘、DPLR 转移与误差修正，对照 Kimi Linear 和 K3 的实际层型，解释 NoPE、输出门控、全局 MLA 与固定状态的分工。
---

# 注意力机制-15 KDA 与 Kimi 的混合注意力

[Gated DeltaNet](/ai-infra/attention-14-gated-deltanet)已经可以先遗忘旧状态，再沿当前 key 修正映射，但一个头的所有记忆方向仍使用同一个遗忘系数。KDA（Kimi Delta Attention，Kimi Delta 注意力）将它细化为按 key 通道控制的向量门，让一个头中的不同方向可以具有不同的保留速度。

Kimi Linear 把 KDA 与全局 MLA 混合，Kimi-K3 进一步在公开架构中使用 KDA 与 Gated MLA。本文先推导共同的记忆更新，再核对两种模型的配置差异；主要依据为[Kimi Linear 论文](https://arxiv.org/abs/2510.26692)、[公开模型配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)与[K3 模型卡](https://huggingface.co/moonshotai/Kimi-K3)，查阅于 **2026-10-14**。

## 00 同一个头，为什么需要不同的遗忘速度

假设一段长任务中同时存在长期约束和临时变量。整体遗忘可以清除临时信息，却也会削弱长期关联。增加头数能提供多种时间尺度，但一个头内部仍可能同时编码不同类型的信息。

KDA 将遗忘从一个标量扩展到一个向量。它不为每个原始 token 单独设置保留位，也不产生稀疏地址；它控制的是压缩状态中的特征方向。关键问题因此变成：门作用于哪一维，如何与 Delta 修正组合？

## 01 先固定矩阵方向，避免把 key 门误写成 value 门

延续上一篇的列向量约定，记忆矩阵为 $S\in\mathbb R^{d_v\times d_k}$，读取为 $Sq$。论文常采用其转置 $d_k\times d_v$，所以本文公式中的乘法方向与论文对应转置，计算含义一致。

标量门控的 Gated DeltaNet 为：

$$
S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

KDA 引入 $\boldsymbol\alpha_t\in(0,1]^{d_k}$，构成对角矩阵 $D_t=\operatorname{Diag}(\boldsymbol\alpha_t)\in\mathbb R^{d_k\times d_k}$：

$$
S_t=S_{t-1}D_t(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top
$$

在本文布局下，$S_{t-1}D_t$ 缩放状态的各列，即 key 特征方向。若在左侧乘一个 $d_v\times d_v$ 的门，就变成了 value 方向控制，不能视为同一个更新。

当 $D_t=\alpha_tI$ 时退化成标量遗忘。KDA 增加的是头内的保留自由度，并未增加状态矩阵的历史长度维度。[定义见论文式 1](https://arxiv.org/html/2510.26692#S3)。

## 02 遗忘、读取旧值、修正：仍是三个连续步骤

将上式展开成容易理解的数据流：

$$
\bar S_{t-1}=S_{t-1}D_t,\qquad
e_t=v_t-\bar S_{t-1}k_t,\qquad
S_t=\bar S_{t-1}+\beta_te_tk_t^\top
$$

先按方向保留旧记忆，再沿当前 key 读取衰减后的旧值，最后写入误差。输出核心仍是 $\widetilde o_t=S_tq_t$。

令旧状态 $S=[10,20]$，遗忘门为 $(1,0.1)$，即第一方向全部保留、第二方向保留十分之一。若当前 key 为 $(0,1)$、目标 value 为 30、$\beta=1$，先得到 $[10,2]$，再修正第二方向，最终为 $[10,30]$。

使用标量遗忘 0.1 再修正同一 key，则得到 $[1,30]$。两者都能覆盖当前关联，但细粒度门保留了第一方向的旧记忆。实际方向由训练学习，不能人为认定某一列永久对应某个自然语言变量。

## 03 细粒度门为何引入 DPLR，又为何不能任意交换

状态转移矩阵可以写成：

$$
A_t=D_t(I-\beta_tk_tk_t^\top)
=D_t-\beta_t(D_tk_t)k_t^\top
$$

它由一个对角矩阵减去秩一项，属于 DPLR（Diagonal-Plus-Low-Rank，对角加低秩）结构。对角部分负责按通道保留，秩一部分负责当前 key 的定向修正。

与完全自由的 DPLR 相比，KDA 把低秩更新的因素约束到当前 key 及其门控形式，保留 Delta 规则的含义，也为高效分块提供更具体的结构。

标量 $\alpha I$ 可以与 $kk^\top$ 交换，向量门 $D$ 一般不能。因此 $SD(I-\beta kk^\top)$ 与 $S(I-\beta kk^\top)D$ 通常不同；把代码中的“先遗忘再修正”随意换序，会改变模型。

论文按块推导这种有序变换的组合，并处理累计遗忘带来的数值问题。固定头维度下，单步仍以矩阵读写与外积为主，约为 $O(d_kd_v)$；不能因为表达成 DPLR，就在每个 token 上显式构造并连乘昂贵的完整转移矩阵。

## 04 最小数值检查：key 方向与乘法顺序

下面的 Python 3 代码只依赖标准库。矩阵只有一行，便于把两列直接看作两个 key 方向；不包含完整 KDA 的模型投影和输出处理。

```python
from math import isclose


def kda_row(state, key, value, decay, beta):
    decayed = [s * a for s, a in zip(state, decay)]
    prediction = sum(s * k for s, k in zip(decayed, key))
    return [s + beta * (value - prediction) * k
            for s, k in zip(decayed, key)]


assert kda_row([10, 20], [0, 1], 30, [1, 0.1], 1) == [10, 30]
assert kda_row([10, 20], [0, 1], 30, [0.1, 0.1], 1) == [1, 30]

old, key, gate, beta, value = [10, 20], [0.6, 0.8], [1, 0.1], 0.5, 7
correct = kda_row(old, key, value, gate, beta)
old_prediction = sum(s * k for s, k in zip(old, key))
wrong_order = [(s - beta * old_prediction * k) * a + beta * value * k
               for s, k, a in zip(old, key, gate)]
assert any(not isclose(x, y) for x, y in zip(correct, wrong_order))
```

当 key 只有一个坐标非零时，一些错误顺序可能碰巧难以区分；检查非轴对齐 key 能更有效地暴露乘法顺序问题。这也说明验证例子应覆盖真正决定机制的条件。

## 05 从核心更新到完整 KDA 层

Kimi Linear 论文的 query/key 生成包含线性投影、短程因果卷积、Swish/SiLU 激活与 L2 归一化；value 不做同样的 L2 归一化。按通道遗忘向量由输入投影产生，$\beta$ 是按头的写入门。

在核心读取后，各头输出经过 RMSNorm，再由输入相关的 sigmoid 输出门控制，最后输出投影。遗忘向量、写入门和输出门分别控制保留、修改、暴露，不能因为都叫 gate 就合并解释。

论文用低秩投影参数化遗忘和输出门，以控制参数成本。K3 的公开配置与代码存在进一步选择，例如 KDA 输出门的 `use_full_rank_gate=true`；不能把早期论文的全部参数化直接套到后来的 checkpoint。[K3 参考实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)还分别给出分块与逐步递归调用。

## 06 为什么混合 MLA，而且全局层可以使用 NoPE

### 6.1 固定状态与可检索历史负责不同工作

KDA 的状态不随序列长度增长，但无法为任意过去 token 保留独立地址。MLA（Multi-head Latent Attention，多头潜在注意力）则用较紧凑的逐位置表示保存历史，当前 query 仍能重新选择关注不同位置。

两种层按深度交替：KDA 持续构造带历史的隐藏状态，周期性的全局 MLA 执行内容匹配。这样比所有层都保存完整显式历史更省状态，同时保留部分独立历史读取机会；是否改善质量仍是训练与评测问题。

### 6.2 NoPE 不等于模型没有顺序信息

NoPE（No Positional Encoding，无显式位置编码）在这里指全局 MLA 不施加 RoPE 等显式位置旋转。KDA 的因果更新、输入相关遗忘与短卷积已经使隐藏状态依赖顺序，因此全局层的内容向量本身并非无序集合的原始 token 表示。

没有显式旋转，还能避免把旋转插入潜在 KV 展开与矩阵吸收之间。它有助于简化 MLA 的读取路径，但不能仅凭 NoPE 就声称任意长序列质量没有下降。

Kimi Linear 与 K3 配置中都可见 `mla_use_nope=true`。即使保留了 `qk_rope_head_dim` 等结构字段，也不代表执行了旋转；K3 的全局 MLA 参考代码明确设置 `rotary_emb=None`，应以实际分支判断行为。

## 07 Kimi Linear 与 K3：哪些延续，哪些改变

| 项目 | Kimi Linear 48B-A3B Instruct | Kimi-K3 |
|---|---|---|
| 主干层数 | 27 | 93 |
| KDA 层数 | 20 | 69 |
| 全局层数 | 7 个 MLA | 24 个 Gated MLA |
| KDA 头数 / 头维度 | 32 / 128 | 96 / 128 |
| 全局位置处理 | NoPE | NoPE |
| 额外全局输出门 | 按对应实现核对 | `mla_use_output_gate=true` |

论文采用约 3:1 的 KDA/MLA 组织原则，但实际公开配置存在尾部例外。Kimi Linear 的 1 起始全局层号为 4、8、12、16、20、24、27；K3 为 4 的倍数至 92，再加 93，其余为 KDA。

```python
linear_full = [4, 8, 12, 16, 20, 24, 27]
k3_full = list(range(4, 93, 4)) + [93]
assert 27 - len(linear_full) == 20
assert len(k3_full) == 24 and 93 - len(k3_full) == 69
```

这里是配置采用的 1 起始编号，代码索引需要转换。不能直接对所有层使用 `layer_idx % 4`，否则会错过末层安排。[Linear 配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)、[K3 配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)。

Gated MLA 是在全局注意力读取后增加输出门控，不表示该全局层变成 KDA，也不表示它只保留固定状态。K3 的 AttnRes（Attention Residuals，注意力残差）处理的是深度方向的残差融合；它与 token 历史的注意力连接属于不同维度，不能混称为新的历史索引器。

## 08 将 Kimi 的公开分支放回时间关系

MoBA 论文发表于 2025 年初，报告用于 Kimi 长上下文服务；这不是对所有后续开源模型的架构声明。K2、K2-Thinking、K2.5 的公开结构保留 MLA 路线；2025 年后期公开的 Kimi Linear 则提供可直接研究的 KDA/MLA 混合模型，后来的 K3 采用扩展后的 KDA/Gated MLA 结构。

因此，Kimi Linear 与 K2 系列存在时间交叉，不能写成“K2 之后所有版本都立刻改用线性注意力”。早期闭源服务没有披露的注意力细节也不应根据产品名推断补齐。

## 09 成本与验证：更多门不意味着无限容量

一个 KDA 层的持久核心状态约为 $Hd_kd_v$ 个元素，加上短卷积的有界状态；全局 MLA 则保留随 $T$ 增长的逐位置潜在 KV。因此混合模型缓存由常量部分与线性历史部分共同组成。

同理，固定维度的 KDA prefill 可线性扩展，完整全局 MLA 层仍有二次匹配项。减少全局层比例缩小该项系数，并不自动改变整个模型的渐近阶数。论文报告的容量和吞吐收益必须保留比较对象、长度与批量条件，不能由“3:1”直接推导全模型快四倍。

验证上，可先检查向量门的轴、遗忘与修正顺序、query/key 归一化和尾部层型，再比较逐步递归与分块输出；随后评测同 key 更新、多 key 干扰、跨距离精确复制和长文多证据检索。本文的算例验证核心代数与配置计数，未执行整模型 GPU 评测。

KDA 将“如何保留旧状态”细化到通道，混合 MLA 又补充显式历史读取。下一篇讨论 **GLM-5.3-Flash**：它把 KDA 与稀疏 MLA 结合，让固定状态与可检索历史在另一种层型比例下协作。

继续阅读：[注意力机制-16 GLM-5.3-Flash](/ai-infra/attention-16-glm-hybrid)。

## 参考资料

1. Kimi Team，[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)，2025。重点见 §3 的向量遗忘与 §4 的完整层和 NoPE。
2. Moonshot AI，[Kimi Linear 配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)、[Kimi-K3 模型卡](https://huggingface.co/moonshotai/Kimi-K3)、[配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)与[参考代码](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)。
3. Moonshot AI，[K2 配置](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)、[K2-Thinking 配置](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/config.json)与[K2.5 配置](https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json)。用于区分时间交叉的模型分支，资料查阅于 2026-10-14。
