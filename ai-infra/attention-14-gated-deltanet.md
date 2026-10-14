---
title: 注意力机制-14 DeltaNet 与 Gated DeltaNet
date: 2026-10-14
description: 从重复 key 的写入冲突出发，推导 DeltaNet 的误差修正、归一化条件与方向选择，再展开 Gated DeltaNet 的遗忘顺序、状态转移和分块训练，验证覆盖、干扰与重置的具体行为。
---

# 注意力机制-14 DeltaNet 与 Gated DeltaNet

[线性注意力](/ai-infra/attention-12-linear-attention)把历史关联累加进矩阵，[Lightning Attention](/ai-infra/attention-13-lightning-attention)进一步用时间衰减管理旧贡献。但面对“变量的新值覆盖旧值”这类任务，直接累加或整体衰减都不够精细。DeltaNet 的变化是：先读取当前 key 对应的旧值，再写入新旧差值。

Gated DeltaNet 在此基础上加入遗忘门，让模型既能沿特定方向修正，也能整体清理旧状态。本文从局部平方误差推导更新，说明 key 归一化、门控顺序和记忆干扰的作用，为下一篇 KDA 建立直接前置知识。

主要依据为[早期 fast-weight Delta 研究](https://proceedings.mlr.press/v139/schlag21a.html)、[DeltaNet 并行化论文](https://arxiv.org/abs/2406.06484)与[Gated DeltaNet 论文](https://arxiv.org/abs/2412.06464)，查阅于 **2026-10-14**。

## 00 新值应该叠加，还是修正已有映射

假设已经记住 key“当前端口”对应 8080，随后上下文说端口改为 9090。基础外积写入把两次关联相加，归一化后可能读到某种平均；整体遗忘又可能削弱其他无关变量。

理想更新应当先问“这个 key 现在读出什么”，再调整这条映射。这里讨论的是模型内部可微记忆，不是把自然语言 key 直接当作哈希表地址；真实 key 来自模型投影，是否能分离不同变量依赖训练。

令 $S_{t-1}\in\mathbb R^{d_v\times d_k}$ 为单头旧状态，$k_t,q_t\in\mathbb R^{d_k}$，$v_t\in\mathbb R^{d_v}$，均采用列向量约定。读取函数为 $Sk$ 或 $Sq$，不同于带 $z$ 分母的基础核注意力。

## 01 从局部误差推导 Delta 更新

对当前关联 $(k_t,v_t)$，定义局部平方误差：

$$
\mathcal E_t(S)=\frac12\|Sk_t-v_t\|_2^2
$$

对矩阵求梯度：

$$
\nabla_S\mathcal E_t=(Sk_t-v_t)k_t^\top
$$

沿负梯度更新一步，步长为 $\beta_t$：

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top
$$

也可以写成：

$$
S_t=S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top
$$

第一种形式解释行为：读出旧值，计算误差，按 key 方向写入修正。第二种形式解释状态转移：对旧状态做一个方向相关的变换，再写入新关联。

$\beta_t$ 通常由当前输入产生，控制这次更新的力度。状态更新发生在每个 token 的前向过程中；它不是推理时对整个模型权重执行一次训练。主网络学习的是如何产生 key、value、query 与更新门。

## 02 为什么 key 的范数影响覆盖与稳定性

将更新后的状态作用于同一 key：

$$
S_tk_t=S_{t-1}k_t+
\beta_t\|k_t\|_2^2(v_t-S_{t-1}k_t)
$$

若 $\|k_t\|_2=1$，则：

$$
S_tk_t=(1-\beta_t)S_{t-1}k_t+\beta_t v_t
$$

当 $\beta_t=1$，当前 key 的读取结果被精确修正为 $v_t$；当 $0<\beta_t<1$，向新值移动一部分。若 key 未归一化，$\beta_t=1$ 不再保证一步覆盖，甚至可能过冲。

单位 key 下，$I-\beta kk^\top$ 在 key 方向的特征值为 $1-\beta$，在正交方向为 1。取 $\beta\in[0,1]$ 时，这是沿 key 方向的非扩张调整。这个解释依赖范数和门范围，不能不加条件推广到任意参数化。

## 03 定向更新如何影响其他记忆

对另一个查询方向 $x$，状态变化为：

$$
(S_t-S_{t-1})x=\beta_t(v_t-S_{t-1}k_t)(k_t^\top x)
$$

若 $x$ 与 $k_t$ 正交，该更新不影响对 $x$ 的读取；若二者相关，就会有干扰。因此，“定向”并不等于“永远只修改一个离散变量”。多个自然语言关联必须映射到足够可区分的方向，有限维状态仍有容量限制。

用最简单的二维 key、标量 value 展示：旧状态 $S=[10,20]$，分别表示两个正交方向的值。用 $k=(1,0),v=30,\beta=1$ 更新，得到 $S'=[30,20]$；第一项覆盖，第二项保留。

若改用与两个坐标都相关的 key，修正会同时改变两者。模型没有获得一张无冲突的无限字典，获得的是更灵活的连续记忆更新规则。

## 04 Gated DeltaNet：先遗忘，再按新状态计算误差

### 4.1 遗忘与修正作用于不同方向

Delta 更新只沿当前 key 清理旧映射。如果长上下文不断写入无关信息，其他方向的陈旧内容仍可能占据状态。Gated DeltaNet 引入 $\alpha_t\in(0,1)$，先构造衰减后的状态：

$$
\bar S_{t-1}=\alpha_t S_{t-1}
$$

再计算误差和修正：

$$
e_t=v_t-\bar S_{t-1}k_t,\qquad
S_t=\bar S_{t-1}+\beta_t e_tk_t^\top
$$

合起来正是论文的 gated delta rule：

$$
S_t=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top
$$

$\alpha_t$ 控制旧状态整体保留程度，$\beta_t$ 控制当前 key 方向的写入力度。$\alpha\rightarrow1$ 接近 DeltaNet；$\alpha\rightarrow0$ 可清空旧状态，再写入当前关联。[依据：论文 §3.1](https://arxiv.org/html/2412.06464#S3.SS1)。

### 4.2 遗忘之后，误差也必须使用遗忘后的旧值

一种常见误写是 $S_t=\alpha S_{t-1}+\beta(v_t-S_{t-1}k_t)k_t^\top$。它对旧状态做衰减，却仍用未衰减的读取值计算误差，与上式不同。

标量例子最容易看出：旧值 10，$\alpha=0.5,\beta=1,k=1,v=30$。正确路径先得到 5，再加误差 25，结果为 30；误写路径得到 $5+(30-10)=25$，无法完成预期覆盖。

因此，不能把任意“衰减 + Delta”两行代码认定为相同模型；相似组件的执行顺序会改变状态转移。

## 05 最小程序：覆盖、遗忘和正交保留

以下 Python 3 程序采用标准库与小矩阵，验证两种等价写法，以及整体遗忘对非当前方向的影响。它不包含模型投影、卷积和训练。

```python
from math import isclose


def update(state, key, value, alpha=1.0, beta=1.0):
    decayed = [[alpha * x for x in row] for row in state]
    predicted = [sum(x * k for x, k in zip(row, key)) for row in decayed]
    error = [v - p for v, p in zip(value, predicted)]
    return [[row[c] + beta * error[a] * key[c] for c in range(len(key))]
            for a, row in enumerate(decayed)]


old = [[10.0, 20.0]]
assert update(old, [1, 0], [30]) == [[30, 20]]
assert update(old, [1, 0], [30], alpha=0.5) == [[30, 10]]
assert update(old, [1, 0], [30], alpha=0.0) == [[30, 0]]
assert update(old, [1, 0], [30], beta=0.0) == old

key, value, alpha, beta = [0.6, 0.8], [7.0], 0.9, 0.4
actual = update(old, key, value, alpha, beta)[0]
prediction = sum(x * k for x, k in zip(old[0], key))
expanded = [alpha * old[0][c] - alpha * beta * prediction * key[c]
            + beta * value[0] * key[c] for c in range(2)]
assert all(isclose(a, b) for a, b in zip(actual, expanded))
```

这里允许 $\alpha=0$ 作为理想重置边界；实际连续门参数化可能只能逼近该值。两个公式数值相同验证代数，不能替代对长序列误差、梯度和模型质量的评测。

## 06 递归更复杂，训练为什么仍可按块组织

记：

$$
A_t=\alpha_t(I-\beta_t k_tk_t^\top),\qquad
B_t=\beta_t v_tk_t^\top
$$

则 $S_t=S_{t-1}A_t+B_t$。连续两步为：

$$
S_{t+1}=S_{t-1}A_tA_{t+1}+B_tA_{t+1}+B_{t+1}
$$

因此，一个片段可以用组合后的仿射变换描述。变换组合具有结合性，但矩阵乘法一般不可交换，token 顺序必须保留。

直接构造每个完整 $A_t$ 再做稠密连乘会很昂贵。DeltaNet 的并行化工作利用“单位矩阵减低秩项”的结构与 WY 类表示，组织块内关系和块间状态，而不是把推理循环照搬成串行训练。[DeltaNet 并行化论文](https://arxiv.org/abs/2406.06484)与[Gated DeltaNet §3.3](https://arxiv.org/html/2412.06464#S3.SS3)详细展开这一点。

模型上的关键变化仍是读取误差与门控；并行化是让这种更新能够高效训练的必要配套，不表示它退化成标准 softmax。

## 07 完整模块与成本边界

论文中的模块还包括短程因果卷积、SiLU 激活、query/key 的 L2 归一化，以及输出归一化和门控。短卷积提供邻近 token 混合，L2 归一化支撑前面的稳定方向解释，输出门控控制暴露的读取结果。它们与状态中的 $\alpha,\beta$ 不是同一个门。

对固定头维度，单 token 的读取、误差和外积更新均约为 $O(d_kd_v)$，持久状态为 $O(d_kd_v)$；短卷积还需要有界历史。所谓常量状态只针对长度依赖，不包含所有训练激活与模型参数。

| 更新规则 | 怎样处理旧记忆 | 怎样写入 |
|---|---|---|
| 基础线性累加 | 全部保留 | 直接加外积 |
| 衰减线性记忆 | 整体或按指定维度衰减 | 直接加外积 |
| DeltaNet | 沿当前 key 修正 | 写入当前读取误差 |
| Gated DeltaNet | 先整体衰减，再沿 key 修正 | 写入衰减后读取误差 |

定向修正改善重复写入，但有限状态仍会干扰；整体遗忘便于清理，也可能删除应长期保留的记忆。论文不同检索场景的结果正说明两者各有作用，而不是某个门越小或更新越强就总是更好。

## 08 从按头遗忘，走向更细的记忆控制

理解这条路线，需要分清“保留多少旧状态”和“沿哪个 key 改正多少”。$\alpha$ 控制整体保留，$\beta$ 控制当前方向的写入，二者互补；输出门只控制当步读取的可见贡献。

接下来还可以追问：一个头的所有 key 特征是否必须以相同速度遗忘？下一篇 **KDA：从 Kimi Linear 到 Kimi-K3** 将把标量遗忘扩展到更细的维度，并分析它与全局 MLA 的混合方式。

## 参考资料

1. Schlag et al.，[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)，2021。用于理解外积记忆、局部学习与 Delta 写入。
2. Yang et al.，[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)，2024。重点见方向更新和并行表示。
3. Yang et al.，[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)，2024。重点见 §3.1 的公式、§3.3 的分块与 §3.4 的模块。资料查阅于 2026-10-14。
