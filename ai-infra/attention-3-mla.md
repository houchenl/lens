---
title: 注意力机制-3 MLA
date: 2026-10-14T10:28:45+08:00
description: 系列第三篇。MLA 将多头 K/V 联合压缩为潜在向量，用矩阵吸收避免展开历史 K/V，再以解耦 RoPE 保留位置匹配。本篇从公式推导到 DeepSeek-V3 的实际维度，解释缓存如何缩小、计算如何改写，以及收益的工程边界。
---

# 注意力机制-3 MLA

> [上一篇](/ai-infra/attention-2-mqa-gqa)讨论了 MQA/GQA：通过共享 K/V 头减少 KV cache。本篇继续追问：能否保留各头不同的 K/V 映射，同时压缩历史信息的表示？DeepSeek-V2 提出的 MLA（Multi-head Latent Attention，多头潜在注意力）给出的方案是联合低秩表示、矩阵吸收与解耦 RoPE。

本文基于 [DeepSeek-V2 论文 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)、[DeepSeek-V3 技术报告 §2.1.1](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)及其[官方推理代码](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。代数推导和显存估算按这些定义展开；数值示例使用 V3 的具体配置，不代表所有 MLA 模型。

## 00 从减少 KV 头数，到压缩 KV 表示

回顾普通 MHA 的计算。当前 token 的 query 与历史 key 匹配，softmax 得到权重，再对历史 value 加权求和：

$$
S_{t,j,i}=\frac{q_{t,i}^{\top}k_{j,i}}{\sqrt{d_k}},\qquad
P_{t,j,i}=\operatorname{softmax}_{j\le t}(S_{t,j,i})
$$

$$
o_{t,i}=\sum_{j\le t}P_{t,j,i}v_{j,i}
$$

沿用本系列的记号，$S$ 是未归一化打分，$P$ 是注意力概率；$t$ 为当前位置，$j$ 为历史位置，$i$ 为头编号。$j\le t$ 表示因果约束，当前位置也参与注意力。

解码时历史 Q 已经用完，历史 K/V 则会被未来查询反复使用，所以需要缓存。设批量大小为 $B$、上下文长度为 $T$、层数为 $L$、头数为 $H$、每个元素占 $s$ 字节：

$$
M_{\mathrm{MHA}}=BTLH(d_k+d_v)s
$$

MQA/GQA 调整的是公式中的 KV 头数：从 $H$ 减少到 1 或 $G$。MLA 改变的是另一件事：**历史 K/V 由一个共享潜在向量表示，各头通过自己的投影读取它。**

理解 MLA 需要连续回答三个问题：潜在向量如何生成 K/V？能否直接用它做注意力？加入位置编码后，这种计算还能成立吗？

## 01 联合低秩压缩：一份潜在表示，两种读取方式

### 1.1 先统一符号与方向

前两篇用整段输入的行向量写法 $Q=XW_Q$。本篇为了逐 token 推导方便，采用论文的列向量写法 $q=W_Qh$。两者互为转置约定，运算含义一致。

| 符号 | 含义 | 维度 |
|---|---|---|
| $h_t$ | 当前层的 token 输入 | $d$ |
| $c_t^{KV}$ | K/V 共享潜在向量 | $d_c$ |
| $k_{t,i}^C$ | 第 $i$ 个头的内容 key | $d_k^C$ |
| $v_{t,i}$ | 第 $i$ 个头的 value | $d_v$ |
| $q_{t,i}^R$、$k_t^R$ | 施加 RoPE 的 query/key 分支 | $d_R$ |
| $c_t^Q$ | query 的潜在向量 | $d_c^Q$ |

上标 $C$ 表示内容分支，$R$ 表示 RoPE 分支。为便于推导，先省略潜在向量上的 RMSNorm，后面再对照实现补上。

### 1.2 K 和 V 共用一个低维瓶颈

MLA 先将隐藏状态降维：

$$
c_t^{KV}=W^{DKV}h_t,\qquad
W^{DKV}\in\mathbb{R}^{d_c\times d}
$$

然后从同一个潜在向量生成所有头的内容 K 和 V：

$$
k_{t,i}^C=W_i^{UK}c_t^{KV},\qquad
v_{t,i}=W_i^{UV}c_t^{KV}
$$

其中 $W_i^{UK}\in\mathbb{R}^{d_k^C\times d_c}$，$W_i^{UV}\in\mathbb{R}^{d_v\times d_c}$。$D$ 表示 down-projection，$U$ 表示 up-projection。

**联合压缩**的含义是：K、V 的降维阶段共享同一个 $c_t^{KV}$，随后采用不同的上投影。将各头拼起来，省略归一化时可以写成：

$$
\begin{bmatrix}k_t^C\\v_t\end{bmatrix}
=
\begin{bmatrix}W^{UK}\\W^{UV}\end{bmatrix}
W^{DKV}h_t
$$

这条联合线性映射经过 $d_c$ 维瓶颈，秩最多为 $d_c$，所以称为低秩表示。

### 1.3 共享潜在向量，为什么仍然是多头

MQA 让所有头直接共享相同的 K/V。MLA 中，各头虽然共用 $c_t^{KV}$，却有自己的 $W_i^{UK}$ 和 $W_i^{UV}$：同一份潜在信息经过不同映射，可以生成不同的内容 K/V。

各头的 query、注意力分布和输出也仍然不同。共享发生在表示的瓶颈处，各头的读取方式保留下来。

这一结构仍有表达约束。MLA 通常在训练时就采用这种参数化，让模型学习适合瓶颈的表示；它不保证能无损重建任意已有 MHA 模型的 K/V。已有模型的迁移需要额外方法，例如 [MHA2MLA](https://arxiv.org/abs/2502.14837)研究了低秩近似、RoPE 调整与后续训练。

## 02 矩阵吸收：直接在潜在空间中计算

如果只缓存潜在向量，却每一步都展开所有历史 K/V，节省显存的同时会增加重建开销。MLA 的关键是改写计算顺序，避开这种重建。

### 2.1 K 上投影移到当前 query 一侧

先暂时忽略 RoPE。当前 query 与历史内容 key 的点积为：

$$
q_{t,i}^{C\top}k_{j,i}^C
=q_{t,i}^{C\top}W_i^{UK}c_j^{KV}
=\left(W_i^{UK\top}q_{t,i}^C\right)^\top c_j^{KV}
$$

定义潜在空间中的 query：

$$
\widetilde q_{t,i}=W_i^{UK\top}q_{t,i}^C\in\mathbb{R}^{d_c}
$$

则内容打分变成 $\widetilde q_{t,i}^{\top}c_j^{KV}$。当前 query 变换一次，就能与所有历史潜在向量匹配，无需显式生成历史的内容 K。

这里用的是结合律和转置关系，**没有交换两个矩阵的顺序**。这是论文所说的 K 上投影吸收在单个头上的展开形式。

### 2.2 V 上投影移到加权求和之后

Value 聚合也可以改写：

$$
o_{t,i}
=\sum_{j\le t}P_{t,j,i}W_i^{UV}c_j^{KV}
=W_i^{UV}\left(\sum_{j\le t}P_{t,j,i}c_j^{KV}\right)
$$

令：

$$
z_{t,i}=\sum_{j\le t}P_{t,j,i}c_j^{KV}
$$

就能先在潜在空间聚合，再做一次 $o_{t,i}=W_i^{UV}z_{t,i}$。由于 $W_i^{UV}$ 对所有历史位置相同，它可以移到求和外面。

进一步把输出矩阵按头拆为 $W^O=[W_1^O\;\cdots\;W_H^O]$ 的横向分块，即各 $W_i^O$ 分别作用于一个头，输出为：

$$
u_t=\sum_{i=1}^{H}W_i^Oo_{t,i}
=\sum_{i=1}^{H}\left(W_i^OW_i^{UV}\right)z_{t,i}
$$

因此 V 上投影在数学上还能与输出投影合并。实现时不必物化这些合并矩阵：矩阵尺寸、权重存储和 kernel 效率都会影响选择。官方代码保留了“先聚合潜在向量，再做 V 上投影，最后输出投影”的计算顺序。

### 2.3 等价性的范围

上述改写对同一组 MLA 参数是代数等价的，softmax 的输入没有改变；不能据此宣称 MLA 与任意 MHA 模型等价。

还要注意：$P$ 由打分经过非线性的 softmax 得到。可以移出求和的是固定的 V 投影，不能随意把投影移过 softmax。

## 03 RoPE 为什么会阻碍吸收

[RoPE](https://arxiv.org/abs/2104.09864)通过位置相关的旋转矩阵作用于 Q/K。若位置 $t$ 和 $j$ 对应 $R_t$ 和 $R_j$，直接对内容分支施加旋转，会得到：

$$
q_{t,i}=R_tq_{t,i}^C,\qquad
k_{j,i}=R_jW_i^{UK}c_j^{KV}
$$

点积变成：

$$
q_{t,i}^{\top}k_{j,i}
=q_{t,i}^{C\top}R_t^{\top}R_jW_i^{UK}c_j^{KV}
$$

与上一节相比，query 与上投影之间多了 $R_t^{\top}R_j$。它依赖历史位置 $j$，一般不能与 $W_i^{UK}$ 交换顺序。

因此无法用同一个 $\widetilde q_{t,i}$ 查询全部历史潜在向量。朴素实现需要重建历史 K；保存完整旋转后 K 则又扩大缓存。

准确的结论是：**常规 RoPE 阻碍了这里的固定 K 上投影吸收方式。** 低秩表示仍能定义，困难在于如何高效计算。

## 04 解耦 RoPE：内容与位置分别打分

### 4.1 单独设置一个较小的位置分支

MLA 对内容 Q/K 不显式施加 RoPE，另设一组携带旋转位置编码的向量：

$$
q_{t,i}^R=R_tW_i^{QR}c_t^Q,\qquad
k_t^R=R_tW^{KR}h_t
$$

完整 Q/K 是两个分支的拼接：

$$
q_{t,i}=[q_{t,i}^C;q_{t,i}^R],\qquad
k_{j,i}=[k_{j,i}^C;k_j^R]
$$

各头有自己的位置 query，但位置 key $k_j^R$ 在头之间共享。拼接后的点积自然拆成两项：

$$
q_{t,i}^{\top}k_{j,i}
=q_{t,i}^{C\top}k_{j,i}^C+q_{t,i}^{R\top}k_j^R
$$

内容项使用矩阵吸收，位置项直接读取小维度的位置 key。由此，每层每个 token 只需缓存：

$$
\boxed{c_j^{KV}\quad\text{与}\quad k_j^R}
$$

对应元素数是 $d_c+d_R$。

“解耦”描述的是计算分支。它不意味着内容向量完全没有位置相关信息：输入 $h_t$ 已经经过前面的层，可能携带上下文与位置信息。

### 4.2 Query 为什么也要压缩

论文还对 query 使用低维瓶颈：

$$
c_t^Q=W^{DQ}h_t,\qquad q_{t,i}^C=W_i^{UQ}c_t^Q
$$

位置 query 同样从 $c_t^Q$ 投影得到。其目的主要是减少训练中的激活内存。**Q 压缩不会进一步减少历史 KV cache**，因为普通增量解码本来就不需要缓存历史 Q。

### 4.3 压缩缓存下的完整注意力

将所有步骤合起来：

$$
S_{t,j,i}
=\frac{\widetilde q_{t,i}^{\top}c_j^{KV}+q_{t,i}^{R\top}k_j^R}
{\sqrt{d_k^C+d_R}}
$$

$$
P_{t,j,i}=\operatorname{softmax}_{j\le t}(S_{t,j,i}),\qquad
z_{t,i}=\sum_{j\le t}P_{t,j,i}c_j^{KV}
$$

随后进行 V 上投影与输出投影。分母使用的是原始拼接 Q/K 的维度；将内容 query 变换为 $d_c$ 维后，不能擅自改为 $\sqrt{d_c}$，否则会改变注意力分布。长上下文方案还可能调整缩放，实际值应以模型实现为准。

## 05 DeepSeek-V3：576 个元素如何替代展开的 K/V

### 5.1 用实际配置计算

[DeepSeek-V3 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)给出的主要维度如下：

| 参数 | 配置字段 | 数值 |
|---|---|---:|
| 隐藏维度 $d$ | `hidden_size` | 7168 |
| 注意力头数 $H$ | `num_attention_heads` | 128 |
| 内容 Q/K 维度 $d_k^C$ | `qk_nope_head_dim` | 128 |
| RoPE 维度 $d_R$ | `qk_rope_head_dim` | 64 |
| Value 维度 $d_v$ | `v_head_dim` | 128 |
| KV 潜在维度 $d_c$ | `kv_lora_rank` | 512 |
| Query 潜在维度 $d_c^Q$ | `q_lora_rank` | 1536 |
| 主模型层数 $L$ | `num_hidden_layers` | 61 |

潜在缓存加位置缓存，每层每 token 需要：

$$
d_c+d_R=512+64=\boxed{576\text{ 个元素}}
$$

如果像官方 `naive` 路径一样，逐头保存完整 K（128 维内容加 64 维位置）与 V（128 维），则需要：

$$
H(d_k^C+d_R+d_v)=128(128+64+128)=40960
$$

这个展开布局逐头复制了共享的位置 K。相对它，压缩布局的元素数缩小约 $40960/576=71.1$ 倍，减少约 **98.59%**。

这个数是特定维度和缓存布局的理论比较，不是所有模型都适用的固定压缩率，也不是模型间的实测加速率。

### 5.2 为什么还会看到 98.24% 和 93.3%

如果比较的是 128 个头、K/V 各 128 维的普通 MHA，基准为 $128(128+128)=32768$ 个元素，减少比例便是：

$$
1-\frac{576}{32768}=98.2421875\%
$$

而 V2 论文报告的 **93.3%** 来自 DeepSeek-V2 与前代 DeepSeek 67B 的模型级比较，两者的层数和注意力配置并不相同。[论文摘要](https://arxiv.org/abs/2405.04434)也明确了比较对象。

| 比较方式 | 基准 | 减少比例 |
|---|---|---:|
| V3 压缩缓存 vs 逐头完整展开 | 每层每 token 40960 个元素 | 约 98.59% |
| V3 压缩缓存 vs 指定的普通 MHA | 每层每 token 32768 个元素 | 约 98.24% |
| V2 vs DeepSeek 67B | 两个具体模型的整体 KV cache | 论文报告 93.3% |

读“减少多少”之前，先看分母是什么。

### 5.3 128K 上下文的缓存显存

假设 $B=1$、$T=131072$（这里的 128K 指 $128\times1024$）、$L=61$，所有缓存元素使用 BF16，每元素 2 字节：

$$
M_{\mathrm{MLA}}
=BTL(d_c+d_R)s
=131072\times61\times576\times2
$$

$$
\frac{M_{\mathrm{MLA}}}{2^{30}}=\boxed{8.578125\text{ GiB}}
$$

同样假设下，逐头完整展开的布局为 610 GiB。这里统一用 $1\text{ GiB}=2^{30}$ 字节。

这是主模型注意力缓存的理想有效载荷，未计权重、临时激活、缓存块的预留与碎片、额外预测层及跨设备复制。多卡部署时，按 query 头做 TP 并不会自动将共享潜在缓存均分到所有卡；实际每卡占用还取决于缓存如何分片或复制。

## 06 对照官方实现：两个缓存，三次关键变换

官方 `inference/model.py` 的 `MLA` 类提供 `naive` 与 `absorb` 两条路径。`absorb` 路径保存的张量形状为：

```text
kv_cache: [batch, sequence, kv_lora_rank]       # V3：512
pe_cache: [batch, sequence, qk_rope_head_dim]  # V3：64
```

下面用公式对应的变量整理其计算顺序。它是说明步骤的伪代码，省略了批量、因果掩码和并行维度：

```python
# 当前 token：生成潜在表示与位置 key
c_kv = kv_norm(W_DKV @ h_t)
k_rope = rope(W_KR @ h_t, position=t)
cache.append(c_kv, k_rope)

# 每个 query 头：内容匹配 + 位置匹配
q_latent = W_UK_i.T @ q_content_i
S = (C_cache @ q_latent + K_rope_cache @ q_rope_i) * scale
P = softmax(S)

# 先聚合潜在向量，再执行 value 上投影
z_i = P @ C_cache
o_i = W_UV_i @ z_i
# 所有头的 o_i 拼接后，再过输出投影
```

代码中的三处对应关系是：K 上投影作用于当前 query；同一个潜在缓存既用于内容打分，也用于 value 聚合；V 上投影发生在聚合之后。

实际潜在向量经过 RMSNorm，所以更准确地写为 $c_t^{KV}=\operatorname{RMSNorm}(W^{DKV}h_t)$。后续 K/V 仍是该向量的线性上投影，第二节的吸收推导仍成立。加上归一化后，从 $h_t$ 到 K/V 的整体映射包含非线性，第一节的低秩线性分解应理解为省略归一化时的结构说明。

另一个值得注意的配置细节是：V3 的 `num_key_value_heads` 也写着 128。只比较这个字段与 `num_attention_heads`，会把它误判成普通 MHA。对 MLA，需要结合潜在维度、解耦 RoPE 字段和模型类实现来判断。

## 07 与 MHA、MQA、GQA 的关系

按每层每 token 的缓存元素数比较：

| 方案 | 共享的对象 | 缓存元素数 |
|---|---|---:|
| MHA | 各头保留自己的 K/V | $H(d_k+d_v)$ |
| GQA | 每组 query 共享一对 K/V | $G(d_k+d_v)$ |
| MQA | 全部 query 共享一对 K/V | $d_k+d_v$ |
| MLA | 各头共用潜在表示，使用不同的上投影；位置 K 共享 | $d_c+d_R$ |

MLA 的缓存并不必然最小。例如 K/V 各 128 维的 MQA 只需 256 个元素，少于 V3 MLA 的 576 个元素。两者的表达约束不同，质量也不能仅由缓存大小推断。

同样，MLA 在特定实验中能达到与 MHA 相当或更好的质量，不意味着对所有规模、数据和训练预算都有这样的保证。结构性能需要通过相应模型的训练和评测来确认。

## 08 工程收益与边界

### 8.1 显存节省不等于同比例加速

缓存有效载荷减少，能为长上下文和更多并发腾出空间；带宽受限时，减少历史数据的搬运也可能改善吞吐。

但吸收后的内容 query 是 $d_c$ 维。V3 中，这意味着从 128 维变成 512 维：内容打分与潜在聚合的运算宽度增加。投影、权重读取、设备通信和 kernel 效率也都需要计入。

因此缓存缩小 71 倍，不能推导出生成速度提升 71 倍。应分别报告缓存显存、吞吐、首 token 延迟与逐 token 延迟，并给出硬件、批量和上下文条件。

### 8.2 Prefill 与 decode 可以采用不同计算路径

Prefill 一次处理许多 query，适合较大的矩阵计算；decode 通常每条序列只处理一个新 query，更容易受到历史缓存访问的影响。两阶段不必选择同一条计算路径。

例如可以在需要时临时展开 K/V 使用合适的注意力 kernel，或直接使用潜在空间的专用 kernel。临时展开会增加工作空间，但不要求持久缓存也采用展开布局。具体选择要结合实现测量，不能只根据公式中的维度判断。

### 8.3 MLA 保留稠密注意力的长度依赖

MLA 仍然对历史位置打分、softmax、加权求和。完整长度为 $T$ 的稠密注意力仍有 $O(T^2)$ 的位置配对；单步 decode 的注意力工作量仍随历史长度 $T$ 增长。

它改变了每个位置的表示宽度和计算顺序，没有把稠密注意力自动变成线性注意力或稀疏注意力。

### 8.4 缓存管理和量化仍然有意义

分页缓存管理与连续批处理可以管理这些压缩向量的分配和调度；缓存量化可以进一步减少每个元素的字节数。但都需要相应后端支持，量化还需要检查误差与性能。

MLA 与 MQA/GQA 都属于注意力结构选择。它们的压缩倍数不能像两个独立开关一样直接相乘；混合设计需要重新定义并验证计算。

## 09 两个等式串起整个设计

MLA 最值得理解的两个等式是：

$$
\boxed{q^{\top}(Wc)=(W^{\top}q)^{\top}c}
$$

$$
\boxed{\sum_j P_j(Wc_j)=W\left(\sum_j P_jc_j\right)}
$$

第一个让当前 query 直接查询历史潜在向量；第二个让 value 上投影推迟到聚合之后。解耦 RoPE 为这种改写保留了一个可处理的位置分支。

因此，MLA 的完整计算链可以写成：

```text
当前隐藏状态
  → 潜在 K/V + 位置 K，写入两个缓存
  → 当前 query 分成内容与位置两支
  → 内容 query 变换到潜在空间
  → 内容点积 + 位置点积 → softmax
  → 对历史潜在向量加权求和
  → value 上投影 → 多头输出投影
```

理解这条链，再读推理引擎中的 MLA prefill/decode 后端、缓存布局和专用 kernel，就能分清哪些步骤由模型结构决定，哪些是实现对显存、计算与通信的取舍。

下一篇 [DSA](/ai-infra/attention-4-dsa)继续讨论：压缩每个历史位置的表示之后，如何用轻量索引器减少主注意力实际访问的位置，以及如何跨层复用筛选结果。

## 参考资料

1. DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024。重点见 §2.1 与附录 C。
2. DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024。重点见 §2.1.1 与 §4.2。
3. DeepSeek-AI, [DeepSeek-V3 官方推理实现：model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。重点见 `MLA` 类的 `naive` / `absorb` 路径。
4. DeepSeek-AI, [DeepSeek-V3 官方模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)。用于本文的维度与显存计算。
5. Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021。
6. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023。
7. Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019。
8. Ji et al., [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/abs/2502.14837), 2025。
