---
title: 注意力机制-6 CSA2
date: 2026-10-14
description: 从层内压缩走向跨层共享。本文解释 CSA2 的 Full、Reindex、Reuse 三种模式，推导主 KV、索引 key 与 Top-K 的状态关系，分析压缩器简化、缓存归属、训练适配及其与 IndexCache 的区别。
---

# 注意力机制-6 CSA2

[上一篇](/ai-infra/attention-5-csa-hca)讨论了如何把多个 token 合并成一条历史记录。即使序列已经压缩，如果数十层仍分别保存全局 KV、分别生成索引 key、分别选择历史，模型依然会维护大量重复的状态和选择过程。DeepSeek-V4.1 的 CSA2（Compressed Sparse Attention 2，第二代压缩稀疏注意力）将压缩推进到层维度：让多个层共享同一份全局历史，同时决定哪些层有必要重新选择。

本文重点回答三个问题：共享的到底是向量还是地址？不同层怎样读取同一份历史却产生不同输出？共享后哪些开销消失，哪些仍然存在？V4.1 的整体因果编码器—解码器和层次化候选搜索留到下一篇。

依据为 [V4.1 技术报告 §2.3](https://arxiv.org/html/2609.19969#S2.SS3)、[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)和[参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)，查阅于 **2026-10-14**。代码的 `main` 分支会更新；以下小例子验证结构，不代表模型质量或速度实测。

## 00 层间相似性，为什么可以有不同的利用方式

假设相邻几层都在处理同一段代码。第一层已经找到了函数定义、调用点和相关约束，后面一层可能继续使用这些位置，也可能转向另一个函数。但“仍看这些位置”和“仍使用同一组历史表示”是两个独立决定。

IndexCache 主要复用 Top-K 地址，减少重复选择；如果各层仍产生自己的 KV，缓存容量不会因此减少。CSA2 同时考虑地址和表示，让后续层既可以沿用选择，也可以在共享历史上重新选择。

这种共享是模型结构的一部分。不同层的隐藏空间不会天然相同，直接拿一个独立训练模型中某层的 KV 替换另一层，通常改变其计算含义。CSA2 的查询、共享源和消费者需要在训练中共同适配。

## 01 四种对象：先把“共享什么”说清楚

令 $l$ 为层号，$t$ 为当前 token，$s$ 为全局条目编号。主头维度为 $c$，索引头维度为 $d_I$；所有向量采用列向量约定。

| 对象 | 记号与维度 | 作用 |
|---|---|---|
| 主全局 KV | $U^{(l)}\in\mathbb R^{n\times c}$ | 保存供主注意力读取的历史，单条向量同时充当 K 和 V |
| 索引 key | $K_I^{(l)}\in\mathbb R^{n\times d_I}$ | 为全局条目建立较低成本的匹配表示 |
| 索引 query | $q^I_{l,t,a}\in\mathbb R^{d_I}$ | 表达当前层、当前 token 的选择需求，$a$ 是索引头 |
| Top-K 地址 | $\mathcal S_{l,t}\subseteq\{0,\ldots,n-1\}$ | 指定本次主注意力读取哪些全局条目 |

主 query $q_{l,t,i}\in\mathbb R^c$ 与局部窗口 KV 是另外两类对象，始终由当前层生成。共享 Top-K 地址也不表示共享主注意力概率；地址只规定候选，概率仍由每个主 query 头重新计算。

为避免把缓存地址当成通用语义，本文将“一份全局 KV 及其对齐的索引 key”视为同一个共享域。地址 $s=3$ 只有相对于这个共享域才有意义；切换全局 KV 来源后，旧地址不能不加判断地继续使用。

## 02 先简化写入路径：CSA2 怎样生成历史

### 2.1 去掉重叠与组内位置偏置

CSA 使用重叠的两组内容投影和门控，并在聚合 logits 中加入组内位置偏置。CSA2 改为非重叠门控池化，去掉这项聚合用的绝对位置偏置。

设压缩步长为 $m$，第 $s$ 组为 $B_s=\{sm,\ldots,(s+1)m-1\}$，输入隐藏维度为 $d$。省略批量，写入投影为：

$$
u_j=W_u h_j,\quad z_j=W_z h_j,\qquad
W_u,W_z\in\mathbb R^{c\times d}
$$

对每个通道 $b$ 独立池化：

$$
\alpha_{s,j,b}=\frac{\exp z_{j,b}}{\sum_{v\in B_s}\exp z_{v,b}},\qquad
v_{s,b}=\sum_{j\in B_s}\alpha_{s,j,b}u_{j,b}
$$

然后得到未施加旋转位置编码的归一化 latent（潜在向量）：$\widehat u_s=\operatorname{RMSNorm}(v_s)$。RMSNorm 指均方根归一化。每完成 $m$ 个 token 产生一条记录，未完成组暂存，不能提前对过去 query 开放。

去掉聚合中的位置偏置，不等于去掉注意力的位置编码。后续主 KV、索引 key 和 query 仍有自己的位置处理。V4.1 的编码部分使用 $m=2$，解码部分使用 $m=1$；后者直接投影、归一化，不需要门控池化。

### 2.2 从同一 latent 派生主 KV 和索引 key

CSA 的索引器有独立压缩器，直接从隐藏状态产生索引 key；CSA2 将索引 key 改为从主压缩 latent 投影：

$$
k_s^{I,\mathrm{pre}}=\operatorname{RMSNorm}(W_I\widehat u_s),\qquad
W_I\in\mathbb R^{d_I\times c}
$$

这使两个历史空间具有明确的来源关系。实现中先从未旋转的 $\widehat u_s$ 产生索引 key，再分别执行索引 key 的位置/量化处理，以及主 KV 的位置/量化处理。不能把“从主 KV 派生”误读成必须读取已经量化、已经旋转的缓存字节再投影。

这些变化可对照[参考代码中的 `Compressor`、`Indexer` 与 `_compress_kv`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)。它们减少独立生成路径，也为主 KV 与索引 key 成对共享建立了基础。

## 03 三种静态模式：谁写历史，谁重新选择

### 3.1 Full：生成共享域，也产生第一组选择

Full 模式生成主全局 KV、由 latent 派生索引 key，并使用当前层自己的索引 query 打分、取 Top-K。它是一个共享域的生产者。

“Full”描述组件齐全，主注意力仍是稀疏读取，不能把它解释成对全部原始 token 做完整注意力。该层同样生成自身的主 query、局部窗口和输出投影。

### 3.2 Reindex：表示保持共享，选择随层变化

Reindex 使用最近可用 Full 层的主 KV 和索引 key，却重新生成索引 query。设共享域来自层 $f$，其打分为：

$$
I_{l,t,s}=\sum_a\omega_{l,t,a}
\operatorname{ReLU}\left((q^I_{l,t,a})^\top k^I_{f,s}\right),\qquad
\mathcal S_{l,t}=\operatorname{TopK}_s I_{l,t,s}
$$

其中 $\omega$ 是当前层产生的索引头权重，缩放并入其中，ReLU 表示保留正部。候选集合还受因果边界约束；V4.1 解码器进一步施加层次化候选限制，下一篇再展开。

改变 query 就能在同一索引空间中改变检索需求。因此，共享 KV 并不要求多层选中相同位置。

### 3.3 Reuse：沿用最近一次选择，继续独立计算主注意力

Reuse 沿用当前共享域及其最近一次由 Full 或 Reindex 产生的 Top-K，不生成索引 query，也不重新计算索引分数。它仍计算本层主 query 与局部 KV，并完成主注意力和输出映射。

| 模式 | 主全局 KV | 索引 key | 索引 query / Top-K | 主 query / 局部 KV |
|---|---|---|---|---|
| Full | 本层生产 | 本层派生 | 本层计算 | 本层计算 |
| Reindex | 共享 Full 来源 | 共享同一来源 | 本层重新计算 | 本层计算 |
| Reuse | 共享 Full 来源 | 无需重新打分 | 复用最近选择 | 本层计算 |

模式由模型层配置静态指定，不是根据当前 token 临时路由。各层的职责见[技术报告 §2.3.1](https://arxiv.org/html/2609.19969#S2.SS3.SSS1)。

## 04 用六层例子跟踪“最新状态”

考虑一个教学安排：Full → Reuse → Reindex → Reuse → Full → Reuse。假设当前 token 可见六个全局条目，$K=2$。

| 层 | 模式 | 使用的共享域 | 本层使用的 Top-K |
|---|---|---|---|
| 0 | Full | 新建 $U_A,K_A^I$ | 算得 $\{1,4\}$ |
| 1 | Reuse | $A$ | 沿用 $\{1,4\}$ |
| 2 | Reindex | $A$ | 重新算得 $\{0,4\}$ |
| 3 | Reuse | $A$ | 沿用 $\{0,4\}$ |
| 4 | Full | 新建 $U_B,K_B^I$ | 算得 $\{2,5\}$ |
| 5 | Reuse | $B$ | 沿用 $\{2,5\}$ |

层 3 复用的是层 2 的选择，而不是一直追随层 0。层 5 必须使用共享域 $B$ 的地址；即使它也恰好选择编号 4，该编号也指向 $U_B$ 的第 4 条。

下面的 Python 3 程序不依赖第三方包，只验证来源与选择的绑定规则，不模拟注意力数值：

```python
modes = ['Full', 'Reuse', 'Reindex', 'Reuse', 'Full', 'Reuse']
fresh = {0: (1, 4), 2: (0, 4), 4: (2, 5)}
kv_owner = index_owner = None
trace = []
for layer, mode in enumerate(modes):
    if mode == 'Full':
        kv_owner = layer
    if mode in ('Full', 'Reindex'):
        assert kv_owner is not None
        index_owner = layer
        selected = fresh[layer]
    trace.append((layer, kv_owner, index_owner, selected))
assert trace[3] == (3, 0, 2, (0, 4))
assert trace[5] == (5, 4, 4, (2, 5))
```

实现层面，状态归属必须跨 token 保持正确。即使某步尚未凑齐新压缩组，Full 层也仍应选择自己的已有索引缓存，不能因为本步没有新 latent，就残留上一个共享域的缓存引用。模型定义和缓存调度分别描述“应共享谁”与“本步实际读到谁”，两者都影响正确性。

## 05 表示共享以后，输出为什么仍然不同

令 $\mathcal W_{l,t}$ 为本层的局部窗口条目，主注意力的候选为共享全局条目 $U_f[\mathcal S_{l,t}]$ 与 $\mathcal W_{l,t}$。两路进入同一次归一化；将条目记为 $x_a$，省略位置处理细节：

$$
p_{l,t,i,a}=\frac{\exp(q_{l,t,i}^\top x_a/\sqrt c)}
{\exp\beta_{l,i}+\sum_b\exp(q_{l,t,i}^\top x_b/\sqrt c)},\qquad
o_{l,t,i}=\sum_a p_{l,t,i,a}x_a
$$

$\beta_{l,i}$ 是该头的可学习 sink logit，相当于零 value 的额外归一化项。即使两个层使用相同全局条目和地址，它们的主 query、局部 KV、sink 与输出投影不同，输出仍会不同。

这可以理解为多个层在同一个历史表示空间里提出不同问题。共享限制了历史写入自由度，却保留了读取与后续变换的自由度；是否足够表达目标任务，取决于训练和共享间隔，不能只从“输出不同”推出没有能力损失。

## 06 缓存与索引分别节约多少

设一个共享组包含 $g$ 层，产生 $n$ 条全局记录，其中 $r$ 层运行索引器，主条目维度 $c$、索引维度 $d_I$、局部窗口 $w$。用统一精度的元素数计，不含临时状态：

$$
M_{\mathrm{independent}}=g\bigl[n(c+d_I)+wc\bigr],\qquad
M_{\mathrm{shared}}=n(c+d_I)+gwc
$$

全局部分从 $g$ 份变成一份，局部窗口仍是每层一份。若 $g=6,n=1000,c=512,d_I=128,w=128$，独立布局需要 4,233,216 个元素，共享布局需要 1,033,216 个元素。全局部分减少至 $1/6$，总量减少比例还受窗口占比影响。

对每个 query，若每次索引仍搜索整个 $n$ 条历史，索引打分约从 $gH_I d_I n$ 变为 $rH_I d_I n$。例如 $g=6,r=2$，索引打分次数减少至三分之一，但六个层的主注意力仍要执行六次。

这两个式子分别对应缓存共享与选择复用，不能互相替代。也不能把它们直接当作全模型显存或端到端延迟的下降比例；权重、激活、投影、局部路径以及数值格式都未包含在内。

## 07 实际配置怎样表达共享关系

V4.1-Flash 的文本配置位于外层 `text_config`。主干有 40 层，层号从 0 开始，关键来源集合为：

| 字段 | 官方值 | 含义 |
|---|---|---|
| `kv_source_layer_ids` | `[2, 8, 14, 20]` | 生成全局主 KV 与索引 key 的 Full 层 |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` | 重新产生选择的 Full 或 Reindex 层 |
| `index_topk` | `512` | 最终全局主注意力最多读取的条目数 |
| `index_n_heads / index_head_dim` | `32 / 128` | 索引打分的头数和维度 |
| `head_dim / sliding_window` | `512 / 128` | 主 KV 条目维度和每层局部窗口 |

由此可读出：0—1 层只有局部窗口；2、8、14、20 是 Full；24、28、32、36 是 Reindex；其余主干层是 Reuse。不能把 `index_source_layer_ids` 理解成“这些层各有一份全局 KV”。

同样，这不是一个 `index_topk_freq` 就能完整表达的周期结构。修改 Hugging Face 配置覆盖值不会自动增加缺失的压缩器或索引参数，也不会把独立层权重变成已经训练适配的共享权重。

## 08 训练与适用边界

共享一个历史表示意味着来自多个消费者的训练信号会作用到同一生产路径。训练不能把它当作若干互不相关的缓存副本；需要保持参数归属、激活引用和梯度关系一致。技术报告 §3.1.2 还介绍了跨流水阶段的共享状态支持，本文只取其模型含义：物理执行副本不等于多个独立学习的共享源。

Reindex 为更深层提供调整选择的机会，Reuse 则接受近期选择的约束。更大的共享组可能节约更多状态，也会减少层间独立写入和选择的自由度。容量推导无法单独决定最佳共享间隔，需要在相同训练预算和目标任务上比较。

CSA2 因而同时改变三条路径：简化单层历史写入，减少多层历史副本，减少多层重复选择。下一篇将把这些层放回 **V4.1 的因果编码器—解码器与层次化索引器**，解释全局历史由哪里产生，以及剩余索引器如何避免反复全范围搜索。

## 参考资料

1. DeepSeek-AI，[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969)，2026。重点见 §2.3.1、§3.1.2、§4.2.1。
2. DeepSeek-AI，[V4.1-Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)与[官方参考模型代码](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)。滚动分支，查阅于 2026-10-14。
3. Bai et al.，[IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)。用于比较索引复用与 KV 表示共享。
