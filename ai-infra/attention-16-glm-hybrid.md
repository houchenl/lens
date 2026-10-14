---
title: 注意力机制-16 GLM-5.3-Flash
date: 2026-10-14T16:52:16+08:00
description: 解析 GLM-5.3-Flash 的 KDA 与稀疏 MLA 混合结构，推导 IndexPool 的逐通道池化、完整组选择、原始地址展开与尾部补齐，区分 NoPE、索引共享和状态容量的不同作用。
---

# 注意力机制-16 GLM-5.3-Flash

[KDA](/ai-infra/attention-15-kda)用固定形状矩阵管理历史，[GLM 的 DSA/IndexShare](/ai-infra/attention-9-glm-indexshare)则保留可检索的逐位置表示。GLM-5.3-Flash 将两条路线组合：多数层使用 KDA，周期性使用带索引器的稀疏 MLA，并通过 IndexPool 降低长上下文的索引候选规模。

本文重点解释“线性状态、主 KV、索引 pool”三种对象的区别，以及它们怎样共同产生当前输出。资料依据为[官方发布说明](https://z.ai/blog/glm-5.3-flash)、[模型配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)和[Transformers 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)，查阅于 **2026-10-14**。

## 00 Flash 是独立混合架构，不能当成主模型缩小版

GLM-5.3 主模型沿用 5.2 的基础模型，采用 DSA 与原生 IndexShare；Flash 从新基础模型训练，文本主干改成 45 层的 KDA/稀疏 MLA 混合，外层还有多模态组件。

| 项目 | GLM-5.3 主模型 | GLM-5.3-Flash 文本主干 |
|---|---|---|
| 架构实现族 | `glm_moe_dsa` | `glm5_next_text` |
| 主干层数 | 78 | 45 |
| 历史计算 | DSA 层与跨层选择复用 | 34 个 KDA、11 个稀疏 MLA |
| 全局位置处理 | 保留旋转位置维度 | NoPE，旋转维度为 0 |
| 索引优化重点 | IndexShare | IndexPool |

官方配置的 `layer_types` 按三个 `linear_attention`、一个 `deepseek_sparse_attention` 循环，最后再加一个线性层。0 起始稀疏层号为 3、7、11、…、43，线性层共 34 个。不能把所有 45 层都计为 DSA，也不能把尾层默认成 MLA。

外层 `Glm5NextForConditionalGeneration` 还封装多模态输入；本篇的层数、状态和成本讨论均针对 `text_config` 的文本主干，视觉编码器不包含在内。

## 01 三种历史对象，各自保存什么

| 对象 | 产生位置 | 未来怎样使用 |
|---|---|---|
| KDA 矩阵状态 | 每个线性层 | 递归更新后直接由 query 读取 |
| 主 MLA 潜在 KV | 每个稀疏全局层、每个 token | 按索引地址读取对应位置的表示 |
| 索引 pool key | 将索引 key 按组聚合 | 用于选择相关组，再展开 token 地址 |

这些对象的压缩维度不同。KDA 合并整个前缀的关联，MLA 压缩每个位置的表示，IndexPool 缩短索引器搜索的序列。将它们统一称作“KV 压缩”，容易误判哪些原始地址仍然存在。

## 02 KDA 层：状态覆盖全部前缀，不只看局部窗口

延续上一章的 $S\in\mathbb R^{d_v\times d_k}$ 约定，核心更新为：

$$
\bar S_{t-1}=S_{t-1}\operatorname{Diag}(\boldsymbol\alpha_t),\qquad
S_t=\bar S_{t-1}+\beta_t(v_t-\bar S_{t-1}k_t)k_t^\top
$$

随后通过 $S_tq_t$ 读取，并进行归一化、输出门控与投影。短程因果卷积提供局部混合，但矩阵状态包含整个已处理前缀经遗忘后的信息，因此不能把线性层简单理解成只看附近几个 token。

公开配置给出 KDA 的 64 个头、头维度 128、短卷积核宽 4。参考实现为单步生成和整段处理提供递归/分块路径，并以 FP32 保存递归状态。模型公式与底层张量可能互为转置，检查遗忘轴时应结合布局判断。

这一层不需要全局 Top-K 地址；稀疏层则负责下一节的显式历史检索。

## 03 稀疏 MLA 层：主历史仍按 token 保存

每个 token 先产生低秩潜在表示 $c_j\in\mathbb R^{r_{KV}}$，本模型 $r_{KV}=512$。各主头通过自己的投影从它产生 key、value：

$$
k_{j,i}=W_i^Kc_j,\qquad v_{j,i}=W_i^Vc_j
$$

在没有显式位置旋转干扰的条件下，主匹配可以写成：

$$
q_{t,i}^\top W_i^Kc_j=((W_i^K)^\top q_{t,i})^\top c_j
$$

这展示了 MLA 矩阵吸收的基本逻辑：可以将变换移到当前 query 侧，历史继续用潜在表示保存。实际缓存是否采用这一紧凑布局，还取决于具体实现，不能把参考路径展开后的张量形状当作唯一部署布局。

本层有 64 个主 query 头，key/value 的主头维度均为 256。选区中的各 token 仍是独立条目，IndexPool 不会将它们的主 value 永久池化成一条。

## 04 IndexPool：压缩索引 key，再选择原始位置组

### 4.1 索引 key 与池化门独立生成

设索引维度为 $d_I=128$，组大小为 $p=4$。从隐藏状态产生索引 key $k_j^I$ 和逐通道门控 logits $z_j\in\mathbb R^{d_I}$；每组还有位置偏置 $b\in\mathbb R^{p\times d_I}$。

对第 $a$ 组 $B_a=\{ap,\ldots,(a+1)p-1\}$，按通道 $u$ 计算：

$$
\alpha_{a,j,u}=
\frac{\exp(z_{j,u}+b_{j\bmod p,u})}
{\sum_{v\in B_a}\exp(z_{v,u}+b_{v\bmod p,u})},\qquad
\bar k_{a,u}^I=\sum_{j\in B_a}\alpha_{a,j,u}k_{j,u}^I
$$

softmax 沿组内 token 方向执行，不同通道拥有不同权重。它是可学习加权池化，不是简单均值，也不是先计算 query 对每个原始 key 的分数再做 max。

当前索引 query 的 32 个头与 pool key 匹配，再将 ReLU 后的点积按头权重汇总为 pool 分数：

$$
I_{t,a}=\sum_{h=1}^{H_I}\omega_{t,h}
\operatorname{ReLU}\left((q^I_{t,h})^\top\bar k_a^I/\sqrt{d_I}\right)
$$

缩放与头权重的具体组织可对照 `Glm5NextTextIndexer`。从内容写入到候选选择，与 MSA 的逐 token 打分后块内 max、MoBA 的主 key 均值路由都不同。

### 4.2 `2048` 是展开后的主条目预算

参考实现将可选择 pool 数设置为 $\lfloor K/p\rfloor$，其中 `index_topk` 为 $K=2048$。因此最多选择 512 个完整 pool，随后展开回最多 2048 个原始 token 地址。

```text
逐 token 索引 key ──每 4 个池化──> pool key
query ─────────────────────────> pool 打分 → Top-512 pool
选中 pool 的成员地址 ───────────> 最多 2048 个 token
主 MLA 逐 token 潜在 KV ────────> 按地址读取 → 主注意力输出
```

与 V4 CSA 相比，二者都对聚合条目做选择，但主读取粒度不同：CSA 直接读压缩后的主 KV；IndexPool 选择后回到逐 token 主 KV。

### 4.3 组完整后才参与竞争，尾部单独补齐

query 只能选择末尾 token 已经可见的完整 pool。否则，pool key 可能聚入未来信息，影响早期 query 的路由。未满一组的可见尾部则作为原始 token 地址补入选区。

若组大小为 4，尾部最多 3 个位置。当前参考实现的输出地址宽度因此最多为 $2048+3=2051$，无效补位用特殊地址表示，不是额外真实 token。不能仅凭字段名就认定每个 query 永远恰好读取 2048 条。

尾部补齐也不是一个固定宽度局部窗口：刚好组完整时，尾部为空，最近完整组仍按索引规则参与竞争。组内局部建模还有 KDA 和短卷积承担。

对于左侧 padding，当前实现从第一个有效 token 开始分组，而不是从原始缓存槽位 0 开始。相同真实文本不应因左 padding 数量不同而改变 pool 边界。[实现入口：`get_pooled_states`、`append_visible_tail`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)。

## 05 用十四个 token 检查地址预算与尾部

令可见 token 数为 14，组大小 4，教学 token 预算 $K=4$。完整 pool 有三组，覆盖 0—3、4—7、8—11；尾部为 12、13。若分数分别为 3、9、1，选择第二组，再附加尾部，主读取位置为 4、5、6、7、12、13。

```python
def expand_selection(visible_count, pool_size, token_budget, scores):
    full_count = visible_count // pool_size
    assert token_budget % pool_size == 0
    selected = sorted(range(full_count), key=lambda a: scores[a], reverse=True)
    selected = selected[:token_budget // pool_size]
    addresses = [j for a in selected
                 for j in range(a * pool_size, (a + 1) * pool_size)]
    addresses += list(range(full_count * pool_size, visible_count))
    return sorted(addresses)


assert expand_selection(14, 4, 4, [3, 9, 1]) == [4, 5, 6, 7, 12, 13]
assert expand_selection(3, 4, 4, []) == [0, 1, 2]
assert expand_selection(12, 4, 4, [3, 9, 1]) == [4, 5, 6, 7]
assert 2048 // 4 == 512 and 2048 + 4 - 1 == 2051
```

此标准库代码验证选择预算与尾部，不模拟门控池化和真实打分。它也展示当历史不足完整 pool 时，尾部仍可保证当前已见 token 可读。

## 06 NoPE、IndexPool 和 IndexShare 不能互换解释

NoPE（No Positional Encoding，无显式位置编码）对应全局 MLA 不旋转位置；配置为 `mla_use_nope=true`、`qk_rope_head_dim=0`。顺序信息仍通过 KDA 的有序更新、卷积及上下文隐藏状态传播。

IndexPool 缩短单个索引器的搜索序列。IndexShare 则让多个层共用一次选择，是层维度的优化。Flash 当前 `indexer_types` 全为 `full`，与主模型四层共享列表不同；不能因为两者同属 GLM-5.3，就自动套用相同索引周期。

同样，`full_attn_layers` 是线性/全局层型分类中的名称，其中的全局层实际有 DSA 选择；“full”不应被误解成每次读取全部原始历史。

mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）作用于层间残差流，不负责选历史 token；MoE 负责前馈专家计算。它们影响整体模型，但都不是 IndexPool 或 KDA 的另一种命名。

## 07 分开统计固定状态、主历史与索引搜索

一个 KDA 层的核心状态约为 $H_Kd_kd_v$ 个元素。34 个 KDA 层、64 头、维度 128，若以 FP32 计，共约 136 MiB，另有短卷积状态和元数据。

稀疏 MLA 层的主历史仍随 $T$ 增长。若使用潜在 KV 布局，其主状态量约为 $11Tr_{KV}$；IndexPool 的理论 pool key 数约为 $11T/p$。后者是否对应持久索引缓存缩小，还需检查在线状态管理。

例如当前 Transformers 通用路径保存索引 key、门控 logits 与有效位，再从这些状态重建 pool；它能实现池化选择语义，却不能仅从 `index_kpool=4` 就推导该路径的持久索引缓存减少四倍。专用推理实现可以保存已完成 pool 与有界尾组，具体容量需按实际布局核对。

对计算量，固定维度下，KDA 总体主要为 $O(34T)$；稀疏主读取约为 $O(11TK)$，忽略头维度；完整 pool 的全范围索引仍含 $O(11H_Id_IT^2/p)$ 项。池化减小搜索系数，没有消除所有二次项。

官方报告的计算与缓存改善来自混合层比例、主模型规模、NoPE、索引池化和数值格式等共同变化，不能只用一个“3:1”或“四个 key 合一”推导端到端速度倍数。

## 08 与 Kimi-K3、GLM 主模型的结构对照

| 设计问题 | GLM-5.3 主模型 | Kimi-K3 | GLM-5.3-Flash |
|---|---|---|---|
| 多数层怎样保存历史 | MLA 逐位置历史 | KDA 状态 | KDA 状态 |
| 全局读取 | DSA 稀疏选择 | Gated MLA 完整历史匹配 | 稀疏 MLA |
| 全局搜索怎样减少 | 跨层 IndexShare | 依全局 MLA 定义 | IndexPool 缩短索引候选 |
| 全局显式位置 | 有旋转维度 | NoPE | NoPE，旋转维度为 0 |

这种比较针对公开设计维度，不构成能力排名。同样是 KDA 混合，保留完整全局匹配与再加一层稀疏选择，带来的成本和遗漏风险不同。

验证时应检查整段/单步 KDA 状态一致性、pool 因果边界、padding 分组、预算换算和尾部补齐，再比较多证据检索与重复 key 更新。仅有启动成功或 mask 正确，不足以证明池外算术、历史展开和缓存字节已经真正省下。

GLM-5.3-Flash 展示了一种组合：递归状态处理连续上下文，稀疏全局层检索明确历史，IndexPool 再降低检索入口的规模。下一篇将按统一坐标比较本系列的全部路线，整理表示、位置、状态和层间共享的共同取舍。

继续阅读：[注意力机制-17 设计取舍](/ai-infra/attention-17-design-tradeoffs)。

## 参考资料

1. Z.ai，[GLM-5.3-Flash: Frontier Intelligence, Flash Cost](https://z.ai/blog/glm-5.3-flash)，官方架构介绍。
2. Z.ai，[GLM-5.3-Flash 配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)。重点见 `text_config` 的层型、NoPE、KDA 与 IndexPool 字段。
3. Hugging Face，[GLM5-Next 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)与[v5.19.0 文档](https://huggingface.co/docs/transformers/v5.19.0/model_doc/glm5_next)。用于核对池化门、地址展开和尾部处理；模型数值以实际 checkpoint 配置为准。
4. Z.ai，[GLM-5 系列官方说明](https://github.com/zai-org/GLM-5)；Moonshot AI，[Kimi-K3 配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)。用于区分主模型与混合分支，资料查阅于 2026-10-14。
