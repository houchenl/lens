# 注意力机制系列推进路线

更新日期：2026-10-14。

本路线依据用户确定的范围安排：聚焦模型设计中的注意力模块，先走完 DeepSeek 的公开注意力演进路线，再展开 GLM、Kimi、MiniMax 的其他设计。按知识依赖组织阅读顺序，在各篇中单独交代真实发布时间、研究分支和模型继承关系。

写作遵循[文章写作规范](writing-guide.md)。以下标题和文件名为计划；已有篇目保持编号，未来新增资料在正式写作前重新核对。

## 已有稿件：建立基础与 DeepSeek 起点

| 篇号 | 文章 | 已建立的知识 |
|---|---|---|
| 1 | [MHA](../ai-infra/attention-1-mha.md) | QK 匹配、softmax 与 V 聚合，多头计算与因果约束 |
| 2 | [MQA/GQA](../ai-infra/attention-2-mqa-gqa.md) | 查询头与 KV 头的共享方式 |
| 3 | [MLA](../ai-infra/attention-3-mla.md) | 潜在 KV 表示、矩阵吸收、解耦 RoPE |
| 4 | [DSA](../ai-infra/attention-4-dsa.md) | 动态 token 选择、索引器训练、索引复用和适用边界 |
| 5 | [CSA/HCA](../ai-infra/attention-5-csa-hca.md) | 重叠与非重叠的序列压缩、压缩条目选择、局部窗口与共享 KV 读取 |
| 6 | [CSA2](../ai-infra/attention-6-csa2.md) | 详见文章中的问题、机制与分析 |
| 7 | [DeepSeek-V4.1](../ai-infra/attention-7-deepseek-v41.md) | 详见文章中的问题、机制与分析 |
| 8 | [NSA](../ai-infra/attention-8-nsa.md) | 详见文章中的问题、机制与分析 |
| 9 | [GLM 与 IndexShare](../ai-infra/attention-9-glm-indexshare.md) | 详见文章中的问题、机制与分析 |
| 10 | [MoBA](../ai-infra/attention-10-moba.md) | 详见文章中的问题、机制与分析 |
| 11 | [MiniMax MSA](../ai-infra/attention-11-minimax-msa.md) | 详见文章中的问题、机制与分析 |

“已有稿件”只表示仓库中存在文章，不据此认定已经提交、推送或线上发布。

## 第一阶段：先完成 DeepSeek 路线（5—8）

| 篇号 | 计划标题 | 核心问题 | 计划文件名 |
|---|---|---|---|
| **5** | **CSA/HCA：DeepSeek-V4 如何压缩历史序列** | 已经压缩 KV 表示、筛选访问位置，为什么还要减少历史条目数？ | `attention-5-csa-hca.md` |
| **6** | **CSA2：从索引复用到跨层 KV 共享** | 多个层如何共享历史表示，同时保留不同的查询与选择？ | `attention-6-csa2.md` |
| **7** | **DeepSeek-V4.1：因果编码器—解码器与层次化索引** | 如何改变全局 KV 的生成来源，并减少后续索引层的搜索范围？ | `attention-7-deepseek-v41.md` |
| **8** | **NSA：回看 DeepSeek 的原生稀疏注意力研究支线** | 压缩、选择和局部窗口如何组合？它与 DSA、CSA/HCA 有何异同？ | `attention-8-nsa.md` |

第 5 篇直接接续 DSA。第 6、7 篇共同解释 V4.1，但职责分开：前者建立跨层状态共享规则，后者说明整体网络中 KV 来源、索引候选域与这些规则的组合。第 8 篇回顾更早发表的 NSA；它不是 V4.1 的后继版本，也不将论文研究方案直接认定为某个旗舰 checkpoint 的实际结构。

### 第 5 篇：序列维度的压缩与多种注意力的分工

- 从 MLA 的“每个位置存得更紧凑”与 DSA 的“每次选较少位置”回顾问题，定义沿 token 序列聚合历史条目的新维度。
- 展开 CSA 的可学习压缩、重叠区间、索引 key 构造与压缩后的 Top-K，说明压缩条目和原始 token 的映射。
- 展开 HCA 的更高压缩率与完整访问压缩历史，以及 SWA（Sliding-Window Attention，滑动窗口注意力）对局部细节的补充。
- 解释共享 KV 的 MQA 式读取和分组输出投影；对照 V4-Flash/Pro 的实际层型安排，避免把 V4 简化成旧 MLA 的参数调整。
- 用同一段示例历史比较条目数、访问数和持久状态，讨论压缩的信息损失与局部补偿。

### 第 6 篇：共享哪些状态，以及何时重新选择

- 定义 main KV、indexer K、indexer Q 和 Top-K 位置，区分表示共享与位置共享。
- 逐层演示 Full、Reindex、Reuse 三种模式；说明各层仍具有自己的主 query 与局部窗口状态。
- 比较 CSA2 与 CSA 的压缩方式变化，解释共享状态的来源、适用范围和兼容条件。
- 与第 4 篇的 IndexCache 做对照：独立索引器训练、共享感知训练、索引结果复用与 KV 共享分别改变什么。
- 模型结构和训练适配为主；命令行参数只用于验证结构判断，不展开框架配置大全。

### 第 7 篇：组合成完整的 V4.1 注意力网络

- 解释 CED（Causal Encoder-Decoder，因果编码器—解码器）的因果约束和全局 KV 来源；不要套用双向编码器的默认直觉。
- 说明编码部分与解码部分如何使用 CSA2，局部状态和全局状态怎样协作。
- 解释首个 Full 层构建候选池、后续 Reindex 层在池内再选择，区分候选池大小与最终 Top-K 大小。
- 分开估算初次全范围扫描与后续受限搜索；候选域固定不能推导出整个模型没有长度依赖。
- 以 V2/V3 的 MLA、V3.2 的 DSA、V4 的 CSA/HCA、V4.1 的 CSA2/CED 收束已发布模型主线。

### 第 8 篇：用 NSA 补齐研究分支

- 回到 NSA 的发表时间，介绍压缩、块选择和滑动窗口三条路径及其融合。
- 比较 token 选择、块选择、压缩条目选择的粒度与表达代价。
- 对照 NSA、DSA、CSA/HCA 的原始论文，明确相似思想、不同机制与公开证据边界。
- 只在解释选择粒度时讨论硬件约束，保持注意力模块设计的主线。

## 第二阶段：扩展已有稀疏注意力知识（9—11）

| 篇号 | 计划标题 | 核心问题 | 计划文件名 |
|---|---|---|---|
| **9** | **GLM 的稀疏注意力：从 GQA、MLA 到原生 IndexShare** | 共享哪些索引、如何训练适配，以及 GLM 各分支怎样演进？ | `attention-9-glm-indexshare.md` |
| **10** | **MoBA：Kimi 如何按内容路由历史块** | 如何把历史分块，并由查询动态选择相关块？ | `attention-10-moba.md` |
| **11** | **MSA：MiniMax-M3 的分组块稀疏注意力** | 为什么每个 GQA 组独立选择历史块，如何产生块级分数？ | `attention-11-minimax-msa.md` |

第 9 篇承接已有的跨层共享知识，第 10、11 篇承接 NSA 的块选择知识。这个阶段讲清“选择和共享历史”的不同方式，随后再引入“把历史写入递归状态”。

- **第 1—11 篇已有稿件，下一篇从第 12 篇开始**，按既定路线推进。
- **第 10 篇**：展开 MoBA（Mixture of Block Attention，混合块注意力）的块表示、内容路由、因果选择与输出计算。区分“论文报告已用于 Kimi 长上下文服务”和“某个公开模型确认内置该模块”。
- **第 11 篇**：展开 MSA（MiniMax Sparse Attention）的 GQA 组、索引分支、块内打分聚合和主注意力；对照 DSA 与 MoBA 的共享范围和选择粒度。以 M2 系列 GQA 到 M3 的变化交代这一分支；01/M1 的线性混合设计在第 13 篇回顾，不能把整个 MiniMax 家族画成单向继承链。

## 第三阶段：递归状态与线性混合设计（12—16）

| 篇号 | 计划标题 | 核心问题 | 计划文件名 |
|---|---|---|---|
| **12** | **线性注意力：从历史 KV 到递归状态** | 如何改变相似度函数，并将历史关联累积为可更新状态？ | `attention-12-linear-attention.md` |
| **13** | **Lightning Attention：MiniMax-01/M1 的线性混合设计** | 带衰减的状态如何工作，为什么还要周期性保留 softmax 注意力？ | `attention-13-lightning-attention.md` |
| **14** | **DeltaNet 与 Gated DeltaNet：记忆的修正与遗忘** | 如何从累加写入发展到误差驱动的定向更新与门控遗忘？ | `attention-14-gated-deltanet.md` |
| **15** | **KDA：从 Kimi Linear 到 Kimi-K3 的混合注意力** | 更细粒度的门控如何管理状态，为什么搭配全局 MLA？ | `attention-15-kda.md` |
| **16** | **GLM-5.3-Flash：KDA 与稀疏 MLA 的组合** | 递归状态层与可检索历史的稀疏层如何分工？ | `attention-16-glm-hybrid.md` |

- **第 12 篇**：从核特征映射和矩阵结合律推导递归形式，解释归一化状态、外积关联与记忆干扰；明确不能直接对标准 softmax 注意力交换乘法顺序。用写入多个 key-value 再读取的示例连接各篇。
- **第 13 篇**：区分线性注意力的模型公式与 Lightning Attention 的执行方式，以状态衰减和混合层安排为重点；解释 MiniMax-01/M1 的 7:1 结构。Lightning Attention 与 DSA 的 Lightning Indexer 是不同组件。
- **第 14 篇**：先推导 DeltaNet 的读取误差与定向修正，再引入 Gated DeltaNet 的遗忘控制。比较记忆覆盖、累积干扰和长上下文检索能力，为 KDA 建立直接前置知识。
- **第 15 篇**：从 Gated DeltaNet 走到 KDA 的细粒度门控；解释 Kimi Linear 的 KDA/MLA 混合与 K3 的 KDA/Gated MLA 组合，核对 NoPE、输出门控和实际层比例。K2/Thinking/K2.5 与 Kimi Linear 的时间交叉必须明确；K3 的 AttnRes 作为深度方向的信息融合另作边界说明。
- **第 16 篇**：在已理解 KDA、MLA、DSA 后，解释 GLM-5.3-Flash 的层型、池化压缩索引、NoPE 与两种状态的分工，并与 GLM-5.3 主模型、Kimi 的线性混合路线对照。关注结构与训练，避免重复整套基础推导。

## 第四阶段：按统一维度比较路线（17）

**第 17 篇：《注意力设计的取舍：表示、位置、状态与跨层共享》**，计划文件名 `attention-17-design-tradeoffs.md`。

以已经讲过的实际机制建立比较框架：

| 比较维度 | 要回答的问题 |
|---|---|
| 历史表示 | 保存独立 KV、潜在向量、压缩条目，还是递归状态？ |
| 访问方式 | 访问完整历史、局部窗口、动态 token，还是动态块？ |
| 状态更新 | 追加、聚合、衰减、误差修正分别会损失什么信息？ |
| 层间关系 | 独立计算、共享索引、共享 KV 与混合层如何配合？ |
| 表达能力 | 精确检索、多证据聚合、局部细节与长期记忆的取舍是什么？ |
| 长度依赖 | 主注意力、索引器、持久状态分别如何随长度增长？ |

画出四家的分支时间线，而不是把“MLA → DSA → 线性注意力”当成所有模型共用的升级顺序。公开架构、论文研究方案和未披露的闭源版本分别标注。

## 本轮范围与维护规则

- **第 5 篇 CSA/HCA 已有稿件，下一篇从第 6 篇 CSA2 开始**，继续完成 DeepSeek 路线。
- 第 5—7 篇解释已发布 DeepSeek 主线，第 8 篇回顾 NSA 研究支线；完成这一阶段后再展开其他模型。
- 本轮关注模型注意力设计。FlashAttention、PagedAttention、缓存卸载、服务调度与通信优化留给后续系统专题；涉及它们时只补充必要的设计约束。
- RoPE、NoPE、门控、归一化、滑动窗口和混合层比例在对应模块附近解释；未来确有独立分析需要时再安排专题，不提前堆积前置篇目。
- MoE、mHC、AttnRes、MTP 等只在影响理解时交代与注意力的边界，不将它们混称为 token 注意力机制。
- 按开源模型演进选择问题，按知识依赖决定篇目顺序。新版本若增加核心机制，先核对证据，再调整未开始的计划；已有文章保持稳定链接。
- 规划资料以本次核对日期为界，正式写作时重新核对模型 revision、配置、论文与实际模型实现；未披露的早期闭源架构不推断补齐。

## 后续写作的主要资料入口

1. DeepSeek：[V4 技术报告](https://arxiv.org/abs/2606.19348)、[V4.1 技术报告](https://arxiv.org/abs/2609.19969)、[NSA](https://arxiv.org/abs/2502.11089)。
2. GLM：[GLM-5 系列官方仓库](https://github.com/zai-org/GLM-5)、[IndexCache](https://arxiv.org/abs/2603.12201)、[GLM-5.3-Flash 配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)。
3. Kimi：[MoBA](https://arxiv.org/abs/2502.13189)、[Kimi Linear](https://arxiv.org/abs/2510.26692)、[K3 官方模型卡](https://huggingface.co/moonshotai/Kimi-K3)。
4. MiniMax：[01 技术报告](https://arxiv.org/abs/2501.08313)、[M1 技术报告](https://arxiv.org/abs/2506.13585)、[MSA](https://arxiv.org/abs/2606.13392)、[M3 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3)。
5. 基础：[Linear Attention](https://arxiv.org/abs/2006.16236)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464)。
