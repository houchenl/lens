---
title: KVCache卸载-1 LMCache
date: 2026-10-14T13:48:41+08:00
description: 《KVCache卸载》系列第一篇。从跨请求复用出发，拆解 LMCache 的分块匹配、外部存储、vLLM Connector 与回载流程，再用成本模型和可复核的实验步骤解释卸载何时值得做。
---

# KVCache卸载-1 LMCache

> 《KVCache卸载》系列讨论 KV cache 的外部存储、跨请求复用与传输调度。第一篇以 LMCache 为例，分析如何延长缓存生命周期，以及如何判断复用的正确性和性能收益。

同一份长文档，换一个问题再次查询，可能需要重新计算相同的上下文。LMCache 将这部分计算结果保存到 CPU 内存、磁盘或外部存储；后续请求匹配到兼容的前缀时，再将 KV 回载到 GPU。**这种设计延长了 KV 的可复用生命周期，性能收益取决于回载成本是否低于被跳过的 prefill 成本。**

这篇文章沿着一条请求的数据路径展开：先弄清 KV 保存了什么，以及 MP 和 Connector 分别承担什么职责，再拆解分块匹配、异步存取与多机共享，最后用成本估算与分层实验判断收益。主线采用独立服务的 MP（multiprocess，多进程）模式，以 CPU 内存作为 L1（一级外部缓存）、文件系统作为 L2（二级外部缓存）。文中的性能数字为假设下的估算，接入步骤未在 GPU 上实测。

## 重复上下文为什么会产生重复计算

在[注意力机制系列](/ai-infra/attention-1-mha)中，我们讨论了 KV cache 如何产生，以及 [MLA](/ai-infra/attention-3-mla)如何压缩每个 token 的表示。这一系列转向存储与调度：缓存放在哪里、保存多久、由谁复用，以及搬运是否比重算划算。

长上下文需要保存更多历史状态，并发请求又会扩大缓存总量。GPU 显存同时承担模型权重、执行工作区和 KV cache，能长期保留的历史前缀有限。如果多个请求重复读取同一份文档或系统提示，丢弃已经生成的 KV，可能意味着下一次再支付一遍 prefill 成本。

CPU 内存和磁盘提供了更大的容量，但访问速度与 GPU 显存不同。因此，把 KV 移到外部只是起点：系统还需要识别可复用的状态，在正确的时间回载，并判断搬运能否赶上计算需求。LMCache 将这组问题抽象为推理引擎之外的缓存层。

要判断这个设计是否成立，需要回答三个问题：哪些历史 KV 可以正确复用，数据何时才能交给 GPU 使用，以及为保存和回载付出的代价是否值得。后面的架构、原理和实验都围绕这三个问题展开。

### “卸载”可能在解决不同的问题

“KV cache offloading”经常被用来描述不同的行为。它们都涉及外部存储，目标却不一样：

| 行为 | 保存的对象与时机 | 主要收益 |
|---|---|---|
| 活跃请求换出与恢复 | 暂停请求的中间状态 | 为其他请求腾出 GPU 空间，之后继续生成 |
| 跨请求缓存复用 | 已计算的上下文 KV，延长其生命周期 | 后续相同前缀减少 prefill 重算 |
| Prefill/Decode 分离 | 将 prefill 产生的 KV 交给 decode 实例 | 分别安排计算资源，并承担 KV 传输成本 |

**本文首先讨论 LMCache 的跨请求保存与复用路径。** 它能把推理引擎产生的 KV 放到引擎之外，并供后续查询使用；它也支持分离式 prefill 等场景，但这需要额外配置。[LMCache 技术报告](https://arxiv.org/abs/2510.09665)、[Disaggregated Prefill 文档](https://docs.lmcache.ai/mp/disaggregated_prefill.html)

启用这种缓存，不等于让 GPU 注意力直接访问 SSD，也不能据此推断任意活跃请求都能自动换出、无损续跑。请求抢占、恢复和状态保留要看具体引擎与 Connector 的实现。

例如两次请求都包含一份很长的规章制度：

```text
请求 A：相同系统提示 + 相同规章制度 + 问题 A
请求 B：相同系统提示 + 相同规章制度 + 问题 B
```

第一次计算公共前缀后，LMCache 将其 KV 保存下来。即使引擎后来不再保留那份 GPU 前缀缓存，外部副本仍有机会用于请求 B。

恢复前缀后，问题 B 的尾部仍需要计算；尾部注意力也仍要读取前缀 KV。复用节省的是已缓存部分的 prefill，并没有消除整个请求的计算。

## 从注意力计算到可复用的缓存状态

### KV cache 保存的是计算状态

在因果自注意力中，当前位置的 query 与历史 key 匹配，再按注意力权重聚合历史 value。对第 $i$ 个头，可以写成：

这里 $t$ 是当前 token 的位置，$j\le t$ 表示当前位置及其历史，$d_k$ 是每个 key 的维度；$P$ 为 softmax 归一化后的注意力权重，$o$ 为该头的输出。

$$
P_{t,j,i}=\operatorname{softmax}_{j\le t}
\left(\frac{q_{t,i}^{\top}k_{j,i}}{\sqrt{d_k}}\right),\qquad
o_{t,i}=\sum_{j\le t}P_{t,j,i}v_{j,i}
$$

后续位置需要反复读取历史 K/V，因此引擎将它们缓存。历史 Q 通常不再使用，无需一并保存。缓存的具体形式取决于注意力结构：MHA、GQA 与 MLA 的每 token 存储量不同，不能用一套固定字节数描述所有模型。

在固定模型、token 序列与兼容布局下，这些历史状态可以成为后续请求的可复用计算结果。它们与完整回答不同：恢复 KV 后，新请求仍需计算自己的后续 token。

### 前缀复用省去了哪些计算

Prefill 处理输入提示，建立上下文 KV，并计算后续生成所需的状态；decode 则逐步生成新 token，读取已有 KV，并追加新状态。

如果公共前缀的 KV 已经存在，新请求有机会跳过这段前缀的 prefill。但普通 GPU 推理路径仍需让相关 KV 在计算前就绪，因此外部缓存会引入回载工作。它通常直接影响首 token 的关键路径，不能由此推断每个 decode token 都会同比例加速。

回到“相同规章制度、不同问题”的例子：把输入拆成公共前缀 $P$ 与问题尾部 $U$。命中缓存后，$P$ 的历史 KV 可以直接恢复；$U$ 仍要执行模型前向，而且它的 query 需要读取 $P$ 的 KV。缓存避免了重新生成前缀状态，后续 token 对前缀的注意力计算仍然存在。

### 数据放在哪里，决定了怎样取回来

GPU HBM、CPU DRAM 和磁盘在容量、带宽及访问延迟上存在差异。这里的分层缓存，是把复用价值不同的状态放到适合的位置，需要时再搬运到计算使用的位置。

还应区分两种生命周期：请求结束后，KV 可以继续作为可淘汰的前缀副本保留；请求仍在生成时，计算所依赖的 KV 则需要受到保护。副本存到外部、GPU block 被释放以及请求可以继续执行，是不同的事件。

### 先约定几个存储与调度术语

| 术语 | 本文中的含义 |
|---|---|
| GPU block | 引擎分配与管理 GPU KV 的分页单位 |
| Chunk | LMCache 匹配和操作外部 KV 的分块单位 |
| MP | Multiprocess，多进程；这里指 LMCache 与推理引擎分进程运行的部署模式 |
| Connector | 协调引擎与 LMCache 的匹配、传输和完成状态的集成层 |
| L1 / L2 | LMCache 的近层与次级缓存；本文选择 CPU DRAM L1 和文件系统 L2 |
| Store / retrieve | 保存 KV 到外部缓存 / 将 KV 回载供引擎使用 |
| Prefetch | 在真正使用前提前准备缓存对象，以尝试隐藏读取延迟 |
| TTFT | 从发送请求到收到首个有效生成 token 的时间 |

这些术语对应着不同的职责：引擎安排计算并管理 GPU blocks，Connector 协调外部状态，缓存服务管理对象与存储。先把分工放到同一张图里，才能顺着请求追踪一次回载。

## LMCache 如何接入推理引擎

LMCache 的定位是推理引擎之外的 KV cache 层。它负责外部 KV 的保存、匹配、搬运与管理；模型执行、GPU block 分配和请求调度仍由推理引擎负责。

先看本文采用的 **vLLM + LMCache MP 服务 + CPU L1 + 文件系统 L2** 路径。图中，上半部分负责计算与 GPU 分页管理，下半部分负责外部对象与存储：

```text
请求 → vLLM Scheduler → vLLM Worker → GPU 上的分页 KV
             │              │                   ⇅
             └── LMCacheMPConnector ──→ LMCache 独立服务
                                          │
                                    StorageManager
                                          │
                                   L1：CPU 内存池
                                          ⇅
                              L2 Adapter：NVMe / 外部存储
```

这张图包含两类交互：查找与完成通知传递控制信息，GPU↔CPU↔L2 搬运实际的 KV 数据。理解这一区别之后，再看 MP 和 Connector 的职责就更清楚。

### MP：独立管理缓存服务的进程与资源

**MP 是 multiprocess，即多进程。** 在 LMCache 的语境中，它描述缓存组件的部署方式：vLLM 运行模型与调度，另一个独立的 `lmcache server` 进程管理外部 KV，双方通过进程间通信协调存取。[MP 架构文档](https://docs.lmcache.ai/mp/index.html)

```text
同一台机器
├─ vLLM 的调度与 Worker 进程：执行模型，持有 GPU KV
└─ LMCache 服务进程：管理外部 KV、内存池和存储后端
```

这里的“多进程”不限定总共只有两个进程：vLLM 本身也可能有多个 Worker。MP 也不是模型并行的缩写，更不意味着已经跨机器部署。

旧的 in-process 模式把 LMCache 嵌入引擎进程，通过 `LMCacheConnectorV1` 接入；MP 模式使用独立服务，并通过 `LMCacheMPConnector` 接入。两者对应的配置方式和生命周期不同。[官方 Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)

拆分之后，缓存服务可以独立配置 CPU 内存、存储和监控，同一节点上的兼容引擎实例也能共享它。独立启动的 LMCache 还可以在 vLLM 重启时继续保留 CPU 缓存。代价是增加进程间通信和服务管理；服务独立并不保证缓存故障时请求一定能无感继续。

### Connector：协调调度、显存位置与数据传输

**Connector 是连接器，在这里是一组由推理引擎调用的集成接口及其实现。** 它把引擎的请求、token 进度、GPU block 分配和执行时序，转换成 LMCache 的查找、保存、回载及完成状态。它作为引擎侧组件工作，不必再启动一个名为 Connector 的独立服务。

例如请求 B 的前缀在 CPU 中命中，连接器需要协助完成三步：向 Scheduler 报告可复用的 token 范围；在引擎分配 GPU blocks 后建立回载映射；启动搬运并反馈完成状态，使后续计算能够安全读取。它连接的是缓存状态与执行时序。

对照 vLLM 的 V1 接口，可以把这些职责映射到 Scheduler（请求调度侧）与 Worker（模型执行侧）：

| 所在位置 | 需要解决的问题 | 典型接口 |
|---|---|---|
| Scheduler 侧 | 外部还能提供多少前缀，是否要等待回载 | `get_num_new_matched_tokens` |
| Scheduler 侧 | 将已经分配的 blocks 与外部 token 对应起来 | `update_state_after_alloc`、`build_connector_meta` |
| Worker 侧 | 注册 GPU KV 布局，按元数据启动回载 | `register_kv_caches`、`start_load_kv` |
| Worker 侧 | 提交保存工作并满足同步约束 | `save_kv_layer`、`wait_for_save` |
| 两侧协作 | 报告传输完成或失败，允许调度继续或回收资源 | 完成结果与 Connector 元数据 |

这些是接口职责，调用细节取决于引擎和 Connector 的版本。例如接口中有 `wait_for_layer_load`，不代表当前 MP 实现一定逐层等待；查阅时的实现中该方法直接返回，回载由其他状态与事件协调。[vLLM Connector 基类](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)、[LMCacheMPConnector 实现](https://github.com/LMCache/LMCache/blob/dev/lmcache/integration/vllm/lmcache_mp_connector.py)

这些接口共同维护一个约束：引擎所认为的“已恢复前缀”，必须与 GPU 目标位置中已经可读的数据一致。仅查询到外部对象，还不足以满足这个约束。

### Connector、GPU 搬运层与存储 Adapter 的分工

LMCache 代码中可能在不同层使用 connector 这个词，阅读时要结合上下文。本文所说的 Connector，主要指引擎侧的 `LMCacheMPConnector`：

| 组件 | 对接哪一侧 | 主要职责 |
|---|---|---|
| 引擎侧 KV Connector | vLLM 调度与执行流程 | 协调 token、blocks、传输任务和完成状态 |
| GPU 数据搬运组件 | 设备张量与缓存布局 | 执行 KV 的收集、复制与回填 |
| L2 Adapter | 文件系统、Mooncake Store 等存储 | 提供对象的查找、读写等存储操作 |

因此，增加一种 L2 后端，通常是在存储接口上扩展；更换推理引擎，则需要相应的引擎集成。它们处于不同层。

MP 服务可以通过 ZMQ 或 gRPC 接收控制请求；RPC（远程过程调用）描述请求如何到达服务端，并不能单独说明张量如何传输。同机 CUDA 路径还涉及 GPU cache 注册、CUDA IPC（进程间通信）与专门的数据搬运。[MP 架构文档](https://docs.lmcache.ai/mp/index.html)

当前实现还有 GDS 等其他 L1 介质选择。本文固定讨论 CPU DRAM L1，后面的 GPU↔CPU↔L2 路径和估算都以这一选择为前提。

### 外部缓存接着 GPU 前缀缓存做什么

GPU 前缀命中时，KV 已在引擎可以使用的位置，通常无需外部回载。LMCache 提供的是 GPU 没有保留的那一部分历史状态。

源码中的 `get_num_new_matched_tokens` 接收引擎已经计算或命中的 token 数，再判断外部缓存能够额外提供多少。随后由引擎分配 blocks，Connector 将数据写入这些目标位置。[LMCacheMPConnector 源码](https://github.com/LMCache/LMCache/blob/dev/lmcache/integration/vllm/lmcache_mp_connector.py)

所以重复请求变快可能有两个来源：GPU 前缀命中，或 LMCache 外部命中。实验必须把它们分开。

## 缓存匹配如何保证复用正确性

### 前面的上下文也参与决定 KV

在因果 Transformer 中，第 $j$ 个位置的隐藏状态受到此前 token 的影响。相同的一段文本，如果前面换了系统提示或其他文档，其深层 K/V 通常也会变化。

这一依赖可以从层间计算看出来。对于普通注意力的第 $\ell$ 层，省略归一化和位置处理细节，有：

$$
k_j^{(\ell)}=W_K^{(\ell)}h_j^{(\ell-1)},\qquad
v_j^{(\ell)}=W_V^{(\ell)}h_j^{(\ell-1)}
$$

这里 $h_j^{(\ell-1)}$ 是上一层在位置 $j$ 的输出。经过此前的因果注意力后，它通常已经包含前缀信息。因此，文档本身相同，只保证当前文本片段相同；它不保证投影的输入隐藏状态相同。

因此，普通前缀复用要求的是相同的有效 token 前缀及兼容的模型状态。语义相似不能替代精确前缀匹配。

```text
请求 A：[系统提示 A][文档 X][问题]
请求 B：[系统提示 B][文档 X][问题]
```

两次都有“文档 X”，但不能直接把 A 中的文档 KV 原样当成 B 的精确前缀 KV。这也解释了为什么聊天模板、空白、分词和特殊 token 会影响命中。

### 用分块哈希保留前缀依赖

LMCache 的普通 token 匹配路径按 chunk 建立滚动前缀哈希。用示意公式表示：

$$
H_i=\operatorname{Hash}\big(H_{i-1},\mathrm{tokens}_i\big)
$$

同一个 chunk 的 token 相同，但前缀哈希不同，得到的键也不同。`TokenHasher.compute_chunk_hashes` 从序列开头累计计算哈希，处理完整 chunk，尾部不足一块的部分不进入这条完整分块路径。[TokenHasher 源码](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/multiprocess/token_hasher.py)

模型、KV 数据类型、布局、并行配置等兼容性也需要核对。跨实例复用要有一致的对象标识与字节语义，不能只看 token 哈希相同就认为任意模型可共用。

### Chunk 大小在粒度与开销之间取舍

GPU block 是引擎分配和管理显存的单位；LMCache chunk 是外部缓存操作与匹配的单位。它们需要满足所选 Connector 的对齐规则，不能把两个配置当成同一个参数。

较大的 chunk 减少对象和操作数量，但短前缀容易不足一块；较小的 chunk 提供更细的匹配粒度，却增加哈希、元数据与传输操作。下面是完整分块情况下的示意：

$$
N_{\mathrm{aligned}}=C\left\lfloor\frac{N_{\mathrm{shared}}}{C}\right\rfloor
$$

如果公共前缀是 1000 tokens，$C=256$，按完整 chunk 可覆盖 768 tokens。实际可跳过的 token 数还受 GPU block、模型缓存分组（cache group）和最后位置的计算需求等影响。普通前缀复用还要求这些状态覆盖连续前缀；较后位置的 chunk 存在，不能自动弥补中间缺失的 KV。[Connector 匹配逻辑](https://github.com/LMCache/LMCache/blob/dev/lmcache/integration/vllm/lmcache_mp_connector.py)

Chunk 大小需要结合共享前缀长度、块对齐和传输开销一起选择。

### 一个贯穿后续流程的分块示例

将前面的请求 A、B 简化为各 1024 tokens，前 1000 tokens 相同，末尾 24 tokens 不同。设 chunk 大小为 256，GPU 当前保留前 512 tokens，而 LMCache 可提供相同前缀的前三个完整 chunks：

| Token 范围，从 0 开始 | 请求 B 的可复用状态 | 处理方式 |
|---|---|---|
| $[0,256)$ | GPU 已有 | 直接保留 |
| $[256,512)$ | GPU 已有 | 直接保留 |
| $[512,768)$ | 仅外部缓存可提供 | 回载至新分配的 GPU blocks |
| $[768,1024)$ | 这一整块不匹配 | 执行 prefill |

在这个简化模型中，前缀共复用 768 tokens，其中只有 256 tokens 需要由 LMCache 回载。最后一块虽然有 232 个相同 tokens，但包含不同的尾部，不能作为相同的完整 chunk 直接命中。具体引擎可能进一步施加对齐和末位置保留要求。

这个例子也说明了三个指标的差别：公共文本长度为 1000 tokens，可复用的完整分块前缀为 768 tokens，外部实际回载量为 256 tokens。

## 一次请求中的保存、回载与同步

匹配解决了“哪些状态可以复用”，下一步是让这些状态在正确的时间抵达正确的 GPU blocks。下面按 MP 文档和 Connector 源码追踪 CPU L1 路径；具体异步事件和调度边界由版本决定。

### 第一次请求：计算并保存

1. 引擎检查已有的 GPU 前缀，Connector 查询外部可用的 chunks。
2. 缺失部分由 GPU 执行 prefill，生成 KV。
3. Connector 根据目标 chunk 与 GPU block 的对应关系提交保存操作。
4. LMCache 为 L1 对象预留空间，将 KV 从 GPU 搬到 CPU，并完成对象写入。
5. 如果配置了 L2，存储控制器可以进一步将 L1 对象写入 L2。

这里的“卸载”可以先建立外部副本。它不意味着复制完成后立刻强制释放 GPU blocks；原有 GPU 缓存何时回收，由引擎与相关策略决定。

### 后续请求：查找并回载

1. 引擎先确定 GPU 已命中的前缀。
2. Connector 查询可额外复用的外部 chunks；L1 缺失时可能提交 L2 预取。
3. 引擎为要回载的 KV 分配目标 GPU blocks。
4. LMCache 将已就绪的数据搬回这些 blocks，并报告传输完成。
5. 引擎跳过可复用部分的 prefill，计算剩余尾部并进入 decode。

**存在、就绪和可被计算使用，是三个不同状态。** L2 中找到对象，不代表它已经到 CPU；到 CPU 不代表 GPU 回载已经完成。Scheduler 必须遵守这些依赖。[MP 请求流程](https://docs.lmcache.ai/mp/index.html#request-flows)

对应上一节的示例，Scheduler 只需要为 $[512,768)$ 建立额外回载任务，并协调 $[768,1024)$ 的剩余计算。已经留在 GPU 的前 512 tokens 不必再次从 CPU 复制。回载过程必须把外部对象写入本次实际分配的 blocks，不能沿用请求 A 当时的物理 block 编号。

### 对象状态与 GPU block 生命周期如何配合

L1 中分配了空间，不代表对象已写完。按照 MP 文档的抽象，缓存对象经历以下状态：

```text
不存在 → 写入中（写锁保护）→ 就绪 → 读取中（读锁保护）
                              ↑                 │
                              └── 读取完成 ─────┘

就绪且无读写保护的对象 → 可按策略淘汰
```

写入方先预留对象并取得写保护，数据复制完成后再发布为就绪；读取方在使用期间持有保护，防止对象被并发淘汰。锁与完成事件分别约束对象访问和异步操作进度，不能相互替代。[L1Manager 状态与存取流程](https://docs.lmcache.ai/mp/index.html#distributed-storage)

这套协议需要满足三条约束：保存结束前，GPU 源 blocks 保持有效；写入完成前，L1 对象不能被作为完整数据读取；回载完成前，GPU 目标 blocks 不能被后续注意力计算使用。若源 block 提前被新请求覆盖，外部副本就可能混入其他请求的 KV；若目标 block 提前被读取，计算就可能使用尚未复制完整的数据。

### 异步传输如何影响计算关键路径

异步传输的收益来自减少计算关键路径上的等待时间，而非消除传输本身。能否重叠，取决于系统是否还有不依赖这份 KV 的工作可执行。

例如，预取可以在请求排队期间把 L2 对象加载到 L1，提前消化部分存储延迟；真正执行该请求时，CPU→GPU 的搬运仍须按依赖完成。可重叠的是彼此独立的工作，仍在等待所需 KV 的注意力计算不能提前开始。

Connector 的保存逻辑及抢占处理涉及这类事件与生命周期协调。阅读 `wait_for_save`、`get_finished` 等方法时，应看具体版本的实现，不能仅凭方法名认定所有存储都是阻塞或全部异步。[Connector 源码](https://github.com/LMCache/LMCache/blob/dev/lmcache/integration/vllm/lmcache_mp_connector.py)

## 分层存储的容量、带宽与持久性

本文选定的 CPU L1 路径中，CPU 是外部缓存的近层，文件系统或远端服务是更大的 L2。当前文档通过 `--l2-adapter` 选择后端，包含文件系统、NIXL、Mooncake Store、S3 等不同实现。[L2 后端列表](https://docs.lmcache.ai/mp/l2_storage/supported_storages.html)

| 层级 | 主要作用 | 需要关注的代价 |
|---|---|---|
| 引擎 GPU 缓存 | 当前注意力使用、原生前缀复用 | 显存容量与 block 回收 |
| LMCache CPU L1 | 较近的外部副本 | CPU 内存、GPU↔CPU 带宽、NUMA 内存位置 |
| 文件系统 L2 | 更大容量的缓存对象存储 | 文件访问、读取与预取延迟 |
| 远端 L2 | 跨节点共享或容量扩展 | 网络、远端队列、后端接口与格式 |

CPU DRAM L1 的数据随 LMCache 服务退出而消失。独立运行 LMCache 可以让它在 vLLM 重启时继续保留 RAM 中的对象，前提是 LMCache 服务本身还活着、对象未被淘汰且布局兼容。

NUMA 指非一致内存访问：CPU 插槽与内存的物理位置会影响访问路径。如果 GPU 的数据搬运还需要跨 CPU 插槽访问远端内存，标称 PCIe 带宽就不足以代表实际回载带宽。

磁盘上存在文件也不代表任意重启后都能自动识别和复用：还要确认对应后端的索引、发现机制与命名兼容性。进程分离与持久化是两个不同属性。

容量用满时，淘汰策略同样重要。本文实验使用 LRU，它有利于时间局部性较强的访问；一次扫描大量长文档，则可能挤掉高复用前缀。存储容量与命中率应一起测量。

## 多机部署中的共享路径与传输边界

**LMCache 支持多机部署。** 需要先明确目标：让不同节点复用已有 KV、把缓存放到共享存储，还是让一个节点 prefill、另一个节点 decode？这些目标可以组合，但不是同一个开关。

### 通过共享 L2，让不同节点读取同一份对象

一种拓扑是在每个推理节点旁部署 LMCache 服务，各自管理本地 L1，同时连接共享的远端 L2。节点 A 写入后，节点 B 可以在对象兼容且已可见时，从 L2 取回：

```text
节点 A：vLLM A ⇄ LMCache A / CPU L1
                          ⇅
                   共享 L2 存储
                          ⇅
节点 B：vLLM B ⇄ LMCache B / CPU L1
```

例如官方 Mooncake Store adapter 对接分布式 KV 存储，并支持相应的 RDMA 配置。共享的是后端中的缓存对象；节点 B 通常仍要将对象准备到本地，再写回本机 GPU。不同后端的配置、传输与发现方式需要分别核对。[Mooncake Store 文档](https://docs.lmcache.ai/mp/l2_storage/mooncake_store.html)

两台机器都设置 `/data/lmcache/l2` 这样的相同路径名，并不会自动共享各自的本地磁盘。必须真正接入共同可见的后端，并保持模型与对象布局兼容。

### 通过 P2P，从其他节点的 L1 获取 KV

LMCache 也提供 P2P KV 共享。本机 L1 未命中时，可以查询其他节点，取得对象并加锁，再通过传输层将远端 KV 读到本机 L1，之后回载到 GPU：

```text
节点 A：vLLM A ⇄ LMCache A / L1
                          ⇅ 节点间 KV 传输
节点 B：vLLM B ⇄ LMCache B / L1

Coordinator：提供成员发现与存活信息
```

在这条 P2P 路径中，Coordinator 帮助节点发现彼此，KV 数据在节点间传输，不经 Coordinator 转发。默认传输引擎使用 NIXL/RDMA；文档还列出 Mooncake Transfer Engine，可按配置使用 TCP 或 RDMA。所需依赖、网络和 L1 内存注册条件必须满足，并非所有 L1 介质都能直接组合使用。[P2P KV Cache Sharing](https://docs.lmcache.ai/mp/p2p.html)、[Multi-Server Coordination](https://docs.lmcache.ai/mp/coordinator.html)

其中 NIXL 提供数据传输能力，RDMA 指远程直接内存访问。节点发现回答“向谁请求”，传输层解决“怎样将字节搬过来”；找到持有对象的节点之后，还需要对象锁定、网络传输和本机 GPU 回载，才算完成一次可供计算使用的复用。

### 跨机 P/D 分离：当前请求的交接与后续复用

Prefill/Decode 分离可以把两个实例放到不同机器。查阅时的官方 MP 方案通过 vLLM 的 `MultiConnector` 组合两类职责：

- `NixlConnector`：传递当前请求从 prefill 到 decode 所需的 KV；
- `LMCacheMPConnector`：访问各实例旁的 LMCache 服务，保存和读取可跨请求复用的 KV。

Router 协调请求先进入 prefill，再进入 decode，并传递握手信息。该方案为 prefill、decode 各自配置 LMCache 服务；如还要让两侧缓存池共享历史前缀，可以再叠加 P2P。具体组合以对应版本的官方配方为准。[Disaggregated Prefill 文档](https://docs.lmcache.ai/mp/disaggregated_prefill.html)

### MP、跨机共享与模型并行不是同一个维度

| 目标 | 主要依赖 | 仅启动本机 MP 服务是否足够 |
|---|---|---|
| 本机引擎与缓存分进程运行 | MP 服务与引擎 Connector | 是，仍需模型与布局兼容 |
| 多节点复用公共 KV | 共享 L2 或 P2P，以及跨机传输 | 否 |
| 跨机 P/D 分离 | Producer/consumer、Router 和 KV 交接协议 | 否 |
| 单模型跨机 TP/PP | 引擎的模型并行，以及各 KV 分片的缓存映射 | 否 |

TP 是 tensor parallelism（张量并行），PP 是 pipeline parallelism（流水线并行）。它们决定模型与 KV 分片如何分布；MP 决定缓存服务是否独立成进程。即使引擎能做跨机 TP/PP，也需要验证 Connector 是否支持对应模型、rank 映射与缓存布局。当前配置文档对多节点 TP/PP 场景建议独立管理 LMCache 服务，而不是依赖引擎自动拉起的子服务。[MP 配置文档](https://docs.lmcache.ai/mp/configuration.html)

同样，**把 `lmcache.mp.host` 改成另一台机器的 IP，只改变了请求连接的目标。** 本文同机 CUDA 路径使用的 CUDA IPC 句柄不能作为跨机 GPU 内存访问机制；请求通了，还需要一条能跨机器传递 KV 的数据路径。

验证多机共享时，可以先让 A 计算并存储一个长前缀，再让没有 GPU 前缀缓存的 B 请求相同前缀，观察 B 的远端读取、网络搬运和 GPU 回载。两边同时核对模型、分词、chunk 大小、KV dtype、命名空间和并行布局，并单独测量网络拥塞、后端不可达时的行为。多机扩大了复用范围，也将网络与远端排队成本带入下一节的回载时间。

## 用成本模型判断复用收益

数据路径可行，并不意味着这条路径总有收益。下面固定 KV 布局与传输假设，建立一个可代入实测数据的简化成本模型。

### 先算要搬多少字节

对全注意力 GQA 模型，忽略 padding 和元数据，整个模型每 token 的 KV 有效载荷为：

$$
m=L\times G\times(d_k+d_v)\times s
$$

假设 $L=32$、$G=8$、$d_k=d_v=128$、BF16 的 $s=2$：

其中 $L$ 是层数，$G$ 是 KV 头数，$d_k$ 和 $d_v$ 分别是每头 K/V 的维度，$s$ 是每个元素的字节数。公式先求一层的 K/V 宽度，再乘层数和存储精度。

$$
m=131072\text{ bytes}=128\text{ KiB/token}
$$

因此 8192 tokens 对应 **1 GiB**，256-token chunk 对应 **32 MiB**。这按模型完整 KV 计算；多卡切分后，每个 rank 的搬运量和并行情况另算。MLA、混合注意力与状态模型需要换用对应布局的容量公式。

用这一假设解释前面的分块示例：复用 768 tokens 并不需要从外部搬运 96 MiB，因为其中 512 tokens 已在 GPU；实际回载的一个 256-token chunk 只对应 32 MiB。估算 I/O 成本时应使用后者。

### 比较关键路径上的回载与重算时间

设可复用的 KV 大小为 $D$、有效搬运带宽为 $b_{\mathrm{eff}}$：

$$
t_{\mathrm{load}}\approx
t_{\mathrm{lookup}}+t_{\mathrm{prepare}}
+\frac{D}{b_{\mathrm{eff}}}+t_{\mathrm{sync}}
$$

四项依次表示对象查找、数据准备、实际传输以及完成同步。$D$ 只包括本次需要从外部恢复的 KV；$b_{\mathrm{eff}}$ 应取端到端测得的有效带宽。

在其他条件相近时，需要比较：

$$
\boxed{t_{\mathrm{load}}<t_{\mathrm{prefill,saved}}}
$$

这里右侧是原本可以被跳过的那部分 prefill 时间，不是整个请求总耗时。重算与尾部计算可能有交互，实际收益还要放回完整调度中测量。

假设 1 GiB 的 CPU→GPU 有效带宽为 12 GiB/s，纯传输约 83 ms。若还需从 SSD 以 3 GiB/s 串行读入 CPU，再搬到 GPU，则这两段约为：

$$
\frac{1}{3}+\frac{1}{12}\approx0.417\text{ s}
$$

预取和流水线可能降低关键路径上的等待，但查找、排队和对象准备也会增加成本。以上带宽是演示假设，应替换成部署实测。

对于上述 1 GiB 回载，如果可跳过的 prefill 原本需要 200 ms，约 83 ms 的 CPU 传输可能有收益空间，而约 417 ms 的串行磁盘加 CPU 传输就可能超过重算成本；若原本需要 1 s，两条路径才都有进一步评估的价值。这个比较还未计入查找、排队、同步与实际重叠，只用于说明存储命中和性能收益必须分别判断。

### 保存成本要靠后续复用摊销

假设一次保存成本为 $t_{\mathrm{store}}$，随后发生 $R$ 次有效复用，每次节省 $\Delta=t_{\mathrm{prefill,saved}}-t_{\mathrm{load}}$，忽略其他变化时需满足：

$$
R\Delta>t_{\mathrm{store}}
$$

所以长而重复的前缀是较自然的候选。一次性文本、极短前缀、低复用率或拥塞的存储后端，可能让卸载得不偿失。异步保存即使不完全落在当次 TTFT 上，也消耗内存与传输资源，可能影响其他请求。

## 实验设计：从 CPU 复用到磁盘回载

前面的成本模型需要两类证据：请求确实跳过了部分 prefill，相关 KV 确实来自外部存储。下面通过 **Linux + CUDA + 单机 vLLM** 实验分别验证 CPU 与磁盘路径，步骤根据官方 MP Quickstart 改编。执行前需要核对所安装版本的参数与运行环境。

### 核对安装与版本

首先按[官方安装指南](https://docs.lmcache.ai/getting_started/installation.html)选择运行时匹配的包或镜像，再依据[兼容性说明](https://docs.lmcache.ai/getting_started/compatibility.html)确认三层条件：Python/PyTorch/加速器 ABI、Connector 加载方式，以及模型与 KV 布局。

```bash
python -m pip show lmcache vllm torch
lmcache server --help
```

当前文档说明：vLLM 0.20.0 起可用 `kv_connector_module_path` 显式选择 LMCache 包提供的 MP Connector。更早版本使用 vLLM 内置实现。**这个版本边界只描述加载机制，不保证任意版本组合都能运行所有特性。**

### 终端 A：独立启动 CPU L1 服务

```bash
lmcache server \
  --host 127.0.0.1 --port 5555 \
  --l1-size-gb 20 --eviction-policy LRU --chunk-size 256
```

这是没有 L2 的基线。内存大小应按实验机器调整；先保持服务独立运行，以便之后单独重启 vLLM，检验外部缓存是否被读取。

### 终端 B：启动 vLLM

以下命令使用支持外部模块加载的 vLLM，并要求所安装的 LMCache 提供对应模块：

```bash
vllm serve Qwen/Qwen3-8B \
  --port 8000 --max-model-len 8192 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "127.0.0.1",
      "lmcache.mp.port": 5555
    }
  }'
```

`kv_both` 允许当前实例承担保存与读取角色，不等于自动建立 prefill/decode 分离。模型需要提前准备，GPU 显存需能容纳模型和实验请求。[官方 Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)

这里的 `kv_connector` 选择引擎侧集成实现，`kv_connector_module_path` 指定它所在的 Python 模块；`lmcache.mp.host` 与 `lmcache.mp.port` 指向独立缓存服务的请求端点。它们各自对应前面的 Connector、模块加载和 MP 服务三个概念。

### 终端 C：构造公共前缀

使用原始 completions 接口，减少聊天模板变化的干扰。下面生成两份不同问题、相同长前缀的请求文件：

```python
import json
from pathlib import Path

prefix = (
    "Service rules: the service opens at 09:00 and closes at 18:00. "
    "Requests outside business hours are processed the next day.\n"
) * 128

questions = {
    "a": "What time does the service open?",
    "b": "How are requests outside business hours processed?",
}
for name, question in questions.items():
    payload = {
        "model": "Qwen/Qwen3-8B",
        "prompt": prefix + "\nQuestion: " + question,
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }
    Path(f"request-{name}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
```

执行请求 A，等待日志确认公共前缀的存储操作完成：

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @request-a.json
```

然后在隔离实验环境中停止并按原命令重启 **vLLM**，保留终端 A 的 LMCache 服务。这样清掉旧的 GPU 前缀缓存，模型与配置保持一致。服务就绪后记录 LMCache 指标，执行请求 B，再检查指标增量：

```bash
curl -s http://127.0.0.1:8080/metrics

curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @request-b.json

curl -s http://127.0.0.1:8080/metrics
```

如果第二次请求确实从 CPU 回载，应看到对应的 retrieve 日志和 L1 读取计数增量。不能仅靠答案返回更快就认定命中。此实验不要同时重启 LMCache，否则 CPU 副本也消失；没有 L2 时更不能通过清空 LMCache 来“只清 GPU”。

### 再增加文件系统 L2

CPU 路径确认后，在新的实验服务中增加文件系统 adapter：

```bash
lmcache server \
  --host 127.0.0.1 --port 5555 \
  --l1-size-gb 20 --eviction-policy LRU --chunk-size 256 \
  --l2-adapter '{"type":"fs","base_path":"/data/lmcache/l2"}'
```

目录需要可写。这是 MP 文件系统 adapter 的配置，不能与旧进程内模式的 `LMCACHE_LOCAL_DISK` 等参数混用。[FileSystem 文档](https://docs.lmcache.ai/mp/l2_storage/fs.html)

**文件出现只证明发生了写入。** 要证明 L2 回载，应确认目标对象已写入 L2，之后在隔离实验中让它离开 L1，并观察 L2 读取、预取和 GPU 回载。不要把文件系统 page cache 命中误当成物理 SSD 性能；是否使用 direct I/O 应结合后端支持与测试目标决定。

## 性能评测与结果归因

跑通存取只是功能验证。性能评估还要排除 GPU 缓存、模型预热和负载变化等因素，确认节省的计算大于新增的存储开销。

### 分开比较四种缓存状态

| 实验状态 | GPU 前缀 | 外部缓存 | 要回答的问题 |
|---|---|---|---|
| 冷缓存 | 未命中 | 未命中 | 第一次计算加保存成本是多少 |
| GPU 命中 | 命中 | 可有可无 | 原生前缀复用的收益是多少 |
| CPU 命中 | 未命中 | L1 命中 | LMCache 外部回载是否有收益 |
| L2 命中 | 未命中 | L1 缺失、L2 命中 | 更大介质的回载是否划算 |

先预热模型和 kernel，再测缓存状态。vLLM 重启后首次执行的编译、图捕获等开销应单独处理，例如用不同前缀预热，避免与目标缓存前缀混在一起。

### 把用户延迟与存储指标对应起来

建议记录以下几组数据，数值必须对应同一测试窗口：

- **用户延迟**：TTFT、请求总耗时、逐 token 延迟，并报告 P50/P95；
- **计算与搬运**：跳过的 prefill token 数、回载字节数、lookup/retrieve/store 耗时；
- **缓存状态**：GPU 命中、LMCache L1/L2 命中、淘汰次数和实际对象占用；
- **负载条件**：模型、KV dtype、并行拓扑、chunk/block 大小、输入输出长度和并发。

流式 TTFT 应测量从发送请求到收到首个有效生成 token 的时间。HTTP 首字节可能只是响应头或空的流事件；非流式请求总时间也不能当 TTFT。

当前 `lmcache server` 默认通过 HTTP 前端的 8080 端口暴露 `/metrics`，文档列有 `lmcache_mp_l1_read_chunks_total` 等指标；具体名称随版本核对。指标可能要发生一次存取后才出现。[Metrics 文档](https://docs.lmcache.ai/mp/observability/metrics.html)

命中率还要注明分母。请求命中率、chunk 命中率、token 命中率和字节命中率并不相同；一个请求命中很短的前缀，与另一个命中几千 tokens，不能只按“两个请求都命中”衡量价值。

### 缓存兼容性也是实验条件

重复测量时冻结模型权重、tokenizer、模板、位置配置与 KV 布局。更换 LoRA、模型修订、KV dtype 或并行布局后，需要确认对象标识是否足以区分这些状态。

不同用户或模型能否共用一个缓存服务，也应依据具体实现的对象命名与隔离机制判断。仅向请求附加一个 `lmcache.tag.*` 字段，并不会自动使它成为缓存键或隔离策略；当前 MP 配置文档明确区分了转发元数据与服务端实际执行的行为。[Configuration Reference](https://docs.lmcache.ai/mp/configuration.html#per-request-lmcache-configuration)

## 讨论：非前缀复用、压缩与 P/D 分离

LMCache 还有更广的能力，但它们应与本篇的普通前缀缓存实验分开理解：

| 能力 | 解决的问题 | 本篇基线是否已经启用 |
|---|---|---|
| 普通外部前缀复用 | 相同前缀反复 prefill | 是 |
| CacheBlend | RAG 文档块处于非前缀位置时的复用与融合 | 否，需要专门配置 |
| CacheGen | KV 压缩与传输成本 | 否，需要单独配置与质量验证 |
| P/D 分离 | Prefill 产生 KV，另一个实例执行 decode | 否，需要节点角色、传输与请求协调 |

CacheBlend 通过选择性重算处理非前缀缓存融合；它并非把不同上下文里的 KV 任意拼接后就得到精确结果。[CacheBlend 文档](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html)、[CacheBlend 论文](https://arxiv.org/abs/2405.16444)

CacheGen 面向 KV 压缩和流式传输，需要把压缩、解压、格式兼容与质量一起评测。[CacheGen 文档](https://docs.lmcache.ai/mp/cachegen.html)、[CacheGen 论文](https://arxiv.org/abs/2310.07240)

前面的多机部署部分说明了 P/D 分离中的两条数据路径：当前请求的 prefill→decode 交接，以及跨请求的外部 KV 复用。评测时也要分别判断两者的收益，不能将网络交接完成视为历史前缀命中。

## 复用的三个前提：状态一致、传输就绪、成本有利

LMCache 将 KV 的生命周期从单个 GPU 缓存扩展到引擎外部。完整的复用路径需要同时满足三个前提：缓存与当前请求具有一致的计算语义，数据在被使用前完成传输，回载及保存的总体成本能被省去的计算抵消。

```text
上下文是否真的相同
  → chunk 和对象标识是否兼容
  → 外部副本是否存在且就绪
  → 能否及时写回目标 GPU blocks
  → 搬运是否比跳过的 prefill 更便宜
  → 保存与共享开销是否值得长期承担
```

系列第一篇先用 CPU L1 建立这条证据链，再用文件系统 L2 检验容量扩展的代价。后续讨论其他卸载实现和传输方案时，也可以沿用同一组问题：缓存如何匹配、数据走哪条路径、调度何时等待，以及收益落在什么负载上。

## 参考资料

资料查阅日期为 **2026-10-14**。官方文档当前以 MP 模式为主要路径，旧的进程内配置归入 Legacy；命令对应查阅时的文档，源码链接指向滚动更新的 `dev` 分支，复现时应记录实际版本或 commit。

1. Cheng et al., [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665), 2025。
2. [LMCache MP 架构与请求流程](https://docs.lmcache.ai/mp/index.html)。
3. [LMCache Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)与[vLLM 兼容性说明](https://docs.lmcache.ai/getting_started/compatibility.html)。
4. [MP Configuration Reference](https://docs.lmcache.ai/mp/configuration.html)。
5. [L2 后端列表](https://docs.lmcache.ai/mp/l2_storage/supported_storages.html)与[FileSystem adapter](https://docs.lmcache.ai/mp/l2_storage/fs.html)。
6. [MP Metrics](https://docs.lmcache.ai/mp/observability/metrics.html)。
7. 源码：[TokenHasher](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/multiprocess/token_hasher.py)、[LMCacheMPConnector](https://github.com/LMCache/LMCache/blob/dev/lmcache/integration/vllm/lmcache_mp_connector.py)、[StorageManager](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/distributed/storage_manager.py)。这些链接指向滚动更新的 `dev` 分支。
8. [Disaggregated Prefill](https://docs.lmcache.ai/mp/disaggregated_prefill.html)。
9. Yao et al., [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444), EuroSys 2025。
10. Liu et al., [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://arxiv.org/abs/2310.07240), SIGCOMM 2024。
11. [vLLM V1 KV Connector 接口](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)。
12. [P2P KV Cache Sharing](https://docs.lmcache.ai/mp/p2p.html)与[Multi-Server Coordination](https://docs.lmcache.ai/mp/coordinator.html)。
13. [Mooncake Store L2 adapter](https://docs.lmcache.ai/mp/l2_storage/mooncake_store.html)。
