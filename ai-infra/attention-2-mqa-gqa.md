---
title: 注意力机制-2 MQA/GQA
date: 2026-10-13
description: 系列第二篇。MHA 的软肋在 KV cache：显存随头数线性膨胀，decode 阶段受内存带宽制约。MQA 用一对共享的 K/V 将 cache 压缩到 1/h，GQA 在两者之间取得折中。本篇讨论这两次改造的动机、定量收益与工程局限。
---

# 注意力机制-2 MQA/GQA

> 系列第二篇。[上一篇](/ai-infra/attention-1-mha)介绍了 MHA：它确立了"Q-K 匹配 → softmax 加权 → V 聚合"的范式，也留下了隐患——KV cache 随头数线性膨胀，decode 阶段受内存带宽制约。本篇讨论两次针对性的改造：MQA 将 K/V 压缩为仅一对，GQA 在质量与 cache 之间取得折中。


## 00 上一篇留下的问题

回顾上一篇第 04 节的结论：自回归推理的 decode 阶段是 **memory-bandwidth-bound**——每生成一个 token，算力需求很小，但要把**全部模型权重**和**该序列的整个 KV cache**从 HBM 读进计算单元。

而 MHA 的 KV cache 恰好占据大头：每 token 每层要存 h 对 K/V，h=64 的 70B 模型在 FP16 下每 token 即达 2.5 MB。小 batch、短上下文时，权重（约 140 GB）占绝对主导，这部分 cache 影响不大；但一旦 batch 增大、上下文拉长，KV cache 就会超过权重，成为带宽开销的主要来源。

上一篇第 3.4 节还留下了另一条线索：研究表明多头之间存在大量冗余，相当一部分头被剪枝后模型性能几乎不变。**既然头本来就冗余，让每个头独享一份 K/V 是否过于奢侈？**

MQA 和 GQA 正是对这两个问题的回答。它们不改变注意力的计算形式，只调整一件事：**Q 头与 K/V 头的配比**。


## 01 定量分析：decode 的带宽瓶颈有多严重

在讨论结构改动之前，先明确瓶颈的数量级——只有这样，才能理解每一次压缩 KV 究竟换来了什么。

### 1.1 每 token 的 KV cache 显存

沿用上一篇的公式，每 token 的 cache 字节数为：

$$\text{bytes/token} = \underbrace{2}_{K\text{和}V} \times L \times g \times d_h \times \text{precision}$$

其中 g 是 K/V 头数。以 LLaMA-3 70B（L=80，h=64，$d_h$=128，FP16）为例：

| 方案 | K/V 头数 g | 每 token KV | 8K 上下文单序列 |
|---|---|---|---|
| MHA | 64 | 约 2.5 MB | 约 20 GB |
| GQA-8 | 8 | 约 0.31 MB | 约 2.5 GB |
| MQA | 1 | 约 0.04 MB | 约 0.31 GB |

### 1.2 什么时候 KV cache 成为主要开销

做一个理想化的 roofline 估算：70B FP16 权重约 140 GB，H100 的 HBM 带宽约 3.35 TB/s。decode 每步需要读取的字节数 ≈ 权重 + 所有序列的 KV cache。

- **batch=1，8K 上下文**：权重 140 GB 占绝对主导，MHA 的 20 GB cache 占比很小。此时压缩 KV 收益有限——这也是小模型、小并发场景无需为此调整结构的原因；
- **batch=32，8K 上下文**：MHA 的 KV cache 总量约 **640 GB**，是权重的 4 倍多。每步读取 780 GB，理论单步耗时约 233 ms，吞吐约 137 token/s。改用 GQA-8，cache 缩至 80 GB，单步约 66 ms，吞吐约 487 token/s——**提升 3.5 倍**。改用 MQA，cache 仅 10 GB，吞吐还可再提升近一半。

结论很直接：**KV cache 的带宽开销与 batch × 上下文长度成正比，生产环境（大 batch、长上下文）下它是主要的带宽消耗项**。这正是 KV 压缩路线针对的问题，也是评估 MQA/GQA 收益的正确参照。

需要注意一个常见的误解：MQA/GQA **不减少计算量**——每个 Q 头仍然执行一次完整的注意力。它们节省的是显存容量与带宽搬运，FLOPs 并未减少。这与 FlashAttention（减少 IO、不改变结果）、MLA（压缩表示）属于不同维度的优化，彼此正交、可以叠加。

### 1.3 原论文的复杂度视角：带宽/算力比

Shazeer 在 MQA 论文中采用了一个统一的分析框架：比较**内存访问量与算术运算量之比**（该比值越低，越适合 GPU/TPU 这类算力远强于带宽的硬件）：

- **训练/批量前向**：该比值约为 $O(\frac{1}{k} + \frac{1}{bn})$（b 为 batch，n 为序列长度，k 为头维度）——分母较大，计算利用率高，因此训练阶段的效率从来不是瓶颈；
- **增量解码**：该比值恶化为 $\Theta(\frac{n}{d} + \frac{1}{b})$。其中 $\frac{1}{b}$ 项对应每步重读全部权重（可通过增大 batch 摊薄），而 $\frac{n}{d}$ 项正是**每步重读 K/V 张量**的代价——它随上下文长度线性增长，且无法通过增大 batch 摊薄（各序列的 cache 相互独立）。

这一分析指出了两个方向：要么限制 n（缩短上下文，代价不可接受），要么压缩 K/V 张量本身的尺寸。MQA 将 K/V 压缩 h 倍，等效于将 $\frac{n}{d}$ 项缩小 h 倍——这正是那 12 倍加速的来源。后续的 GQA、MLA、KV 量化，也都是围绕同一项展开。


## 02 MQA：一个写头就够了

### 2.1 做法：Q 头保留，K/V 全部共享

2019 年，Transformer 原始论文的作者之一 Noam Shazeer 发表了[《Fast Transformer Decoding: One Write-Head is All You Need》](https://arxiv.org/abs/1911.02150)，提出 Multi-Query Attention（MQA）。

改动非常直接：**h 个 Q 头全部保留，但 K、V 各只保留一份投影，由所有 Q 头共享**：

$$Q_i = XW_Q^{(i)},\quad K = XW_K,\quad V = XW_V$$

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K^\top}{\sqrt{d_h}} + M\right) V$$

每个 Q 头仍然拥有自己的 $W_Q^{(i)}$，输出各自的注意力分布——**多样性保留在"提问"一侧**；但它们查询的是同一套"档案"。KV cache 随之缩小 h 倍：h=32 即缩至 1/32，h=64 即缩至 1/64。

### 2.2 标题的含义：为什么是"写"头

论文标题使用了一个双关：write-head（写头）对应 read-head（读头）。将 K/V 视为"写入 cache 的档案"，将 Q 视为"读取档案的查询"——MQA 的主张即：**读的方式可以多样（多个 Q 头），但存档一份即可（一个 K/V 头）**。

这一视角比"共享权重"更准确：它指出 K/V 的真正成本不在计算，而在于**被反复从显存中读取**。

### 2.3 论文的实验结果

Shazeer 在 WMT'14 英德翻译等任务上进行了对照实验：

| 指标 | MHA | MQA |
|---|---|---|
| WMT'14 EN-DE BLEU | 28.4 | 28.5 |
| 语言模型 perplexity | 29.9 | 30.2 |
| 增量解码单 token 耗时 | 46 μs | **3.8 μs（约 12 倍加速）** |

质量几乎无损（BLEU 的 0.1 差异属于噪声范围），解码速度却提升了一个数量级。论文还排除了"模型变小"这一平凡解释：单纯减少头数或压缩头维度均无法达到同样的效果——**共享才是关键，缩小不是**。

### 2.4 三个局限

实验结果之外，MQA 还存在三个局限，也解释了它为何没有成为最终方案：

**① 质量损失与训练不稳定。** 单一 K/V 意味着所有 Q 头只能从同一"视角"读取历史。在聚合类 benchmark 上该损失几乎不可见，但在需要多角度检索上下文的任务上可以测量。GQA 论文附录给出了更具体的描述：从头训练的 MQA 模型在预训练中**频繁出现 loss 尖峰**，且在长输入任务上微调时**直接发散**；即使经 uptraining 转换的 MQA，训练方差也大到需要对不稳定任务报告**三次微调的平均分**。作为对照，uptraining 得到的 GQA 训练过程稳定。MQA 的质量问题不只是指标下降，还涉及训练动力学的稳定性。

**② 张量并行（TP）适配困难。** 多卡推理时，MHA 的 h 个 K/V 头可以自然地按卡切分；MQA 只有一个 K/V 头，要么在每张卡上**完整复制**（浪费显存），要么引入专门的 KV 并行调度（增加通信）。节省的显存会被并行策略部分抵消。

**③ 提出时间早于需求。** 2019 年推理成本尚非关注焦点，BERT 时代的研究重心是训练与理解任务，该论文影响有限。直到 2022 年 Google 的 PaLM（540B）率先在大模型上采用，随后 Falcon、StarCoder、Gemma 1 跟进，MQA 才得到大规模验证。但单一 K/V 的质量上限是客观存在的——Gemma 2 重新换回了 GQA。MQA 的历史意义在于验证了"共享 K/V"的可行性；此后，更均衡的方案接过了这一方向。


## 03 GQA：不是新结构，而是一个旋钮

### 3.1 机制：分组共享

2023 年，Google 的 Ainslie 等人发表[《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》](https://arxiv.org/abs/2305.13245)（EMNLP 2023），提出 Grouped-Query Attention。

思路可以概括为：把 h 个 Q 头分成 g 组（$1 < g < h$），**每组共享一对 K/V**：

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_{\lceil i \cdot g / h \rceil}^\top}{\sqrt{d_h}} + M\right) V_{\lceil i \cdot g / h \rceil}$$

需要强调的是：**GQA 不是一个新结构，而是一个旋钮**。设 K/V 头数为 g：

- g = h：每个 Q 头独享 K/V —— 就是 MHA；
- g = 1：所有 Q 头共享一对 K/V —— 就是 MQA；
- 1 < g < h：中间的任意点都是 GQA，g 越小 cache 越小、质量损失越大。

MHA 与 MQA 之间不再是二选一，而是一条连续的权衡曲线。

### 3.2 论文结果：接近无损的加速

GQA 论文在 T5-Large / T5-XXL 上做了系统实验，在摘要、长文摘要、翻译、问答共 7 个任务上取平均分（XXL 行均为 5% 算力 uptraining 后的结果）：

| 方案 | 平均任务得分 | 单样本推理耗时 |
|---|---|---|
| MHA-Large | 46.0 | 0.37 s |
| MHA-XXL | **47.2** | 1.51 s |
| MQA-XXL | 46.6 | 0.24 s |
| GQA-8-XXL | **47.1** | **0.28 s** |

这张表包含的信息不止一处：

- **GQA-8 的质量代价可忽略**：47.1 vs 47.2，质量差 0.1 分（噪声级别），速度却比 MHA-XXL 快约 5.4 倍，与 MQA 基本持平；逐任务看也同样接近，例如 TriviaQA F1 为 81.6 vs 81.9。GQA 后来被开源社区广泛采用，依据正是这一结果：它将 MQA 的"质量代价"压缩到了噪声级别；
- **更大的 MQA 模型优于更小的 MHA 模型**：MQA-XXL（46.6 分、0.24s）相比 MHA-Large（46.0 分、0.37s）**质量更高且速度更快**。论文据此提出一个实用的部署策略：与其部署较小的 MHA 模型，不如将同样的预算用于训练更大的 MQA 模型——相同时间预算下获得更高的质量；
- **MQA-XXL 与同规模 MHA 仍相差 0.6 分**：MQA 的损失在小尺寸上被噪声掩盖，在 XXL 规模的严格对照下显现出来——这正是 GQA 要弥补的差距。

论文自身声明的局限也应说明：实验全部基于 encoder-decoder 架构的 T5，decoder-only 模型未验证（作者预期 GQA 在 decoder-only 上的优势更大，因为自注意力每层都需要读取 cache）；摘要任务采用的 Rouge 指标本身存在缺陷；uptraining 模型也未与同等条件下从头训练的模型对比。这些留白后来由开源社区的大规模实践填补。

### 3.3 另一项贡献：uptraining 配方

GQA 论文实际包含两个贡献，常被提及的只有"分组"。另一项同样重要：**无需从头训练，即可将已有的 MHA 模型低成本改造为 GQA/MQA**。

配方分两步：

1. **Checkpoint 转换**：把同一组内 h/g 个头的 K/V 投影矩阵**逐元素平均（mean pooling）**，作为该组共享 K/V 投影的初始化。消融实验表明，平均池化优于"选取组内第一个头"，更远优于随机初始化——直觉上也合理：平均最大程度保留了原模型的信息。顺带指出，该实验本身也是"头冗余"的佐证：若各头的 K/V 投影差异很大，平均得到的结果本应没有意义；
2. **Uptraining**：用原预训练数据和设置继续训练约 **5%** 的原始算力。实验显示 5% 已足够恢复质量，增加到 10% 则收益递减。GQA 甚至转换后不训练（α=0）即可使用，MQA 则必须经过 uptraining 才可用。

这意味着 2023 年已训练完成的大量 MHA 模型，都可以约 5% 的额外成本迁移到新结构。

### 3.4 为什么折中点在 8 组

论文的组数消融实验揭示了一个反直觉的规律：**从 MQA（g=1）开始增加组数，最初几乎不损失速度**——大模型头数多，K/V 尺寸下降很快，此时 KV 读取已不是唯一的带宽项；越接近 MHA，每增加一组的代价越大。于是 g=8 成为质量与速度的折中点，被 LLaMA-2 70B、LLaMA-3 全系列、Mistral 等广泛采用。

另有一个并行策略上的实际考虑：**让 g 能被 TP 度数整除**（例如 TP=8 配 g=8），每个 rank 恰好分到完整的 K/V 头，避免 MQA 那种跨卡复制的浪费。GQA-8 与 8 卡 TP 的组合并非巧合。


## 04 参考实现

在上一篇的 MHA 实现基础上改为 GQA，核心改动只有投影维度与一处广播：

```python
class CausalGQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.h, self.g = n_heads, n_kv_heads     # 要求 h 能被 g 整除
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)  # 只投影 g 个头
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)  # (B,h,T,dh)
        k = self.wk(x).view(B, T, self.g, self.dh).transpose(1, 2)  # (B,g,T,dh)
        v = self.wv(x).view(B, T, self.g, self.dh).transpose(1, 2)  # (B,g,T,dh)
        # （此处省略：对 q、k 施加 RoPE，k 须在写入 cache 前完成）

        if kv_cache is not None:             # cache 只存 g 个头——省显存的关键
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)
            kv_cache.k, kv_cache.v = k, v

        # 广播：每个 KV 头复制给组内 h/g 个 Q 头
        k = k.repeat_interleave(self.h // self.g, dim=1)  # (B,h,T,dh)
        v = v.repeat_interleave(self.h // self.g, dim=1)
        # 之后与 MHA 完全一致
        S = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if kv_cache is None:
            mask = torch.triu(torch.full((T, T), float('-inf'),
                                         device=x.device), diagonal=1)
            S = S + mask                     # 解码时 T=1，天然无需掩码
        P = S.softmax(dim=-1)
        o = (P @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(o)
```

三个实现细节：

- **cache 只存 g 个头**：`repeat_interleave` 发生在读取 cache 之后、注意力计算之前。省显存的关键在于"存得少"，而不是"算得少"；
- **`h % g == 0` 是硬约束**：组内头数必须均匀。这也是主流模型头数取 8 的倍数的原因之一；
- **生产 kernel 不做显式 repeat**：FlashAttention、vLLM 等原生支持 MQA/GQA，直接接收 (B, g, T, dh) 形状的 K/V，在 kernel 内部完成广播，避免物化复制后的张量。上面的写法用于对齐公式，便于理解。


## 05 工程实践要点

### 5.1 快速判别：查看 config 的两个字段

对于任何 HuggingFace 模型，无需查阅论文，从 config 即可判断其注意力类型：

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
print(cfg.num_attention_heads, cfg.num_key_value_heads)  # 32, 8 → GQA-8
```

判别规则：`num_key_value_heads == num_attention_heads` 是 MHA；`== 1` 是 MQA；介于两者之间是 GQA。

### 5.2 参数量与补偿

MHA 的 K/V 投影共 $2d^2$ 参数；GQA 缩为 $2g \cdot d \cdot d_h = 2g d^2 / h$。以 h=64、g=8 为例，K/V 参数只剩 1/8，每层节省约 $1.75d^2$。相对注意力模块的总参数 $4d^2$ 而言比例不大，但有些模型会将节省的参数补回 FFN 以保持总参数量可比——Yi 系列即采用这一做法。阅读模型卡时若发现 FFN 尺寸不合惯例，多半与此有关。

### 5.3 张量并行的适配问题与 Pope 等人的解法

TP 切分时，K/V 头是最小的切分单位：

- g ≥ TP 度数且整除：每个 rank 分得完整的 K/V 头，最理想；
- g < TP 度数（MQA 必然如此）：按头维度无法切分，只能将唯一的 K/V 头**复制到每张卡上**——节省的显存被完全抵消。

这一问题在 PaLM 540B 的部署中真实出现过。Google 的 Pope 等人在[《Efficiently Scaling Transformer Inference》](https://arxiv.org/abs/2211.05102)（MLSys 2023）中给出的解法很直接：**MQA 的注意力计算不按头切分，改按 batch 维度切分**——每张卡只存储自己负责的那部分序列的 KV cache，显存节省按卡数等比兑现。代价是每步增加一次 all-to-all 通信以重排激活，但论文的量化分析表明：KV cache 比每步的 Q/K/V 激活大几个数量级，**以小型张量的通信时间换取大型张量的搬运时间，收益明确为正**。效果：PaLM 540B 在 64 芯片上，batch=512 时支持的最大上下文从 MHA 的 330 token 提升至 10,700 token（**约 32 倍**）。

该论文还包含两个值得借鉴的工程细节：

- **prefill 与 decode 使用不同布局**：prefill 时 K/V 的读取被整段 prompt 摊薄，不构成瓶颈，仍按头切分；只有 decode 阶段切换到按 batch 切分。一个推理系统中两套布局并存，按阶段切换；
- Falcon 40B 采用了另一条路径：不使用纯 MQA，而保留 8 个 K/V 头（本质上已是 GQA，只是 2023 年 5 月时这一名称尚未普及），使 TP=8 时每卡恰好分到一个 K/V 头，问题在结构选型阶段即被规避。

选择 g 时应与部署的 TP 拓扑一并考虑，这是推理部署的基本要求。

### 5.4 与其他优化的正交性

MQA/GQA 改变的是"存多少"，与下列优化互不冲突、效果叠加：

- **FlashAttention**：减少注意力内部的 IO，支持 GQA 形状直接输入；
- **PagedAttention / 连续批处理**：管理 cache 的内存布局与调度——需要注意，它解决的是**显存碎片与调度**问题，并不减少每步 KV 的读取量，带宽问题仍需 MQA/GQA 这类结构方案；
- **KV cache 量化**（INT8/FP8）：在 g 决定的尺寸之上进一步压缩精度；
- **MLA**：从低秩压缩的角度解决同一问题（下一篇的主题）。

Shazeer 在 2019 年即验证过这种正交性：MQA 与局部注意力（local attention）叠加时，收益是相加的。


## 06 演化时间线：采用情况与选型

| 年份 | 模型 | 方案 | Q 头 / KV 头 | 备注 |
|---|---|---|---|---|
| 2019 | Shazeer 论文 | MQA | h / 1 | 提出早于需求，影响有限 |
| 2022 | PaLM 540B | MQA | 多 / 1 | MQA 首次大规模落地 |
| 2022 | Pope et al.（PaLM 部署） | MQA + 按 batch 切分 | — | 支持上下文提升 32 倍 |
| 2023 | Falcon 7B、StarCoder | MQA | 71 / 1、48 / 1 | 开源社区跟进 |
| 2023 | Falcon 40B | 8 个 KV 头 | 多 / 8 | 事实上已是 GQA，早于论文命名 |
| 2023 | LLaMA-2 | 混合 | 70B：64 / 8；7B/13B：MHA | GQA 首次用于旗舰模型，仅最大规格 |
| 2023 | Mistral 7B | GQA | 32 / 8 | 发布即采用 GQA |
| 2024 | Gemma 1 → Gemma 2 | MQA → GQA | Gemma 2 每 2 个 Q 头一组 | Google 从 MQA 退回 GQA |
| 2024 | LLaMA-3 | GQA | 8B：32 / 8；70B：64 / 8 | GQA 推广至全尺寸 |
| 2024+ | Qwen、Yi 等 | GQA | 多为 4:1 ~ 8:1 | 国内模型全面跟进 |
| 2024+ | DeepSeek-V2/V3 | MLA | 低秩压缩 | 更换思路：不减头数，压缩表示 |

几个细节：

- **小模型一度不采用 GQA**。LLaMA-2 只在 70B 上使用 GQA，7B/13B 仍为 MHA——小模型的 KV cache 本来就不大，压缩收益有限。到 LLaMA-3 时代，长上下文与大 batch 推理普及，GQA 才推广至全尺寸；
- **Gemma 的反复是一次有参考价值的公开实验**。Gemma 1 使用 MQA，Gemma 2 换回 GQA，且技术报告明确说明依据是消融实验：GQA 在提升推理速度的同时保持了下游指标。Google 的实际选型说明，单一 K/V 的质量损失在严格的评测体系下是真实成本；
- **g 的选择远比"8"多样**。Gemma 2 每 2 个 Q 头一组，Qwen2.5-7B 为 28/4（7:1），LLaMA 家族均采用 8。8 并非定律，应结合自身 TP 拓扑与质量要求选择。


## 07 结语

本篇的关键结论：

1. decode 阶段是带宽瓶颈：内存访问/算力比中的 $\frac{n}{d}$ 项（每步重读 K/V）随上下文线性增长，且无法通过 batch 摊薄，是生产环境下的主要矛盾；
2. MQA 的核心洞察：多样性的价值集中在 Q 侧（如何查询），K/V 侧（存储内容）可以大幅共享——cache 缩小 h 倍，解码速度提升一个数量级；
3. MQA 的代价不止于指标下降：从头训练 loss 尖峰频发、长输入微调直接发散、uptrained 版本需要多次微调取平均；TP 下还有 KV 复制的浪费（Pope 等人以按 batch 切分解决，代价是每步一次 all-to-all）；
4. GQA 是旋钮而非新结构：g=1 即 MQA，g=h 即 MHA。XXL 规模上 GQA-8 质量仅差 0.1 分、速度快 5 倍以上；MQA-XXL 则验证了"更大的 MQA 模型胜过更小的 MHA 模型"的部署策略；
5. uptraining 配方（mean pooling 初始化 + 5% 算力续训）使存量 MHA 模型得以低成本迁移，是 GQA 论文中同样重要、却较少被提及的贡献；
6. 这一切的学理基础仍是上一篇的伏笔：**头本来就冗余**——mean pooling 之所以有效，恰因为各头的 K/V 投影本就高度相似。

但无论 g 如何调整，cache 中存储的始终是完整的 K/V，只是数量减少。能否对每份 K/V 本身也进行压缩？2024 年 DeepSeek-V2 的 MLA 给出了回答：将 K/V 联合低秩压缩为潜向量存入 cache，通过矩阵吸收避免展开历史 K/V——论文报告相对 DeepSeek 67B 的 KV cache 减少 93.3%。[下一篇讨论 MLA](/ai-infra/attention-3-mla)。


## 参考文献

1. Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
2. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), EMNLP 2023.
3. Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), MLSys 2023.
4. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
5. Voita et al., [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://aclanthology.org/P19-1580/), ACL 2019.
6. Touvron et al., [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288), 2023.
7. Gemma Team, [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118), 2024.
8. DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
