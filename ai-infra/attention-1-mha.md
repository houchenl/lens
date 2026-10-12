---
title: 注意力机制-1 MHA
date: 2026-10-12
description: 这是一个关于注意力机制的系列文章。作为开篇，我们从最经典的形式讲起：decoder-only LLM 中基于 MHA（Multi-Head Attention，多头注意力）的注意力模块。不堆术语、不跳步骤，把从公式到显存的每个细节讲透。
---

# 注意力机制-1 MHA

> 这是一个关于注意力机制的系列文章。作为开篇，我们从最经典的形式讲起：decoder-only LLM 中基于 MHA（Multi-Head Attention，多头注意力）的注意力模块。不堆术语、不跳步骤，把从公式到显存的每个细节讲透。


## 00 为什么第一篇讲 MHA

2017 年，Google 的八位作者发表了[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)，MHA 由此诞生。它最初是为**机器翻译的 encoder-decoder 架构**设计的：8 个头、$d_{\text{model}}=512$、$d_k=64$。几年后，这套为翻译而生的组件成为了 GPT、LLaMA、Qwen、DeepSeek 等所有大模型的共同骨架。

为什么必须先讲 MHA？因为在 decoder-only Transformer 的骨架里，**注意力模块是唯一负责 token 之间信息交换的组件**（FFN 是逐 token 独立加工的）。理解 MHA，就握住了后续一切的钥匙：FlashAttention 是在 MHA 上做 IO 优化，GQA/MQA 是在 MHA 上砍 KV cache，MLA 是在 MHA 上做低秩压缩——它们全是"MHA 的变体与改良"。


## 01 全景：一个 Token 的诞生

在钻进注意力内部之前，先用一分钟建立全局视角。以 LLaMA 类模型为例，从一段文本到输出下一个 token，完整流程是：

```
"今天天气"
  → ① 分词（BPE/SentencePiece）：[id₁, id₂, id₃]      # 文本切成离散 token id
  → ② Embedding 查表：X⁰ ∈ ℝⁿˣᵈ                        # 每个 id 映射成 d 维向量
  → ③ L 个 Block 循环：x ← x + MHA(RMSNorm(x))         # 注意力交换信息
                       x ← x + FFN(RMSNorm(x))         # FFN 加工信息（SwiGLU）
  → ④ 最终 RMSNorm
  → ⑤ LM Head 线性投影：logits ∈ ℝⁿˣⱽ（V 为词表大小）
  → ⑥ 取最后一个位置的 logits，softmax → 词表上的概率分布
  → ⑦ 采样（argmax / top-k / top-p / temperature）→ 下一个 token
  → ⑧ 拼回输入，自回归循环（后续步走增量解码 + KV cache）
```

两个容易混淆的细节，先在这里埋个伏笔：

- **位置信息不在 embedding 处"添加"**。GPT-2/BERT 式模型把可学习位置向量加到 embedding 上，而现代主流模型 embedding 里没有任何位置信息——位置完全靠每层注意力内部对 Q/K 施加的 **RoPE** 旋转注入；
- **训练时不需要显式 softmax**。交叉熵损失在数学上把 softmax 吸收了，只有推理采样时才真正需要概率分布。


## 02 单头注意力：Q、K、V 的三重角色

设某层输入为 $X \in \mathbb{R}^{n \times d}$（n 个 token，每个 d 维）。

### 2.1 一次输入，三种投影

三个可学习矩阵把**同一份输入**投影成三种角色：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

直觉记忆法：

- **Q（Query）**："我想找什么样的信息"
- **K（Key）**："我能被什么查询匹配到"——相当于索引标签
- **V（Value）**："我实际携带的内容"

Q 与 K 的相似度决定**往哪看**，V 决定**看什么**。三者同源，所以叫自注意力（self-attention）。

### 2.2 缩放点积：为什么是 $\sqrt{d_k}$

$$S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

$S_{ij}$ 是 token i 对 token j 的未归一化关注程度。

除以 $\sqrt{d_k}$ 的原因：若 q、k 各分量独立、均值 0、方差 1，则点积 $q \cdot k$ 的方差约为 $d_k$。维数越大，打分绝对值越大，softmax 就越深入饱和区，梯度趋近于 0，训练停滞。除以 $\sqrt{d_k}$ 把方差拉回约 1，保住梯度健康。

### 2.3 因果掩码：Decoder-only 的灵魂

构造上三角掩码 $M$（$j > i$ 时为 $-\infty$，其余为 0），加到打分上再做 softmax：

$$P = \text{softmax}(S + M)$$

$-\infty$ 经过 softmax 后精确变为 0，效果就是：**第 i 行的注意力分布只在列 0..i 上非零**——每个 token 只能"看到"自己和历史，看不到未来。

这一步的意义怎么强调都不为过：

- 训练时，整段序列可以**一次性并行**前向，却不泄漏"答案"；
- 推理时，模型天然给出条件概率 $p(x_{i+1}\,|\,x_{\le i})$，与自回归生成严丝合缝。

没有因果掩码，就是 BERT 那样的双向 encoder；有了它，才是 GPT 这条路线。

### 2.4 加权聚合，输出结果

$$O = PV \in \mathbb{R}^{n \times d_k}$$

每个输出 $o_i = \sum_{j \le i} P_{ij} v_j$，是前缀所有 token 的 value 的**凸组合**（softmax 保证权重非负且和为 1）。信息就这样从上下文流入当前位置。

### 2.5 一个记号问题：S、P 还是 A？

细心的读者会发现，不同资料里注意力矩阵的记号五花八门。这里说清来历，以后读文献不迷路：

| 记号 | 出处 | 含义 |
|---|---|---|
| **S / P** | FlashAttention 论文及系统社区 | S = softmax 前的打分矩阵，P = softmax 后的概率矩阵 |
| **A** | 经典教材、一般论文 | 通常指 softmax 后的注意力权重，但有时也指打分，需看上下文 |

为什么 FlashAttention 要刻意区分 S 和 P？因为它的核心贡献恰恰是**不物化这两个 $n \times n$ 矩阵**——分块计算 + online softmax，全程避免把它们写进 HBM。所以在推理引擎、kernel 优化语境下，**S → P → O 是事实标准记号**，本系列后续也统一用它。

### 2.6 隐藏细节：softmax 的数值稳定性

工程实现从不直接算 $\exp(s)$，而是先减行最大值：

$$\text{softmax}(s_i) = \frac{\exp(s_i - \max_j s_j)}{\sum_j \exp(s_j - \max_j s_j)}$$

数学等价，但避免指数上溢。这个技巧不是可有可无的工程补丁——FlashAttention 的 online softmax 正是在它之上，把分块算出的局部 max/sum 增量合并，实现了"边算边归一"的精确注意力。这个我们后续细讲。


## 03 MHA：一个头不够，那就来一群

### 3.1 动机

单头注意力里，每个位置只能学**一种** Q-K 匹配模式。但语言中的关系是多元的：指代、句法依存、位置邻近、语义关联……一个头装不下。

多头的思路：把 d 维表示空间切成 h 份，每个头在**自己的子空间**里独立做完整注意力，各自捕捉不同模式。

### 3.2 计算流程

设头数 h，每头维度 $d_h = d/h$（如 $d=4096, h=32 \Rightarrow d_h=128$）：

$$\text{head}_i = \text{softmax}\left(\frac{XW_Q^{(i)}(XW_K^{(i)})^\top}{\sqrt{d_h}} + M\right) XW_V^{(i)}$$

$$\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O$$

最后过输出投影 $W_O \in \mathbb{R}^{d \times d}$ 融合各头信息，回到残差流维度。**$W_O$ 不可省略**：拼接只是简单堆叠，头间信息的混合全靠它。

### 3.3 三个反直觉的事实

**① 多头不增加参数量。** 四个投影 $W_Q, W_K, W_V, W_O$ 共 $4d^2$ 参数，与 h 无关。工程上也不是真做 h 次小投影，而是一个 $d \times d$ 大矩阵一次算出，再 reshape 成 $(b, h, n, d_h)$，在头维度上 batched 并行——对 GPU 张量核非常友好。多头改变的是信息的**组织方式**，不是计算量。

**② 每个头有低秩瓶颈。** 单头的注意力矩阵秩受 $d_h$ 限制：$d_h$ 太小损害表达力，太大则头数太少、模式单一。经验甜点在 $d_h = 64 \sim 128$，主流模型几乎全部落在这个区间——2017 年原始论文的 $d_k=64$ 至今仍是常见默认值。

**③ RoPE 只加在 Q/K 上，不加在 V 上。** 位置信息只需影响"匹配"（Q·K），不应污染"内容"（V）。且 K 必须**先加 RoPE 再写入 KV cache**——推理框架里这是一个经典 bug 高发点。

### 3.4 冷知识：大部分头其实是"摸鱼"的

多头设计的一个隐含假设是"每个头都有用"。但 2019 年的两篇经典研究把这个假设打了个问号：

- Voita 等人在 ACL 2019 的[《Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned》](https://aclanthology.org/P19-1580/)中发现，Transformer 的头可以按功能分为三类：**位置型**（只看邻近 token）、**句法型**（追踪依存关系）、**稀有词型**（关注低频词）——只有这些"专业化"的头在干重活。在英俄翻译任务上，**剪掉 encoder 48 个头中的 38 个，BLEU 仅下降 0.15**；
- Michel 等人在 NeurIPS 2019 的[《Are Sixteen Heads Really Better than One?》](https://arxiv.org/abs/1905.10650)中做了更激进的头剪枝实验：相当一部分层在测试时**只保留一个头**都不会显著掉点。

这两个结论当时看是"可以剪枝压缩模型"，多年后回看却是另一层含义：**头与头之间存在大量冗余，每个头独享一份 K/V 是奢侈的**。这正是 MQA/GQA 敢于大幅共享 K/V 的底气——第 06 节我们会看到，这条"头冗余"的研究线索，如何一路通向今天推理时代的注意力设计。


## 04 训练 vs 推理：两张完全不同的面孔

同一个 MHA 模块，在训练和推理时的计算形态截然不同，理解这一点是理解一切推理优化的前提。

### 4.1 训练：并行 + 掩码

整段序列一次前向，因果掩码防止泄漏。注意力部分 FLOPs 约 $O(n^2d)$（QK^T 与 PV 各一份），加上投影的 $O(nd^2)$。**序列长度 n 是训练成本的核心瓶颈**，这就是 FlashAttention、序列并行存在的意义。

### 4.2 推理：逐 token + KV Cache

自回归生成时每步只有一个新 token。关键观察：位置 j 的 $k_j, v_j$ 只依赖 $x_j$ 与权重，**与后续 token 无关**。于是历史的 K、V 全部可以缓存，新 token 只算自己的 q、k、v，再与缓存做注意力：

$$o_t = \text{softmax}\left(\frac{q_t K_{\le t}^\top}{\sqrt{d_k}}\right)V_{\le t}$$

每步复杂度从 $O(n^2d)$ 降到 $O(nd)$。注意此时**不需要因果掩码**——query 只有最新一个位置，它能看到的恰好全部是历史。

还有一个常被忽略的问题：**为什么不缓存 Q？** 因为解码时只需要"当前目的地 token"的 query 去查询历史，历史 token 的 query 再也用不上；而 K/V 要被未来所有 token 反复查询。Q 是"问完即弃"，K/V 是"持久档案"。

### 4.3 KV Cache 的代价

MHA 下每 token 的 cache 显存：

$$\text{bytes} = \underbrace{2}_{K\text{和}V} \times L_{\text{层数}} \times h \times d_h \times \text{precision}$$

以 7B 模型（L=32，h=32，$d_h$=128，FP16）为例：**每 token 约 0.5 MB，128k 上下文就是约 64 GB 显存**。

### 4.4 Prefill 与 Decode：两种瓶颈

- **Prefill**（处理输入提示）：计算密集，compute-bound；
- **Decode**（逐 token 生成）：每步算力需求小，但要完整读一遍 KV cache 和全部权重，**memory-bandwidth-bound**。

早在 2019 年，Shazeer 在 MQA 论文里就点破了这个本质：增量推理慢的根源不是算力，而是**反复搬运庞大的 K/V 张量的内存带宽开销**——这个判断直接催生了后来整条 KV cache 压缩路线。vLLM 的 PagedAttention、连续批处理、分页 KV 管理，本质上也都是围绕 decode 阶段的带宽瓶颈设计的。这也直接引出了 MHA 的最大软肋，见第 06 节。


## 05 参考实现：三十行看懂全部

```python
class CausalMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)  # 合并投影
        self.wo   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        # (B, T, D) -> (B, h, T, dh)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        # （此处省略：对 q、k 施加 RoPE，k 须在写入 cache 前完成）

        if kv_cache is not None:             # 推理：拼接历史 KV
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)
            kv_cache.k, kv_cache.v = k, v

        S = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,h,T,T_past)

        if kv_cache is None:                 # 训练/prefill：因果掩码
            mask = torch.triu(torch.full((T, T), float('-inf'),
                                         device=x.device), diagonal=1)
            S = S + mask                     # 解码时 T=1，天然无需掩码

        P = S.softmax(dim=-1)                # 内部先做减最大值稳定化
        o = (P @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(o)
```

对照前面的公式逐行读，每一个设计决策——合并投影、reshape 出头维、掩码的位置、cache 的拼接——都能在前文找到出处。


## 06 MHA 的局限与演化地图

MHA 的历史地位在于确立了 **"Q-K 匹配 → softmax 加权 → V 聚合"** 这一范式。它的软肋在第 04 节已经暴露：KV cache 随头数线性膨胀，长上下文推理时显存和带宽双双告急。再叠加第 3.4 节的发现——头之间本就冗余严重——改造的方向就清晰了：**让 Q 头共享 K/V**。后续所有变体，本质都是在**保表达力**与**压 cache**之间做权衡：

| 变体 | 年份/出处 | K/V 头数 | KV cache | 质量 | 思想 |
|---|---|---|---|---|---|
| **MHA** | 2017，Vaswani et al. | h | 最大 | 最好 | 每个 Q 头独享一对 K/V |
| **MQA** | 2019，Shazeer | 1 | 缩小 h 倍 | 有损 | 所有 Q 头共享一对 K/V |
| **GQA** | 2023，Ainslie et al.（EMNLP） | g（1<g<h） | 缩小 h/g 倍 | 接近 MHA | Q 头分组共享 K/V |
| **MLA** | 2024，DeepSeek-V2 | —（低秩压缩） | 最小 | 不损甚至更好 | KV 联合低秩压缩为潜向量 |

几个值得记住的历史细节：

- **MQA 生不逢时又先知先觉**。2019 年 Shazeer 提出时，推理成本还不是焦点，论文影响有限；直到 2022 年 Google 的 PaLM（540B）率先采用，后来 Falcon、Gemma 1 跟进。但单一对 K/V 的质量损失是真实的，Gemma 2 又换回了 GQA——MQA 基本完成了历史使命；
- **GQA 的论文其实有两个贡献**，常被记住的只有"分组"。另一个同样重要：**uptraining 配方**——把已训练好的 MHA checkpoint 的 K/V 投影按组平均池化初始化，再用约 **5% 的原始预训练算力**继续训练，就能平滑转换成 MQA/GQA 模型，不必从头训练。这让存量 MHA 模型也能低成本拥抱新结构；
- **GQA 是一个"旋钮"而非新结构**：设 K/V 头数为 g，g=1 时就是 MQA，g=h 时就是 MHA，中间任意取值都是可调的权衡点。LLaMA-2 70B、LLaMA-3 全系列、Qwen3、Mistral 都选择了 g=8 附近的甜点；
- 用真实数字感受一下差距：LLaMA-3 70B（h=64，GQA-8）在 8K 上下文、FP16 下，MHA 的 KV cache 约 20 GB，GQA-8 压缩到约 2.5 GB（**省 87.5%**），MQA 则只剩约 0.31 GB；
- **MLA 换了条思路**：不减少头数，而是把 K/V 联合低秩压缩成一个潜向量存入 cache，推理时再还原。DeepSeek-V2 论文报告 KV cache 相比前代减少 93.3%，质量不输 MHA。


## 07 结语

回顾一下本篇的关键结论：

1. 注意力是 Transformer block 中**唯一**的 token 间信息交换通道；
2. Q/K/V 同源异投影：匹配靠 Q·K，搬运靠 V，缩放 $\sqrt{d_k}$ 保梯度；
3. 因果掩码是 decoder-only 的灵魂，它让并行训练与自回归推理完美统一；
4. 多头不加参数量，改的是信息组织方式；$d_h \in [64,128]$ 是经验甜点，且研究证明大部分头本就冗余；
5. 训练靠并行 + 掩码，推理靠 KV cache + 增量解码，decode 阶段是带宽瓶颈；
6. MHA 之后的所有变体，都是围绕 KV cache 的显存-质量权衡——而"头冗余"正是这场变革的学理基础。


## 参考文献

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
2. Voita et al., [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://aclanthology.org/P19-1580/), ACL 2019.
3. Michel et al., [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650), NeurIPS 2019.
4. Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
5. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), EMNLP 2023.
6. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
7. DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
