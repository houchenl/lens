---
title: 注意力机制-9 GLM 与 IndexShare
date: 2026-10-14
description: 从 GLM 的分组 KV、MLA 和 DSA 分支出发，深入解释 GLM-5.2/5.3 的原生 IndexShare：跨层共享地址如何训练、为何不等于共享主 KV，以及怎样读懂 indexer_types、index_topk_freq 与架构类名。
---

# 注意力机制-9 GLM 与 IndexShare

在[DSA](/ai-infra/attention-4-dsa)中，轻量索引器为每个 query 选择历史位置，但相邻层常常重复选择相似内容。GLM-5.2 将 IndexShare 引入原生模型训练，让一组层共用一次选择；GLM-5.3 主模型延续这一基础。其核心问题与 CSA2 有交集，却保留了不同的状态边界：共享地址，并不自动共享各层主 KV。

本文先通过公开配置梳理 GLM 的注意力分支，再解释共享选择的数据流、训练目标、复杂度和配置含义。资料以 **2026-10-14** 核对的官方模型卡、配置、IndexCache 论文与 Transformers 实现为准；GLM-5.3-Flash 的稀疏/线性混合结构留到本系列第 16 篇。

## 00 同一家族，为什么不能只看版本号判断注意力

同一模型家族可能同时维护大模型主线、小型 Flash 分支和多模态分支。名称相近，不代表注意力模块完全一致。尤其不能从 `num_key_value_heads` 一个字段，就判断模型一定是普通 GQA，或从 `Flash` 后缀推断其采用某种执行算法。

以下配置例子展示结构差异，而不是声称覆盖所有闭源产品：

| 分支 | 可核对的结构 | 证据入口 |
|---|---|---|
| ChatGLM-6B | 独立多头 Q/K/V 的早期结构 | [官方模型代码](https://huggingface.co/zai-org/chatglm-6b/blob/main/modeling_chatglm.py) |
| ChatGLM2-6B | 32 个 query 头、2 个 KV 组，配置名为 multi-query | [配置](https://huggingface.co/zai-org/chatglm2-6b/blob/main/config.json) |
| GLM-4.7 主模型 | 96 个 query 头、8 个 KV 头的 GQA | [配置](https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json) |
| GLM-4.7-Flash | 带低秩 KV 和解耦位置维度的 MLA 分支 | [配置](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json) |
| GLM-5 | MLA 历史表示与 DSA 动态选择 | [配置](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)与[官方说明](https://github.com/zai-org/GLM-5) |
| GLM-5.2 / 5.3 主模型 | 在 DSA 上原生采用 IndexShare | [5.2 配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)、[5.3 配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) |

ChatGLM2 的字段虽叫 `multi_query_attention`，但 KV 组数为 2；按本系列统一术语，它属于分组共享，不能机械解释成全模型只有一个 KV 头。GLM-4.7-Flash 的配置虽然也声明多个 KV 头，却具有潜在表示和上投影，不能只按头数估算实际压缩缓存。

这些分支可以归纳成两条相关路线：分组共享减少 KV 头数；MLA/DSA 主线进一步减少表示成本和主注意力访问范围。IndexShare 接着处理层间重复选择，而不是重新定义 token 的含义。

## 01 从一次 DSA 选择回顾需要共享的对象

令层号为 $l$、query 位置为 $t$、历史位置为 $j$，索引头数为 $H_I$、维度为 $d_I$。索引器计算：

$$
I_{l,t,j}=\sum_{a=1}^{H_I}\omega_{l,t,a}
\operatorname{ReLU}\left((q^I_{l,t,a})^\top k^I_{l,j}\right),\qquad
\mathcal S_{l,t}=\operatorname{TopK}_{j\le t}I_{l,t,j}
$$

缩放并入头权重 $\omega$。索引器给出历史位置集合，主注意力再用本层 query 与本层 KV，在这些位置上计算自己的概率和输出。

如果一组层 $G_f$ 共用来源层 $f$ 的选择，就有：

$$
\mathcal S_{l,t}=\mathcal S_{f,t},\qquad l\in G_f
$$

这只约束地址。它不要求主 key $k_{l,j}$、value $v_{l,j}$ 或 query $q_{l,t}$ 与来源层相同，也不要求两个层输出相同。共享的位置仍对应同一序列的 token，但实际读取的向量由各层生成。

```text
来源层：本层索引 Q/K → 选择历史地址 ─┬→ 本层主 Q/K/V → 输出
                                  ├→ 下一层主 Q/K/V → 新输出
                                  └→ 再下一层主 Q/K/V → 新输出
```

因此，与 CSA2 共享主 KV、索引 key 并可重新选择相比，IndexShare 把主要节约集中在选择路径。[参考实现的 `full/shared` 分支](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)可直接验证这一边界。

## 02 为什么需要让索引器学会服务多个层

### 2.1 相邻选择相似，不等于可以任意替换

假设层 A 关注函数定义与参数，层 B 更关注异常处理。两层的 Top-K 可能大部分相同，但 B 额外需要的少量位置恰好决定答案。集合重叠率高，并不能保证删除另一个索引器没有影响。

免训练的 IndexCache 可以在校准数据上搜索保留哪些来源层；训练感知方法则让保留的索引器学习多层需求。[IndexCache 论文 §3](https://arxiv.org/html/2603.12201#S3)将两者明确区分。GLM 的官方发布说明将 IndexShare 链接到该工作，但模型原生训练安排与论文所有实验配置不能混称。

### 2.2 平均多层目标，改变的是选择任务

用一个抽象蒸馏目标解释多层适配。设 $p_l(j)$ 是第 $l$ 层在某 query 下给出的归一化目标分布，$\pi_f(j)$ 是共享索引器的概率分布；目标停止梯度。一个共享组有 $g$ 层，取平均 KL 散度：

$$
\mathcal L_f=\frac1g\sum_{l\in G_f}
D_{\mathrm{KL}}\left(p_l\Vert\pi_f\right)
$$

定义 $\bar p(j)=\frac1g\sum_l p_l(j)$，展开得到：

$$
\mathcal L_f=-\sum_j\bar p(j)\log\pi_f(j)+C
$$

$C$ 与索引器参数无关，因此其梯度与使用平均分布 $\bar p$ 作为目标相同。若用 KL 的求和而非平均，梯度多一个 $g$ 倍系数。这个目标让索引器服务整个组，而不是只模仿来源层。

例如 $p_A=(0.7,0.2,0.1)$、$p_B=(0.1,0.2,0.7)$，平均目标为 $(0.4,0.2,0.4)$。当预算允许选两个位置时，兼顾两端比只复制 A 的单层偏好更合理；预算只有一个时，冲突依然存在，训练无法消除固定预算的表达约束。

该推导说明 IndexCache 的训练感知原理，不代表 GLM-5.2 发布了完整逐项相同的训练配方。官方能确认的是：GLM-5.2 从 128K 序列的中期训练开始使用 IndexShare，而非仅在推理时临时启用。[官方发布说明](https://z.ai/blog/glm-5.2)。

## 03 从四层示意，读到真实的 78 层配置

### 3.1 周期共享与边界例外

对一组四层，常见安排是 F、S、S、S：F 层生成当前 query 的 Top-K，后面三个 S 层沿用它，下一个 F 层再更新。这里 F 指完整执行索引器，不表示主注意力变成稠密。

但 GLM-5.2/5.3 的实际 `indexer_types` 并非从第 0 层起简单重复这个四元组。按公开配置，0、1、2 层为 `full`，随后 3—5 为 `shared`，6 为下一个 `full`，再每四层设置一个来源。

```python
full_layers = [0, 1] + list(range(2, 78, 4))
owners = []
owner = None
for layer in range(78):
    if layer in full_layers:
        owner = layer
    owners.append(owner)
assert len(full_layers) == 21
assert owners[3:6] == [2, 2, 2]
assert owners[6:10] == [6, 6, 6, 6]
assert owners[77] == 74
```

标准库示例仅重现主干来源关系。78 层中有 21 个产生选择的层、57 个共享层；边界例外使全主干索引次数下降比例约为 $57/78\approx73.1\%$，不能不加条件地写成恰好 75%。MTP（Multi-Token Prediction，多 token 预测）还有自身共享安排，不在这个主干统计中。

### 3.2 架构类名为什么保持相同

GLM-5、5.2 和 5.3 都可能声明 `GlmMoeDsaForCausalLM`。其中 `Glm` 是模型族，`Moe` 表示混合专家前馈模块，`Dsa` 表示稀疏注意力模型实现族，`ForCausalLM` 是自回归语言建模任务封装。

这个名字是加载模型实现的重要线索，但不是逐层结构的完整规范。同一实现类仍可根据配置决定哪些层拥有索引器、维度如何设置、位置编码怎样处理。必须把类名、配置、权重中的模块和实际执行路径一起读。

官方说明还明确指出 GLM-5.3 主模型沿用 5.2 的基础模型，改进来自后训练；不能凭版本号自行补出一次注意力架构更换。[官方仓库](https://github.com/zai-org/GLM-5)。

## 04 为什么不能只改 `index_topk_freq=8`

`hf-overrides` 指推理框架在加载 Hugging Face 模型配置时应用覆盖值。它改变的是软件读到的配置，不会重写权重，也不会自动训练缺失组件。

在 IndexShare 语境中，至少需要核对三个层次：

| 对象 | 需要确认的事实 |
|---|---|
| 模型本身 | 是否具有 DSA/索引复用结构，以及对应权重 |
| 实现的配置规则 | 使用 `indexer_types`、pattern，还是频率与 offset 生成安排 |
| 实际执行 | 哪些层真的运行索引器，哪些层复用哪个来源的地址 |

原生配置包含显式 `indexer_types`、`index_topk_freq=4` 和 `index_skip_topk_offset=3`。如果某实现优先使用显式列表，只覆盖频率可能没有实际效果。若实现用新频率重建列表，也必须确认新的来源层具有权重；原生共享层可能根本不包含独立索引模块。

即使新安排只删除部分已有索引器，它仍把训练时的四层共享改成更长的共享，质量影响需要验证。不能从“DSA 模型”推出“任意模型都可安全使用频率 8”。相关部署细节已在第 4 篇展开，本篇把它落回模型与权重约束。

## 05 成本如何变化：索引省了，主 KV 仍在

令 $L$ 为层数、$F$ 为运行索引器的层数、$T$ 为长度、$J=H_I d_I$ 为索引打分维度成本。若基础索引器全范围扫描，单 query 索引匹配量从 $O(LJT)$ 变成 $O(FJT)$；prefill 仍有 $O(FJT^2)$ 项。

每层主注意力仍在最多 $K$ 个位置上计算，约为 $O(LTHD K)$ 的 prefill 量级，$H,D$ 概括主头数和有效匹配/聚合维度。主 KV 仍需各层保存，以便未来不同 query 选择不同历史。

如果不共享前索引计算占总时间比例为 $a$，其余开销不变，仅按索引次数估算的理想加速为：

$$
S\approx\frac1{(1-a)+aF/L}
$$

例如 $a=0.6,F/L=1/4$，得到 $S\approx1.82$。这是解释局部加速为何不能直接等于整体四倍的成本模型；实际占比随长度、设备和实现变化，不能当作 GLM 的实测数据。

GLM-5.2 官方在 1M 上下文给出 2.9 倍的每 token FLOPs 改善，应保留其计算量与上下文条件，不能改写成任意负载下的 2.9 倍延迟改善。MTP 的收益也应与索引共享分开归因。

## 06 如何检验共享是否合理

结构检查先确认每个 Shared 层都能找到同一请求、同一 query、同一历史边界的有效来源，随后核对主 KV 是否仍按层独立生成。来源地址必须和当前序列位置一致，不能跨请求或跨生成步沿用旧选择。

质量比较应包含多层需要不同证据的任务，而不只看单个针检索。可以分别比较每层独立索引、原生四层共享、人工更大共享组，记录长文多证据、代码跨文件修改和推理链的失败案例。配置生效、索引减少、任务质量三种证据各自回答不同问题。

GLM 的演进说明，模型设计不仅可以修改相似度函数，也可以修改“谁负责选择、选择服务哪些层”。下一篇讨论 **MoBA**：将选择粒度从 token 转为连续历史块，用块表示完成内容路由，再读取块内细节。

继续阅读：[注意力机制-10 MoBA](/ai-infra/attention-10-moba)。

## 参考资料

1. Z.ai，[GLM-5 系列官方仓库](https://github.com/zai-org/GLM-5)、[GLM-5.2 发布说明](https://z.ai/blog/glm-5.2)与[GLM-5.3 发布说明](https://z.ai/blog/glm-5.3)。
2. Bai et al.，[IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)，2026。重点见 §3 的校准搜索、多层蒸馏与平均目标推导。
3. Z.ai，[GLM-5.2 配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)、[GLM-5.3 配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)。其他分支配置见第 00 节链接。
4. Hugging Face，[GLM MoE DSA 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)。滚动分支，以上资料查阅于 2026-10-14。
