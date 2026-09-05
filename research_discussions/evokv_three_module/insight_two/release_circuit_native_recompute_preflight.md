# Release-circuit native recomputation：head/circuit 候选审计

日期：2026-09-03  
状态：**严格 NO-GO；不进入 UID preflight，不作为 Insight 2 / Design 1 候选**  
范围：现有 Yambda Medium `v0..v5`、单次 Parent→Current、KV-only persistent interface、
`<20% Exact-All`；只做代码图、sealed summary、checkpoint parameter 与 primary-source 审计，未读
label、未读 confirmation population、未运行 Current-Exact selector、未训练

## 1. 裁决先行

本轮检验的候选不是“把某些 head 的 Current Exact K/V 诊断性贴回 Parent cache”，而是更强的命题：

> 跨版本变化是否由少量 Transformer attention heads/circuits 承担，并且这些 circuits 能从现有
> Parent persistent K/V 出发，以原生 Transformer 算子形成 dependency-closed Current continuation？

结论是 **NO-GO**。在当前 HSTU 计算图里，attention head 只在单层 `QK/AV` 内并行；一旦经过 dense
`W_O`、全宽 gate、residual 与下一层 normalization，head 就不是独立的 causal state module。任意一个
upper-layer Current head 的精确输入都依赖下方所有 head 的 response。实际六个 checkpoint 也没有训练出
可作为例外的 block-diagonal head lanes。

严格成本给出两个互补结论：

1. **Exact Current closure**：触达 cache layer 2（0-based，即第三个 K/V layer）的任意一个 head，
   最乐观也需 `1,674,313,728 FLOPs/user = 35.0915% Exact-All`。
2. **Parent-K/V hybrid closure**：即使未选择的 head 直接读 Parent K/V，形成下一层 dense hidden 仍必须
   为所有 head 计算 historical Q、QK/AV、`W_O` 和 gate。跨两个 block 时，把 selected-head work
   乐观降为零后的固定下界已经是
   `1,347,158,016 FLOPs/user = 28.2347% Exact-All`。

唯一留在 `<20%` 内的 exact window 是：

- 只生成 cache layer 0 的任意 head；或
- 完整运行 Current block 0，再生成 cache layer 1 的至多三个 head，三个 head 时为
  `944,111,616 FLOPs/user = 19.7874% Exact-All`。

第一项只是 layer-0 head K/V repair；第二项最终状态是“完整 Current layer 0 + 部分 Current layer 1 +
其余 Parent K/V”。它字面上就是 dependency-closed 的 early-layer/head-wise KV replacement，仍以
token/layer/head locality 为迁移抽象。它与本轮“不是 head-wise KV splice”的候选边界冲突；现有
full-head 功能证据也没有给出正向先验，prior art 则已覆盖其方法成分。因此不值得用一个 subset oracle
把它美化成新 Design。

可以保留的 Transformer-specific **负面观察**是：

> **Head salience is not migration modularity.** Attention heads 是单层内并行的读取算子，却不是跨层
> 隔离的状态电路；dense merge、gate 和 residual stream 会把一个 upper-head 的原生重算闭包扩展到
> 下层全部 heads。

这可以补强 Insight 1 对 locality 的解释，但没有给出新的低成本迁移对象，不能单独充当用户要求的
Insight 2 / Design 1。

## 2. 当前 HSTU 的真实 head dependency graph

对 layer `l`、head `a`，现有 legacy Medium block 为

\[
Z_l=\operatorname{RMSNorm}(X_l),
\]

\[
Q_l^a=Z_lW_{Q,l}^a,\quad
K_l^a=Z_lW_{K,l}^a,\quad
V_l^a=Z_lW_{V,l}^a,
\]

\[
R_l^a=\phi(Q_l^aK_l^{a\top})V_l^a,
\]

\[
O_l=[R_l^1;\ldots;R_l^h]W_{O,l}^{\top},\qquad
G_l=\operatorname{SiLU}(Z_lW_{G,l}^{\top}),
\]

\[
X_{l+1}=X_l+O_l\odot G_l.
\]

这一语义直接对应：

- `src/hstu_kvcache/models/attention.py` 的 `_aggregate()`：每个 head 独立做 `QK` 与 `AV`；
- 同文件的 `_finish()`：拼接全部 head 后执行 full dense `out_proj`；
- `src/hstu_kvcache/models/block.py` 的 `HSTUBlock.forward()`：full-width gate、Hadamard product 与
  residual update；
- `src/hstu_kvcache/models/hstu.py` 的 `forward_embedded()`：上述 dense residual 成为下一 block 输入。

所以对任意 `l>=1` 的目标 head `b`，其 `K_l^b,V_l^b` 需要完整 `Z_l`；完整 `Z_l` 又依赖
`X_l` 中全部 lower-head output。除非 `W_O`、gate 与下一层 projection 具有结构性 block diagonal，
否则不能从计算图中删除某个 lower head 并仍声称得到 exact Current head。

### 2.1 六个 checkpoint 没有 accidental head separator

为排除“代码是 dense，但训练后恰好形成独立 head lanes”的例外，本轮只读取六个 frozen checkpoint 的
小型 block weights。对每个相邻 layer、source head `a`、target head `b` 和下一层 `Q/K/V` projection，
检查未计 gate 时的跨 head composition：

\[
M_{l,a\rightarrow b}^{Q/K/V}
=W_{O,l}[:,a]^{\top}W_{Q/K/V,l+1}[b]^{\top}\in\mathbb R^{32\times32}.
\]

每个 checkpoint 共 `5*6*6*3=540` 个矩阵。结果是：

| checkpoint | 最小 normalized Frobenius coupling | 最大 coupling | 最少 nonzero entries |
| --- | ---: | ---: | ---: |
| v0 | .054501 | .232226 | 1024 / 1024 |
| v1 | .055345 | .216221 | 1024 / 1024 |
| v2 | .056163 | .217421 | 1024 / 1024 |
| v3 | .056434 | .217124 | 1024 / 1024 |
| v4 | .056528 | .222962 | 1024 / 1024 |
| v5 | .056564 | .213897 | 1024 / 1024 |

这里的 normalization 是 `||AB||_F/(||A||_F||B||_F)`。这不是 head importance 指标，也不声称每条
功能路径同样强；它只验证所有 source-target head pairs 都存在非零 dense path，没有可供 exact executor
利用的训练后 block sparsity。数据相关 gate 与 RMSNorm 还会增加 coupling，不能提供对所有用户成立的
separator 保证。

## 3. Exact Current dependency closure 与 FLOP 下界

固定 Medium：

\[
N=1024,\quad H=192,\quad h=6,\quad d=32,\quad L=6,
\]

\[
P=N(N+1)/2=524800.
\]

沿用仓库 multiply-add=`2 FLOPs` 口径：

\[
C_H=2NH^2=75,497,472,
\]

\[
C_{in}=2N(2F)H+C_H=88,080,384,
\]

\[
C_{block}=5C_H+4PH=780,533,760,
\]

\[
C_{Exact-All}=C_{in}+6C_{block}=4,771,282,944.
\]

设最高目标 cache layer 为 `m`（0-based），只需该层 `s` 个 Current heads。为了让这些 heads 真正来自
Current trajectory，必须完整执行下面 `m` 个 block；terminal layer 只做 selected K/V projection，
不计算无人消费的 Q/attention/output/gate。因此最有利的精确下界为

\[
C_{exact}(m,s)=C_{in}+mC_{block}+2\frac{s}{h}C_H.
\]

前面完整 block 已经计算了其全部 head K/V；选择更少 lower heads 不会进一步节省其 FLOPs。

| 最高 Current cache layer | selected heads at terminal | FLOPs/user | Exact-All | 裁决 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 113,246,208 | 2.3735% | 纯 layer-0 head KV |
| 0 | 6 | 239,075,328 | 5.0107% | full layer-0 KV；现有 mean recovery 低 |
| 1 | 1 | 893,779,968 | 18.7325% | 算术可行，但仍是 early head splice |
| 1 | 3 | 944,111,616 | 19.7874% | `<20%` 内最大 exact head 数 |
| 1 | 4 | 969,277,440 | 20.3148% | FAIL cost |
| 1 | 6 | 1,019,609,088 | 21.3697% | FAIL cost |
| 2 | 1 | 1,674,313,728 | 35.0915% | 任意 deeper circuit 已失败 |
| 5 | 1 | 4,015,915,008 | 84.1685% | 几乎完整 terminal-KV replay |

该账本仍没有给 RMSNorm、activation、cache read/write 和 kernel inefficiency 额外计价，因此只能对候选更
有利。

## 4. 读取 Parent K/V 也不能形成 cheap hybrid circuit

一种表面上更便宜的解释是：selected heads 使用新算的 Current K/V，未选择 heads 直接读取 Parent K/V，
然后逐层产生一条 hybrid trajectory。这条路径仍不能只算 selected heads。为了得到下一层 `X`，每个
nonterminal block 必须执行：

- 全部 heads 的 Current/hybrid Q projection：`C_H`；
- selected heads 的 K/V projection：`2fC_H`，其中 `f=s/h`；
- 全部 heads 的 QK 与 AV：`4PH`；
- full dense `W_O`：`C_H`；
- full gate：`C_H`。

所以每个 traversed block 至少为

\[
C_{hybrid,block}(f)=(3+2f)C_H+4PH.
\]

跨 `r` 个 block、最后形成 selected K/V 的总成本下界是

\[
C_{hybrid}(r,f)=C_{in}
+r\{(3+2f)C_H+4PH\}+2fC_H.
\]

| traversed blocks | selected heads/layer | FLOPs/user | Exact-All |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 767,950,848 | 16.0953% |
| 1 | 4 | 918,945,792 | 19.2599% |
| 2 | **optimistic zero-head floor** | 1,347,158,016 | 28.2347% |
| 2 | 1 | 1,422,655,488 | 29.8170% |
| 5 | 1 | 3,386,769,408 | 70.9824% |

因此 Parent K/V 只省下未选择 heads 的 K/V projection；它不保存 historical Q、head response、
`W_O` output、gate 或 residual hidden，无法免费提供 dense next-layer input。若省略未选择 heads 的
QK/AV，就得到一个 pruned model，不是 Current 或上述 hybrid reader 的 dependency closure；若额外保存
Parent hidden/response/gate tape，则回到
`migration_ready_source_tape_preflight.md` 已否决的大状态接口；若从 Parent K/V 解码这些对象，则回到
`parent_anchored_delta_scan_preflight.md` 的 `>=25.3173%` source floor，尚未开始 circuit delta。

## 5. 与现有 functional evidence 的交叉判定

已有 Insight 1 population 结果已经给出 full-head layer Exact-KV intervention。它是对 head/circuit
候选非常有利的上界，因为 selector 和 Current Exact cache 都免费：

| exact cache layers | edge-equal recovery | minimum edge | executable native closure |
| --- | ---: | ---: | ---: |
| layer 0 | .455614 | .157834 | 5.0107% |
| layers 0--1 | .483584 | .230331 | 21.3697% |
| layers 0--2 | .843743 | .717618 | 37.7287% |
| layers 0--3 | .974351 | .944096 | 54.0877% |

前三个 cache layers 首次达到约 `.84` mean recovery，但其真实 native closure 已是 `37.73%`，而不是
诊断图上的 `50% KV coordinates`。`<20%` 只允许 layer 0 加 layer 1 的三个 heads；即便完整六个
layer-1 heads 的理想 splice，mean 也只有 `.484`。

不能从“完整六个 heads 较差”严格推出某个三-head subset 一定更差，因为 head effects 可能相消；但把
subset 选到更好恰好需要一个 selector。若 selector 来自 Current-Exact output，它违反协议；若来自
parameter/attention importance，它退化为已有 head pruning/locality。无论数值是否偶然改善，这条路线都
过不了机制创新门。

### 5.1 Parameter-only concentration sanity check

本轮还做了一个不读用户数据的必要条件检查。对每个 `(layer,head)`，把相邻 release 的 `Q/K/V` row slice
和 `W_O` column slice 的 squared parameter delta 相加。下面只是在 head-attributable weights 内的能量
覆盖，不是功能 importance：

| edge | per-layer top-1 | per-layer top-2 | per-layer top-3 | global top-6 / 36 |
| --- | ---: | ---: | ---: | ---: |
| v0→v1 | .2777 | .5186 | .6831 | .3376 |
| v1→v2 | .2408 | .4486 | .6372 | .2829 |
| v2→v3 | .2676 | .4642 | .6347 | .2914 |
| v3→v4 | .3300 | .5360 | .6869 | .3457 |
| v4→v5 | .2711 | .4656 | .6449 | .2644 |

版本变化没有在参数坐标中稳定集中到一两个 heads；top head identity 也随 layer/edge 改变。更重要的是，
该统计完全遗漏不能归属某一 head 的 input、norm、gate、residual 和 readout drift。因此它不能作为
selector，也不支持“少数 release heads 承担大部分模型变化”的强先验。

## 6. 无 Current-Exact selector 审计

| selector | 是否合法/可执行 | 本质与裁决 |
| --- | --- | --- |
| 逐 subset 最大化 Current Exact−Reuse recovery | 否 | 直接使用 target output 的 oracle；禁止 |
| 用 validation label、loss gradient 或下游 AUC 排 head | 否 | future-label / qualification tuning；禁止 |
| `||Delta W_{Q/K/V/O}^h||` | 可作 control | release-shared、label-free，但只是 weight-magnitude structured pruning；且上表不集中 |
| Parent attention mass/entropy | 可作 control | 衡量旧 head utility，不衡量 Current contextual defect；属于 head importance/KV budgeting |
| 固定 candidate probes 上的 response drift | 需要生成 Current response | 若全算，selector 先支付欲节省的 circuit work；若近似，回到 PRO/response compression/controller |
| learned gate/router | 当前不授权 | 需要训练，属于 dynamic head pruning/MoA family |
| 只更新少数 heads 的 adapter/LoRA | 当前不授权 | model adaptation，不是既有 `v0..v5` cache migration；已有直接 prior art |

不存在一个同时满足“无需 target、无需训练、低成本、版本差异特异、非已有 importance metric”的 selector。
更根本地，即使 selector 免费，depth closure 已经使中深层 circuit 超出预算。

## 7. Primary-source collision matrix

| 工作 | 已覆盖内容 | 对本候选的边界 |
| --- | --- | --- |
| [Michel et al., NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html) | 单 head ablation、importance score、greedy head pruning；论文也讨论逐层 oracle 与跨层 compounding | “少数重要 heads 足够”与 head subset search 已有；重要性不等于跨版本 causal closure |
| [Voita et al., ACL 2019](https://aclanthology.org/P19-1580/) | specialized heads、stochastic gate 与 differentiable `L0` head pruning | 用 learned gates 找少数 functional heads 已有 |
| [Differentiable Subset Pruning, TACL 2021](https://aclanthology.org/2021.tacl-1.86/) | 在给定 hard head budget 下学习精确 subset | 固定 head 数的优化选择不是新机制 |
| [Mixture of Attention Heads, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.278/) | router 对每个 token 动态选择 `k` 个 attention heads | query/input-dependent circuit routing 已有，而且需要架构与训练改动 |
| [HiFi, ACL 2023](https://aclanthology.org/2023.acl-long.475/) | 用 head information/correlation graph 与 PageRank 选择少数 heads 做 parameter-efficient adaptation | “release 只改变重要 head”会落入 selective-head PEFT，而非状态迁移 |
| [DuoAttention](https://arxiv.org/abs/2410.10819) / [Ada-KV](https://arxiv.org/abs/2407.11550) / [HeadKV](https://arxiv.org/abs/2410.19258) | retrieval/streaming head 分类、head-wise KV policy 与 adaptive cache budget | head heterogeneity 驱动 cache 管理已经是直接 KV prior art |
| [DroidSpeak, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) | 跨 fine-tuned model variants 识别 critical layer groups，选择性重算部分 layer KV、复用其余 layer | 与 cross-model selective recomputation 最接近；把粒度从 layer 改到 head 并不足以形成新贡献 |
| [Houlsby adapters](https://proceedings.mlr.press/v97/houlsby19a.html), [LoRA](https://arxiv.org/abs/2106.09685), [Tiny-Attention Adapter](https://arxiv.org/abs/2211.01979) | bottleneck adapter、attention weight low-rank delta、极小 attention adapter | 若把 release circuit 改写成小 delta branch/adapter，已离开 cache migration且撞上成熟 PEFT family |

因此，“DroidSpeak 式 selective recomputation + head pruning/HeadKV 式 selector + persistent recommender
workload”是已有方法的场景组合，正是用户明确不接受的论文形态。

## 8. 为什么本轮不运行唯一 subset canary

从算术上可以写出一个唯一 `<20%` 配置：每条 edge 用 parameter-delta energy 选 layer 1 的三个 heads，
完整 native recompute block 0，并把生成的 layer-0 full K/V 与 layer-1 selected K/V 写入 Parent cache。
它无需 Current-Exact selector，成本为 `19.7874%`。

本轮仍明确**不运行**，原因不是预期数字差，而是它在运行前已经违反方法准入：

1. 输出对象就是 head-wise exact K/V replacement；
2. 只在 early two layers 有算术窗口，重新把 Design 绑定到 layer/head locality；
3. selector 是普通 weight-magnitude head pruning；
4. 即使 recovery 很高，也只能说明这个 checkpoint 上存在一个有利 sparse hybrid，不能证明新的
   Transformer migration law；
5. 现有 full layer-0/1 oracle 已给出较低稳定性，新增 subset search 的最高风险是事后挑中 cancellation。

继续跑它会产生一个容易包装、但不能通过 novelty review 的数字。按照 hard innovation gate，应在纸面
阶段停止，而不是消耗 discovery UID 后再退休。

## 9. 最终边界与对主线的意义

本轮只允许形成以下结论：

1. **不再声称 head selection 尚未检查。** 现在可以精确写成：head-wise exact splice 的质量尚未穷举，
   但在当前 dense residual architecture 下，跨深度的原生 dependency closure 已被结构与成本否决；
   唯一低成本区退化为 early locality。
2. **不把 parameter concentration 当功能证据。** 它只是一个 label-free necessary-condition sanity check。
3. **不把 sparse hybrid 的潜在好数值当新 Design。** head pruning、dynamic routing、head-wise cache budget、
   selective cross-model recomputation 与 adapters 均有直接先例。
4. **设计含义只到 architecture requirement。** 真正可迁移的 sparse circuit 需要模型从结构上提供
   head-isolated residual lanes 或显式 merge boundary；给当前 checkpoint 事后套 mask 不会创造这种
   separator。该方向需要新架构/训练和独立 prior-art 审计，不能用于现有 `v0..v5` Design 1。

所以 `release-circuit native recomputation` 在当前接口下正式结束。它提供的是对 Insight 1 的
Transformer-specific closure explanation，不是 Insight 2 的正向迁移对象，也不是可冻结的 Design。
