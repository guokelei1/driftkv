# 天然 causal suffix 自证：信息源、等价性与 NO-GO 审计

日期：2026-09-03  
状态：**严格 NO-GO；不建立实验合同、不运行新 preflight、不读取 confirmation 或 label**  
范围：只研究单个 `Parent -> Current` edge；现有 Yambda-500M Medium `v0..v5`；当前 KV-only persistent interface

## 1. 裁决先行

本轮检验一个看似不同于固定 probe 的 Current-information source：把同一用户历史末端的一段真实、已发生
causal suffix 当作 release-time 自证序列。Current reader 从 Parent prefix 出发递归消费这些历史事件，
希望利用它们的中间 response 或新生成 K/V 构造 persistent release defect，而不读取 Current-Exact upper
state，也不使用 request label。

严格结论是 **NO-GO**：天然 suffix 是很好的、与用户自身请求分布对齐的 query panel，但它没有提供一个
独立的 Current-state target。它不能把 Parent-conditioned Current trajectory 变成 Current-Exact trajectory，
也不能从模型的 causal self-consistency 中检验初始 cache 是否来自正确版本。

最核心的 Transformer 事实是：

> **Autoregressive append is lineage-blind.** 对任意形状合法的初始 K/V，Current Transformer 按整段或
> token-by-token 消费同一 causal suffix 都会得到同一条 hybrid trajectory。错误的 Parent 初始 cache
> 也完全满足这条自洽性，所以 chunk/roll-forward consistency 不能成为 cache-version certificate。

在当前仓库中，所有能从 suffix 得到非零信号的变体只落入三类已经存在的机制：

1. 用 suffix token 当 query：是旧 PRO / time-aligned probe 的 observation-window 扩展；
2. 用 Current 在 Parent prefix 上重放 suffix：就是已经执行并失败的 `Tail-128 functional estimator`；
3. 用 suffix 上成对的 Parent K/V 与 hybrid-Current K/V 拟合旧 prefix：是局部 paired mapping、cache
   distillation 或 locality extrapolation，不是新的 Transformer state law。

因此本轮没有满足“版本特异、非 mapping、非 probe/distillation、可在 `<20%` 内执行”的命题，也就没有
唯一值得运行的数值 preflight。继续扫描 suffix width、聚合方式、probe count 或拟合 loss 只会调已有方法，
不进入 Design 1。

## 2. 把可观测量写准确

将历史分为旧 prefix `a` 与天然 suffix `b`：

\[
h=a\Vert b.
\]

令 `S_v(x)` 表示版本 `v` 从空状态对历史 `x` 做完整 causal execution 后得到的 K/V；令
`A_v(S,b)` 表示版本 `v` 从任意初始 cache `S` 继续 append `b`。三个相关状态是：

\[
S_P(h),\qquad S_C(h),\qquad
S_{C\mid P}(h)=A_C(S_P(a),b).
\]

`S_{C|P}` 是合法、dependency-closed 的 hybrid state，但一般不等于 `S_C(h)`。需要迁移的 reader effect
是

\[
d(q)=R_C(q,S_C(h))-R_C(q,S_P(h)).
\]

suffix replay 实际能直接产生的是

\[
\widetilde d_b(q)=R_C(q,S_{C\mid P}(h))-R_C(q,S_P(h)),
\]

以及 suffix positions 上的

\[
S_{C\mid P}(b)-S_P(b).
\]

两者都不包含把旧 prefix 从 `S_P(a)` 改成 `S_C(a)` 后引起的 contextual propagation。只有在
`S_P(a)=S_C(a)`、`a` 为空，或另有已经证明的有限状态闭包时，hybrid endpoint 才是 Current endpoint。
这些条件恰好删除了本问题要解决的跨版本旧历史误差。

## 3. 为什么 causal self-consistency 对 lineage 完全失明

### 3.1 Chunk equivalence 对任何初始 cache 都成立

把 suffix 再分成 `b=b_1||b_2`。标准 causal K/V append 满足

\[
A_C(A_C(S,b_1),b_2)=A_C(S,b_1\Vert b_2)
\]

对**任意**合法初始 `S` 成立，而不是只对 `S=S_C(a)` 成立。其原因只是 causal mask 与 append-only
cache 的执行语义：后一个 token 读取相同的已有 K/V 和相同的 earlier-suffix K/V，batch/chunk 边界不改变
计算图。

所以，下列常见“自证”比较在数值精度内必然为零：

- 整段 suffix 与 token-by-token suffix；
- 一个 chunk 与多个 chunk；
- 在任意中间位置 seal cache 后继续，与不 seal 连续执行；
- 从同一个 Parent prefix 开始的不同合法执行分块。

它们只能检查 executor 正确性，不能检查 initial state lineage。仓库已有 same-model Full/append continuity
与 append-only K/V tests；再为错误 Parent cache 重跑一次只会证明同一个 API invariant，不会产生 release
defect。

### 3.2 Parent-reader / Current-reader 差不是 cache defect

在相同 Parent state 上运行两版 reader 可以得到

\[
R_C(q,S_P)-R_P(q,S_P).
\]

它测量的是 reader parameter change 加 query embedding/readout change，而目标是

\[
R_C(q,S_C)-R_C(q,S_P).
\]

前者在 `S_C=S_P` 时仍可非零；后者在某个 query 对 state difference 不敏感时可以为零。两者没有
Transformer 恒等式可互相推出。仓库的 producer/reader commutator oracle 也已经表明，只有额外读取
`R_P(q,S_C)` 这条 Exact-Current reverse path 才能形成近交换关系；该 path 正是本轮禁止的 Current-state
信息。

### 3.3 Suffix query panel 仍有不可观测的 prefix response

即使不用 scalar score，而保留每层、每 head、每个 suffix token 的完整 attention response，有限 query
panel 也不自动决定任意 future query 的 response。

固定一层的 old-prefix keys，令 `A_b` 是 `m` 个 suffix queries 对 `n` 个 old-prefix positions 的 attention
weight/activation matrix。只考虑 value defect，就存在

\[
A_b\,\Delta V=0
\]

但对某个未来 query `q` 有

\[
A_q\,\Delta V\ne0.
\]

当 `m<n` 时，`A_b` 的 position-space nullspace 至少有 `n-m` 维；Medium 的 `Tail-128` 对 896-position
old prefix 正处于这个情形。真实 release defect 不必充满整个 nullspace，但当前模型没有结构约束保证它
避开 nullspace。于是“suffix 上完全一致”至多是 empirical query coverage，不是 causal sufficiency。

这也是为什么增加真实 probe 数量不能创造 target-state information：它只增加 observation rows。若再用
SVD、ridge、optimized virtual KV 或一个 learned lift 从这些 rows 外推，就进入 response fitting / cache
distillation，而不再是新的 exact self-certificate。

## 4. 四种实现化尝试及其准确归类

| 构造 | 真正读取的信息 | 是否得到 `S_C(a)` | 准确归类 | 裁决 |
| --- | --- | ---: | --- | --- |
| 用 suffix items 作为 Current queries 读取 Parent K/V | `R_C(q_b,S_P)` | 否 | history-derived / time-aligned probe bank | 旧 PRO family |
| Current 从 `S_P(a)` 递归 replay suffix | `S_{C|P}(h)` | 否 | dependency-closed Tail replay | 已执行负结果 |
| 比较 suffix 的 stored Parent K/V 与 hybrid Current K/V | paired local states on `b` | 否 | local cross-version map / extrapolation | mapping + locality |
| 从 Parent K/V 恢复 suffix source rows，再评价 Current-minus-Parent residual | exact Parent row response | 否 | sampled source-residual test | 已执行同类负结果 |
| 调 correction 使 suffix response、logit 或 hidden 匹配 | finite-query target outputs | 否 | cache/feature distillation | prior-art collision |
| 比较不同 suffix chunkings 的结果 | 同一 hybrid computation 的两种调度 | 否 | executor consistency test | 恒为零、无信号 |
| 用真实 later item/action 检验 earlier prediction | historical outcome target | 否 | self-supervised task loss | 不再 label-free |

最后一行需要特别区分。已发生事件在 wall-clock release 时不是“未来数据”，但一旦把 later item、action 或
like/dislike 当作 prediction target，它就是用户行为监督，而不是本文冻结的 label-free state constructor。
如果只把 item identity 输入 query encoder、从不以其发生与否计算 loss，它仍只是 query panel，回到第一行。

## 5. 仓库内最接近的实证反例

### 5.1 Tail-128 已经完整实现了最强 label-free 版本

[Tail functional estimator](../../../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/estimator_tail_functional_v1/canary/analysis/report.md)
从完整 Parent cache 中切出 recent 128 raw events，用 Current weights 对它们做 dependency-closed replay，
旧 896 positions 保持 Parent lineage；随后只持久化 1,152-scalar S4 sidecar。Current Exact 仅用于事后评价。

结果为：

| suffix path | Exact-All compute | edge-equal recovery | min edge | positive edges | cosine | norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tail-128 + P1 | 18.2842% | -0.0876 | -0.3503 | 2/5 | 0.8834 | 0.1219 |
| Tail-128 + P2 | 18.5303% | -0.0865 | -0.3458 | 2/5 | 0.8832 | 0.1217 |
| Tail-128 + P4 | 19.0224% | -0.0878 | -0.3518 | 2/5 | 0.8834 | 0.1219 |

高 cosine 与约 `0.12` norm ratio 很符合本轮的信息分析：suffix replay 能看到一个方向相近的局部 effect，
但没有看到旧 prefix 的主要 Current producer defect。增加 probe 没有改变这一点。

### 5.2 固定 history probe 与时间对齐也已经失败

Medium functional-probe canary 用 `Parent K/V + Current parameters + history-derived probes + compact Current
carriers` 生成 S4 correction；最佳固定配置仍为 `-0.9288` recovery。把 probe time 从零修正到真实 cutover
delta 只把 C32 从 `-1.2993` 改到 `-1.0590`，没有一条 edge 达 80%。因此把 suffix token 换成更多真实时间
query，不足以改变 information source。

### 5.3 纯 suffix replay 的成本虽然可进预算，但没有 target

按已冻结 Medium 账本，单做 Current hybrid replay、尚未额外读取 probe 的理论成本为：

| suffix width | replay / Exact-All |
| ---: | ---: |
| 32 | 4.6579% |
| 64 | 9.2169% |
| 96 | 13.6770% |
| 128 | 18.0382% |

这些点只说明 suffix 是一个便宜的合法重算边界，不说明它能估计旧 prefix defect。改变 width 会重新扫描
Insight 1 已否定的 recent locality；在没有新 invariant 前，不应把一个成本 grid 当成新方法。

### 5.4 把 suffix 当 exact source residual rows 也不是新信息源

[Source-certified reduced execution](source_certified_reduction_preflight.md) 已利用 joint Parent K/V 在少量
positions 恢复 exact Parent normalized query，再对完整 causal Parent prefix 计算 exact source response；
其更强的 finite release-defect residual 版本有 zero-release/full-rank exact limit，成本 `19.4726%`，
五边 mean recovery 仍只有 `.662`。算法骨架又是 sampled nonlinear residual 加 trial-space lift，与
DEIM/hyper-reduction 直接重叠。

把四个 DEIM test rows 换成 recent causal suffix rows，只改变 sampled-row rule，并没有提供
`S_C(a)` 或新的 attention守恒量。若保存所有 suffix rows后再拟合如何 lift 到 old prefix，则回到本节前述的
response fitting；若只把 suffix residual 平均成 sidecar，则回到 PRO/Tail functional correction。

## 6. 与原始相关工作的碰撞

### 6.1 Recent suffix 作为 observation queries 已有直接先例

[SnapKV](https://arxiv.org/abs/2404.14469) 明确用 prompt 末端的 observation window 形成 recent queries，
并由这些 queries 对旧 prefix 的 attention pattern 选择 cache positions。EvoKV 的版本语义不同，但
“天然末端 query 能代表后续 query、据此处理旧 cache”不是新的方法原则。若本轮根据 suffix attention
选择 token、head、region 或 carrier，还会同时退回 Insight 1 的 locality/sparsity family。

### 6.2 用 recent real queries 匹配 attention output 已被写成 cache distillation

[KVSculpt](https://arxiv.org/html/2603.27819) 直接把 recent retain-zone queries 作为 future queries 的
自然 proxy，并加入 synthetic future queries；它优化一组 compact K/V，使这些 queries 下的 attention
output 与 full cache 匹配。因而下列骨架已经有清楚归属：

```text
recent real queries -> cache/sidecar parameters -> match reader response
```

把 full single-model cache target 换成 Parent/Current release response，定义了不同应用问题，却没有让
query-panel fitting 变成新的 Transformer mechanism。若 target 只来自 `S_{C|P}`，它蒸馏的还是 hybrid
teacher；若 target 来自 `S_C`，则违反本轮 no-Current-Exact 条件。

### 6.3 Suffix paired states 再外推就是 cross-model mapping

[Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) 已用成对 source/target token states
拟合 per-head、cross-layer cache mapper。只在每个用户的 suffix 上生成
`(KV_P,KV_{C|P})`，再把关系应用到 old prefix，是 online/local calibration 版本的同一表示迁移骨架；
它还额外假设 Parent-conditioned recent token relation能外推到更老、拥有不同 causal context 的 positions。
无论使用线性、MLP、nearest neighbor、kernel 或 closed-form solve，都没有越过用户明确排除的 mapping。

## 7. 什么时候这条路线才可能重新打开

天然 suffix 要成为真正新的 Current-information source，必须额外提供目前不存在的边界条件，而不是增加
query 数。至少需要下面之一：

1. suffix 起点存在一个 Exact Current state checkpoint；这会把未解问题推回旧 prefix rematerialization；
2. 架构保证旧历史到 suffix/future 的所有路径经过一个 version-compatible causal separator；这属于新的
   model--system co-design，且 memory/port prior art 与跨版本 fiber condition需要独立解决；
3. release family 给出解析 transition homomorphism，使 Parent initial state 可无 target 地变成 Current
   initial state；若只学习该变换，仍是 mapping；
4. 使用 later outcomes 对 correction 做优化；这变成 label/self-supervised adaptation，并违反当前 protocol。

在现有 `v0..v5` KV-only interface 下四项都不成立。autoregressive causality只规定信息从 prefix 流向
suffix；它不会让 suffix 反向成为 Current prefix state 的免费 witness。

## 8. 最终冻结边界

本轮可以保留一个准确的负面观察，但不能把它包装成 Insight 2 或 Design 1：

> **天然 causal suffix 提供 query coverage，不提供 target-state information；Transformer 从任意初始
> cache 出发的 append 都内部自洽，因此 suffix consistency 无法认证 cache lineage。**

据此冻结以下执行决定：

- 不新增 suffix-self-probe runner、合同或 formal canary；
- 不扫描 suffix width、query aggregation、loss、rank、mapper 或 probe count；
- 不把 Tail replay、time-aligned history probe 或 suffix-paired mapping改名后重新接纳；
- `Tail-128` 保留为最直接的 executable negative evidence；
- Design 1 仍未冻结。若继续寻找论文级机制，必须寻找一个真正新增 Current state information 的
  Transformer computation/interface，而不是另一组更自然的 probes。
