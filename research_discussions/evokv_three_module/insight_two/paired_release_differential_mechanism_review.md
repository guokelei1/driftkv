# Paired finite-release differential：Insight 2 / Design 1 机制高度审查

日期：2026-09-03  
状态：**独立机制审查；未运行 GPU、未修改冻结证据或合同、不是 Insight 2 / Design 1 冻结稿**

## 1. 裁决先行

当前 `rank4 Parent + rank4 Current -> rank8 layer-0 basis -> exact Parent + signed cores`
值得保留，但它目前只给出一个**可检验的机制线索**，还不能写成论文结论。

真正可能达到论文高度的命题不是“两个 rank-4 相减以后至多 rank-8”，也不是“一个 layer-0 basis
可以跨层存 cores”，而是：

> 对同一历史的相邻 Transformer release，采用相同的有界数值协议分别执行 Parent 与 Current
> 的逐层计算时，两条绝对近似轨迹即使各自不够准确，其 contextualization error 仍可能在版本间
> 保持相关；因此取两版的有限差分会消去主要的共享误差，得到比任一绝对近似更准确、并对 Current
> reader 足够的 release difference。

这是一个可以被直接否证的 Transformer 命题。它讨论的是误差如何穿过多层 attention、gate 和 residual
而在两个真实模型端点之间保持相关，不是低秩分解本身。推荐系统提供的特殊条件是：已有 exact Parent
persistent cache 可以充当不动的基准状态，而同一个用户级差分会被后续多个请求反复读取。

与此对应的 Design 1 候选是：使用 matched Parent/Current bounded replay 逐层形成有限版本差分，
只把该差分编译为 exact Parent cache 上的紧凑 correction；不拟合 Parent-to-Current mapper，也不把一份
compressed Current cache 改名为 migration。

但是必须同时记录三个否定性事实：

1. 当前 D 是**两条独立的 data-dependent reduced trajectories**。两臂共享 rank、numerical rules 和
   Gaussian seed，但不共享实际 basis/projector；在证明层间依赖以前，不应称为“耦合 quotient closure”。
2. 单 UID 五边 `0.870/0.901/0.985/0.869/0.706` 只说明路线值得做机制诊断。它没有证明 population
   stability，也没有证明 paired replay 优于 single-arm compression。
3. dense-input prospective 完整账本为 `25.2952% Exact-All`，原 last-layer KV-only 路径为
   `21.8226%`。后续等价的 matrix-free input factor 已把后者降到 `18.3264%`，所以 arithmetic cost
   blocker 已解除；但该 executor rewrite 不是机制，而且 paired recovery 仍低于 generic single-arm。
   当前仍不能冻结 Design 1。

结论是：**可以把“逐层近似误差在相邻 release 间相消”作为 Insight 2 的唯一主假说；当前 rank4/rank4
路径只是它的 mechanism witness。只有误差相消、版本必要性、跨层因果性和严格成本同时通过，才能把它
提升为 Design 1。**

## 2. 为什么这不是给 low-rank compression 换名字

下列组件已经有直接 prior art 或属于标准数值工具，不能承担论文新意：

- token/history-axis SVD、randomized range finder、QR、rank 4/8；
- sequence-specific 或 user-specific basis；
- 一个 history basis 跨 K/V、head 或 layer 共享；
- exact base cache 加 low-rank residual/core；
- native reader 直接消费 dense base 与 factorized delta；
- same-architecture cross-model KV reuse；
- common random numbers、control variate 或“两个近似相减”的一般数值思想；
- parameter delta、LoRA、JVP 或 tangent propagation。

因此，下面这句话本身不构成机制：

> Parent 与 Current 各做一次低秩 replay，然后相减并加回 Parent。

若没有逐层证据说明“相减为什么比单臂更准确”，reviewer 完全可以把它归类为两次普通 compression 加
residual encoding。真正必须成立的是一个**迁移特异的差分优势**：在 matched compute/storage 下，
Parent/Current 两臂的近似误差应当比任一单臂误差更相关，且这种相关性必须沿 Transformer depth 转化为
更高的 Current-reader functional recovery。

同样，`U0` 不是主机制。它是把差分编译成 persistent state 的接口。xKV 等工作已经覆盖跨层共享
history basis 和 per-layer cores；当前论文只有在证明 paired finite-release evolution 产生了
single-arm compressor 没有的功能增益时，才有新的研究内容。

## 3. 一个精确且可分解的机制命题

令 `m in {P,C}` 表示 Parent 和 Current。对用户历史，记第 `l` 层 exact K/V state 为

\[
Z_l^m=[K_l^m\mid V_l^m],
\]

matched bounded replay 产生

\[
\widetilde Z_l^m=Z_l^m+E_l^m.
\]

这里的 “matched” 只表示两臂使用同一组事前冻结的 rank、oversampling、power iteration、seed 和
compression schedule；由于 range finder 是 data-dependent，`P`、`C` 两臂仍可有不同 basis。

从合法的 approximate layer-0 difference 得到 `U0`，令

\[
\Pi_0=U_0U_0^\top.
\]

第 `l` 层 logical migrated state 为

\[
Z_l^{M}=Z_l^P+\Pi_0(\widetilde Z_l^C-\widetilde Z_l^P).
\]

那么相对 Current Exact 的误差有一个不依赖经验拟合的恒等分解：

\[
Z_l^{M}-Z_l^C
=-(I-\Pi_0)(Z_l^C-Z_l^P)
+\Pi_0(E_l^C-E_l^P).
\]

这把当前假说拆成两个不能互相替代的条件：

1. **表示条件**：真实 release difference 在 reader-relevant directions 上必须主要落在 `U0` 中，
   即第一项不能太大。
2. **递推条件**：两条 bounded trajectories 的误差必须在版本间相消，即第二项应显著小于
   `Pi0 E_l^C` 或相应 single-arm error。

当前输出 recovery 无法区分这两项。即使第一项接近零，也不表示 paired recurrence 有效；它可能只说明
“exact Parent + layer-0 low-rank release correction”是一个好 representation oracle。反过来，即使
`E_l^C-E_l^P` 很小，`U0` 漏掉真实差分时仍无法迁移。

还有一个必须排除的内部闭合假象：rank-4 Parent K/V 与 rank-4 Current K/V 的差在构造上至多 rank 8，
再从这个 approximate difference 自己构造 rank-8 `U0`，必然容易重建**算法自己的差**。这不证明
`U0` 能承载真实 Current-minus-Parent difference，更不证明其跨层稳定。只有 Current Exact oracle 和
Current-reader intervention 能裁决表示条件。

最后，K/V tensor decomposition 不是最终评价对象。Current reader 对 corrected K 的 attention weight
是非线性的，所以每一项都必须同时报告 tensor error、query-conditioned response、post-block residual、
final user representation 和 recommendation-gap recovery。

## 4. 建议的 Insight 2 与 Design 1 表述

### 4.1 Insight 2（待证版本）

一段可以在证据通过后用于论文的克制表述是：

> **Insight 2.** 相邻版本 Transformer 对同一历史做有界重放时，主要 approximation error 可以沿
> attention--gate--residual 链在两个版本之间共同演化。因而两条绝对轨迹无需各自精确；对两版实际
> 参数做 matched finite-release replay 后取差，可以消去版本共享的近似误差，留下对 Current reader
> 足够的历史状态差异。

这里的 `finite-release` 只表示直接使用 Parent、Current 两个真实参数端点，不把一次 Parent-point JVP
当成有限模型更新。`matched` 也不是声称两臂共享 basis，而是强调相同 approximation protocol 使
`E^P` 与 `E^C` 有机会相关。

该 Insight 被下列任一结果直接否定：

- `E_l^C-E_l^P` 并不比单臂 error 小；
- common protocol/seed 与 independent 或 deliberately mismatched protocol 没有稳定区别；
- paired 优势只出现在最终投影，沿层的 Q/K/V、context、residual 中不存在；
- paired recovery 不优于 matched-FLOP Current-only replay；
- 现象只存在于一个 UID、一个 edge 或一次 numerical seed。

### 4.2 Design 1（条件性候选）

一个不依赖新造名称的设计描述是：

1. 保留现有 exact Parent K/V 作为 persistent base；从同一 raw history 分别形成 Parent 和 Current
   的 model-specific input。
2. 两版按相同的 rank、range-finder schedule 和 seed 逐层执行 bounded native Transformer replay。
   两臂允许拥有各自的 factor/basis，不强迫 RMSNorm、attention 和 gate 后仍共享 column space。
3. 只从 dependency-free approximate layer-0 `Delta[K,V]` 形成用户级 `U0`。这一过程不读取上层
   Current Exact、label、candidate/query 或 future event。
4. 每层写入

   \[
   C_l^K=U_0^\top(\widetilde K_l^C-\widetilde K_l^P),\qquad
   C_l^V=U_0^\top(\widetilde V_l^C-\widetilde V_l^P),
   \]

   不持久化两条 reduced working trajectories。
5. 后续 Current reader 原生消费

   \[
   K_l^P+U_0C_l^K,\qquad V_l^P+U_0C_l^V,
   \]

   并由真实 query 决定 attention，不把 correction 变成 candidate-shared output offset。

从论文机制看，步骤 2--4 的核心不是“生成一份 low-rank Current cache”，而是**显式保留 Parent arm，
使每层写出的对象都是两个有限模型端点在相同近似协议下的差**。如果删除 Parent arm、只运行 Current
compressor，最后再减 Parent coefficient 后结果不变，那么该实现应降级为 generic control。

最终可执行版本还应把 pair 写成 base/difference state recurrence，以便直接审计 Parent 是否在每层
参与以及哪些算子可以共享：

\[
\widetilde B_{l+1}=\mathcal C_l^P(F_l^P(\widetilde B_l)),
\]

\[
\widetilde D_{l+1}=\mathcal C_l^\Delta\left[
F_l^C(\widetilde B_l+\widetilde D_l)-F_l^P(\widetilde B_l)
\right].
\]

这只是最终 Design 应满足的接口，不是当前 D 已经证明的代数等价。低 rank 时，直接压缩 difference 与
分别压缩两臂再相减并不等价，必须通过 matched ablation 决定。full rank、关闭截断时则应逐层恢复
真实 finite-release Current trajectory；该 exact-limit 是 correctness 条件，不是 novelty 证据。

## 5. 当前五边数字能说明什么

[mode-space cost preflight](mode_space_current_rematerialization_preflight.md) 记录的当前 D 单 UID
五边结果为：

| edge | paired rank4/rank4 D |
| --- | ---: |
| `v0->v1` | 0.870 |
| `v1->v2` | 0.901 |
| `v2->v3` | 0.985 |
| `v3->v4` | 0.869 |
| `v4->v5` | 0.706 |
| **edge mean** | **0.866** |

它说明：用很小的两臂工作状态仍能留下相当多的 functional release signal，值得继续问“为什么”。它不
说明误差相消，因为没有同时给出 `E^P`、`E^C`、`E^C-E^P`。

还必须正视一个不利的交叉文档信号：[earlier history-mode preflight](history_mode_replay_preflight.md)
记录的 single-arm rank-8 shared-layer0 splice 为
`0.861/0.917/0.985/0.947/0.975`，edge mean 约 `0.937`。这两组数字不是来自一个冻结的 matched
runner，不能据此做正式优劣裁决；但字面上它们**没有提供 paired 优势**，尤其最后两条 edge 反而更差。
因此 formal diagnostic 必须把 two-arm、single-arm 和 seed/operator controls 放进同一次执行，不能只
报告 D 的绝对 recovery。

原始 dense-input 成本同样不利；后续 executor 更新必须单列：

| path | Exact-All fraction | 当前身份 |
| --- | ---: | --- |
| paired D, six full blocks | 25.2952% | over-budget diagnostic |
| paired D, original final KV-only | 21.8226% | 原路径超 20% |
| paired D, matrix-free input + final KV-only | 18.3264% | 成本通过；机制仍未通过 |
| single-arm C + `U0` cores | 22.8028% | generic control，仍超预算 |

所以当前结果最多支持“保留机制假说”，不支持“Design 已经成立”。matrix-free 路径完整计算了
operator apply、QR 和截断，是合法等价 executor；但它只回答“能否进入预算”，没有回答“paired 为何
优于 single-arm”。后续主门是 matched functional gain，不再通过省略 input formation、`N^2`
activation、gate/residual boundary 或 range-finder passes 美化成本。

## 6. 必须通过的五个机制实验 / ablation

以下实验应先形成 prospective protocol，再做 `32 users x 5 edges` canary；通过后才允许扩大到冻结配置
的 population confirmation。32-user 结果仍不是最终论文 population evidence。

### M1. 逐层直接测量 approximation-error cancellation

对每个 user/edge/layer/stage，诊断性计算 exact 与 bounded 的 Parent/Current state：

\[
E_l^P=\widetilde Z_l^P-Z_l^P,\qquad
E_l^C=\widetilde Z_l^C-Z_l^C.
\]

同时报告 `||E^P||`、`||E^C||`、`||E^C-E^P||`、error cosine/covariance，以及把三种 error 分别注入
Current reader 后的 response/readout error。对照至少包括：共同 seed/rules、独立 seed、故意错配
rank 或 compression schedule，并报告多个 numerical seeds。

**支持机制的结果**：两条绝对 replay 可以明显不准，但 `E^C-E^P` 在 reader-relevant directions 上
稳定更小；共同 protocol 的差分恢复优于 independent/mismatched protocol，且优势沿多个层、至少
4/5 edge 出现。

**否证结果**：差分 error 与 single-arm error 同量级或更大；shared seed/rules 没有作用；优势只在
某个 layer/edge；或者 paired 的好结果完全来自两条 arm 各自已经很准。此时“error cancellation”不能
写成 Insight 2。

### M2. 用二项分解隔离 representation 与 recurrence

在同一 user/edge 上做一个事前固定的 2x2 intervention：

| `U0` | upper-layer core | 回答的问题 |
| --- | --- | --- |
| exact layer-0 `U0` | exact `Delta KV` core | 固定早期 boundary 的表示 ceiling |
| approximate layer-0 `U0` | exact `Delta KV` core | 合法 basis estimation 损失 |
| exact layer-0 `U0` | paired approximate core | paired recurrence 本身的损失 |
| approximate layer-0 `U0` | paired approximate core | 最终 executable semantics |

另报每层 independent best rank-8 exact-delta oracle，防止把“差分本来低秩”误写成“layer-0 跨层
稳定”。所有 exact 项只用于评价，不进入 Design。

**支持机制的结果**：fixed layer-0 representation ceiling 高，paired core 相对 exact core 只带来小的
额外 functional loss，实际路径仍接近 ceiling。

**否证结果**：fixed `U0` exact-core oracle 在五边中不能至少四边达到既定功能门（现有 preflight
建议 `0.80`）；或 oracle 很高而 paired-core 路径明显崩塌。前者否定 layer-0 compiler，后者否定 paired
recurrence。不能用提高 rank 或换 per-layer oracle 掩盖失败。

### M3. matched-compute 的“版本必要性”对照

同一 runner、同一 users/requests、同一输出 sidecar 语义下比较：

1. paired rank4/rank4 finite-release replay；
2. 相同**实际总 FLOPs**的 Current-only replay；
3. 两臂使用 independent seeds/operators 后的普通 compressed-cache subtraction；
4. ordinary per-layer 或 xKV-like Current compression；
5. 用 matched approximate Parent 替换 exact Parent base、但保留同一 `U0` projected delta 的
   ablation；另以未投影恒等式
   `approximate Parent + (approximate Current - approximate Parent) = approximate Current`
   作为 sanity control。

不能只匹配 nominal rank；必须匹配 strict FLOP、storage 和 reader semantics。对 user-edge recovery 做固定
seed paired bootstrap，五条 edge 全报。

**支持机制的结果**：paired path 在相同成本下稳定优于 strongest single-arm/ordinary compression
control，增益的 bootstrap lower bound 大于零，并至少 4/5 edge 方向一致；exact Parent base ablation
显著变差。

**否证结果**：Current-only 或普通两份压缩后相减达到相同/更好 recovery；independent seed 不损失；
或者 approximate Current replacement 与 exact-Parent-plus-delta 等价好。此时跨版本 paired mechanism
没有必要性，剩下的只是 compression/residual encoding。

### M4. Transformer depth 的因果干预

沿 depth 逐层打断 paired dependency，而不是只看最终 score：

- 在 layer `k` 后给两臂换成 independent numerical randomness；
- 在 layer `k` 后删除 Parent arm，只继续 Current reduced replay；
- 只保留 layer-0 delta、停止 upper-layer difference formation；
- 用 Parent-point tangent/linear propagation 替代真实 Current endpoint，作为 finite-release control；
- 分别在 Q/K/V、attention context、gate update 和 residual 后注入 exact/approximate difference。

同时做两个 correctness invariant：`P=C` 且 protocol 相同时差分必须严格为零；full rank、关闭截断时必须
逐层恢复 Current Exact。

**支持机制的结果**：越早打断配对，后层差分 error 与 functional loss 越明显；关键变化能定位到原生
Transformer stage，而不是只在最终 readout 才出现；finite endpoint 稳定优于 tangent control。

**否证结果**：删除 upper-layer Parent arm、换 independent randomness 或只用 layer-0/final-layer
correction 几乎不改变结果。那就不存在“沿 Transformer contextualization 的 paired evolution”，只能
把方法解释成早期 basis selection 或最终 cache splice。

### M5. release specificity 与 persistent-reader sufficiency

在不改变主设计单个 Parent->Current edge 边界的前提下，增加诊断 controls：

- 五条相邻 edge 全部报告，并用 skip-release pair 检查 cancellation 是否随 model distance 退化；
- `P=P`、错配 user history 和独立 numerical seed 作为 negative controls；
- 同一 sidecar 在不同 candidate/query panel 上复用，覆盖 ranking 与 next-item readout；
- 固定旧 `U0`，append 1/8/32 个 native Current event，禁止每次重算 whole-history basis。

**支持机制的结果**：相同用户、相邻 release 的 paired error correlation 最强；correction 不依赖某个
candidate，并在短期 append 后仍保留主要 functional gain。

**否证结果**：unrelated/non-adjacent pair 同样有效，说明只是 generic compressor；换 query 就需要重新
拟合 correction，说明它不是 persistent user state；少量 append 后迅速坍塌，说明 release-time object
没有推荐系统价值。

## 7. 三个非机制但不可放宽的 Design admission gate

即使 M1--M5 支持 Insight 2，Design 1 仍必须独立通过：

1. **质量**：冻结配置后，user-equal 再 edge-equal 的 functional recovery 达到论文目标
   `>=0.80`，至少 4/5 edge 稳定正向；所有 edge、seed 和 failure 完整报告。
2. **成本**：release-time total generation cost 严格 `<=20% Exact-All`。neural、selection/basis、
   sidecar build、I/O、reader overhead 分开报；不能用请求摊销掩盖 per-user generation。
3. **合法性**：constructor 不读取上层 Current Exact、target response/score、future event 或 label；
   whole-history projection 只能称 release-time compiler，不能称逐位置 causal rematerialization。

> 2026-09-03 executor 更新：保持 paired rank/seed/power/trajectory 语义不变的 matrix-free input
> factor 已把 migration-sufficient 成本降到 `18.3264%`。因此第 2 条的算术阻塞已经解除，但这不是
> scientific mechanism。paired 单 UID recovery 仍低于 single-arm rank-8；在 matched functional
> comparison 证明独立收益以前，仍不启动 Design-scale qualification。

## 8. 结果对应的明确决策

| 观察 | 结论 |
| --- | --- |
| fixed `U0` exact-core oracle 失败 | 拒绝 layer-0 persistent boundary；不再调 paired replay 美化结果 |
| oracle 高，但 `E^C-E^P` 不小、paired core 失败 | 拒绝 paired finite-release recurrence；保留 representation oracle |
| paired 与 Current-only/mismatched controls 无显著差异 | 归类为 generic low-rank compression + delta encoding，不作为 Design 1 |
| M1--M5 通过，但成本仍 `>20%` | Insight 2 可以成立，当前 Design realization 仍失败；只允许研究结构性省算 executor |
| 机制、functional recovery、合法性和 `<=20%` 全部通过 | 才进入 prospective Design 1 freeze 与更大 population confirmation |

特别地，如果最终只有 single-arm Current replay 表现最好，应诚实地把它保留为系统 baseline。不能为了
维护 paired 叙事而降低 novelty gate，也不能把两个 rank 参数继续调到某个 edge 好看。

## 9. 论文写作边界

若上述机制全部通过，可以写：

> 我们发现，相邻版本 Transformer 的 bounded historical replay error 在逐层计算中具有版本共享成分；
> matched Parent/Current finite difference 能消去该成分。基于此，我们以 exact Parent cache 为基准，
> 只生成并持久化 reader-sufficient release correction，而不重建完整 Current KV。

在证据通过前，只能写成 research hypothesis。以下说法现在都越界：

- “rank4/rank4 证明了 paired closure”；
- “两臂共享一个 history quotient”——当前两臂实际 basis 不同；
- “layer-0 basis 跨层稳定”——除非 exact-core oracle 在 population 上通过；
- “我们提出新的 low-rank KV representation / residual attention”；
- “方法无需 Current replay”——当前路径明确执行 approximate Current history replay；
- “成本进入 0--20% 就证明了 Design”——`18.3264%` 只证明 executor 可行；
- “已泛化到 Transformer recommenders”——当前完整证据仍是 legacy HSTU/Yambda；
- 用单 UID 五边结果或 full-rank exact-limit 作为论文效果证据。

## 10. 最短后续路径

下一步不应继续调 rank。最短路径是先实现一个只读 diagnostic runner，一次性输出 M1 和 M2 所需的
逐层 exact/approximate error decomposition，并把 M3 controls 放在同一执行中。先用 1 UID 做语义
preflight，只检查 invariant，不读取它来改配置；随后才为 `32 users x 5 edges` 建 prospective contract。

matrix-free input 已把同语义 executor 降到 `18.3264%`，因此不再需要为跨过 20% 调 rank。只有当
paired path 在同一次 matched runner 中显示出 single-arm 没有的稳定差分优势，才值得进入 formal
canary；否则这条路线已经得到清楚否证，应停止，而不是继续用新 basis、更多 power iteration或另一组
rank 组合延长探索。
