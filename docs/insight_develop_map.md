# EvoKV Insight-Driven State Refinement Develop Map

更新日期：2026-08-28

> **历史文档（2026-09-02 起 superseded）。** 本文保留 Small/seed17 的完整推导、PRO 数字与
> negative results，但其中“论文主设计仍为 C32 lightweight PRO”“AV correction 已冻结”等表述
> 不再代表当前研究裁决。当前权威方向是 Medium Insight 1/Insight 2 与 Migration Sketch Design；入口见
> [paper design](paper_design.md)与
> [Insight 2 / Design 1 统一稿](insight2_design1_expert_brief.md)。不要用本文
> 的旧结论反向选择新的 boundary、estimator 或 action。

本文是论文第 3 章 `Insight-Driven State Refinement` 的研究支撑文档，回答：

> 推荐系统中一份 persistent user state 被整个 candidate bank 重复读取；跨版本 compatibility
> 是否存在候选共享的 user-evidence structure，并能否据此推出比通用 token repair 更贴合
> recommendation workload 的状态迁移？

论文不再单独设置一个服务所有设计的 Insight 章节，也不在 Insight 和机制之间增加独立
`Design Principles`。本文件的内容映射为：

~~~text
Chapter 3: Insight-Driven State Refinement
  3.1 Design Insights: Insight -> Design implication
  3.2 PRO Main Design and Design 0 comparison boundary
  3.3 Reader-Pushed Version Transform
  3.4 Compact Contextual Carriers
  3.5 Cost-Aware Plan Selection
~~~

它不负责证明多次 transition 可以安全组合，也不负责 GPU 执行。后续两章分别是
`Debt-Bounded Continuous State Evolution` 和 `GPU Transformation Runtime`。现有三条
机制观察足以推出一次转换的 strong baseline；新的 candidate-broadcast observation 已找到更
recommendation-specific 的 headline structure。signed causal 与真实 exposed candidate 复核已经通过；
唯一的 matched-cost evidence-measure basis canary 为 0/5，不进入正式 quality qualification。它
说明当前已证实对象应称为 reader compatibility correction，而不是 history basis；仍没有合格的
新 action。最新 stage/persistence gate 已定位 `qK·V/AV` 并以五边通过；唯一 compact-probe AV
sidecar 的无标签 score canary 为 4/5，同时保留 `v3→v4` 反例，尚未进入 formal quality。
最新 lightweight PRO 已将 joint map 推入一次固定 reader probe，action 内不物化 translated
prefix；held-out 正确性/成本门通过，primary 32-carrier 为 Full 理论 FLOPs 的 9.1%。冻结机制的
五边全人口 formal quality 也已完成：AUC 5/5、log-loss 3/5、五边均值两项均改善。总体 Design
viability 通过；严格双门因 log-loss edge count 未过，仍不准入 serving/seed/runtime qualification。
最新 progressive 增量也已完成：双固定 probe 在 5/5 edge 几乎等价，纯幅值与 segment-decay
解释未过冻结门。held-out C32/C48/C64 frontier 中，C64 relative L2 对 C32 在 cutover/rolling
均 5/5 改善，但 absolute rolling direction 为 0/5 过门，且 C48/C64 非单调；因此不选择升级，
论文主设计仍为已完成真实质量验证的 C32 lightweight PRO。
Continuous 需要验证 bounded-debt estimator、Exact-shadow feedback 和
Rebase 闭环，Runtime 需要真实执行数据。

## 最新主裁决：candidate-shared reader compatibility correction

固定 3,000 用户（Small 的 30%）、五条 `v0→...→v5` 相邻边、512-event cutover history 和每用户
64 个 label-free candidate probe 的 observation 表明：

- 所有 60,000 个 candidate×history influence matrix 均为 rank-1@90%，各 edge/layer 的第一
  candidate-shared 方向平均占 99.9681%–99.9992% energy；
- Exact−Reuse influence delta 在 59,999/60,000 个 user-edge-layer 上为 rank-1@90%，最终
  readout delta 在 15,000/15,000 个 user-edge 上为 rank-1@90%；
- held-out-user factorization 中，item/action coordinate 在 combined input 与 layer-0 K 上很强，
  但 `AV × U` update 后 item-specific predictability 大幅下降；
- same-item/typed pairing 虽显著增加 semantic matched pairs，却只在 3/5 edge 胜过 positional
  pairing，不能准入 raw semantic GROUP。
- signed、逐 head、无 candidate normalization 的 oracle 干预在 controlled width-64 上恢复
  97.98%–99.64% 概率缺口；真实 exposed candidates 的 20/20 edge×width 组合均由 shared 优于
  residual，shared-only 平均 absolute logit gap 为 `5.58e-5`，Reuse 为 `1.55e-2`；
- 一个 matched-cost `CAST value measure + Current anchor residual` 实现在五边 label-free canary 上
  0/5 不弱于 Design 0，按合同停止，说明 causal structure 不能直接用 signed V 相加来可执行化。
- correction 最早在 query-dependent `activated(qK)·V` 形成；AV correction 在 11,364 对相邻
  真实请求上以 5/5 通过方向与 coverage-scaled recovery 门；
- 唯一 compact-probe AV sidecar 在 1,805 个无标签请求上 4/5 不弱于 Design 0，未自动进入 quality。
- lightweight PRO 的 fused AV 最大相对 L2 为 `4.73e-6`、replay error 为 `3.58e-7`，translated-
  prefix positions 为 0；32-carrier sidecar 相对旧 extractor 的五边方向 cosine mean 为
  `0.9983–0.9993`，理论成本为 Full 的 9.1%。
- lightweight PRO 在 217,584 个正式 rolling 请求上相对 Reuse 的 AUC 为 5/5 正向、log-loss 为
  3/5 正向，五边非加权平均为 `+0.06641` AUC pp、`−7.65e-5` log-loss。
- progressive decomposition 的 C32 direction 门为 cutover 2/5、rolling 0/5，amplitude-dominant
  门为 0/5+0/5；双 probe 一致性为 5/5，segment decay 仅 2/5，不替换 global decay。
- held-out 10.52%/14.54%/18.64% Full-FLOPs frontier 中，C64 对 C32 relative L2 为 5/5+5/5
  改善，但 absolute direction 和 C48 intermediate 单调性门失败；按事前规则保留 C32。

因此当前论文最有辨识度的 Insight 不再是“tail 比 prefix 更重要”，而是：

~~~text
Cross-version HSTU history error is distributed and contextual, but its effect
converges in the Current reader to a candidate-shared user compatibility correction.
State generation is user-contextual; state consumption is candidate-amortized.
~~~

**Design implication：** 主方法收敛为 Per-user Reader Offset：用一次 reader-pushed version read
和 32 个 contextual carrier 生成 AV sidecar，再在 bounded horizon 内跨 candidate 摊销。不应把
每个 candidate 当成独立 RAG query 做 request-time token Route；总体可行性已通过，但严格上线
qualification 尚未通过。完整数字与边界见
`results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/expert_discussion_summary.md`。

当前论文不再把 `CAST / PATCH / GROUP / SCALE` 包装成四条并列 headline insight，也不把其固定
组合包装成最终机制。它们是由三条机制观察推导出的 **Design 0 / strong baseline**；PRO 不与
这条 pipeline 串联：

~~~text
PLAN(repair width r, carrier count c)
  -> CAST(large stable region)
  -> GROUP(repair region r -> c) -> PATCH(c Current carriers)
  -> SCALE(represented mass) -> UNION/COMMIT
~~~

其中 `PLAN` 是选择器而不是状态算子；`GROUP + SCALE` 是同一条“mass-aware compact
rematerialization” Insight 的两个实现语义。`CAST / PATCH / GROUP / SCALE` 仍保留为底层 typed
plan IR，完整定义和机制数字见
[Typed State Refinement Algebra](typed_state_refinement_algebra.md)。

## 0. Design 0：三条机制观察直接推导可执行 baseline

### Insight 1：分布式失配与非对称修复

~~~text
Cross-version mismatch propagates across a multi-layer dependency chain;
a recent suffix is a useful and cheap legal repair boundary, but tail replay alone is incomplete.
~~~

当前直接证据是：只替换 layer 0 不充分并可能恶化，而 lower 3/4 Current layers 联合恢复
83.4%–96.0% output gap；dependency-closed Tail-128 在五条 edge 都正恢复，但只恢复
19.4%–25.8%。删除 recent-128 比删除 oldest-128 破坏更大，说明近期证据具有较高读取效用；
同时 causal dependency 使 suffix 成为比 middle/old segment 更便宜的 exact replay 边界。

这还没有证明 recent tail 是等大小区域中“最敏感”的区域。现有 old384/recent128 和
old480/recent32 对照宽度不匹配，也没有 old/middle/recent/random-128 的统一位置干预。因此当前
准入的结论是“Tail 有用、便宜但不完整”，不是“Tail 全局最重要”。同样，当前只否定
layer-0-only，不声称所有 layer/head 选择都已被否定。

**Design implication：** 不采用单层或 Tail-only 方案；组合大范围便宜修复与小范围
dependency-closed contextual PATCH。

### Insight 2：共享且可转换的版本变化

~~~text
Part of the mismatch is shared across users and can be repaired by a parameter-only,
layerwise joint K/V translation without replaying raw history.
~~~

parameter-only CAST 在五条 edge 全部正恢复，范围 21.6%–64.1%，平均 43.0%。它只由 Parent/
Current 每层的 K/V projection 和 normalization 参数构造，不读取 raw history、label 或 Current
target K/V。CAST 与 Tail PATCH 组合平均恢复 57.0%，same-scope residual 干预平均恢复 61.2%，
说明共享格式变化与用户上下文变化是互补成分。

当前只证明“对完整 layered state 应用 layerwise CAST 的总体效果稳定”，尚未逐个证明每个 layer
和每个 token region 都有正贡献。因此论文可以写 distributed transformable component，不能写
“所有 token、所有层都同样可转换”。

**Design implication：** 为版本对构造共享 joint K/V `CAST`，将不读 raw history 的转换用于大范围
state，把用户上下文相关的剩余误差留给 PATCH。

### Insight 3：保持证据质量的紧凑重算

~~~text
Current-state repair need not materialize one state per event, but every compact carrier
must preserve how much ordered evidence it represents.
~~~

将 recent-128 映射到 64 个 carrier 时，`GROUP->PATCH` 与先 dense PATCH 再 GROUP 的 output-gap
recovery 平均只差 0.49 percentage point；前者只物化 64 个 Current states。匹配的 SCALE 在
40/40 个非平凡消融中都优于 unscaled 路径。旧 capability 对照中，`CAST + weighted
Landmark-64` 平均恢复 51.9%，而 `CAST + dense Tail-128` 为 57.0%：重算状态数减半后，平均少
5.1 个 recovery points，而不是损失一半效果。

因此 GROUP 和 SCALE 在论文里是一条联合 Insight：GROUP 减少 expensive PATCH carrier，SCALE
保留 HSTU 非归一化聚合中的 occurrence mass。SCALE 不能补回被过度压缩的具体语义；8/16
carriers 在部分 edge 仍失败。

**Design implication：** 在 expensive PATCH 前 GROUP 为较少有序 carrier，并用 SCALE 保留
coverage/mass；carrier density 作为 `(r,c)` 质量—成本轴，而不是无限压缩。

## 1. 统一问题和证据层级

整条链回答五个问题：

1. Reuse harm 是否真的侵蚀 release value，而非随机输出扰动？
2. mismatch 中是否存在不读 raw history 就能修复的版本坐标分量？
3. CAST 之后是否仍有必须由 contextual residual 修复的部分？
4. 少量 Current-space carrier 的价值能否被表达成连续的 coverage/density 轴？
5. 压缩后的状态是否必须显式保留 occurrence mass？

证据分三级：

- **Task observation**：AUC/log-loss 和 release-gain erosion；
- **Mechanism intervention**：CAST/PATCH/GROUP/SCALE 对 Current–Reuse output gap 的干预；
- **System qualification**：完整 rolling quality、理论计算、真实 Runtime 资源和 held-out seed/scale。

当前前两级已经足够闭合 **One-Release** instruction semantics。第三级中的固定计划 rolling quality
和理论 FLOPs 已完成，额外 seed/scale 和跨 edge 安全性尚未完成；真实 GPU/I/O 由独立 Runtime
章节负责。这些证据还不能闭合 multi-release state evolution。

## 2. 已观察的核心 Insight

| Insight / observation | 规模 | 主要结果 | 设计含义 |
| --- | ---: | --- | --- |
| candidate-shared reader correction | 3,000 controlled users + 15,338 real request groups | signed causal 通过；最早在 qK·V，AV 跨请求 5/5 持久；唯一 sidecar score canary 4/5 | 按用户生成 AV sidecar 并跨候选摊销；不做 per-candidate Route |
| typed coordinate → contextual residual | 3,000 users；UID-disjoint fit/held-out | layer-0 K item/action R² 86.7%–92.6%；AV×U 后 item-specific component 大幅下降 | shared typed coordinate + user-context residual，而非 isolated embedding operator |
| raw semantic coreset boundary | 3,000 users × 5 edges | same-item matching 约 3.5%→30%，但仅 3/5 edge 胜 positional | raw item/action equality 不足以定义 GROUP；需 contextual/functional substitutability |
| release-benefit targeting | 217,584 requests，5 edges | `G/H` Spearman 0.323–0.606；matched probe 0.489–0.889 | 流水线目标是恢复 release value，不是只降 tensor distance |
| Current-state anchoring | 32/64 users per edge | natural/current anchor 明显优于 pure eviction；pre-cutover tail replay 稳定降 gap 17.8%–23.4% | 需要 Current-space state refinement，不能只等 eviction |
| recommendation sensitivity | 217,584 requests | novel-to-prefix AUC harm 在五条 edge 都高于 recent repeat | 进入 evaluation/failure analysis，当前不足以准入 Route |
| contextual origin | 23,051 requests + 128 probes/edge | isolated item embedding 弱且变号；Parent blocks 产生 64.6%–92.6% gap | 需要 contextual PATCH，不需要 embedding-specific operator |
| layered K/V propagation | 128 probes/edge | K-only/V-only 都不充分；lower 3/4 Current layers 恢复 83.4%–96.0% gap | CAST 必须 joint K/V；PATCH 必须带 dependency contract |
| coordinate repair | 1,267 requests，5 edges | parameter-only joint map 恢复 21.6%–64.1%，平均 43.0% | 直接推导 `CAST` |
| contextual residual | 同上 | CAST+tail PATCH 恢复 38.7%–72.6%，比较好单项多 8.5–18.6 points | 直推 `PATCH`，但旧证据可能只是 prefix/tail 分区 |
| same-scope CAST/PATCH | 同上 | Parent-generated tail residual 加到 CAST base 恢复 37.7%–79.6%，比较好单项多 10.3–23.5 points | 确认版本 base 与 contextual delta 可在同 scope 叠加 |
| carrier-density refinement | 同上 | 8->16->32->64->128 的 40 个密度增量中 39 个非负 | 直推 `GROUP`，但不冻结单 seed 阈值 |
| state-mass contract | 同上 | 8/16/32/64 carriers、两个顺序、五 edge 的 40/40 SCALE 消融均为正 | 直推 `SCALE` |
| fixed one-release rolling qualification | 217,584 requests，5 edges | 固定 `CAST384 + GROUP/PATCH 128->64 + SCALE2` 在 4/5 edge 提高 Reuse AUC；保守理论 compute 为 Exact 的 48.0%；v4->v5 失败 | 流水线可执行且理论计算减少 52.0%，但固定全局 `(r,c)` 不能无条件进入 Continuous |

### 2.1 固定流水线的完整任务质量结果

机制实验之后冻结的第一条可执行路径是：

~~~text
CAST(old prefix 384)
  -> GROUP(recent 128 evidence -> 64 carriers)
  -> PATCH(64 Current carriers)
  -> SCALE(represented mass = 2)
~~~

它在五条正式 D14/E14 edge、217,584 个请求上与既有 Recompute/Reuse 路径同请求评估。沿用
Full-only `Current-Parent` 发布收益作为分母时，前三条 edge 的 Our retained gain 分别为
97.2%、117.9% 和 87.3%；v3->v4 虽提高 Reuse 0.090555 AUC point，但发布收益分母过小；v4->v5
则比 Reuse 低 0.265765 point，是不能隐藏的真实反例。五条 edge 都满足正式
`Recompute AUC > Reuse AUC`。

这项结果完成了“Insight -> typed semantics -> 固定 one-release path -> rolling AUC”的可执行链，
同时给出一个重要的新边界：sampled output-gap recovery 不能保证每条 release edge 上的 AUC 恢复，
一个固定 `(r=128,c=64)` 配置也不是普适 policy。该反例可以动机化未来的事前安全判断和
Exact-shadow/Rebase，但不能用本次 qualification label 事后调 action。完整公式、逐 edge 表和协议见
[核心 Motivation 与 Observation](motivation_observations.md#41-固定-one-release-方案的完整-rolling-auc)。

在完整 512-position state 上，按理想 causal attention 的有效 pair 公平计数，Exact-All 为
0.625 GFLOPs/user，固定计划为 0.301 GFLOPs/user，即使用 48.0% compute、理论减少 52.0%。其中
CAST 占 Exact 的 32.2%，compact GROUP/PATCH/SCALE 占 15.9%。当前 dense PyTorch 图的 companion
为 34.1%，但不作为主值，也不推出 GPU runtime 加速。

## 3. Design 0 observation -> Pipeline 与底层语义

三条机制观察与四个底层语义不是一对一关系。分布式失配观察确定“大范围便宜翻译 + 小范围
依赖闭合重读”的分工；共享 coordinate 解释 CAST 为什么可行；evidence mass 联合推导 GROUP
和 SCALE。选择 `r/c` 的 PLAN 位于这些语义之上，不是第五种修复原语。新的
candidate-broadcast headline 改变未来机制的优先级，但不会事后修改这些已验证 instruction semantics。

### 3.1 `CAST`

~~~text
Observation:
  Parent K/V can be partially repaired using only Parent/Current parameters.

Inference:
  mismatch contains a version-coordinate component independent of raw replay.

Operator requirement:
  change version/read coordinate while preserving coverage and mass.

Instruction:
  CAST
~~~

这条链已由五条 edge 的 parameter-only intervention 直接支持。

### 3.2 `PATCH`

~~~text
Observation:
  contextual blocks dominate the mismatch;
  CAST leaves residual harm;
  raw-history residual adds recovery on the same CAST scope.

Inference:
  coordinate repair is insufficient; state needs a typed contextual delta.

Operator requirement:
  modify payload without changing evidence coverage or represented mass.

Instruction:
  PATCH(base_lineage, target_version, dependency_scope, delta)
~~~

最新 same-scope 干预补上了旧证据的关键缺口：CAST/PATCH 互补不再只能用 prefix/tail
作用区域不同解释。但 payload 可加不代表 quality 单调，PATCH 仍必须是 base-aware typed delta。

### 3.3 `GROUP + SCALE`：一条论文 Insight，两个实现语义

~~~text
Observation:
  small Current state anchors stale history;
  carrier density gives an almost monotone output-fidelity refinement curve.

Inference:
  evidence coverage/cardinality is an independent budget axis.

Operator requirement:
  explicitly map ordered evidence occurrences to carrier scopes.

Instruction:
  GROUP(ordered_coverage, carrier_map)
~~~

GROUP 不产生 K/V；它必须在 expensive PATCH 之前减少 carrier 数，才能降低主要重算。
当前 8/16 carriers 在部分 edge 仍为负 recovery，因此只能准入“密度是可控轴”，
不能准入固定密度。

~~~text
Observation:
  compact state without multiplicity systematically changes HSTU reads;
  mass-aware variants improve every non-trivial matched ablation.

Inference:
  a state carries aggregation measure in addition to K/V payload.

Operator requirement:
  change read mass without changing payload, version, or coverage.

Instruction:
  SCALE(alpha_or_represented_mass)
~~~

SCALE 是 compact replay 的必要接口语义，但不单独作为 headline mechanism。它不能补偿过度
GROUP 造成的 heterogeneous evidence 丢失。

## 4. 不从 Insight 推导新原语的观察

### Novel-to-prefix

novel candidate 的 aggregate harm 稳定更大，但不同 repair plan 在 novel/repeat cohort 上没有稳定的
差异恢复排序，Route 也多数恶化。3,000-user candidate bank 进一步显示三类 probe 的 influence
support 与绝对 logit shift 接近，Exact−Reuse readout delta 对 candidate bank 几乎 rank-1。
因此它当前只定义 evaluation weighting 和 failure analysis，不产生 `NOVEL_ROUTE` 或
request-time repair primitive；优先修复的是 candidate-shared reader compatibility correction。

### Embedding drift

raw item-embedding drift、candidate/history geometry 与 harm 的关系弱且跨 edge 变号；isolated item
embedding 只产生 Parent-all gap 的 0.5%–4.0%。因此当前不产生 `EMBEDDING_REPAIR`。若未来
contextual exposure 能在 held-out edge 预测 PATCH value，它只改变 SLICE/value estimator。
新的 held-out-user factorization 同时说明不能把这条负结果误写为“item identity 不重要”：item/action
coordinate 在 combined input 和 layer-0 K 很强，只是经 aggregation/gating 后不再能由 isolated
embedding operator 完整修复。

### History Utility cohorts

recent/old Utility、repeat、diversity 和 user activity 的方向不稳定，不准入 semantic-region scheduler，也不产生
新原语。

## 5. 新机制的抽象评估

当前裁决：typed IR 在抽象层面上明显好于旧 action catalog，足够保留为 Design 0 的稳定 plan IR；
但固定流水线不是最终 recommendation-specific mechanism。candidate-broadcast structure 已通过
signed causal/真实 candidate 复核；一个合法、matched-cost 的 signed value-measure plan 已执行并在
canary 上失败。后续 `qK·V/AV` stage 与跨请求 persistence gate 已通过；唯一 Current-HSTU AV
sidecar 的无标签 score canary 为 4/5，但尚未进入正式质量。

| 标准 | 裁决 |
| --- | --- |
| 底层语义可分解 | 通过；CAST/PATCH/GROUP/SCALE 分别修改 coordinate/payload/coverage/mass |
| 同一状态可组合 | 通过 mechanism probe；same-scope CAST+PATCH 五 edge 都有额外恢复 |
| 旧宏计划可表达 | 通过；Tail/Exact/Translate/Landmark/Retire/Route/Fuse 均可编译 |
| 能进行编译优化 | 通过；被 exact PATCH 覆盖的 CAST scope 实测误差为 0 |
| workload insight 不污染 instruction set | 通过；user/item/candidate 信号仅作为 scope/value 候选 |
| 完整 recommendation-quality 资格 | **总体可行性通过、严格资格未过**；lightweight PRO 完整 rolling AUC 5/5、log-loss 3/5，五边均值两项改善；progressive 增量未形成可冻结升级；严格双门要求各 4/5 |
| 理论计算优势 | **通过**；完整计划含 CAST 后为 Exact 的 48.0% causal FLOPs，理论减少 52.0% |
| 实际 GPU/I/O 优势 | 留给 Design III Runtime；Design I 不用理论 FLOPs 代替 runtime 结论 |
| target-free budget compiler | **未通过**；还没有 held-out residual-value estimator |

因此当前准入的是一次四阶段流水线的 mechanism semantics 和底层 plan IR，不是
scheduler、multi-release policy、GPU runtime 或已经证明更快的端到端系统。

## 6. 当前机制与人口级 observation

当前代码：

~~~text
scripts/insight/probe_refinement_algebra.py
~~~

当前结果：

~~~text
results/yambda500m_small_seed17/insight_refinement_algebra_v1/
~~~

它只保留四个核心对照：

1. `CAST / PATCH / CAST+PATCH / same-scope additive PATCH`；
2. exact PATCH 覆盖 CAST 的死代码消除；
3. `GROUP->PATCH` 与 `PATCH->GROUP`；
4. 8/16/32/64/128 carrier density 和 matched SCALE ablation。

旧 primitive discovery 脚本的 reader bridge、Retire、Route 和大量动作组合不再执行。其历史结果保留为负证据，
但不定义新代数。

人口级 recommendation-state observation 使用：

~~~text
scripts/insight/probe_recommendation_state_structure.py
scripts/insight/adjudicate_recommendation_state_structure.py
configs/contracts/yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml
results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/
~~~

它固定 3,000 用户而不是扩大 request 数来伪装 population coverage，完整报告五条边，并只保留
compact aggregates。该 observation 不训练新模型、不读取 label、不把 probe candidate 当作负样本。

## 7. 理论计算成本已知什么

在当前 4-layer/context512 机制 probe 中，固定 recent-128 scope 的理论结构成本为：

| 计划 | 重算 token-layer / Exact-All | causal attention-pair work / Exact-All |
| --- | ---: | ---: |
| Exact-All | 100% | 100% |
| dense Tail-128 PATCH | 25.0% | 43.7% |
| GROUP(128->64) -> PATCH -> SCALE | 12.5% | 20.3% |

GROUP 本身是线性 coverage-map/gather；SCALE 为 64 个 carrier 的线性 read-contribution
处理。因此它们不会在算子数量上把减少的 causal replay 全部吃回。将 CAST 也计入后，完整
512-token 固定计划为：

| 组件 | 保守 causal FLOPs / user | 相对 Exact-All |
| --- | ---: | ---: |
| Exact-All | 0.625 GFLOPs | 100.0% |
| Reuse state conversion | 0 | 0.0%* |
| CAST 384 positions | 0.201 GFLOPs | 32.2% |
| GROUP/PATCH 128->64 + SCALE | 0.099 GFLOPs | 15.9% |
| **完整 Our** | **0.301 GFLOPs** | **48.0%** |

`*` 这里只计算 release-time neural arithmetic，不计算 state read/storage。主值使用理想 causal kernel
的有效 pair，因而不会利用当前 Exact dense mask 的额外浪费来夸大收益。当前 dense graph 的 Our
比例为 34.1%，只作为实现图 companion。两种口径都不是 CUDA latency、GPU utilization、KV bandwidth
或 makespan 证明；这些指标属于 Design III Runtime。

## 8. 接下来的关键补强

不继续增加大动作或大量正确性测试。下一阶段围绕新的 recommendation-specific Insight 和
Design 0 边界补强：

1. **位置与依赖闭包**：在同一批请求上做等宽 old/middle/recent/random-128 诊断性
   Current-K/V 区域干预，同时单独报告各自可执行 causal closure 成本。前者回答位置敏感性，
   后者回答为什么 suffix 是便宜边界；诊断 splice 不进入 scale action set。
2. **CAST 贡献分解**：对 normalized layer bundle 和 token quartile 做 leave-one-region-out/
   cumulative CAST，区分“整体 CAST 有效”与“每层、每位置都有效”。未测 head 之前不声称
   head selection 已被否定。
3. **完整计划理论成本**：已计入 CAST、compact PATCH 和 SCALE；固定计划为 Exact 的 48.0%
   causal FLOPs。CUDA 时间、raw-history/state I/O 和 makespan 不在 Design I 提前证明，统一交给
   Design III Runtime。
4. **跨 edge 失效边界**：rolling AUC 已完成且暴露 v4->v5 反例；下一轮只允许在新 prospective
   contract 下验证事前固定的安全判断或回退，不用本轮 label 反向调 `r/c`。
5. **AV broadcast residual**：stage/persistence 与 score-replay canary 已完成；下一步是否冻结
   rolling quality 由专家另行裁决，不从 4/5 canary 调 probe、group size 或 coverage scale，
   也不拟合 target K/V。

前两项补强 Design 0 的机制边界；第三项和 rolling AUC 已完成，第四项来自其真实反例，第五项
是 candidate-amortized reader-correction headline 的首个机制化候选，但尚未获得质量资格。
固定计划已经证明“一次转换可以在正式任务指标上产生收益，并理论减少 52.0% 计算”，但还没有满足
跨 edge 稳定性。后续不把复杂 PATCH-value scheduler 当作前置条件；若人口预算或安全边界确实需要
选择性分配，再在独立 development evidence 上寻找 held-out、label-free marginal-value proxy。

位置和 CAST 分解都可复用现有五条 edge checkpoint 和已封存请求，是 focused inference probe，
不需要新长训练。它们仍需报告全部 edge 和固定 cohort，不按结果选位置、layer 或 carrier density。

Medium/Large 长训练、theta3 和 RecFlow Medium 仍保持现有门禁。新代数不授权新的 scale action、训练或
qualification tuning。

## 9. 最终逻辑链

~~~text
Release gain is selectively eroded by stale persistent state
  -> the problem is worth repairing

Mismatch spans a multi-layer dependency chain and Tail replay is useful but incomplete
  -> split broad cheap repair from local dependency-closed replay

Parameter-only joint K/V translation recovers a stable shared component
  -> CAST(large stale region) + PATCH(context-specific repair region)

Current repair can use fewer carriers, but unnormalised aggregation exposes lost mass
  -> GROUP before PATCH + SCALE represented occurrence mass

Typed refined segments form one legal state view
  -> UNION -> COMMIT
~~~

这条链已经可以从 Insight 正向推导四阶段流水线，并已用一个固定计划完成 full-population rolling
AUC 执行，而不是只有算子名称后的事后解释。剩余未闭合的是 Tail 位置比较、CAST 粒度归因、
以及 v4->v5 所揭示的跨 edge 安全边界，不是继续发明新动作。实际端到端性能由 Runtime 章节承接。
target-free allocation 是可选增强；最小的保守回退则是进入 Continuous 前必须面对的可靠性问题。

## 10. 向 Continuous 和 Runtime 的交付边界

Design I 输出一次可执行的近似转换。Continuous controller 不重新打开底层操作集合，只为每段状态
维护三个控制量：

~~~text
last_exact_or_rebase_version,
approximation_depth,
estimated_compatibility_debt
~~~

候选单次计划仍写成 `p=(r,c)`。Continuous 的 provisional rule 是选择满足
`D_hat(S,p) <= tau` 的最低成本计划；若没有近似计划满足阈值，或 approximation depth 达到硬上限
`H`，则 Exact/Rebase 并清零 debt/depth。

每个 release 使用事前固定的小规模 Current-Exact shadow，立即比较无标签 fidelity。Normal 状态保持
计划，Warning 状态增加 repair、降低压缩，Invalid 状态 Rebase 并禁用失效配置。延迟行为质量只做
配置级 canary 审计和后续 release 的保守回退，不能进入同一请求或单个用户的 future-label scheduling。

现有 Insight 对这条链的支持与缺口是：

- long-age direct Reuse harm 随 producer age 增大，说明持续不处理会积累风险；它没有证明
  当前 `D_hat` 能准确估计 debt；
- One-Release 的 recovery/cost frontier 提供 `(r,c)` 候选计划，却没有证明多 release 误差受控；
- output fidelity 与 rolling quality 不完全等价；固定计划在 v4->v5 的失败进一步支持 Exact shadow
  和质量回退的必要性；
- `tau`、`H`、shadow rate、hysteresis 和回退范围仍需 prospective experiment，不是现有结果。

因此当前逻辑链的准确终点是：

~~~text
three observed Insights
  -> one executable one-release refinement
  -> bounded-debt plan selection
  -> sampled Exact feedback and quality-triggered Rebase
  -> deferred GPU execution problem
~~~
