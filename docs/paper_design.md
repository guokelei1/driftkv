# EvoKV 论文总体设计（概念层）

更新日期：2026-08-27

本文只定义论文长期稳定的概念边界。它回答“在什么场景下解决什么问题、为什么这个问题重要、和哪些方向比较、用什么指标判断是否成功”，不记录某一次训练的超参数、版本窗口或具体结果。

## 1. Background & Motivation

### 1.1 Persistent state across model releases

推荐模型持续发布新版本，但系统中已经存在大量由旧模型产生的用户持久化 K/V：

~~~text
旧模型 theta_(t-1) + 用户历史 H_u
        -> persistent state C_u^(t-1)

新模型 theta_t + 旧 state C_u^(t-1)
        -> 当前请求预测
~~~

每次发布都对完整历史执行 Current 模型，状态语义最干净，但会产生巨大的后台计算、历史读取和
迁移时间；直接复用旧状态成本最低，却无法保证其与 Current reader 的表示和聚合语义兼容。

### 1.2 Motivation

EvoKV 首先将两个问题分开：上游通过 Full-only validation 决定模型是否值得发布；随后才评估
旧 persistent state 是否阻碍这个已接受模型兑现发布收益。Current Full 是状态演进的 reference，
不是 release admission 判定器。被拒绝的 candidate 不成为 cache producer，lineage 保持不变。

当前 motivation 已在多条相邻 edge 上观察到 release gain 与 Reuse harm 同时存在，并发现 direct
Reuse harm 随 producer version age 增大。具体数字和反例只记录在
[核心 Motivation 与 Observation](motivation_observations.md)。

### 1.3 Design goals

EvoKV 的目标是：

> 在上游已经接受 Current 模型后，以有限的后台计算和 I/O，使活跃用户状态接近 Current Full，
> 同时在连续 release 中限制近似债务，并提供可度量的 GPU 执行路径。

设计必须满足四个边界：

1. **Quality**：恢复被旧状态侵蚀的 release value，而不只降低 tensor distance；
2. **Cost**：完整计划的 compute、history I/O、state I/O 和 writeback 必须低于 Exact-All；
3. **Reliability**：连续近似必须有 bounded debt、Exact shadow 和 Rebase 回退；
4. **Protocol**：release-time 决策保持 target-free、label-free，行为标签只用于封存评价和安全审计。

## 2. System Overview

EvoKV 由三个相互承接的部分组成：

~~~text
Insight-Driven One-Release State Refinement
  -> Debt-Bounded Continuous State Evolution
  -> GPU Transformation Runtime
~~~

1. **One-Release Refinement** 以 candidate-amortized reader compatibility 为主要功能单位，
   为每个用户一次性生成 Current-reader AV offset，并在 bounded post-release horizon 内由整个
   candidate bank 摊销消费；
2. **Continuous Evolution** 在这条流水线反复执行时限制 approximation debt，并用 sampled Exact
   feedback 触发加固或 Rebase；
3. **GPU Runtime** 将单次与连续控制器产生的 typed plan 分桶、批处理并安全提交。

这一节只给全局数据流，不单列 `Design Principles`。recommendation-specific Insight 与 Design 0
机制观察直接放入第 3 章；Continuous 和 Runtime 各自说明其额外控制与执行问题。

## 3. Insight-Driven State Refinement

### 3.1 Design Insights

#### Insight 1：跨版本误差形成 candidate-shared reader compatibility correction

推荐 ranking 不是一条 query 只读取一次 context；同一份 persistent user state 会被大量 candidate
重复消费。在固定 3,000 用户、五条版本边和 64-candidate bank 上，candidate×history influence 与
Exact−Reuse readout delta 几乎都由一个 candidate-shared 方向主导。后续 signed、逐 head、无
candidate normalization 的因果干预在四种 controlled width 与真实 same-UID/same-timestamp exposed
candidate bank 上均确认 shared-only 决定主要 score gap，排除了原始 norm/normalization 与受控 bank
造成这一结构的主要替代解释。因此兼容性首先是一份用户证据对候选集合的广播失配，而不是每个
candidate 独立的 token-retrieval failure。这里已证明的是 query-dependent reader correction，
不是历史 token 可以线性物化为 evidence basis。

**Design implication：** correction 的最早稳定边界已定位到 query-dependent `activated(qK)·V`，
AV sidecar 的跨真实请求方向与 coverage-scaled recovery 已在五边过门。主设计直接生成 per-user
AV offset，不迁移完整 prefix state，也不采用 request-time per-candidate token Route。旧 compact-
probe score canary 为 4/5；取消 translated-prefix 物化的 lightweight PRO 已通过 held-out 正确性/
成本门，并在 217,584 个全人口请求上取得 AUC 5/5、log-loss 3/5、五边均值两项均改善。总体
Design viability 已通过，但事前严格双门未过，因此还不是 serving-qualified action。
按专家增量建议完成的 label-free error decomposition 与 C32/C48/C64 frontier 没有替换这一主设计：
C64 对 C32 的 internal relative L2 在 cutover/rolling 均 5/5 改善，但 absolute rolling-direction
门为 0/5，C48/C64 也不单调。按事前规则保留已验证质量的 C32 PRO，不在旧五边重测 quality。

#### Insight 2：typed entity coordinate 经过聚合变成 user-context residual

UID-disjoint factorization 表明，item/action coordinate 在 combined input 与 layer-0 K/V 中可跨用户
泛化；但经过 HSTU pointwise aggregation 与 U gate 后，item-specific predictability 明显下降。
parameter-only joint K/V mapping 的稳定正恢复与 same-scope contextual residual 的额外恢复进一步
说明，共享 version coordinate 和用户上下文变化是互补的。

**Design implication：** 版本对共享的 joint coordinate map 可以推入少量 reader probe，而不必
逐用户、逐位置物化 translated K/V。recent contextual carriers 只在 Current 模型中做 dependency-
closed replay；isolated embedding replacement 不是完整接口。

#### Insight 3：历史是带 mass 的 evidence measure，raw identity 不等于可合并

Current repair 使用少于原事件数的 carrier 仍能保持大部分 recovery；但不保留 represented mass 时，
HSTU 的非归一化聚合会系统性改变。过低 carrier density 仍会丢失异质语义。人口级 matched-budget
对照还显示：same-item/item-action pairing 显著增加语义匹配，却不能跨五条 edge 稳定胜过 positional
pairing，因此 contextual/functional substitutability 不能由 raw identity 代替。

**Design implication：** 先 `GROUP` evidence、再对较少 carrier 执行 expensive `PATCH`，并用
`SCALE` 显式保留 coverage/mass；SCALE 不能补偿过度 GROUP，未来 compact state 的 carrier
关系必须由 contextual function 验证，而不是只看 item/action equality。

三条 Insight 不再被包装为四个 operator 的同义改写。现有 `CAST + compact PATCH` 是历史 strong
baseline（Design 0），不是主方法的前置阶段；candidate-shared reader correction 的定位与持久性验证已经完成。一个
matched-cost 的 `CAST signed value measure + Current anchor residual` 最小实现已在五边 canary
上 0/5 不弱于 Design 0，不能准入。这个反例说明共享 reader effect 不能直接退化为历史 V 相加、
PCA/SVD 或固定 anchor 换名；它不构成 history basis 存在性的证据。后续 stage/persistence gate
通过后，直接来自 Current reader AV 的 compact-probe sidecar 已取得 4/5 score canary 正结果，
同时保留 `v3→v4` 反例。进一步的 lightweight PRO 把 joint map 融入一次 reader read，使用
Parent-conditioned 32 个 Current carrier，action 内 translated-prefix positions 为 0；其 held-out
数值/成本门通过且理论 FLOPs 为 Full 的 9.1%。正式 rolling quality 已显示 AUC 5/5、log-loss 3/5
和总体均值改善；额外 seed、label-free admission 与 runtime 仍未验证。
后续 progressive 增量已完整保留为负资格结果：双 probe 几乎等价，纯幅值与 segment decay 门
未过；10.52%/14.54%/18.64% carrier frontier 虽有 fidelity 收益，但不满足 absolute-direction 与
单调性联合门，因此不产生新主配置。
完整证据、反例和未闭合缺口见
[Insight-Driven State Refinement Develop Map](insight_develop_map.md)。

### 3.2 PRO Main Design 与 Design 0 比较边界

主设计统一为 Per-user Reader Offset（PRO）：

~~~text
Parent persistent K/V
  -> recent raw evidence -> 32 Parent-conditioned Current carriers
  -> one fixed Current-reader probe
       [reader-pushed joint version read of old Parent state
        + native carrier read - Parent Reuse read]
  -> persist four layerwise AV offsets
  -> bounded-horizon, candidate-shared injection
~~~

joint version map 在 key-side 被推到 probe query，在 value-side被推到 history aggregation 之后；不生成、
不持久化 translated prefix K/V。carrier 是一次性 sidecar generator 的内部对象，生成后丢弃。最终
持久对象只有每层一个 hidden-width AV vector。

当前已执行的 Parent-to-Current strong baseline 仍表示为：

~~~text
PLAN(repair width r, carrier count c)
  -> CAST(large stale region)
  -> GROUP(repair region r -> c) -> PATCH(c Current carriers)
  -> SCALE(represented mass) -> UNION/COMMIT
~~~

`CAST / PATCH / GROUP / SCALE` 是底层 typed semantics，不是四条并列 Insight；`SLICE`、`UNION`
和 `COMMIT` 分别负责寻址、typed read-view 组合和 lineage 事务。完整 IR 见
[One-Release State Refinement 与 Typed Plan IR](typed_state_refinement_algebra.md)。
该 pipeline 只作为质量/成本比较和 typed-state 语义证据；它不与 PRO 串联，也不是 Gate 的默认
fallback。安全 admission 只能在少量 Exact shadow 上事前比较 Reuse、PRO 及必要对照，未过门则
保持 Reuse 或 Exact/Rebase。

### 3.3 Reader-Pushed Version Transform

版本 map 仍只处理 Parent/Current 之间可共享的 coordinate mismatch，不声称恢复用户特定的
contextual hidden drift。它每条 release edge 构造一次，但每用户仍需流式读取 Parent state 并执行
融合聚合；不能把其 FLOPs 或 bandwidth 写成零。主 action 不写回 mapped K/V。

### 3.4 Compact Contextual Carriers

recent `r` 个 evidence 先固定 GROUP 为 `c` 个有序 carrier，Current 模型在 Parent prefix 上生成
dependency-closed contextual payload，SCALE 声明每个 carrier 的 occurrence mass。当前 primary
固定 `r=128,c=32,mass=4`；16-carrier 只保留为成本诊断，不进入质量主路径。

### 3.5 Cost-Aware Plan Selection

`r` 控制 contextual evidence 范围，`c` 控制 sidecar generator 实际执行的 Current carrier 数。
第一版冻结单一 primary `(128,32)`，不要求先训练复杂 scheduler。Design I 分别报告解析 FLOPs、
结构 token/pair work 和 rolling quality；真实 CUDA time、raw-history/state I/O、write bytes、storage
和 makespan 留给 Design III Runtime。理论计算减少不自动等于实际加速。

旧 Design 0 已完成 full-population one-release rolling qualification并暴露真实失败 edge；lightweight
PRO 已通过零物化、数值与 `<20%` 理论成本门，也完成了五边 full-population quality。PRO 的总体
可行性为正，但严格 log-loss edge-count 门未过；二者都不能据此成为 always-on 配置。具体数字只记录
在[核心 Motivation 与 Observation](motivation_observations.md)；本章只保留 release-level Exact-
shadow admission 与 Reuse/Exact fallback 接口，不提前扩展复杂 scheduler。

## 4. Debt-Bounded Continuous State Evolution

一次转换并不保证连续执行时误差不会累积。Continuous controller 对每段状态只消费：

~~~text
(last_exact_or_rebase_version,
 approximation_depth,
 estimated_compatibility_debt)
~~~

对候选单次计划 `p=(r,c)`，选择满足剩余 debt 阈值的最低成本方案：

~~~text
p* = argmin_p Cost(p)
     subject to EstimatedDebt(state, p) <= tau
~~~

若近似计划均不满足阈值，则扩大 repair 或 Exact/Rebase；达到硬上限
`approximation_depth = H` 时也强制建立 Exact anchor。sampled Current-Exact shadow 立即提供无标签
fidelity feedback：Normal 保持计划，Warning 加固 repair，Invalid 执行 Rebase 并禁用失效配置。

延迟行为质量只用于配置级 canary 审计和后续 release 的保守回退，不能成为同一请求或单个用户的
future-label scheduler。`EstimatedDebt`、`tau/H`、shadow rate 和 hysteresis 都是待验证规划，不是
当前结论。

## 5. GPU Transformation Runtime

Runtime 接收 Design I/II 的 typed plan，再按 execution signature 聚合为 GPU micro-batch：

~~~text
(source version, target version, version-map type,
 carrier-count bucket, sequence-length bucket, sidecar dtype)
~~~

逻辑上不同用户可采用不同配置；物理执行时将相同或相近配置分桶。执行路径包括 state/history
prefetch、fused Parent-state reader reduction、Current carrier replay、sidecar scale/write 和
atomic COMMIT。

Runtime 当前只冻结接口。只有 profiling 证明真实瓶颈，并且相对朴素后台重算显著改善吞吐、
makespan 或 serving isolation，才作为独立系统贡献；否则只是实现章节。

## 6. 相关工作与比较对象

论文需要和以下几类工作明确区分：

### 6.1 持久化 KV、prefix cache 与 serving cache

已有工作关注在同一模型或相近模型之间复用 prefix/KV 以减少重复计算。EvoKV 的重点不是单次请求的 cache hit，而是模型版本改变后，已经持久化的用户状态是否仍然兼容，以及如何管理整个人口的跨版本状态。

### 6.2 Continual learning、模型更新与 release pipeline

持续训练、增量更新、checkpoint lineage 和模型发布流程解决的是模型参数如何产生与上线。它们不能自动解决旧模型状态如何迁移。EvoKV 接收一个已经确定发布的当前模型，把 release 后的 persistent-state convergence 作为独立系统问题。

### 6.3 推荐系统的用户状态、序列表示与特征缓存

推荐系统中的用户历史长、用户数多、item/embedding 持续变化，状态既有长期偏好又有近期行为。EvoKV 关注这些表示在模型版本切换时的跨版本语义兼容，而不是只比较一次请求的排序模型精度。

### 6.4 近似计算、部分重算与资源分配

近似推理、分层重算、预算分配和 learned scheduler 提供了降低计算成本的工具。EvoKV 的问题在于：哪些状态可以安全复用、哪些状态需要何种依赖闭合的演进，必须由跨版本兼容性和状态风险驱动，并在完整活跃人口上满足总预算约束。

### 6.5 系统迁移、缓存一致性与 version debt

传统缓存一致性通常围绕数据更新、失效和重新获取展开。EvoKV 面向的是神经网络内部持久状态的版本债务：每次模型发布都可能留下由旧 producer 产生的状态，连续 No-op 会使状态年龄和兼容性风险累积。

论文中的比较对象不是某一篇工作的实现复刻，而是这些方向共同覆盖的基线边界。

## 7. 论文要回答的研究问题

- **RQ1：存在性** 新模型已经带来模型发布收益时，直接复用父版本 persistent state 是否会损害这个收益？
- **RQ2：结构** 跨版本风险能否分解为 candidate-shared reader compatibility correction、typed entity/action
  coordinate、user-context residual 与 represented mass？
- **RQ3：单次转换** 不物化 translated prefix 的 lightweight PRO 能否以低于 Exact-All 20% 的
  release-time FLOPs 恢复 rolling quality；bounded-horizon AV sidecar 是否跨 edge 安全？
- **RQ4：连续演化** Debt-bounded incremental refinement 加 sampled Exact feedback，能否在多个
  release 中限制近似深度和质量偏差，并以低于每版 Exact-All 的摊销成本触发必要 Rebase？
- **RQ5：物理执行** 异构的 typed transition plan 能否被编译成高吞吐 GPU workload，并在
  state/history I/O、迁移 makespan 和 serving 隔离上优于 Exact-All？
- **RQ6：适用边界** 上述 Insight、单次转换和持续演化结果能否跨 seed、模型规模和第二 workload 保持？

RQ1 是 motivation 的最低成立条件；RQ2 推导 Design I；RQ3 验证一次转换。RQ4 是下一阶段的
核心算法与状态语义问题，RQ5 是最后实现的系统执行问题，RQ6 负责外推边界。target-free
allocation 可以服务 RQ3/RQ4，但不是当前必须先完成的独立 headline。

## 8. 成功标准与主要比较

One-Release 至少比较以下状态处理策略：

- No-op / direct Reuse；
- Exact-All / Current Full；
- 固定宏计划：Translate-All、Tail-128、weighted Landmark-64 和 Translate+Tail-128；
- 固定的 `CAST + GROUP/PATCH + SCALE` Design 0（strong baseline）；
- lightweight PRO `(recent128, carriers32, latest-item probe)`；只有获得新的 prospective quality
  contract 后才可加入正式 rolling 比较。

Continuous 只需围绕闭环比较：每次 Exact-All、连续近似但不反馈、固定周期 Rebase，以及
debt-bounded + Exact-shadow-triggered Rebase。Runtime 完成后再比较
未分桶执行、execution-signature batching 与 Exact-All 后台重算。Random、metadata-only、
target-free selector 和 offline oracle 只在需要回答分配问题时加入，不作为当前 Design I 的前置条件。

主要指标分为三层：

### 任务质量

在同一当前模型、同一请求、同一 causal history 和同一 readout 下，报告：

- ROC-AUC、PR-AUC；
- event log-loss、Brier；
- user-equal 与 event-weighted 的配对差异；
- 若使用开放 catalog，再报告 Top-K、NDCG、MRR 等 ranking 指标。

Motivation 的直接质量量是：

~~~text
Reuse harm = Quality(Current Exact) - Quality(Reuse)
~~~

对 loss 指标则使用 Loss(Reuse) - Loss(Current Exact)，正值代表旧状态带来损害。

### 状态语义

报告 Bernoulli JS、normalized score RMS、probability shift、Top-K overlap、margin/pairwise disagreement，以及用户级尾部。状态 fidelity 是解释机制和指导迁移的 companion，不能替代任务空间质量。

### 资源与系统代价

报告 exact-equivalent compute、token-layer work、KV read/write bytes、history I/O、新增 storage、后台 worker-hours、
迁移 makespan 和跨版本 state debt。计算、raw-history I/O、state I/O 和 storage 必须分别报告，
不能用任意权重混成一个没有解释的总分。

## 9. 论文贡献结构与当前边界

论文完成时的目标贡献结构是一个跨版本 persistent neural state evolution 系统，而不是单独一条
Tail 优化：

1. 明确模型 release、状态兼容性和状态演进的分离边界；
2. 用同一当前模型下的 Current Full/Reuse 对照建立版本化状态不兼容的任务质量证据；
3. 观察分布式失配、共享可转换版本变化和 mass-aware compact replay，并将它们编译为
   PLAN→CAST→GROUP/PATCH→SCALE/COMMIT 状态迁移流水线；
4. 用 bounded debt、最大近似深度、sampled Exact shadow 和质量触发 Rebase，将一次转换扩展为
   有明确失效检测与回退路径的连续演化闭环；
5. 将 typed transition plan 编译成可度量的 GPU 数据面，并在连续版本、不同状态年龄和更大人口上
   验证质量与摊销成本。

当前完成度必须分开写：Motivation、Small/seed17 mechanism observation、第一条固定 Design I 路径的
完整 rolling qualification 和理论 FLOPs 已有证据；该固定路径跨 edge 结果混合，事前安全边界仍未
闭合。真实 GPU/I/O 结果属于尚未实现的 Design III，而不是 Design I 的缺失理论值。
当前不冻结 residual estimator、carrier-density 安全阈值、predictor、scheduler、debt estimator、
`tau/H`、shadow rate、rebase policy 或 GPU executor。这些组件只能在相应实验后准入。

## 10. 不应过度声称的内容

- 一条有害边不能推出所有模型更新都必然有害；
- One-hop 直接复用不能单独证明 recursive debt 或最终迁移策略；
- One-release refinement 正恢复不能证明其近似可无损跨多个 release 组合；
- 一个固定 one-release plan 在多数 edge 改善 Reuse，不能证明它适合每条 release 或允许隐藏失败 edge；
- producer-age harm 单调增加只能动机化 Continuous，不能证明 debt estimator、`tau/H` 或 Rebase
  规则已经正确；
- sampled Exact fidelity 是安全反馈，不自动等价于真实推荐质量保证；
- 延迟行为标签不能被用于同一请求、单个用户或 qualification slice 的未来标签调度；
- 版本对 CAST operator 可跨用户共享构造，但仍需逐用户读取、转换和写入 state；
- 状态 fidelity 提升不能直接等同于线上排序质量提升；
- Yambda 单一 workload 不能代表所有推荐场景；
- 训练规模、checkpoint 大小或离线 GPU 数不能单独定义系统规模；
- 诊断性 KV splice 不能被写成可部署 action；
- payload residual 在张量层可加，不能被写成任务质量必然单调；
- Tail replay 稳定正恢复不能被写成“Tail 已证明是等宽历史中最敏感的位置”；
- aggregate CAST 正恢复不能被写成每层、每个 token 或每个 head 都同样可转换；
- GROUP+SCALE 的结构 work 下降不能替代整体 plan 的 CUDA latency、I/O 和 makespan 实测；
- Runtime 的 batching/pipeline 设想不能替代 GPU throughput、utilization、tail latency 和 serving
  interference 实测；
- Small/seed17 的 carrier-density frontier 不能直接冻结为跨规模阈值；
- 模型发布质量提升不能被归因于 EvoKV 的状态管理。

具体架构、数据、版本链和阶段性结果统一记录在
[具体实验设计](experimental_design.md) 与
[核心 Motivation 与 Observation](motivation_observations.md) 中。Insight 到四阶段 pipeline 和底层 IR 的推导见
[Insight-Driven State Refinement Develop Map](insight_develop_map.md) 和
[Typed State Refinement Algebra](typed_state_refinement_algebra.md)。
