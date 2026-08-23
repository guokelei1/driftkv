# EvoKV 论文设计：发布期预算化状态收敛

更新日期：2026-08-22。本文定义稳定的论文问题边界；当前证据见
[截至 P11.4 的统一总结](evidence_summary_through_p11.md)，执行授权见
[当前路线](current_route.md)。P7–P11 已完成从 `H → S → legal partial → scheduler →
recursive quality` 的 development 闭环；下一阶段是冻结方法的规模验证，paper
qualification 仍未执行。

## 一句话定义

> **EvoKV is a release-time state evolution system that transitions an entire active population of persistent user KV states across model versions under a finite background compute and I/O budget, minimizing deviation from the active model’s full-execution semantics.**

中文：EvoKV 面向模型发布期间的持久化用户状态演进：在有限后台计算与 I/O 预算下，对整个活跃 KV 状态集合完成兼容复用、局部演进或精确重算，并尽可能逼近当前模型的完整执行语义。

## 问题边界

模型从 `θ_(t-1)` 变为已决定发布的 `θ_t`。在 `T_snapshot`，系统只根据发布前信息冻结活跃状态集合：

```text
U_t = { C_u^(t-1) | C_u^(t-1) is materialized at T_snapshot }
```

其中 `C_u^(t-1) = F_(θ_(t-1))(H_u)`。Yambda 主定义不臆设 TTL：它包含发布时所有已物化、具有至少一个合法 effective prefix token 的 exact-parent 状态；recent activity / TTL 只作为敏感性 cohort。对每个状态选择动作 `a_u`，产生 `C~_u^t(a_u)`；Current Full 的 reference 为 `C_u^t = F_(θ_t)(H_u)`。

主优化形式是：

```text
minimize   mean_u D_u(C~_u^t(a_u), C_u^t)
subject to sum_u Cost_u(a_u) ≤ B_t
```

`B_t` 是发布控制面的后台资源预算，不是请求数据面的 latency SLO。对偶形式的固定 fidelity 阈值只用于 companion operating point，不声称其为行业线上标准。

状态处理的完成定义为“逻辑版本切换完成”：每个 active state 被确认兼容、快速演进、局部重算或精确重算。它不要求每个 KV tensor 都物理重写成 exact KV。

## 为什么不是请求时 controller

Yambda 可以提供时间、用户状态、发布切换和离线排名评测；它不能提供真实 P99、QPS、迁移 worker 余量、缓存淘汰策略、后台带宽或业务可接受排序变化。因此论文不写“线上 Top-10 overlap SLO”，也不使用发布后的请求数、append 数或是否活跃作为主 scheduler 输入。

请求 trace 仍有两种合法用途：

- 作为 served-user fidelity evaluation subset；
- 作为 trace-aware companion，分析逻辑切换后的真实请求与 append dilution。

但系统总体与主工作量由 `T_snapshot` 的发布前 materialized-state snapshot 定义。

## 三层架构

### 1. Release compatibility profiler

EvoKV v1 使用 1% deterministic target-free sparse probe，在发布前 canary states 上离线
运行 Current Full、Reuse 和冻结 partial paths，构造版本级与状态级风险画像。Exact 只用于
小样本 profiling ground truth；不用于拟合任意自由度 old-KV→new-KV mapper。1% 是当前
主配置，不宣称是百万/亿级人口的普适比例；规模点同步报告 fixed-count/capped-rate。

### 2. Budget-aware state scheduler

当前冻结实现使用发布时可得的 state features、`StandardScaler + Ridge(alpha=1.0)` 和
concave-hull greedy allocator，预测各动作的连续收益，并按 benefit/cost 在
5%/10%/25% exact-equivalent budget 下分配。它不使用未来 request 或 label。更复杂
predictor 不属于下一阶段。

合法特征类别包括：

- effective prefix length、cache age、pre-release recent activity、history recency；
- old KV layer-wise norm/sketch；
- cache-producing layer parameter delta 与 canary sensitivity；
- pre-release canonical probe 的 reuse-only entropy、Top-K boundary margin；
- estimated exact work 与 KV storage tier。

目标不是训练 safe/unsafe 请求分类器，也不根据 θ3 结果选择 feature 或 Ridge 参数。

### 3. State transition executor

EvoKV v1 已冻结六种 dependency-closed action：`No-op`、
`Layer0-Recent128`、`Layer0-Middle`、`Layer0-Full`、`Hybrid-Tail128` 和
`Exact-All`。Grouped executor 只优化执行效率，不改变 UID action。诊断性任意 KV splice
不属于可执行动作，也不得进入成本 frontier。

## 主指标与预算

### Primary fidelity

主 fidelity 必须匹配冻结 workload。当前 F 是显式 like/dislike 的 Bernoulli candidate-conditioned task，主 endpoint 使用 Current Full 与 action 输出的 Bernoulli JS，并同步报告 normalized score RMS、absolute probability shift 和 tails。未来多候选 workload 可使用 Current-model Top-K regret。两者都不使用 future label 做状态选择。

### Fidelity companions

- Top-10 overlap loss；
- margin-weighted pairwise disagreement；
- normalized score RMS；
- JS divergence；
- mean 与 P95/P99 tail。

Current Full 是当前模型执行语义 reference，不是 future ranking quality 的理论上界。

### Primary budget

使用相对 Exact-All 的 exact-equivalent token-layer/FLOP work：

```text
work_ratio = EvoKV recompute + migration work / Exact-All work
```

主 operating points 冻结为 `[.05, .10, .25]`；`0` 与 `1` 分别是 No-op/Exact endpoint，
完整曲线作为 companion。不得根据新时间边选择一个最有利预算。

### System companions

- old KV read bytes、new KV write bytes、history input bytes；
- compute worker-hours 与 pipeline makespan；
- worker 数量下的 rollout completion；
- state-version debt：若每次 transition work 超过周期后台容量，债务会累积。

不在第一版用任意权重混合 FLOPs 与 I/O bytes。

### Downstream validation

控制器不使用 future label 决策；在独立质量 manifest 上报告 Candidate CE、AUC、Conditional NDCG@10、HR@10 与 MRR，相对 Current Full 的差异。

## Cutover 与 dilution

zero-append cutover 是最保守的 fidelity endpoint：旧 prefix 占比最高，当前模型尚未 append 行为。它通过 target-independent candidates 和 canonical readout token 离线评测。

随后报告 append count `0,1,2,4,8,16` 的 dilution curve。Yambda 没有 serving request log，因此验证 endpoint 是首个发布后 observed listening event 的 trace proxy；需要报告 cutover-probe risk 对该 endpoint divergence 的 Spearman correlation / high-risk recall，且不可将 canonical probe 说成未来请求的替代品。

## 研究问题

1. **RQ1 — Characterization**：哪些 workload、release 与状态兼容，哪些形成 H/S？
2. **RQ2 — Executor**：dependency-closed Partial 能否以低于 Exact 的成本恢复 fidelity/quality？
3. **RQ3 — Scheduler**：发布前 sparse probe + state features 能否在同成本下优于固定、元数据与随机基线？
4. **RQ4 — Rollout**：连续 No-op 后的 version debt 是否保留，frozen policy 是否仍有效？
5. **RQ5 — Qualification**：上述链条能否在 8L scale point 和未见 θ3 时间边复现？

## 实验纪律

- P7–P11 是 development substrate；当前六动作、profiler、Ridge、allocator、executor、
  budgets 和统计规则必须先封存为 EvoKV v1。8L 只做规模复现；θ3 才承担 frozen-policy
  blind temporal qualification。
- 模型发布质量与状态兼容性分开报告；EvoKV 不负责 model admission。
- quality manifest 可以 target-inject，profiler manifest 不可 target-inject。
- active snapshot 必须只依赖 pre-release materialization；主定义不设 TTL，TTL/activity 仅作 sensitivity cohort；不可按未来 served users 定义。
- `0.2` / `0.5` overlap loss 只是诊断切片，不是 serving SLO。
- 失效的 batch-alignment 或 release-lineage artifact 仅保留审计价值。

## 贡献边界

EvoKV 的差异化不是单一 KV mapper，而是：连续模型 lineage、长期用户持久状态、旧 prefix 与新模型 append 的混合语义、cutover 风险与 append dilution、面向完整 active population 的预算调度，以及多版本状态债务控制。

## 规模化资格边界

Yambda-50M 是机制开发平台：用于稳定协议、合法 lineage、风险定义、预算 scheduler 和 migration executor；它不是最终“大规模生成式推荐系统”结论的充分依据。当前模型规模和有效历史长度不能替代真实的大模型、大数据和 persistent-state 总量。

规模化资格分层如下：

- `8L/H256/context1024`：当前首个同源更大模型复现点，只复现
  `H→S→legal partial→same-cost scheduler`；
- θ3 blind edge：完全未查看时间边上的 frozen-policy temporal qualification；
- 同源 Yambda-5B：上述通过后的大数据/状态规模扩展；
- VK-LSVD：在百万到千万 materialized states 上验证 KV footprint、迁移工作量、I/O、调度吞吐和 state-version debt；
- RecFlow：补充真实多阶段 candidate workload，检验 candidate set 对 compatibility risk 和 scheduler 的影响。

可跨规模复用的是 snapshot、lineage、future-information exclusion、exact-equivalent work、动作接口和 accounting；用户级风险模式、节省比例、candidate-set 效应和 release regression 必须在新规模上重新验证。完整的分阶段准入条件和禁止外推的表述见[规模化扩展路线](scaling_extension.md)。

## 论文图表骨架

在 qualification 完成前只维护紧凑 paper jig，固定准备四类核心图表：系统图、
N/R/F × R0/R1/R2 phase diagram、同成本成本—fidelity—quality frontier，以及并列展示
development、8L、recursive lineage 和 θ3 blind 的 qualification 表。
