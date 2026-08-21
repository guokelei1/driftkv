# EvoKV 论文设计：发布期预算化状态收敛

更新日期：2026-08-18。本文是当前论文边界的正式定义。

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

在少量发布前 canary states 上离线运行 Current Full、Reuse 和预定义 partial paths，构造版本级与状态级风险画像。Exact 只用于小样本 profiling ground truth；不用于拟合任意自由度 old-KV→new-KV mapper。

### 2. Budget-aware state scheduler

预测连续风险 `D^_u`，并按“预期 semantic benefit / exact-equivalent cost”排序，在任意预算点选择状态动作。主特征必须在发布时得到，例如：

- effective prefix length、cache age、pre-release recent activity、history recency；
- old KV layer-wise norm/sketch；
- cache-producing layer parameter delta 与 canary sensitivity；
- pre-release canonical probe 的 reuse-only entropy、Top-K boundary margin；
- estimated exact work 与 KV storage tier。

第一版可采用规则、线性 ranker 或浅层树；目标不是训练 safe/unsafe 请求分类器。

### 3. State transition executor

第一阶段只实现 `No-op` 与 `Exact`。随后引入 `Fast Migration`、`Selective Recompute`，按预计 fidelity gain / incremental work 进行多动作预算分配。

## 主指标与预算

### Primary fidelity

Current-model Top-K regret：用 Current Full 分数衡量迁移路径 Top-K 相对 Current Full Top-K 的模型效用损失。它不使用 future label，并对近似同分项交换较不敏感。

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

报告全区间 `[0, .1, .25, .5, .75, 1]`，而非选择一个人为“正确预算”。

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

1. **RQ1 — Characterization**：哪些预定义 release 类型与状态群体兼容，哪些不兼容？是否存在版本、用户、层与 cutover/dilution 异质性？
2. **RQ2 — Oracle opportunity**：同一 exact-equivalent budget 下，state-level oracle 是否优于 Reuse All、Exact All、version-level gate、随机、长度/活跃度优先？
3. **RQ3 — Scheduler**：只用发布前 feature 的连续 risk ranker 能否逼近 oracle frontier，并跨版本泛化？
4. **RQ4 — Executor**：Fast Migration / Selective Recompute 能否把 no-op/exact frontier 向更低工作量推进？
5. **RQ5 — Rollout**：在连续发布与有限 worker/IO 容量下，能否避免 state-version debt 累积？

## 实验纪律

- `θ0→θ1` 与 `θ1→θ2` 仅为 controller development；冻结后 `θ2→θ3` 是 blind qualification。
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

- 同源 Yambda-5B：在方法冻结后重新训练更大的 foundation 和版本链，承担“大模型、大数据”的主 qualification；
- VK-LSVD：在百万到千万 materialized states 上验证 KV footprint、迁移工作量、I/O、调度吞吐和 state-version debt；
- RecFlow：补充真实多阶段 candidate workload，检验 candidate set 对 compatibility risk 和 scheduler 的影响。

可跨规模复用的是 snapshot、lineage、future-information exclusion、exact-equivalent work、动作接口和 accounting；用户级风险模式、节省比例、candidate-set 效应和 release regression 必须在新规模上重新验证。完整的分阶段准入条件和禁止外推的表述见[规模化扩展路线](scaling_extension.md)。
