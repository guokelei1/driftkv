# CohortKV Design 3 未来方向：Organic mixed-version bounded renewal

日期：2026-07-28

状态：**未冻结的 future direction**。它不是当前实施计划，不包含新实验结果，也不能作为论文
claim 的证据来源。

当前优先级：先完成 `DESIGN2_FINAL_PLAN.md`。D3 至少等 D2 的 action/report 接口和 integrated
cost ledger 稳定后再启动。

---

## 0. 当前判断

D3 最值得保留的问题不是“再做一个 scheduler”，而是：

> 当 cache fleet 同时包含多个 source version、不同 migration depth、不同长度和不同 deadline
> 时，如何用一个紧凑的 version-program graph 在有限 GPU、embedding collective 和 HBM
> transaction 预算下持续 catch up，同时保证 exact renewal 可达、债务有界，并且不发布
> mixed/伪造版本状态？

它包含两个尚未被当前系统解决、但彼此耦合的问题：

1. **Multi-version execution**：不能为所有 \((source,target)\) 版本对长期保存 \(O(V^2)\)
   programs，也不应为每个中间版本反复 materialize K/V。
2. **Bounded renewal**：exact 是稀缺且会形成 refresh wave 的资源；controller 必须在多资源预算下
   保证 hard depth/deadline，而不是预测哪条 cache “安全”。

当前不把 D3 写成定稿，是因为三个前提还没有证据：

- adjacent program composition 的 serialized sequential equivalence；
- organic mixed-version workload，而不只是固定或受控 source-version mix；
- D2 给出的真实 multi-GPU cost、collective、capacity 和 failure vector。

---

## 1. 与 D2 的单向接口

```text
D3 controller
  version graph + renewal state + resource budgets
                         │
                         ▼
                    D2ActionPlan
                         │
                         ▼
D2 runtime
  physical-wave compilation + segmented compiled/exact/append + transaction
                         │
                         ▼
                    D2WaveReport
                         │
                         └── measured costs feed the next D3 decision
```

D3 负责：

- source-to-target program path；
- requested action；
- exact/replay budget；
- migration depth/deadline；
- cross-wave backlog 和 program-cache policy。

exact fraction、requested `compiled|exact` 和 communication-aware semantic admission 都是
D3/policy 变量，不是 D2 runtime knobs。D2 可以把 measured communication/movement cost 写入
`D2WaveReport` 供下一 wave 使用，但当前 wave 的 requested actions 已冻结。

D2 继续负责：

- within-wave placement；
- owner-compute、sharded exact 和 append；
- preflight/fallback；
- COW、commit、abort 和 readback；
- 实际 cost/communication/capacity/failure report。

D3 只能生成冻结的 `D2ActionPlan`，不能绕过 D2 transaction，也不能在 runtime 内临时重选
record action。

---

## 2. 必须继承的研究不变量

以下不是未来可重新讨论的候选，而是当前 roadmap/evaluation protocol 已经确定的边界：

1. **Task quality 不是 admission oracle。** recommendation labels、per-user drift、JVP、Fisher
   或预测的 ranking gain 都不能路由 cache。
2. **Version cohort 是 compilation、batching、placement 和 scheduling key，不是“该版本可安全
   reuse”的预测器。**
3. **Unconditional repair。** 对仍在 cache fleet、进入 target epoch 的每个 stale cohort，
   至少执行声明的 compiled repair；随后才按预算 progressive replay 或 exact。
4. **Exact renewal 不可被 composition 取消。** exact 重置 last-exact version 和 migration
   depth；compiled/composed repair 不重置。
5. **Depth 按跨越的 model edges 计数。** 五条边即使被压成一个 kernel，semantic debt 仍增加五。
6. **Append 必须使用 target model。** delta/latest tokens 不得错误地套用旧版本 program。
7. **只有逻辑完整的 post-append cache 可以发布。** 它可以是 coverage 完整、顺序明确的
   `{retained,suffix}` segmented manifest，不强制物理 contiguous；若 consumer 必须拼接，
   拼接成本属于 D2 boundary。
8. **不得发布 stale-as-current。** capacity 不足必须在 transaction 前显式 reject、evict 或走
   声明的 cold/exact 路径，不能静默 defer 后伪装成 current version。
9. **现有 per-cache threshold route 是 refresh-wave negative result。** 不恢复 adaptive-risk 或
   global-optimality claim。

“访问概率/hotness”最多用于 storage admission、placement 或后台工作排序；它不能成为
quality/safety signal。没有真实 request trace 时，eviction/tiering 不进入 D3 主设计。

---

## 3. 候选机制 A：Certified adjacent-program graph

### 3.1 保守主路径

只持久保存经过当前 provenance/certificate 检查的相邻程序：

\[
T_{v\rightarrow v+1},T_{v+1\rightarrow v+2},\ldots
\]

任意 source cohort catch up 时，首先必须支持按 lineage 顺序执行 adjacent programs。这个
sequential graph 是 composition 失败时仍成立的保守路径。

每条 graph edge 至少记录：

- source/target version；
- program、parent/checkpoint 和 certificate hashes；
- layout、dtype、operator ABI；
- calibration provenance；
- compatible retained-prefix/crop/append contract。

### 3.2 可证伪候选：on-demand affine composition

若相邻程序保持兼容的 affine 形式，可以候选地组合：

\[
M_{a\rightarrow c}=M_{a\rightarrow b}M_{b\rightarrow c},
\]

\[
c_{a\rightarrow c}
=c_{a\rightarrow b}M_{b\rightarrow c}+c_{b\rightarrow c}.
\]

目标是：

- 减少中间 K/V materialization；
- 减少 kernel launch 和中间 transaction；
- 只缓存活跃的 \((source,target,layout)\) 组合，避免物化 \(O(V^2)\) programs。

composed artifact 还必须记录 parent IDs/hashes、composition dtype/time、serialized bytes 和
sequential-vs-composed error。

### 3.3 Composition 不能推出什么

即使理想 affine 代数成立，也不能推出：

1. composed output 等于 exact current-model K/V；
2. source-target 距离增加时近似误差不增长；
3. serialized FP16/BF16 与 FP32 composition 完全一致；
4. crop、mask、target append、calibration 和 program ABI 不影响等价边界；
5. composed execution 可以绕过 exact-renewal deadline。

因此 composition 是一个独立 gate，不是 D3 的前置真理。

---

## 4. 候选机制 B：Reserved-exact multi-resource renewal

### 4.1 Action ladder

对每个 stale cohort，候选 ladder 保持当前方法方向：

```text
unconditional compiled repair
  → budgeted progressive residual replay
  → mandatory exact recomputation
```

controller 只决定 replay/exact 的预算和顺序；它不根据预测质量决定是否执行基础 repair。

建议状态：

```text
source_version
target_version
last_exact_version
migration_depth
deadline_slack
record/token/KV bytes
layout and owner summary
candidate program path
measured compiled/replay/exact cost vector
embedding collective demand
COW capacity demand
failure/fallback state
```

### 4.2 资源模型

每个 epoch 至少显式约束：

- compiled/replay/exact GPU time；
- requested/unique/remote embedding vectors；
- HBM old+new+transient capacity；
- COW/commit/reclaim margin；
- reserved exact capacity；
- per-rank makespan 和 imbalance。

controller 先满足：

1. artifact、lineage 和 transaction capacity；
2. mandatory exact depth/deadline；
3. every-admitted-cohort repair；
4. no mixed visible epoch。

再优化：

- exact refresh-wave amplitude；
- exposed embedding collective；
- wave makespan 和 tail；
- steady-state backlog；
- program-cache hit 和 serialized storage。

exact budget 必须是保留资源，不能被 compiled work 挤占。若 capacity 无法满足 hard constraints，
必须在 D2 transaction 开始前返回明确 infeasible/reject 结果。

### 4.3 当前 baseline

当前 frozen bounded-renewal policies 是 D3 的 baseline，不是新的贡献：

- Stage 4.6 balanced age/deadline + program-level edge severity；
- Stage 4.9 staggered renewal H12；
- fixed periodic exact；
- token debt cost endpoint；
- matched random refresh；
- per-cache threshold refresh-wave negative baseline；
- all exact 和 all compiled endpoints。

Stage 4.6 的 depth-four fixed-history结果与 Stage 4.9 的 H12 growing-history结果属于不同 protocol，
不得合并。Stage 4.9 只有 11 updates，尚未覆盖完整 H12 renewal cycle。

---

## 5. 尚未解决的关键分歧

### 5.1 Composition 是否值得成为核心

支持理由：D1 的 deployed operator 是 affine，组合可能直接减少中间 state movement 和 program
library growth。

反对理由：实际 program 带 serialization、crop/append、lineage 和 Stage 4.10 calibration
provenance；代数形式相同不等于完整 runtime contract 可组合。

当前裁决：先保留为最便宜的 falsification gate。失败后 D3 退回 certified sequential graph，
不影响 bounded renewal 问题本身。

### 5.2 “Organic” workload 从哪里来

现有数据能够构造 canonical-date growing histories，也有受控 mixed-version assignments，但没有
被验证的生产 request/arrival trace。

当前裁决：

- controlled version/arrival generator 可以做 stress characterization；
- frozen chain 可以做可比 control；
- 没有真实 trace 时不得声称生产 hotness、eviction 命中率或真实 steady-state arrival law。

### 5.3 Admission、eviction 与 defer

旧讨论曾提出 `defer and serve stale`。这与当前 unconditional-repair/current-version publication
边界冲突，不能作为主 action。

允许的容量行为只有显式状态转换：

- 保留在 admitted fleet 并执行 repair；
- transaction 前拒绝/驱逐，后续访问 cold/exact；
- 进入有真实 source-version 标签、不能冒充 current 的独立 serving mode。

第三种若要进入论文，需要单独 serving semantics 和质量实验，当前不纳入 D3。

---

## 6. 恢复 D3 时的实施顺序

### D3-P0：前置条件

- D2 至少完成 P0–P4；
- `D2ActionPlan`/`D2WaveReport` schema 冻结；
- 1/2/4 GPU compiled/exact/append cost vector 可用；
- D2 capacity/failure ledger 可被离线 controller 消费。

### D3-P1：Composition falsification

使用现有相邻 programs，不重训：

1. FP32 sequential vs FP32 composed；
2. serialized runtime dtype sequential vs composed；
3. fixed retained、crop 后 retained、不同 token birth versions；
4. target-model delta/latest append；
5. K/V、hidden、score、Top-100 和 task diagnostics；
6. composition time、program bytes、中间 K/V movement。

通过才实现 on-demand composed-program cache；失败则冻结 sequential graph。

### D3-P2：Organic mixed-version workload contract

- 冻结 source-version arrival、update interval、burst 和 churn 参数；
- 分开 controlled stress 与真实/回放 trace；
- 至少覆盖一个完整 renewal horizon；steady-state claim 最好覆盖两个；
- 记录每条 cache 的 birth/source/last-exact/depth/deadline lineage。

### D3-P3：Offline controller

先只消费 frozen D2 cost reports：

- hard constraints first；
- reserved exact budget；
- multi-resource feasibility；
- action-plan exporter；
- deterministic replay 和 hash。

不要一开始把 scheduler 嵌进 distributed runtime。

### D3-P4：D2 integration

- D3 生成 immutable `D2ActionPlan`；
- D2 执行并返回 `D2WaveReport`；
- 下一 wave 只从 committed report 更新 state；
- abort/reject 不推进 target version 或 migration depth；
- 完整运行至少一个 renewal horizon。

### D3-P5：Replication

只有形成新的 Pareto/system point 后才扩展：

- additional seeds；
- 必要的第二数据集；
- 2/4 GPU；
- update interval、burst、capacity 和 exact-budget sweeps。

---

## 7. 实验闭环

### 7.1 Workloads

- frozen single-source chain control；
- controlled organic mixed source versions；
- growing histories 和不同 token birth versions；
- update burst、active-cache churn 和 capacity pressure；
- 若未来获得真实 request trace，再加入 trace replay。

### 7.2 Baselines

- all exact；
- all compiled without reserved exact；
- sequential adjacent graph；
- current frozen balanced/H12 policies；
- fixed periodic exact；
- token debt；
- matched random exact；
- no-reserved-exact ablation；
- on-demand composition，仅在 composition gate 通过后。

### 7.3 Metrics

- cumulative GPU time 和 integrated wall time；
- D2 phase-tagged embedding collective/P2P/HBM ledger；
- maximum/mean migration depth；
- deadline violations；
- exact fraction 和 per-wave amplitude；
- backlog size/age 和 infeasible/reject count；
- minimum/q10/q50/q90 K/V fidelity、score、Top-100 和 task delta；
- program composition/cache time、hit rate 和 serialized bytes；
- commit/abort/fallback 和 visible-version correctness。

统计 replication unit 仍是 training seed；同一 model 下的 record/wave 只能作为 systems
diagnostics。

---

## 8. Go/no-go gates

### G0：Protocol boundary

- D3 只通过 `D2ActionPlan`/`D2WaveReport` 与 D2 交互；
- action selection 与 runtime timing 不互相污染；
- 新实验使用独立 protocol/result family。

### G1：Composition

- serialized composed 与 sequential program 在部署 tolerance 内一致；
- parent hashes、edge count、dtype/layout lineage 完整；
- composition 实际减少 movement、launch、transaction 或 program-storage cost。

若前两项失败，删除 composition claim，使用 sequential graph。若只有第三项失败，composition
保留为正确但无用的 negative result。

### G2：Bounded progress

- every admitted stale cohort 获得 repair；
- hard exact depth/deadline 零违反；
- abort/reject 不错误推进 lineage；
- 不出现 stale-as-current 或 mixed visible epoch；
- 至少覆盖一个完整 renewal horizon。

### G3：System benefit

相对当前 frozen balanced/H12 baseline，至少改善一项：

- refresh-wave peak；
- exposed collective/full-wave time；
- steady-state backlog；
- intermediate K/V movement；
- program storage/build cost。

同时不能破坏 D2 correctness、capacity、tail 和 task/fidelity gates。

### G4：Paper-strength

D3 只有在 organic mixed versions、完整 renewal horizon、D2 integrated movement 和 bounded
progress 同时成立时，才可作为独立设计进入论文。

否则按证据收缩：

| 结果 | 处置 |
|---|---|
| composition 失败，renewal 通过 | `bounded renewal over a certified sequential program graph` |
| composition 通过，scheduler 无系统收益 | composition 作为 extension/negative，D3 不成 design |
| 只有 controlled generator | 只称 mixed-version stress，不称 production-organic |
| bounded progress 或 integrated movement 失败 | D3 停止，论文保留 D1+D2 |

---

## 9. 明确不做

- per-user drift/JVP/Fisher/risk predictor；
- recommendation-label routing；
- task-quality admission oracle；
- global-optimal scheduler claim；
- `defer and serve stale as current`；
- 无真实 trace 的 anti-LRU/LFU superiority claim；
- SSD/object-store tiering；
- training DDP、gradient sync 或 checkpoint distribution；
- 为了 D3 重新打开 arbitrary interval/layer search；
- 把 COW/manifest 重新包装成 D3 novelty。

---

## 10. 以后恢复时的第一步

不要先写 scheduler。顺序是：

```text
freeze D2 measured cost/report interface
  → run composition falsification
  → freeze mixed-version workload contract
  → build offline bounded controller
  → integrate through D2ActionPlan
```

在这些前提完成前，D3 保持为 future direction，不与当前 D2 实现并行扩张。
