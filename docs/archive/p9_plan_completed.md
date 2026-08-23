# Archived: P9 Staleness Tomography and Action-Space Qualification

P9 的目的不是继续证明 `S` 或把 gap 调大，而是将局部陈旧结构转化为依赖合法、质量保持且成本低于 Exact-All 的真实动作。

## 已完成

- **P9.0**：P8 evidence seal。
- **P9.1**：24-cell 用户级 H/S 分布、尾部和 cohort 描述；`v1` 的零值 tail 定义已判无效，后续只用 `v2`。
- **P9.2**：24/24 layer-only 与 segment-only diagnostic scan。R0 recovery 为零；R1/R2 结构跨 seed 可重复。详见 [P9.2 结果摘要](p9_2_result_summary.md)。

P9.2 的 exact-KV splice 全部是 `diagnostic intervention, not executable migration action`。

## 当前收口任务

### P9.2-Q：quality companions（complete）

在不重跑模型、不改变区域的前提下，将已有 diagnostic logits 与封存 F quality labels 按请求连接，统一计算：

- aggregate log loss；
- ROC-AUC；
- dislike PR-AUC；
- Brier；
- dislike-only log loss。

这一步验证“恢复 JS”是否也恢复任务质量；不得用 label 选择 state/action。

结果显示 R2 的局部 fidelity recovery 同步恢复 aggregate quality；R1 更混合。完整结果见 [P9.2 Closure Result](p9_2_closure_result_summary.md)。

### P9.1-C：risk concentration（complete）

Top 1%/5%/10% 用户分别贡献约 7.2%–10.0% / 24.2%–31.2% / 38.6%–49.1% 总 `S`。风险异质性成立，但 scheduler 仍须等待合法 action frontier。

## 后续顺序

### P9.3：代表语义的 2-D tomography（complete）

预先固定四类语义格子：

1. R0 negative control；
2. M0-F R1 edge1 routine update；
3. M0-F R2 encoder refresh；
4. M1-F R2 shared-state refresh。

三个 seed 全部保留。M0-F R2 的 layer-0 大区段结构稳定；M1 的 aggregate 结果受 seed 异质性影响。详见 [P9.3 Result](p9_3_result_summary.md)。任何发现仍须通过合法 executor 回放到完整 P8 matrix。

### P9.4：dependency-closed executor（complete）

对候选 tail/layer/segment 动作明确列出：

- 输入 raw history 与边界 hidden state；
- 必须重算的上游/下游层；
- old KV reads、new KV writes；
- 是否可由线上真实持久化状态执行。

缺少 hidden boundary 时，不能用直接拼接 exact KV 代替 executor。优先实现最小的 No-op、合法 tail/interval partial 和 Exact。

### P9.5–P9.8：rolling executor 与全人群 cutover profiler（complete）

真实 cutover state、逐事件 append/evict 和 uid-keyed executor 已通过。P9.8 对 24 cells 的全部 8,229/8,488 个状态完成 label-free action profiling：HybridTail128 跨非 R0 cells 全 seed 正恢复；release-specific Layer0-Full 在若干格子以 25% token-layer work 恢复约 98%–99.6%。完整结果见 [P9.7–P9.8 Full-Population Result](p9_8_full_population_result.md)。

### P9.9：held-out rolling quality（complete）

24-cell full evaluation 已完成。M1-R2 HybridTail128 在三 seed 上均降低 No-op log loss，并保持稳定 fidelity recovery；R1 quality 较小且混合。详见 [P9.9 Held-out Rolling Quality](p9_9_heldout_rolling_quality_result.md)。

### P9.10–P9.11：runtime 与正式 development frontier（complete）

全人群 migration runtime、逻辑 I/O 和 uniform/oracle state×action frontier 已完成。5% token-layer budget 下，near-optimal oracle 在非 R0 cells 恢复约 38%–94%，random Exact 约 5%，说明 state-level opportunity 明确。详见 [P9.10–P9.11 Runtime and Frontier](p9_10_11_frontier_result.md)。

### 下一阶段：cheap profiler qualification（authorized）

只允许使用发布时可得特征和明确计费的 sampled probes，预测 action benefit。先封存 policy assignment，再连接 held-out quality labels。未经该门，不训练或宣称 deployable scheduler。

记录每个合法 action 的：

- exact-equivalent token-layer work；
- old-KV read、new-KV write、raw-history read bytes；
- batched runtime 与 rollout completion estimate；
- target-free fidelity 与全部 F quality companions。

下一步补齐完整 held-out rolling quality 与 full-population batched runtime，然后比较 Reuse All、Exact All、uniform legal partial、version-level best action、random state allocation与 state×action near-optimal allocation。先做 oracle/near-optimal frontier，不训练 controller。

## 停止门

- Partial 明显推开 frontier，且 state-level allocation 优于 version-level policy：人工授权 P10 scheduler。
- Partial 有效但风险近似均匀：只做 release-level action selector。
- 只有 Exact 有效：保留 No-op/Exact，收缩 selective migration 叙事。
- 未经人工授权，不训练 controller、theta3、blind edge，也不扩大模型规模。
