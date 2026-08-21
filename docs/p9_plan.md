# P9: Staleness Tomography and Action-Space Qualification

P8 已证明 F workload 的 H 与跨版本 S 在 development 中存在。P9 不再调整基础模型链，也不训练
controller；它只回答陈旧性位于哪里、哪些合法动作可恢复，以及 state-level action 是否值得做。

## 冻结边界

- 固定 P8 的模型、seed、workload、release recipe、评分和结果；禁止为放大 S 调基础链。
- M0-F 是主隔离模型；M1-F 是共享状态 companion；M1-N/R 只作共享状态对照。
- `feedback_history_strata_v2` 只前向用于 P9，绝不回写 P7.8。
- rare-dislike companion 在每个 P9 action 上完整报告；不用 future label 选择状态或训练决策。

## 执行顺序

1. **P9.0 — P8 evidence seal**：记录 P8 全矩阵 hash、model/seed/release role 与结果摘要。
2. **P9.1 — H/S distribution**：全 P8 cells 做用户级 mean/P50/P90/P95/P99、H-S joint、
   prefix length/activity/state age/feedback cohort 分桶，区分普遍小风险与少数高风险状态。
3. **P9.2 — coarse tomography**：全部 P8 cell、全部 seed 做 layer-only 与 frozen history-segment-only
   recovery scan（oldest half、middle、recent-128/32/8/1）。所有任意 KV exact 拼接均标为
   *diagnostic, not executable action*。
4. **P9.3 — 2-D tomography**：对 R0、M0-F R1、M0-F R2、M1-F R2 四个按 release semantics
   预选的代表 cell 做 layer × segment map；发现必须回放到全 P8 cells/seed。
5. **P9.4 — executable actions**：只实现 dependency-closed 的 No-op、合法 tail/layer/segment
   recompute 与 Exact；不能从仅保存的 K/V 假装执行缺少 hidden boundary 的 patch。
6. **P9.5 — action frontier**：在同一 exact-equivalent token-layer work 下比较 No-op、Exact、
   uniform legal partial、random allocation、version-level best action 与 near-optimal state×action
   allocation，并记录 bytes、history read、measured time 和 F quality companions。

## P9 停止门

- 若 partial action 推开 fidelity–work frontier，且风险/恢复存在异质性：人工授权 P10
  budget-aware state scheduler。
- 若只有 Exact 有效：保留 No-op/Exact 的 release-level selector，收缩 selective migration 叙事。
- 若风险近似均匀：只做 release-level action，不训练 state-level controller。
- 未经新的人工授权，不训练 controller、θ3 或 blind edge。
