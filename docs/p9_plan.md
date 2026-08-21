# P9: Staleness Tomography and Action-Space Qualification

P9 的目的不是继续证明 `S` 或把 gap 调大，而是将局部陈旧结构转化为依赖合法、质量保持且成本低于 Exact-All 的真实动作。

## 已完成

- **P9.0**：P8 evidence seal。
- **P9.1**：24-cell 用户级 H/S 分布、尾部和 cohort 描述；`v1` 的零值 tail 定义已判无效，后续只用 `v2`。
- **P9.2**：24/24 layer-only 与 segment-only diagnostic scan。R0 recovery 为零；R1/R2 结构跨 seed 可重复。详见 [P9.2 结果摘要](p9_2_result_summary.md)。

P9.2 的 exact-KV splice 全部是 `diagnostic intervention, not executable migration action`。

## 当前收口任务

### P9.2-Q：quality companions

在不重跑模型、不改变区域的前提下，将已有 diagnostic logits 与封存 F quality labels 按请求连接，统一计算：

- aggregate log loss；
- ROC-AUC；
- dislike PR-AUC；
- Brier；
- dislike-only log loss。

这一步验证“恢复 JS”是否也恢复任务质量；不得用 label 选择 state/action。

### P9.1-C：risk concentration

对每个 model × release × seed 计算 Top 1%/5%/10% 用户贡献的总 `S`、Lorenz/集中度 companion，以及 risk 与 candidate action recovery 的联合分布。只有风险与收益均有异质性，用户级 scheduler 才可能有价值。

## 后续顺序

### P9.3：代表语义的 2-D tomography

预先固定四类语义格子：

1. R0 negative control；
2. M0-F R1 edge1 routine update；
3. M0-F R2 encoder refresh；
4. M1-F R2 shared-state refresh。

三个 seed 全部保留，扫描 layer × frozen history segment。选择依据是发布语义和模型角色，不是 P9.2 最大数字。任何发现的动作必须回放到完整 P8 matrix。

### P9.4：dependency-closed executor

对候选 tail/layer/segment 动作明确列出：

- 输入 raw history 与边界 hidden state；
- 必须重算的上游/下游层；
- old KV reads、new KV writes；
- 是否可由线上真实持久化状态执行。

缺少 hidden boundary 时，不能用直接拼接 exact KV 代替 executor。优先实现最小的 No-op、合法 tail/interval partial 和 Exact。

### P9.5：fidelity–cost frontier

记录每个合法 action 的：

- exact-equivalent token-layer work；
- old-KV read、new-KV write、raw-history read bytes；
- batched runtime 与 rollout completion estimate；
- target-free fidelity 与全部 F quality companions。

比较 Reuse All、Exact All、uniform legal partial、version-level best action、random state allocation 与 state×action near-optimal allocation。先做 oracle/near-optimal frontier，不训练 controller。

## 停止门

- Partial 明显推开 frontier，且 state-level allocation 优于 version-level policy：人工授权 P10 scheduler。
- Partial 有效但风险近似均匀：只做 release-level action selector。
- 只有 Exact 有效：保留 No-op/Exact，收缩 selective migration 叙事。
- 未经人工授权，不训练 controller、theta3、blind edge，也不扩大模型规模。
