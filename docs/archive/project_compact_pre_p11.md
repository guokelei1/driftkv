# Archived: EvoKV 项目全程 Compact（P9 时点）

更新日期：2026-08-21。

## 一句话现状

项目已经从“长期状态与跨版本陈旧性是否存在”推进到“如何把可定位的陈旧状态转化为低成本、依赖合法的迁移动作”。

当前在 Yambda-50M Explicit Feedback workload、两类模型、三个独立 seed 和三种发布语义上建立了 development 证据链：

```text
长期状态 H 存在
→ output-only release 的 S 位于数值地板
→ cache producer 更新后 S 稳定非零
→ 强发布条件下 Reuse 损害任务质量
→ stale error 具有 layer / history-position 局部结构
```

这授权系统方法开发，但尚未构成 paper-qualified evidence。

## 当前论文问题

EvoKV 接收一个已通过发布质量门的新模型，在发布期后台预算内，为全部已物化状态选择动作，使其逼近 Current Full：

\[
\min_{\{a_u\}} \sum_{u\in\mathcal S_t} L_u(a_u)
\quad\text{s.t.}\quad
\sum_{u\in\mathcal S_t} C_u(a_u)\le B_t.
\]

- `L_u(a_u)`：相对 Current Full 的执行 fidelity loss；
- `C_u(a_u)`：token-layer work、KV I/O、history read 与运行时间；
- 动作空间最终可能包含 No-op、Dependency-closed Partial、Exact，之后才考虑 Fast Migration。

Current Full 是当前模型的执行语义参考，不是未来 ranking 的理论上界。模型 admission 与 cache compatibility 是两道独立的门；No-op 是一等结果。

## 已建立的核心证据

| 阶段 | 结果 | 证据边界 |
| --- | --- | --- |
| P7 / `H` | F 通过；M0-F 为 2/3 seed、聚合正；M1-F 为 3/3 seed | development long-state object |
| P7 controls | N 未通过，符合短期控制；R 被 Frozen Base 主导，未通过 | workload phase boundary |
| P8 / R0 | 最大 JS `4.44e-15`，低于 `1e-8` 地板 | output-only release 可 No-op |
| P8 / R1-R2 | cache-producing path 更新后 `S` 稳定非零 | cross-version staleness established in development |
| P8 / M1-R2 | Full 相对 Reuse：log loss `+0.003274`、ROC-AUC `+0.01331`、dislike PR-AUC `+0.01773`、Brier `+0.000999` | stale KV 可损害实际任务质量 |
| P9.2 | 最佳诊断区域恢复约 `78%–99%`；recent-128 在全部非 R0 条件和 seed 上正恢复 | local structure exists; splice is not executable |

P9.2 还显示 recent-128 恢复约 `45%–97%`，recent-32 约 `13%–42%`，recent-1 仅约 `1%–2%`。部分单层 splice 会使 fidelity 变差，说明跨层依赖闭包是下一阶段的生死门。

用户风险呈长尾；部分 cell 的 P99/P50 达 `62×–162×`。但 Top 1%/5%/10% 风险贡献尚未正式计算，因此还不能断言用户级 scheduler 必要。

## 路线如何演变

1. QK/QB 与 KuaiRand 建立了容量和机制动机，也暴露出 raw score、candidate protocol、lineage 与模型发布质量不能混为一谈。
2. Yambda 平台修复了 timestamp 单位、batch alignment、release cutoff 和 suffix time delta 四类关键错误；修复前数字永久失效。
3. Neutral readout 被证明会在 suffix 出现后绕过长期 prefix，不是旧 KV 被修复。
4. Sampled next-listen 被 repeat、count、recency 和 proposal rank 主导；P5/P6 将该协议判为 No-Go，并永久停止继续造 negatives。
5. N/R/F workload suite 与 Frozen Base + CC residual 将简单统计显式交给 Base，只让 CC 状态证明额外增量。
6. F 在 P7 通过 `H`；P8 用 R0/R1/R2 建立兼容、中等陈旧、强陈旧的发布语义相图；P9 开始定位可恢复结构。

这些失败与负控制是当前问题定义的一部分，不得删除、翻案或重新包装。

## 当前系统结构

1. **Release/Staleness Profiler**：识别 cache producer 是否变化，以及 release、state、layer 和 position 风险。
2. **State Transition Executor**：实现真实可执行的 No-op、dependency-closed Partial 和 Exact。
3. **Budget Scheduler**：只有在 frontier 证明 state-level allocation 优于 version-level policy 后才授权。

主要优化量是 Current-Full fidelity 与迁移成本，而不是 future click 或虚构的线上 P99 SLO。

## 当前未完成的关键问题

- diagnostic recovery 是否同步恢复 log loss、AUC、PR-AUC、Brier 和 rare-dislike companion；
- 风险是否集中到足以支持用户级调度；
- 哪些 layer/segment 操作满足真实 hidden/KV 依赖闭包；
- legal partial 是否在 token-layer work、I/O 和实测 runtime 上优于 Exact-All；
- version debt、8L/H256/context1024、blind temporal edge 与外部 workload 是否复现。

## 当前唯一执行路线

1. 用已有 diagnostic logits 与封存 F labels 补齐 quality companions，不重新运行或选择区域。
2. 计算 Top 1%/5%/10% 用户的总 `S` 贡献和风险—收益异质性。
3. 对预先按语义选定的 R0、M0-R1 edge1、M0-R2、M1-R2 全 seed 做 layer × position map。
4. 审计依赖闭包，区分 diagnostic intervention 与 executable action。
5. 实现最小合法 partial executor，并测 token-layer work、I/O、history read 和 batched runtime。
6. 绘制 No-op / Partial / Exact frontier，比较 uniform、version-level 与 state-level near-optimal allocation。
7. 只有 frontier 明确推开且异质性足够，才授权 scheduler。
8. 方法冻结后再做 version debt、更大模型和 blind qualification。

当前禁止调 P8 基础链来放大 `S`、筛 seed/edge、训练 controller、训练 theta3、将 arbitrary exact-KV splice 放入成本 frontier，或把 development 结果写成论文最终结论。
