# 03 根因探索：跨数据结论与路线决定

状态：development decision，非 scientific/formal result。

| Gate | QK | KuaiRand | 跨数据判断 |
|---|---|---|---|
| L0 | pass | pass | pass |
| L1 | pass_development | pass_development | pass_development |
| L2 | strong_pass | short_horizon_only | mixed_by_window_semantics |
| L3 | pass_one_edge | pass_two_edges | pass_development |
| L4 | weak_or_mixed | partial_only | cache_safe_subspace_is_material |
| L5 | adjacent_fail_age2_ce_only | early_ce_pass_broad_ranking_fail | original_d1_opportunity_gate_not_met |
| L6 | not_reopened | not_measured_in_this_revision | pending_only_after_model_revision_confirmation |
| L7 | blocked | blocked | do_not_resume_d1_d2_d3 |

## 主要方向

采用 **KV-invariant streaming updates + periodic full refresh**。当前原生子空间包含 untied output head、最后一层 Q/gate/output projection 和 final norm；所有产生 resident K/V 的参数冻结。

两条自然日边 pooled CE 收益保留 `52.23%`；Reuse 与 Exact 的 cache/hidden/NLL 最大误差均为 `0`。
两条边的逐 target ranks 完全一致：`True`。

## 明确停止

停止把“所有相邻版本 K/V 的通用近似迁移”作为当前论文主线；不再通过负样本、窗口、epoch 或指标筛选制造更大的 Reuse–Exact 差值，也不恢复旧 D1/D2/D3 性能扩展。

## 仍需完成

- Repeat the entire base/full/KV-invariant chain with an independent training seed; update-only stochastic repeats do not count.
- Extend the natural-day chain to four to six ordinary updates and predeclare the periodic full-refresh cadence.
- Measure the quality-versus-cache-renewal-cost curve for invariant updates between exact full refreshes.
- Confirm the native invariant parameter boundary on the large QK model before using QK as the capacity stressor.
- Refresh the closest-work screen before making any novelty claim about cache-compatible training.

当前结果只有一个 base-model seed，排名收益弱于 CE；因此这是下一阶段候选路线，不是论文结论。
