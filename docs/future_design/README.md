# future_design

这里只保留当前 D2 实施基线和尚未冻结的 D3 未来方向。

| 文档 | 状态 |
|---|---|
| [DESIGN2_FINAL_PLAN.md](DESIGN2_FINAL_PLAN.md) | D2 当前实施 source of truth |
| [DESIGN2_FOUR_STAGE_EXECUTION.md](DESIGN2_FOUR_STAGE_EXECUTION.md) | D2 的输入/输出、风险与回退式执行方案 |
| [DESIGN2_STAGE_A_HANDOFF.md](DESIGN2_STAGE_A_HANDOFF.md) | 已冻结的 Stage A 边界、裁决、风险与 Stage B 入口 |
| [DESIGN2_STAGE_B_HANDOFF.md](DESIGN2_STAGE_B_HANDOFF.md) | Stage B 当前交接；W1/W2、W3 与 C0 开发闭合已完成，正式 W4 gate 尚待安全 GPU2 |
| [DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md) | 当前双 Gate 台账；C0 与 W3 physical-lowering discovery 完成，正式 Stage-C evaluation 仍 BLOCKED |
| [DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md) | D3 暂存方向；未冻结，不应立即实现 |

这里的内容不是实验结果。发生冲突时，以 `docs/08_core_insights_and_roadmap.md` 和
`docs/eval_protocol.md` 为准。

当前统一叙事是：D1 生成 immutable action plan 和 logical sparsity，D2 通过
`(S,R)` shape-aware extents、segmented suffix-only destination、merged exact physical pool
和 row-sharded execution 将它兑现为 physical sparsity。D2 不重新选择 exact/compiled；
communication-aware semantic selection 属于 D3。Motivation 2 不依赖 serving trace，synthetic
lookup contention 只作为 supporting characterization。

2026-07-28 已将旧的 D2 中间稿和 D2/D3 混合讨论分别收敛到
`DESIGN2_FINAL_PLAN.md` 与 `DESIGN3_FUTURE_DIRECTION.md` 后删除；随后增加四阶段执行文档，
用于控制 D2 的实际推进而不改变其设计 source of truth。2026-07-29 已完成 Stage A；
其 handoff 和 `configs/cohortkv_d2/stage_a_summary.json` 是进入 Stage B 前必须重新审计的
非科学实现边界。Stage B 的代码、W1/W2 normal、W1 repeat、W2 hard-failure 和
GPU1/GPU3 cross-island supplemental 已完成；物理 GPU0/GPU1/GPU3 上的 W3 NCCL normal 与
hard-failure 也已作为 `scientific_result=false` 的开发诊断完成。C0 随后在固定 16-record
fixture 上完成 W1/W2/W3 normal 和 W3 pre-commit abort，只闭合 development
wave/state-machine，不含 timing、full-cohort、capacity 或正式 epoch publication。W3/C0 都不能
替代四张独立 A40 的 W4 normal/hard-failure；Stage B 冻结和正式 Stage-C evaluation 仍由 W4
与 checked summary 阻塞。另一个独立 W3 development family 已完成 full682 v1→v5
physical-lowering discovery 和全 payload validation；它冻结候选机制但仍不是 formal Stage C
或 paper evidence。
