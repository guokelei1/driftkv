# EvoKV

EvoKV 研究模型发布后持久化用户 K/V 状态的预算化收敛：哪些状态可以继续复用，哪些需要依赖合法的部分重算，哪些必须 Exact Recompute。

当前项目已经在 Yambda-50M Explicit Feedback workload 上建立 development 证据链：长期状态 `H` 存在；output-only 发布的陈旧性位于数值地板；更新 cache-producing path 后出现稳定陈旧性 `S`；部分发布中 Reuse 会损害任务质量；陈旧误差具有 layer/history-position 局部结构。

当前工作不再追求更大的基础 gap，而是验证诊断局部恢复能否转化为成本低于 Exact-All 的合法迁移动作。

## 入口

- [项目全程 Compact](docs/project_compact.md)：问题演变、核心证据、未决问题与完整路线。
- [当前执行路线](docs/current_route.md)：现在允许做什么、停止门和最近任务。
- [P8 结果摘要](docs/p8_result_summary.md)：冻结的 `H → S → quality` development 证据。
- [P9.2 结果摘要](docs/p9_2_result_summary.md)：全矩阵 coarse tomography 结果。
- [P9 计划](docs/p9_plan.md)：tomography、合法 partial action 与成本 frontier。
- [37D 技术规格](docs/newset.md)：lineage、population、fidelity、预算与 qualification 协议。

所有结果必须按 workload、release、lineage、seed、metric 和证据等级解释。不得跨协议拼接结果、筛选有利 seed/edge，或把 diagnostic K/V splice 当作可部署动作。
