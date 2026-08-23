# Archived: P10.5 Executor Optimization and P10.6 Full-Stack Freeze

在 P10.4 scheduler 冻结后，executor 只按 `(effective prefix length, operation signature)` 重排相互独立的状态，并跳过零工作 No-op。所有 UID 的 probe、action 和预算保持不变。

128-state canary 的 per-UID final K/V 最大绝对差均为 `2.62e-6`，低于 `1e-5` 冻结容差。十个预注册 runtime 条件全部完成：

- speedup：`1.12×–2.56×`；
- 十条件几何平均：`1.60×`；
- operation batch 数减少：`21%–61%`；
- 优化后相对 Exact-All runtime 节省：`35.8%–80.4%`。

因此 development full stack 现已冻结：1% deterministic sparse probe、固定 Ridge benefit predictor、六动作 allocator 与 grouped executor 不再调整。2% probe 保留为完整 companion。

下一阶段依次为 version-age-2/recursive lineage、8L/H256/context1024 scale point、fixed-count/capped-rate scale sensitivity，之后才一次性打开 θ3 blind edge。
