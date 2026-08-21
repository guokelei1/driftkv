# P9.2 Coarse Tomography Result Summary

更新日期：2026-08-21。证据层级：development diagnostic；不是 executable migration 结果。

## 完整性

- 24/24 frozen F cells：4 releases × 2 models × 3 seeds。
- 每个 cell 均包含 4 个 layer-only 与 6 个 history-segment-only exact-KV splice。
- Current Full / Reuse 相对 P8 sealed raw 的最大误差为 `4.77e-7 / 0`。
- raw matrix 已封存为 `results/p9/p9_2_tomography_raw_seal_v1.json`；统一聚合为
  `results/p9/p9_2_coarse_tomography_v1.json`。

## 三 seed 聚合结果

下表只列每个 condition 的诊断性最佳 splice。`Recovery` 定义为 Reuse JS 减去 splice 后的
residual JS；正值表示向 Current Full 恢复。

| Release | Model | Diagnostic splice | Stale JS | Residual JS | Recovery | Positive seeds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| R0 | M0-F | layer_0（并列） | ≈0 | ≈0 | 0 | 0/3 |
| R0 | M1 | layer_0（并列） | ≈0 | ≈0 | 0 | 0/3 |
| R1 edge1 | M0-F | middle | 9.067e-5 | 1.966e-5 | 7.101e-5 | 3/3 |
| R1 edge1 | M1 | layer_0 | 5.375e-5 | 7.116e-6 | 4.664e-5 | 3/3 |
| R1 edge2 | M0-F | oldest_half | 3.614e-4 | 7.705e-5 | 2.843e-4 | 3/3 |
| R1 edge2 | M1 | layer_0 | 2.672e-5 | 6.194e-7 | 2.610e-5 | 3/3 |
| R2 | M0-F | layer_0 | 4.019e-4 | 4.047e-6 | 3.979e-4 | 3/3 |
| R2 | M1 | middle | 1.354e-3 | 1.383e-5 | 1.340e-3 | 3/3 |

## 当前判断

P9.2 说明 staleness recovery 具有 layer/history-position 结构，并且该结构在多数组合中跨三 seed
稳定；R0 的严格零恢复继续排除 evaluator/lineage 假象。R2 的恢复最强，R1 也存在中等结构。

但任意 exact-KV layer/segment splice 仍需要当前模型生成的内部 K/V，且不保证依赖闭包，因此只能
用于定位。它不能进入成本 frontier，也不能被称为 partial migration。下一步是冻结语义代表格子的
P9.3 layer×position map，之后再审计哪些结构能导出 dependency-closed executor。
