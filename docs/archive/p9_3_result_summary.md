# Archived: P9.3 Layer × Position Tomography Result

更新日期：2026-08-21。

P9.3 已完成并封存 4 个预注册语义条件、3 个 seed、24 个 layer×segment intervention，共 12 cells 和 5,517,504 条 raw rows。所有选择依据是 release semantics，不是 P9.2 最大结果。

## 主要结构

R0 的所有二维 recovery 为零，负控制通过。

M0-F R2 的结构最稳定：`layer-0 × middle` 在 3/3 seed 为正，恢复 92.06% JS；`layer-0 × recent-128` 也恢复 77.6%。更短 recent-32/8/1 分别约为 30.4%/8.7%/1.1%。

该二维区域也具有质量意义。M0-F R2 的 `layer-0 × middle` 相对 Reuse：

- log loss 改善 0.001231；
- ROC-AUC 提高 0.002963；
- dislike PR-AUC 提高 0.003396；
- Brier 改善 0.000310。

M1-F R2 的等 seed 聚合 `layer-0 × middle` 恢复 91.20%，但只有 2/3 seed 为正：seed 17 为 -35.1%，seed 71 接近 99.6% 且主导聚合。它说明 shared-state 模型的结构具有强训练随机性，不能包装成 robust universal map。

M0-F R1 edge1 的 `layer-0 × middle` 聚合恢复 46.9%，同样只有 2/3 seed 为正。不同 seed 的 24-action rank correlation 整体仅低到中等，M1-R2 甚至有负相关 pair。

## 科学裁决

二维结果没有支持一个跨模型、跨 seed 通用的“最佳格子”。它支持的是更窄、但可执行价值更高的观察：

> 在单任务 M0-F encoder refresh 中，绝大部分陈旧误差集中于 layer-0 的大历史区段；这一结构跨三个 seed 稳定，并同步影响任务质量。

Layer-0 K/V 只依赖当前位置的 current-model input embedding、normalization 和 K/V projection，因此 layer-0 segment 有机会成为真正依赖闭包完整的 action。上层 K/V 依赖下层 causal hidden，任意上层 exact splice 仍只是诊断。

因此 P9.3 只授权最小 executor canary；不授权 controller，也不把二维 splice 直接放入成本 frontier。
