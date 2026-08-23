# Archived: P10.3 Baseline Gate and P10.4 Scheduler Freeze

P10.3 在相同 Exact-equivalent 总预算下，将 1% Ridge policy 与以下固定非学习基线比较：最佳 release-level uniform action、全部可行 uniform partial、random Exact、按 prefix length/state age/30-day activity/7-day unique items 排序的零-probe Exact，以及 P9.11 offline oracle。

6/6 个非 R0 release×model 条件通过预注册 gate。5% 预算下，Ridge 相对每格最强固定非学习基线的三-seed平均 recovery 优势约为：

| 条件 | Recovery 绝对优势 |
|---|---:|
| R1 edge1 / M0-F | +26.9 pp |
| R1 edge1 / M1 | +28.2 pp |
| R1 edge2 / M0-F | +25.0 pp |
| R1 edge2 / M1 | +30.4 pp |
| R2 / M0-F | +43.0 pp |
| R2 / M1 | +29.4 pp |

这六格均为 3/3 seed 正向，且 paired user-bootstrap CI 为 3/3 正。高预算时固定 Hybrid/Layer action 已经很强，Ridge 不总是继续领先；这说明用户级 scheduler 的主要价值位于受限发布预算区，而不是任何预算下都必须胜过版本级策略。

dislike-only 归因显示，M1-R2 seed17/71 的 CurrentExact 相对 No-op 已分别恶化约 0.070/0.210 dislike log loss。多数 mixed policy 处于 No-op 与 CurrentExact 之间，因此主要 caveat 属于当前模型完整语义/发布质量，而不是 scheduler 独立制造；个别 Policy 相对 Exact 的恶化仍需在 blind non-inferiority gate 中监控。

P10.4 据此冻结最小 scheduler：1% deterministic probe 为主、2% 完整 companion；固定 cutover features、StandardScaler+Ridge、六动作集合、5%/10%/25% 预算、probe 计费、concave-hull greedy allocator 和 tie-breaking。后续不允许增加 predictor 复杂度或新 action。

下一步只允许不改变 UID 动作选择的 executor batching 优化，并要求优化前后数值等价。θ3 blind edge 仍未使用。
