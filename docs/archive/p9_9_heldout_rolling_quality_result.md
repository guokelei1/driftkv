# Archived: P9.9 Held-out Rolling Quality Result

P9.9 在完整冻结 F quality requests 上，以真实 cutover materialization、每 uid/动作只迁移一次、逐事件 append/evict 和跨请求状态复用，完成全部 24 个 `release × model × seed` cells。raw logits 在计算指标前封存。

## 覆盖

- 24/24 cells；
- 469,176 次 cell-request evaluation；
- 2,815,056 条 request-action rows；
- edge1 每 cell 19,158 requests，edge2 每 cell 20,722 requests；
- R0 全动作差异保持数值零；
- 每 cell wall time 约 9.95–10.86 分钟，GPU0/1 双队列总时长约两小时。

这些 future requests 只用于质量验证，不定义 migration population 或成本分母。全人群分母仍是 P9.7 的 8,229/8,488 个 cutover states。

## Fidelity

真实 rolling-lineage 下的 Full-fidelity recovery 与 P9.8 cutover probes 方向一致：

- M1 R1-edge2 Layer0-Full：99.5% / 91.7% / 97.4%；
- M0-F R2 Layer0-Full：97.8% / 97.2% / 99.8%；
- M1 R2 HybridTail128：76.4% / 68.4% / 97.6%；
- M1 R2 layer-0 actions 在 seed17 为负，不能作为通用动作；
- HybridTail128 在所有非 R0 cells 的三 seed 上保持正恢复。

## 最强质量结果：M1 R2

No-op 相对 CurrentExact 的 log-loss 损害为：

```text
seed17: +0.001779
seed37: +0.000619
seed71: +0.003486
equal-seed mean: +0.001962
```

HybridTail128 后残余损害为：

```text
seed17: +0.000499
seed37: +0.000308
seed71: +0.000177
equal-seed mean: +0.000328
```

因此 HybridTail128 在三 seed 上都降低 No-op log loss，按等权 seed 总量约恢复 83.3% 的 stale-quality harm。它同时恢复约 94% 的平均 ROC-AUC harm；dislike PR-AUC 的恢复跨 seed 不完全一致，但最强 seed71 的差距由 `-0.03933` 缩小到 `-0.01509`。

M1-R2 的绝对 log loss：

| Seed | No-op | HybridTail128 | CurrentExact |
|---:|---:|---:|---:|
| 17 | 0.343839 | 0.342559 | 0.342060 |
| 37 | 0.343350 | 0.343038 | 0.342731 |
| 71 | 0.351685 | 0.348375 | 0.348198 |

用户级 bootstrap CI 对单 seed 的小效应仍较宽；正式稳健性主要来自三个独立训练 seed 的一致方向，而不是把请求数当作独立重复。

## Release 与模型异质性

- R1 的 fidelity staleness 清楚，但 aggregate quality 多数很小且跨 seed 混合；不能声称 routine update 每次都显著损害任务质量。
- M0-F R2 的质量方向跨 seed 混合：一个 seed 的 No-op 甚至优于 CurrentExact，另一个 seed 明显受损。因此它支持 fidelity state evolution，但不是最强质量 substrate。
- M1 R2 是当前最完整的 `S -> quality harm -> partial recovery` development case。

## Rare-dislike caveat

M1 R2 的 dislike-only log loss 并不随 aggregate fidelity 单调恢复。No-op 在该 slice 上有时优于 CurrentExact；HybridTail128 虽显著改善 aggregate log loss、AUC 和 Brier，仍不能保证 rare-class calibration 完全对齐。

所以后续每个 frontier point 必须同时报告：aggregate log loss、ROC-AUC、dislike PR-AUC、Brier 和 dislike-only log loss。future dislike label 不得用于 action selection。

## 裁决

P9.9 是正面 development 结果：合法 partial action 在真实 rolling lineage 下不仅恢复 target-free fidelity，也在最强 release/model 条件中恢复实际任务质量。下一步可进入 migration-only batched runtime、I/O/rollout measurement 和正式 No-op / Partial / Exact frontier；scheduler 仍未授权。
