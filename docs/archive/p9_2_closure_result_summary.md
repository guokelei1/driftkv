# Archived: P9.2 Closure Result Summary

更新日期：2026-08-21。

P9.2-Q 与 P9.1-C 已在全部 24 个冻结 cell、10 个 diagnostic action 和三个 seed 上完成。标签只用于事后质量 companion；state 和 action 均未按标签选择。

## 用户风险集中度

R0 的总风险低于数值地板，因此不人为定义 Top-K share。六个非 R0 model × release 条件的等 seed 均值为：

| 用户比例 | 总 S 贡献范围 |
| ---: | ---: |
| Top 1% | 7.15%–10.04% |
| Top 5% | 24.22%–31.19% |
| Top 10% | 38.63%–49.07% |

这证明 state risk 具有明显长尾，并为 state-level allocation 提供了机会信号；但它还不能授权 scheduler，因为当前 recovery 仍来自不可部署的 exact-KV splice。

## Diagnostic recovery 的质量意义

R0 的所有 action quality effect 都为零，负控制继续通过。

R2 中，fidelity recovery 与任务质量恢复高度一致：

| 条件 | Diagnostic region | JS recovery | Log-loss gain vs Reuse | 相对 Full gain | 其他质量恢复 |
| --- | --- | ---: | ---: | ---: | --- |
| M0-F R2 | layer 0 | 98.99% | 0.001461 | 101.6% | ROC 约 95.0%，dislike PR 约 112.0% |
| M1-F R2 | middle history | 98.98% | 0.003050 | 93.2% | ROC 约 88.0%，dislike PR 约 100.4% |

预注册的固定 `recent-128` 也不是只恢复数值：

- M0-F R2：恢复 81.6% JS，log-loss 对 Reuse 为 3/3 seed 正；
- M1-F R2：恢复 96.8% JS，log-loss、ROC-AUC 和 dislike PR-AUC 均为 3/3 seed 正。

R1 的质量恢复更混合。这与 P8 一致：routine update 可以产生可测 fidelity staleness，但不保证每条边都形成稳定业务质量损失。

## Caveat 与裁决

- dislike-only log loss 的异质性继续存在；aggregate fidelity 或质量恢复不能替代该切片。
- 单个 action 的 quality recovery ratio 在 Full–Reuse 质量差接近零时不稳定，只作 companion。
- 最佳区域是描述性诊断，不是 action selection，也不是成本 frontier。

当前裁决：局部 stale structure 在 R2 中具有明确质量意义，用户风险也足够集中，授权事前按发布语义固定的 P9.3 二维 tomography；仍不授权 executable partial、frontier 或 controller。
