# Archived: P10.2 Mixed-Policy Runtime Result

## 范围

P10.2 对 P10.0 已封存的 state-action assignments 做真实 batched GPU rollout。采样状态执行五种计费 probe 并保留 Exact 结果；其余状态只执行封存动作。No-op 不启动 kernel。策略、probe 比例和预算均未根据 runtime 或 quality 修改。

十个预注册条件全部完成，且只使用 GPU 0/1。逻辑 recomputed token-layer work 均不超过 5%/10%/25% 冻结预算；离散 action 导致的最大欠用约 `2.1e-6` Exact fraction。

## 实测结果

全部 mixed policies 的 GPU kernel rollout 为 Exact-All runtime 的 36.4%–72.2%，即节省 27.8%–63.6%。逻辑 work fraction 明显小于 runtime fraction，因为 prototype executor 仍受：

- 多 action 小 batch 的碎片化；
- cache clone 与 launch overhead；
- sampled probe 的多路执行；
- ragged prefix-length 分组。

因此不能把 5% token-layer budget 写成 95% wall-time saving。

代表性的 M1-R2 seed17：

| Probe | Budget | Runtime / Exact | Target-free recovery | No-op→Policy log-loss改善 |
|---:|---:|---:|---:|---:|
| 1% | 5% | 44.7% | 12.5% | +0.000229 |
| 1% | 10% | 55.5% | 29.8% | +0.000957 |
| 1% | 25% | 68.5% | 75.7% | +0.001009 |
| 2% | 5% | 50.0% | 9.8% | +0.000010 |
| 2% | 10% | 54.3% | 26.5% | +0.000730 |
| 2% | 25% | 72.2% | 74.6% | +0.001115 |

这里的 runtime 只包括 transition action kernels 和返回 cache materialization；raw-history reconstruction、H2D 和 parent reference build 分开记录。它仍是 prototype PyTorch executor，不是生产优化 kernel或存储吞吐。

## 裁决

```yaml
sealed_mixed_policy_execution: passed_10_of_10
logical_budget_conservation: passed
measured_runtime_saving_vs_Exact: 27.8_to_63.6_percent
cost_fidelity_quality_frontier: established_in_development
logical_work_equals_wall_time: false
production_executor: not_yet
controller_method_frozen: not_yet
blind_qualification: not_authorized
```

P10.2 证明 state-level action assignment 不只是离线数学 frontier：封存策略可以由真实 executor 完成，并在 M1-R2 上用低于 Exact-All 的实测 kernel 时间恢复 fidelity 和 aggregate quality。下一决策点是是否将 1% sparse-probe + low-capacity benefit predictor 冻结为最小 scheduler，或先做 executor batching 优化。该选择会改变最终方法，适合在进入 blind edge 前人工/专家裁决。
