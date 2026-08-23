# Archived: P10.0–P10.1 Cheap Release-Time Profiler Result

## 问题

P9.11 的 state×action frontier 使用每个状态的 CurrentExact loss，是不可部署的 oracle。P10 首次检验：发布时不知道未来请求时，只看 cutover 已有状态特征，并对确定性抽取的 1%/2% 状态执行显式计费的 target-free probe，能否为全人群分配 No-op、Partial 或 Exact。

策略在连接任何 P9.9 label 前已封存。特征只包括 prefix length、state age、发布前活跃度、unique item、organic ratio 和 repeat ratio。预测器固定为 StandardScaler + Ridge；没有搜索特征、系数、阈值或 action。

## Target-free frontier

24 cells、全部三个 seed 和两种 probe 比例均完整保留。R0 根据 release metadata 全部选择 No-op，零 probe、零迁移成本。

1% probe 在非 R0 条件的等权三-seed平均恢复率如下：

| Release / Model | 5% budget | 10% budget | 25% budget |
|---|---:|---:|---:|
| R1 edge1 / M0-F | 38.0% | 54.1% | 72.9% |
| R1 edge1 / M1 | 41.6% | 60.8% | 87.9% |
| R1 edge2 / M0-F | 39.7% | 54.9% | 78.5% |
| R1 edge2 / M1 | 41.3% | 67.5% | 95.4% |
| R2 / M0-F | 54.1% | 79.2% | 95.8% |
| R2 / M1 | 38.1% | 58.2% | 83.4% |

5% budget 下，策略约达到 P9.11 offline oracle recovery 的 54%–66%；更高预算下逐渐逼近 oracle。所有 probe 计算均已从总 token-layer budget 中扣除。

2% probe 也全部为正，但低预算下额外 calibration cost 通常抵消了更多样本带来的收益；这两条冻结曲线不能在当前 development cells 上事后择优后冒充预注册主配置。

## Held-out rolling quality

1,194,300 条 state-action assignments 封存后，才与 P9.9 held-out rolling logits 做 uid/action join，共评测 144 个冻结策略。

最重要的 M1-R2 条件中，1% probe 的 No-op→policy aggregate log-loss改善为：

| Budget | seed 17 | seed 37 | seed 71 | 等权均值 |
|---|---:|---:|---:|---:|
| 5% | +0.000229 | -0.000146 | +0.003879 | +0.001321 |
| 10% | +0.000957 | -0.000118 | +0.003232 | +0.001357 |
| 25% | +0.001009 | +0.000187 | +0.003332 | +0.001509 |

25% budget 下三 seed 的 aggregate log loss 均优于 No-op；AUC 的等权改善约 0.0117。R1 的质量差异较小且方向混合，符合 P8/P9 的 release-semantics picture。

dislike-only log loss 仍不单调，部分 seed 即使 aggregate fidelity 和 quality 改善也会恶化 rare-class calibration。它是强制 companion，不能用 future dislike label修正 cutover 动作。

## 裁决

```yaml
release_time_information_suffices_for_nontrivial_allocation: established_in_development
r0_release_metadata_noop_gate: passed
charged_sparse_probe_frontier: positive_across_all_non_r0_seeds
heldout_quality_recovery: strongest_and_three_seed_positive_at_M1_R2_25pct
rare_dislike_calibration: unresolved_companion
deployable_controller: not_yet
paper_qualification: not_yet
```

P10 已回答“未知未来请求时能否决策”：可以在模型发布 cutover 时，用全人群已有状态特征和少量计费 probe 预先分配动作；后续请求只消费已迁移状态。下一门是测量 mixed-policy 的真实 batched runtime，并冻结最小 scheduler 方法，之后才进入新版本边或 blind qualification。
