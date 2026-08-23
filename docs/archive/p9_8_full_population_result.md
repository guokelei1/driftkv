# Archived: P9.7–P9.8 Full-Population Cutover Result

P9.7 将 migration population 从“未来会出现 F 请求的用户”纠正为 release cutover 时所有已经物化的状态。该总体完全由发布时信息定义：edge1 有 8,229 个状态，edge2 有 8,488 个状态；未来 F served users 仅占 36.14%/36.86%，因此不能代替全人群分母。

P9.8 在全部 24 个 `release × model × seed` cells 上，对每个状态使用 16 个由发布前 F train candidate pool 确定性生成的无标签 probes，一次性计算 No-op、四个合法 Partial 和 Exact。共封存 199,050 次 state-cell evaluation 与 19,108,800 条 candidate-action raw rows。Exact 是离线 oracle，不是部署时全人群必须执行的 profiler。

## 全人群恢复

下表使用每个 seed 的 population MSE，再做等权 seed 汇总。括号内是动作的 Exact-equivalent token-layer work；edge1/edge2 的比例近似相同。

| Release / Model | L0 Recent128 (6.8%) | L0 Middle (12.5%) | L0 Full (25.0%) | HybridTail128 (27.2%) |
|---|---:|---:|---:|---:|
| R1 edge1 / M0-F | 34.3% | 40.6% | 26.9% | 56.9% |
| R1 edge1 / M1 | 44.8% | 61.2% | 87.7% | 56.5% |
| R1 edge2 / M0-F | 9.5% | 17.0% | 23.0% | 42.8% |
| R1 edge2 / M1 | 57.5% | 77.1% | 98.0% | 58.6% |
| R2 / M0-F | 82.3% | 95.2% | 99.6% | 85.5% |
| R2 / M1 | 85.2% | 87.9% | 82.1% | 94.2% |

R0 的 No-op MSE 为零；partial 的残余 MSE 仅约 `1.4e-17`–`8.7e-17`，仍是数值地板。R1/R2 的 No-op population MSE 则为 `0.00429`–`0.22237`，说明 target-free cutover probes 上的 release staleness 明确存在。

## Seed 边界

聚合均值不能掩盖 seed 异质性：

- M0-F R1-edge1 的三个 layer-0 action 均有一个 seed 为负恢复；HybridTail128 为 54.3%/56.4%/60.3%，3/3 seed 为正。
- M1 R2 的 layer-0 actions 在 seed17 为负，但 HybridTail128 为 84.8%/70.9%/95.1%，3/3 seed 为正。
- M1 R1-edge2 的 Layer0-Full 为 99.5%/93.8%/97.6%，是稳定的低工作量 release-specific action。
- M0-F R2 的 Layer0-Full 为 99.1%/97.9%/99.9%。

因此当前结果支持 release/model-aware action，而不支持一条 layer splice 对所有模型和 seed 通用。HybridTail128 是跨格子最稳定的统一 partial candidate；Layer0-Full 在部分 release/model 条件下更优。

## 全人群风险集中度

非 R0 cells 的 No-op MSE Gini 为 0.316–0.803，Top 10% 状态贡献约 18.4%–70.5% 总风险。以 HybridTail128 只迁移 offline oracle 排出的 Top 10% 高风险状态：

- 成本约为 Exact-All token-layer work 的 1.85%–2.92%；
- 捕获总体 stale-error recovery 的约 11.3%–52.0%；
- risk 与 action benefit 的 Pearson 相关系数在非 R0 cells 为 0.71–0.99。

这证明 state-level allocation 存在潜在空间，但该排序读取了 CurrentExact probe loss，只能作为 oracle opportunity upper bound，不能作为部署策略。P10 scheduler 仍未授权。

## 证据边界与下一步

P9.8 已解决两个问题：全人群分母不依赖未来请求；合法动作在 release cutover 的无标签 probes 上能以显著低于 Exact 的逻辑工作量恢复 fidelity。它尚未完成：

1. 全量 held-out future F 请求上的 rolling-lineage quality companion；
2. 全人群 batched migration 的实测 runtime、KV I/O 和 rollout completion；
3. 将 fidelity、quality 和实测成本合并成正式 No-op / Partial / Exact frontier；
4. 仅使用发布时 cheap features 的 deployable profiler。

在这些完成前，不训练 scheduler，不启动 blind edge，也不将 P9.8 升格为 paper-qualified evidence。
