# EvoKV 当前执行路线

更新日期：2026-08-21。完整项目脉络见[项目全程 Compact](project_compact.md)，协议细节见[37D 技术规格](newset.md)。本文只定义当前事实、授权和下一步。

## 当前状态

```yaml
workload:
  primary: Yambda-50M Explicit Feedback (F)
  negative_control: Natural Next-Listen (N)
  not_qualified: Return-to-Familiar (R)

development_evidence:
  long_state_H: established
  cross_version_staleness_S: established
  release_semantics_heterogeneity: established
  quality_harm: strongest_in_M1_F_R2
  diagnostic_local_recovery: established

completed:
  - P7 workload/base/theta0/H qualification
  - P8 R0/R1/R2 release chain and H-S adjudication
  - P9.0 evidence seal
  - P9.1 user-level H-S distributions
  - P9.2 24-cell coarse layer/segment tomography

authorized_now:
  - diagnostic splice quality companions
  - user-risk concentration analysis
  - representative layer-by-position tomography
  - dependency-closure audit
  - legal partial executor and cost frontier

prohibited:
  - tuning P8 models/releases to enlarge S
  - controller training
  - theta3 or blind qualification
  - paper-qualified claims
```

## 冻结结论

- F 是当前唯一进入版本链的长期状态 workload。N 是 No-op 负控制；R 未通过长期质量门。
- P8 的模型、三个 seed、release recipe、lineage 和指标永久冻结，不再优化基础现象。
- R0 最大 JS 为 `4.44e-15`；R1/R2 稳定非零；M1-R2 的 Reuse 对 log loss、ROC-AUC、dislike PR-AUC 和 Brier 均有稳定损害。
- P9.2 的最佳诊断区域可恢复约 `78%–99%` stale error；recent-128 跨全部非 R0 条件和 seed 正恢复。
- 任意 KV splice 只是诊断干预。部分 layer splice 会变差，未经依赖闭包与真实 executor 验证不得称为 migration action。

数值与 caveat 分别见 [P8 结果摘要](p8_result_summary.md) 和 [P9.2 结果摘要](p9_2_result_summary.md)。

## 紧接着要做的事

1. **P9.2 companion closure**：将已封存 diagnostic logits 与 F quality labels 连接，统一计算 aggregate log loss、ROC-AUC、dislike PR-AUC、Brier 与 dislike-only log loss。不得据结果更换区域。
2. **Risk concentration**：计算 Top 1%/5%/10% 用户贡献的总 `S`，区分普遍小风险与少量高风险状态。
3. **P9.3 2-D map**：固定 R0、M0-F R1 edge1、M0-F R2、M1-F R2，三个 seed全部保留，扫描 layer × position。
4. **Dependency closure**：列出每个候选动作所需 raw history、hidden boundary、K/V 读写和下游重算范围。
5. **Executor + frontier**：只让合法动作进入 No-op / Partial / Exact 的 fidelity–work、I/O 和 runtime frontier。

P9 结束后人工裁决：若 state-level action 推开 frontier，才进入 P10 scheduler；若 uniform/version-level policy 已足够，则收缩为 release-level selector；若只有 Exact 有效，则保留 No-op/Exact 系统。

## 证据纪律

- Current Full 是当前模型执行参考，不是 future-ranking 理论上界。
- 发布质量 gate 与 cache compatibility gate 分开；低 `H/S` 的合格模型是合法 No-op。
- 全部冻结 seed 与负结果保留；不能按 `H`、`S` 或 controller 效果筛 seed。
- scheduler/profiler 不能读取 future label、future activity 或 target K/V。
- dislike-only calibration 是强制 companion，不能隐藏，也不回写改变 P7/P8 gate。
- P5/P6 的 next-listen No-Go、neutral-readout bypass 和所有已记录实现错误均不得翻案。

## 资源边界

当前 GPU 实验仅使用 GPU 0/1；CPU join、aggregation 和 bootstrap 应安全多线程。长任务仍需遵循冻结 contract，并先通过 focused canary。
