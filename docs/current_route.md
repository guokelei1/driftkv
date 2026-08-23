# EvoKV 当前执行路线

更新日期：2026-08-23。完整 development 证据见
[截至 P11.4 的统一总结](evidence_summary_through_p11.md)，下一轮规模实验见
[规模化扩展路线](scaling_extension.md)。

## 当前结论

Yambda-50M、4L/H128/context512 上已经完成一轮端到端 development 实验：

```text
N/R/F workload 与 Frozen Base
→ F 的长期状态 H
→ R0/R1/R2 的 release-dependent S
→ dependency-closed partial actions
→ 全人群成本—fidelity frontier
→ 1% target-free sparse profiler + Ridge scheduler
→ grouped executor
→ θ0→θ1→θ2 true recursive lineage
→ 封存策略后的 rolling quality validation
```

关键状态：

```yaml
development_full_round: complete
F_long_state_H: established
cross_version_S: established
R0_noop_control: passed
legal_partial_frontier: established
minimal_scheduler: frozen_and_positive
recursive_version_debt: established
recursive_quality_recovery: positive_but_not_universal

scale_8l_m0_f_seed17_H: passed
scale_8l_m0_f_seed17_rolling_S:
  r1_edge1: passed
  r1_edge2: passed
  r2: passed
scale_8l_R0: passed
scale_8l_frozen_partial_actions: passed
scale_8l_scheduler_fidelity_frontier: positive_8_of_9_budget_cells_vs_strongest_nonlearning
scale_8l_policy_quality:
  R1: fidelity_positive_quality_not_stable
  R2_25pct: improves_over_Noop_and_matches_Exact
scale_cross_seed_or_M1: not_yet

next_phase: scale_replication_scope_adjudication_before_theta3_contract
theta3_blind_edge: untouched
paper_qualification: not_yet
```

## 冻结方法

- Workload：F Explicit Feedback；N 是短期负控制，R 未通过长期资格。
- 模型：M0-F 单任务与 M1 N/R/F 共享多任务；seed 17/37/71 全部保留。
- Release：R0 output-only、R1 routine continual、R2 encoder refresh。
- 动作：No-op、Layer0-Recent128、Layer0-Middle、Layer0-Full、
  Hybrid-Tail128、Exact-All。
- Scheduler：1% deterministic target-free probes 为主，2% companion；固定
  cutover features、StandardScaler + Ridge(alpha=1)、5%/10%/25% token-layer
  budgets、concave-hull greedy allocator。
- Executor：按 prefix length 和 operation signature grouping；不得改变 UID action。

以上方法统一命名为 **EvoKV v1**。在规模训练前必须封存代码 commit、全部配置与
合同、manifest、Frozen Base、模型 checkpoint、assignment/raw-result hashes 和统计
实现。此后不得再根据 θ0–θ2、规模点或 θ3 的结果修改。

## 下一步：更大模型复现点

先运行一个预注册的同源规模点：

```text
8 layers / hidden 256 / context 1024
Yambda-50M F workload
M0-F + M1
seed 17/37/71
R0/R1/R2
```

规模点只复现以下冻结证据链，不重新探索：

```text
H → S → legal partial recovery → same-cost scheduler advantage
```

1. Full/Append 与 rolling-cap correctness；
2. F 的 Full-1024 vs Recent-32 长期 H；
3. R0 是否仍处于数值地板，R1/R2 是否仍有 S；
4. 冻结动作的 recovery 与 exact-equivalent cost；
5. 冻结 scheduler 相对同成本基线的 target-free frontier；
6. 代表性 rolling quality companions；
7. 1% probe 与预先冻结的 fixed-count/capped-rate sensitivity。

S0–S4 pilot 已完成。资源/覆盖、真实 rolling correctness、四卡 FSDP backward/AdamW、
trainer/checkpoint round-trip、M0-F seed17 的长期 H，以及 R1 两边和 R2 的 rolling S
均通过。用户本轮只启动了 R1/R2，因此 **8L R0 尚未复现**，4L R0 仍是当前唯一严格
No-op 负控制。当前结果只能授权冻结 action replay，不能升级为跨 seed 或 paper 结论。

8L M0-F seed17 的关键规模结果：

| Edge | H JS | S JS | S/H | Exact 相对 Reuse 的 log-loss gain |
| --- | ---: | ---: | ---: | ---: |
| R1 edge1 | 8.503e-4 | 1.483e-4 | 17.44% | +0.001523，CI 为正 |
| R1 edge2 | 1.223e-4 | 3.485e-5 | 28.50% | +0.000146，CI 跨 0 |
| R2 | 9.774e-4 | 1.509e-4 | 15.44% | +0.001139，CI 为正 |

三条边的 H/S 用户 bootstrap CI 均高于数值地板。R1 edge1 与 R2 还复现了 stale Reuse
对 aggregate log loss 的稳定伤害；R1 edge2 证明 S 存在，但 aggregate log-loss 质量
影响尚不稳定。dislike-only log loss caveat 在规模点继续存在，必须完整报告。

S5/S6 的 M0-F seed17 方法 pilot 也已完成：R0 全动作处于数值地板；最佳冻结 partial
在 R1-edge1、R1-edge2、R2 分别恢复 `43.4% / 74.9% / 96.35%` 的全人群
target-free stale MSE。1% Ridge 在 9 个 release×budget 点中的 8 个优于最强同成本
非学习基线。按 5%/10%/25% logical budget，实测 grouped transition runtime 相对
Exact-All 节省约 `69.5%–93.4%`。

Rolling quality 并非所有边都同步改善：R1 两边主要是 fidelity 结果；R2 的 25% policy
相对 No-op 的用户等权 log-loss gain 为 `0.001234`，95% CI
`[0.000117, 0.002311]`，同时与 Exact 的差异 CI 跨 0。第一版未按 request_weight
汇总的 quality 表已显式作废，正式 scale quality 结果为
`results/scale_8l_v1/policy_quality_adjudication_v2.json`。

## 规模点之后：θ3 blind qualification

规模点完成后、训练或读取 θ3 结果之前，必须冻结 blind contract。合同至少写死：

- θ3 数据窗口、release recipe、model-admission gate；
- H、R0 identity、R1/R2 staleness 和同成本 scheduler 的主判定；
- aggregate quality non-inferiority/improvement 规则；
- dislike PR-AUC、dislike-only log loss 等强制 companion；
- logical work、I/O、runtime 和统计汇总方式；
- 各种失败形态的裁决与报告规则。

θ3 是新的、尚未查看的时间发布边。揭盲后不得修改 action、feature、Ridge、probe、
budget、阈值或指标。通过后证据才能从 `established in development` 升级为
`reproduced on a previously unseen temporal release under a frozen policy`。

## 当前禁止

- 调整 P8 release recipe 或训练幅度来放大 S；
- 重新设计 N/R/F、history length、task weights 或 candidate protocol；
- 按 H/S/quality 筛 seed；
- 增加新 action、复杂 predictor、GBDT 或 target-KV mapper；
- 使用 future request/label 做发布期动作选择；
- 启动 θ3 或宣称 paper qualification；
- 将 Current Full 写成 future quality 理论上界；
- 隐藏 dislike-only calibration caveat。

## 资源边界

- 当前规模实验 GPU allowlist 为 0/1/2/3；24 亿参数训练采用单模型四卡 FSDP，
  seed/版本由队列串行推进；
- CPU-only join/aggregation/bootstrap 应安全多线程；
- 长任务先 canary，再冻结合同，再启动自动队列；
- 不默认保留重复 checkpoint、临时展开数据或重复 raw artifacts。
