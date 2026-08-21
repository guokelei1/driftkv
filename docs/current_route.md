# EvoKV 当前路线（37D）

更新日期：2026-08-21。

本文是当前执行入口；[论文设计](paper_design.md) 定义论文问题边界，[完整技术规格](newset.md) 定义术语、协议与研究问题。除非本文件明确保留，旧材料不是当前事实或论文结果。

## 核心问题

EvoKV 是一个**发布期预算化状态收敛系统**。模型团队已经决定发布 `θ_t`；EvoKV 在有限后台计算和 I/O 预算下，对发布时冻结的 active persistent-KV snapshot 完成逻辑版本切换，使状态尽可能逼近当前模型的完整执行语义。

它不解决模型是否应发布，也不假设公开数据能够给出真实生产请求 P99、QPS 或业务排序变化 SLO。

允许的状态动作是：

- `No-op Reuse`
- `Fast Migration`
- `Selective Recompute`
- `Exact Recompute`

“完成迁移”指每个 active state 都获得明确的版本处理结果，而非每个 tensor 必须被物理改写为 exact KV。

## 当前证据状态

- Yambda-50M 的时间语义、长历史 workload、数据机会与 HSTU Full/Append correctness 已通过。
- batch-alignment 与发布 cutover 均已修正；修正前的 medium θ0、两边链与 profiler 输出永久标为失效 development artifact。
- 修正后的 θ0 通过 history-conditioned sanity gate。
- 已冻结无 TTL 的 release-state snapshot：`θ0→θ1` 有 7,868 个状态、`θ1→θ2` 有 7,890 个状态；两者均只由发布前已物化的 exact-parent KV 定义。
- **2026-08-18 temporal-continuity audit：当前所有 batch-fixed-v3 的 release-fidelity、multi-panel、frontier、ranker、large-candidate 与 dilution 数值均降级为失效 development artifact。**原因是部分 evaluator 把交给 `forward_with_cache` 的首个 suffix/readout token 的 time delta 重置为 0，而 Full 路径保留它相对 cached-prefix 最后事件的真实 delta；两路不再只差 prefix KV version。已加入连续时间边界回归测试，必须在修复后的合法 lineage 上重跑，才可恢复任一数值结论。
- 严格 next-release proxy 的总体定义、coverage 守恒与 cohort-bias 审计仍是有效**协议设计**，但其中的 Spearman/recall 数值须随合法 lineage 重算。它无论结果如何都只能表述为 observed-active cohort 的 trace proxy，而不是 serving request。
- 原 panel A/B 是 popularity rank 的不同确定性切片，跨 panel 排序失败，不能作为候选无关的风险证据。为此已冻结 `Q_main_rank_decay_v1`：对同一 pre-release proposal 独立无放回抽取 32 个 target-free 100-way panel，前 16 个 development、后 16 个 held-out。
- 修复前的 `Q_main` reliability、external validity、opportunity、panel-free、large-candidate、single-panel 和 GBDT frontier 文件全部保留用于可追溯性，但不再是当前证据，也不得用于 controller/ranker/model-selection 决策。
- 修复后的 `Q_main` multi-panel baseline 已重跑：split-half Spearman 为 0.957 / 0.930、Top-10% recall 为 94.5% / 92.4%。这仅恢复 `Q_main` 内 release-fidelity label 的测量可靠性；proxy、opportunity 与 controller/ranker 仍需在合法 lineage 下重算。
- 在两条边的 complete-case、无 physical-cap-eviction cohort（126 / 128 states）上，repair-vs-bypass/fixed-query v1 显示**neutral readout bypass**：fixed-query `k=0` 的 stale error 为 0.00122 / 0.00080，而 Full 相对 Suffix Only 的长期-prefix contribution 为 0.41493 / 0.11971；加入一个 current-model event 后该贡献降至 0.02310 / 0.00311，但 Full relative to latest-append-masked 的变化只有 0.00013 / 0.00001。故现象不是“一个 token 将旧 KV 修复为当前 KV”，而是当前 neutral readout 在有一个 suffix event 后几乎不再读取旧 prefix。它是 development tomography evidence，只适用于当前 4L/H128/512、one-hop、Q_main 与 neutral readout。

Yambda-50M 当前定位为 development platform：它负责把协议、风险定义、scheduler 和 migration mechanism 做对，不承担最终“大模型、大数据”结论。规模化扩展的分阶段边界见[规模化扩展路线](scaling_extension.md)：方法冻结后由同源 Yambda-5B 承担主 qualification，再由 VK-LSVD 做 population/system-scale 验证，RecFlow 补充真实 candidate workload。

这些都是 development evidence，不是 controller qualification 或论文最终结果。

## P5 授权状态（2026-08-19）

P4 已永久冻结为 development diagnosis：evaluator 一致，CE/Top-K 冲突为真，旧 candidate protocol 被 seen-membership / long-term count shortcut 污染。它推翻的是“当前 v1/v2 gate 已证明长期高维 KV 必要”，不是论文假设本身。

```text
candidate_protocol_identifiability: failed
paper_hypothesis: unresolved
cc_theta1_theta2_authorized: false
```

授权链现为 `Protocol Validity → H → S → Heterogeneity → Opportunity → Design`。当前停在 Protocol Validity 与 H 之间。旧 v1/v2 gate 不得通过更换候选后重算来恢复资格。

P5.1 已按覆盖率冻结 `m_recent=16, m_old=24, m_discovery=59`（完整配额覆盖 94.1%）。P5.3 未通过且不得改判。P6 完成了最后一次、不训练的 conditional identifiability 裁决，结论是 **分支 B / Yambda next-item No-Go**：

- P5 的 0.789 已确认是 request-conditioned cAUC；按 uid 分组 OOF 为 0.811，不是泄漏。
- 连续配平在 holdout 上 SMD 达标、覆盖未坍缩（1481 请求），但 simple-feature cAUC 仍为 0.736 > 0.60；残差主要是 Q_main rank 形态。
- 无注入时自然进入 Q_main@100 的 target 仍可被简单特征识别（cAUC 0.795）。

不得训练 CC-θ0-v3，不得再构造第五套 Yambda next-item negatives。P6 的 No-Go 只适用于
`next-listen + sampled candidate` 的长期 KV identifiability，不得解释为 Yambda 整体 No-Go。
Yambda 的时间链、snapshot、lineage 和状态演进平台资格保留；论文假设仍未决。

P7 改为 [Yambda Multi-Regime Stateful Workload Suite](yambda_stateful_workload_suite.md)：
Natural Next-Listen 作为 No-op/短期负控制，Return-to-Familiar 作为长期状态主机制候选，
Long-Horizon Feedback 优先审计显式 like/dislike。simple count/recency 不再被要求失效，
而是进入冻结 base ranker；CC-HSTU 只承担其外的 candidate-history residual。

## P8 结果与当前授权（2026-08-21）

P8 的 F-workload release chain 已完整封存：`R0`、两条 `R1` routine-continual
edge 与一条 `R2` periodic encoder-refresh edge 均覆盖 M0-F、M1 与 seed 17/37/71；
所有 18 个 update checkpoint 都通过既有 admission rule，所有 seed 均保留。
完整数值与 caveat 见 [P8 结果摘要](p8_result_summary.md)。当前可成立的 development
因果链是：

```text
F 的长期状态 H 存在
→ R0 output-only 更新时 S 位于数值地板
→ cache-producing path 更新后 R1/R2 的 S 非零
→ M1-F 的 R2 中 Reuse 还稳定损害 F 质量
```

因此当前授权状态为：

```yaml
F_long_state_H: established_in_development
cross_version_staleness_S: established_in_development
release_semantics_heterogeneity: established
system_design_opportunity: authorized_for_development
paper_qualification: not_yet
tomography_and_action_space_qualification: authorized
controller: not_yet_authorized
theta3_blind_edge: preserved
```

P8 基础模型、workload、release recipe、seed 与评分协议永久冻结；不得为放大 S 再调
学习率、任务权重、刷新幅度、candidate protocol 或 K/V。当前下一阶段是
[P9：Staleness Tomography and Action-Space Qualification](p9_plan.md)，而不是 controller。

## 不承担的证明负担

- 不证明 HSTU 优于 SASRec、BERT4Rec 等算法。
- 不要求每个自然版本边都出现正的 Recompute-over-Reuse NDCG gap。
- 不把 Current Full 当作未来用户质量的理论上界；它只是当前模型的 fidelity reference。
- 不把 `Top-10 overlap loss ≤ 0.2/0.5` 声称为真实线上 SLO；它们只是诊断 operating points。
- 不使用发布后活跃、未来请求次数、future target 或 future append 作为发布期 scheduler 的主特征。
- 不用事后分数混合、挑选有利边/用户群或 target-KV fitting 制造结果。

## 最小系统闭环

1. **Release profiler**：在少量 pre-release canary states 上离线比较 Current Full、Reuse 和 partial paths，产生版本与状态风险画像。
2. **Budget-aware scheduler**：只使用发布时可得的版本差异、状态与成本特征，预测连续 semantic risk，并在发布预算内排序分配动作。
3. **Transition executor**：先实现 no-op/exact；oracle 与 learned ranker 均证明必要后，再加入 fast migration 与 selective recompute。
4. **Rollout accounting**：报告 exact-equivalent work、KV read/write、worker-hours 和 state-version debt，而非虚构请求级 SLA。

## 当前执行顺序

1. Yambda 旧 next-item candidate-protocol 搜索已停止（P6 分支 B），P5/P6 不改判。
2. P7.1 全量审计已完成：R 按 coverage-only 规则冻结为 3-day inactivity gap；F 按标签
   覆盖与 shortcut 门冻结为 explicit like/dislike。两项选择均未读取 H/S。
3. P7.4 已冻结 base fitter、feature transform/L2 blocked CV、M0/M1 objective 和 R exact
   variable-candidate batching；细分窗口覆盖审计通过。P7.5 的 N/R/F quality/fidelity canary
   各 128 个独立用户并通过因果、label separation 与完整候选集断言。
4. P7.5-Full compact materialization 已通过并封存：124,214 个唯一 query spec，R 的
   quality-rankable / fidelity-all-eligible / rankable-companion 总体已分离；28 组 compact
   与 raw-expanded 重建逐字段等价，覆盖守恒与 M0/M1 非 compute-matched 预算表通过。
   qualification 默认 loader guard 已生效，且未评分；完整结果见
   `configs/contracts/p7_5_full_result_v1.yaml`。
5. P7.6 frozen Base fitting 与 development-only sanity 已通过。N/R/F 的 L2 分别冻结为
   0.001/0.01/0.01；streaming-vs-expanded canary、参数有限性、非零分数方差、R 完整候选、
   F 标签方向和 qualification guard 均通过。结果仅证明 Base 正常可用，不是 H 证据；见
   `configs/contracts/p7_6_base_fit_result_v1.yaml`。
6. P7.7 M0/M1 θ0 training 与 development-only sanity 已完成：M0-N/R/F 和共享 M1 各有
   3 个独立 seed，共 12 个 Full-512 run；Frozen Base、训练/开发 manifest 与 checkpoint
   均已按 hash 封存，qualification 未读取。所有 M0 seed 的部署 loss 均优于对应 Base；
   M1 的 N/F 三个 seed 均优于 Base，但 R 仅 seed 17 优于 Base，seed 37/71 出现多任务干扰。
   该负面异质性完整保留，不影响预注册的 P7.7 最低门（每个 seed 的 R 或 F 至少一个正增量，
   且各任务优于无信息控制），但它不是 H 证据。完整结果见
   `configs/contracts/p7_7_theta0_training_result_v1.yaml`。
7. P7.8 已按开锁前封存的 run plan 对全部 12 个 checkpoint 一次性完成；42 份 raw-score
   Parquet 在计算指标前统一封存，all-seed 与既有 dev admission 下的 released subset 均完整报告。
   M0-F 以 2/3 seed、seed-equal aggregate 为正的 `provisional` 等级通过；M1-F 以 3/3 seed
   `robust` 通过。M0-R/M1-R 均有 target-free 输出变化，但 Full-512 相对 Recent-32 的质量
   主指标 CI 不为正，因此不形成可进入版本链的长期状态对象。M0-N 同样有输出变化但无稳定
   质量增量，保留为 No-op/负控制。完整结果见
   `configs/contracts/p7_8_h_qualification_result_v1.yaml`。
8. F 中 aggregate log loss、ROC-AUC、dislike PR-AUC 与 Brier 均通过冻结门，但 dislike-only
   log loss 在所有 seed 上变差，必须作为异质性同步报告。F history-position companion 的
   `never_seen`/`seen_only_before_512` raw 分层实现有误，已标为无效且不用于任何 gate；其余
   预注册 cohort 有效。见 `results/p7/h_qualification/post_reveal_audit_v1.json`。
9. P8 已完成：R0 blocking control 通过；R1 edge1、R1 edge2 与 R2 的六个训练、六个 raw
   score、封存与 H/S adjudication 均完成，且没有 admission rejection。R1/R2 均满足
   target-free edge-staleness candidate 门；M1-F R2 还出现稳定的 aggregate log loss、ROC-AUC
   和 dislike PR-AUC 损失。P8 结果是 development evidence，不能直接升格为 paper claim。
10. 现在仅授权 P9 tomography、dependency-closed partial action 与 No-op/Partial/Exact
    frontier。不得调 P8 基础链、训练 θ3、启动 controller 或按 P8 结果筛 seed。RecFlow 保留为
    以后真实 request/stage candidate 的外部验证，不是当前路线的前置条件。

## 资料规则

- 当前事实源依次为本文、[论文设计](paper_design.md)、[完整技术规格](newset.md)、冻结 contract。
- 每项结果必须记录 workload、lineage、snapshot rule、候选协议、seed、指标和证据等级。
- 不跨协议拼接历史结果；失效 artifact 保留原因，但不能进入论文表格。
