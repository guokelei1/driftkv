# Archived: Yambda Multi-Regime Stateful Workload Suite

本文定义 P7 的执行边界。它把 P6 的失败限制在旧的 natural
next-listen sampled-reranking 实例，不把该失败外推成 “Yambda 不适合作为状态演进平台”。

```yaml
yambda_next_listen_sampled_reranking:
  long_kv_qualification: no_go
yambda_as_state_evolution_platform:
  status: retained
paper_hypothesis:
  status: unresolved
```

旧 next-item candidate engineering 永久停止。P5/P6 的门槛、配额和失败结论均不改判。
P7 改为 assumption-conditioned systems evaluation：先事前定义一个确实使用长期持久化
状态的推荐任务，在未查看 H/S 的数据窗上验证其行为与覆盖，再研究跨版本状态债务。

## 三种 workload 角色

- **N — Natural Next-Listen**：保留旧自然任务作为短期、repeat/recency 主导的负控制；
  `H≈0` 或 `S≈0` 是 No-op region，不是需要修补的失败。
- **R — Return-to-Familiar**：在按 query-time inactivity gap 定义的 session start，
  对历史中多个 familiar items 做 competing-candidate reranking。正例仍是实际发生的
  return 行为；其他候选不声称是真实负反馈。
- **F — Long-Horizon Feedback**：优先使用严格因果的 like/dislike query；若覆盖或
  shortcut 门不通过，才另立 future-window preference 合同。两者不能按 H 大小选择。

所有任务共享一份 persistent prefix state。轻量 base ranker 显式承担 item/artist count、
recency、popularity 和 proposal rank；CC-HSTU 只输出 candidate-history residual。部署分数
固定为 `s_base + s_CC`，不设置可事后调节的 residual 系数。Full/Recent 比较中 base score
必须逐候选完全相同。

## 当前授权

P7.1 全量只读审计已经完成。9238 个 listen users 上，五个注册 gap 均达到 coverage 门，
因此按“最大合格 gap”规则冻结 R 为 3 天：rankable train/dev/qualification 请求分别为
25,639 / 1,189 / 1,281；familiar candidate 中位数为 278 / 327 / 329。简单特征的 optimistic
best-feature target midrank 约为 0.14，说明它们很强，应进入 base ranker，而不是被当作失败门。

显式反馈也通过预注册门：causal dev/qualification query 为 18,668 / 19,967，其中 dislike
为 1,886 / 1,865；latest-item shortcut 为 11.9% / 12.3%。约三分之一 label 在此前 30 分钟
有同 item listen，且约 8%--9% 与 listen 同 timestamp；前者必须作为 cohort 完整报告，后者
由严格 `listen_timestamp < feedback_timestamp` 规则排除在 persistent prefix 外。

审计没有训练模型、生成 workload manifest 或读取 H/S。R 已按 coverage-only 规则冻结为
3 天，F 已按标签合法性、覆盖和 shortcut 门冻结为 explicit like/dislike。精确 workload
语义见 [P7.2 合同](../../configs/contracts/p7_workload_suite_v1.yaml)。

## 后续门序

P7.3 的最小 primitives 已实现：冻结线性 base scorer 的系数与标准化量不进入 optimizer，
部署组合 API 只接受 `base + residual`；R 的 target-free familiar universe 和 F 的严格因果
feedback query 由独立小模块构造。聚焦回归与现有测试合计 40 项通过。
P7.4 已完成此前待冻结的 base fitter/regularization、细分训练窗覆盖与
variable-size R candidate batching：R residual-train 有 8,407 个
eligible requests，其中 3,939 个 quality-evaluable familiar returns、2,728 个独立用户；
完整候选集 P99 为 503、最大 512，无需 cap。M0/M1 每任务预算据此冻结为 `K=3939`。

P7.5 六份 quality/fidelity canary 各包含 128 个独立用户并通过：所有 history 严格 causal，
fidelity 不含 label/target 字段，R 不采样且 target 自然出现一次，F 排除 coincident listen。
这些只是接口证据，不是质量或 H 证据。user-level evaluation sampling、streaming base fit、
Parquet sharding 和 retention 已在 `p7_5_materialization_contract_v1` 中冻结；下一步只授权
生成和封存 compact manifests，仍不授权 base fitting 或训练。完成 manifest conservation/hash
review 后才可拟合 base，随后才可只训练 θ0 的 M0/M1，完整报告 N/R/F 的
Base Only、Recent-32、Full-512 和 Compact Summary。

只有至少一个事前冻结的 R/F workload 同时通过 rank-sensitive quality H 与 target-free H，
才授权 R0 output-only、R1 routine continual、R2 periodic encoder refresh 的 development
版本链。最终证据对象是 `workload × release type` 的 H–S 矩阵，而不是要求每个 workload
或每条发布边都产生正 gap。
