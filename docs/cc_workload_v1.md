# CC Workload Qualification v1

本文冻结 candidate-conditioned (CC) workload 的实现边界；它不产生实验结果，也不启动训练 runner。

## Query 与 scoring

每个 candidate 的 transient query token 是：

```text
candidate-item embedding
+ shared query-type embedding
+ RESERVED_QUERY_ACTION embedding
+ query-time embedding
```

query type/action 表与 behavior/PAD/MASK 表独立；`query_action_id` 由
`HSTUConfig` 显式配置，action 表不设置 padding index。query 只在评分时
append，不写入或改变输入的 persistent prefix KV。`score_cc_full` 先为每个
用户生成一次当前模型 prefix KV，再把每个 candidate 展平为独立的
`B×C` one-token 序列；`score_cc_reuse` 使用 parent prefix KV 执行相同的
current-model append。两路都使用相同的 query time、relative-time bias 和
scoring head，candidate 之间不能互相 attention。

公开的最小 API 位于 `hstu_kvcache.models.HSTU`：

- `embed_query_tokens`
- `score_cc_full`
- `score_cc_reuse`
- `conditional_reranking_loss`

## Candidate protocol

`build_q_main_rank_decay` 只接收已经按 causal cutoff 排序的 candidate IDs，
生成 target-free、无放回的 proposal，并记录每个 candidate 的
`proposal_rank`、`weight`、`log_q_main` 和 `causal_cutoff`。seen item 不会被
额外排除；负例只排除当前 positive。quality manifest 可以在模型查询之外
显式注入 positive，不能把 future label、organic 或 played-ratio 放入 query。

`conditional_reranking_loss` 是 candidate-panel 内的 CE，只表示
`Q_main`-conditional reranking，不声明 full-catalog 概率校准。

## Qualification gates

当前只允许实现并验证 CC-θ0。固定 development 时间边界、4L/H128/4 heads/
context 512、1-day schedule 和既有 optimizer/step/token/replay 规则后，先检查：

1. Full-512 相对 Empty 的 history sanity、非随机 conditional CE，以及用户
   bootstrap target log-prob 增益为正；
2. Full-512 相对 Recent-32 或 Last-2 在预注册连续质量指标上稳定为正。

Gate 2 未通过即停止，不训练 CC-θ1/θ2。通过后才可按 one-hop exact-parent
lineage 重训 CC-θ0 → CC-θ1 → CC-θ2，并严格分离 fidelity 与 quality manifests。

## P5 seen-aware 协议（2026-08-19）

当前停在 `Protocol Validity → H`。旧 v1/v2 quality gate 因 seen-membership shortcut 永久降级为 development diagnosis，不得恢复资格。合同见 `configs/contracts/cc_p5_seenmix_v1.yaml`。

四个冻结中的候选协议：

- `Q_train_seenmix_v1` / `Q_quality_seenmix_v1`：positive + 固定 recent-seen / old-seen / discovery 配额；历史已见 item 是 competing candidates，不是真实负反馈。
- `Q_fidelity_seenmix_v1`：相同配额，不注入 target，供未来 Full–Reuse 与 target-free (H)。
- `Q_history_matched_diag_v1`：按 seen stratum、count、recency、artist familiarity、popularity 匹配，只做机制诊断。

配额只按完整覆盖率冻结，不按 Full–Recent-32 gap 选择。跨层 backfill 禁止；填不满的请求直接丢弃。P5.1 选中 `16 / 24 / 59`，覆盖率 94.1%。P5.3 未通过且不得改判。

P6 在不改配额、不训练的前提下完成了 grouped cAUC、dropped-cohort、连续配平和无注入审计。连续匹配平衡了单特征 count/recency，但 holdout cAUC 仍为 0.736；自然检索到的 repeat target 同样可识别。裁决为 **Yambda next-item long-KV No-Go**。不得再为这个 next-item gate 人工设计 negatives。

## 当前 θ0 裁决状态（2026-08-19）

`cc_theta0_gate1_v1` 已通过；`cc_theta0_gate2_v1` 未通过长期历史资格门。
这不是 Last-1/Last-2-only 结论：Full 相对 Empty 的 target log-prob 增益为
`0.14669`，Full 相对 Last-2 为 `0.09016`，说明第 3--32 个最近行为仍有实质
贡献。Full 相对 Recent-32 仅为 `0.00290`，bootstrap 95% CI 为
`[-0.00470, 0.01041]`，`R_long=0.01974`。

冻结解释为：CC-θ0 是 history-conditioned reranker，当前可证实的有效范围是
中短期（最近 2--32 tokens）；第 33--512 个 token 的稳定边际价值尚未建立。
当前 qualification manifest 已因 Gate 2 观察而降级为 development evidence，
不得用于任何后续改变训练协议的独立资格门。此状态不授权 θ1/θ2、staleness、
tomography、controller 或 θ3。下一步仅是 long-horizon opportunity adjudication：
在不训练的条件下测有效长度、Old-480 输出干预、数据级机会和训练覆盖。

## H/S 与机制门

History utility 的路径至少包括 Empty、Last-1/2、Recent-8/32/128、Full-512、
Shuffled Full 和 Reuse Full。Shuffled Full 保留单调 timestamp slots，只置换
item/action 内容；reverse-content-on-fixed-time-slots 是 companion。

S 的资格地板由 identity、output-only、split-half 的联合 P99 测量定义，不能
使用事后分位数。H 先由群体 Full-512 相对 Recent-32 的 bootstrap 质量增益
定义资格，状态级 H 还必须超过预注册的 split-half 标准误倍数。只有 H/S
高状态稳定存在，才继续 fixed-query/natural-service dilution、layer×position
tomography、version-age 1/2 与 recursive mixed-lineage debt screen；否则如实
停止 EvoKV controller 路线。
