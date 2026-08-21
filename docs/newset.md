# EvoKV 37D 完整技术规格

更新日期：2026-08-21。本文保留 37D 的完整技术协议与协议演变；当前事实和执行授权以[项目全程 Compact](project_compact.md)与[当前路线](current_route.md)为准。本文中 P7 以前的 Q_main、neutral-readout、旧 oracle/ranker 数值仅用于解释协议为何被替换，不是当前系统结果。

> **执行状态更新（2026-08-21）。** P8 的 F-workload R0/R1/R2 release chain 已完成：R0
> 处于数值 identity floor，R1 两条边与 R2 均出现跨版本 S，且 M1-F R2 还出现稳定的 F quality
> 损失。基础链永久冻结；当前只授权 P9 staleness tomography、dependency-closed partial action
> 和 fidelity–work frontier。controller、θ3 与 blind qualification 仍暂停，见
> [当前路线](current_route.md)、[P8 结果摘要](p8_result_summary.md) 与 [P9 计划](p9_plan.md)。

> **P9 更新。** P9.0、P9.1 和 24/24 coarse tomography cells 已完成。最佳诊断区域可恢复
> 约 78%–99% stale error，但 arbitrary exact-KV splice 不是合法迁移动作。当前先补齐 splice
> quality companions 与用户风险集中度，再做预选语义格子的二维 map、依赖闭包和真实成本 frontier。

> **执行状态更新（2026-08-18）。** release evaluator 曾将传给
> `forward_with_cache` 的首个 suffix/readout token 的 time delta 重置为
> 零，而 Full 保留该 token 相对 cached prefix 末事件的真实时间间隔。故
> batch-fixed-v3 的 release-fidelity、multi-panel、frontier、controller 和
> dilution 数字均已降级为失效 development artifact，待 continuous temporal
> lineage 重跑；snapshot membership、candidate manifests、θ0 history sanity
> 和 synthetic same-model cache test 仍有效。当前路线是修复后先完成 State
> Staleness Tomography；controller freeze 和 θ3 均暂停。

## 1. 对象与 lineage

发布时刻 `T_t`，旧模型 `θ_(t-1)` 已生成 active snapshot 中用户的 prefix cache：

```text
C_u^(t-1) = F_(θ_(t-1), H_u^pre)
```

`H_u^pre` 包含 `T_t` 前按 parent model 在线服务并 append 的所有可用行为。发布后新行为由 `θ_t` 依时间顺序 append。合法四路语义：

```text
Previous Full        θ_(t-1), complete history
Current Full         θ_t, complete history; fidelity reference
Reuse                θ_(t-1) prefix KV + θ_t post-release append
Current Suffix Only  θ_t, post-release append only
```

`Current Suffix Only` 不是 empty-history；empty-history 只用于 θ0 sanity。

同模型 Full/Append 必须在冻结 precision tolerance 内一致。identity 与 output-only transition 不应产生 cache/hidden/score drift；cache-producer update 必须能产生可测 drift。

## 2. Population 与时间协议（通用规则；Q_main 内容为历史协议）

主总体是 `T_snapshot` 前已经物化的全部 exact-parent 状态，不是发布后恰好请求的用户。Yambda 主定义不设 TTL 或 recent-activity 门槛；这些仅用于 sensitivity cohort。冻结 snapshot 必须记录：

- snapshot cutoff；
- pre-release recent-activity（仅供分组，不作主总体过滤）；
- 最小 prefix 条件；
- catalog/OOV policy；
- raw 与 effective prefix lengths。

Yambda-50M v2 的时间单位是秒（5 秒精度），主 update window 为 1d，3d 仅 robustness。每条边的 release 必须位于其 update window 结束加 30 分钟后；update-window 行为属于 parent-model prefix，不属于 current-model suffix。

## 3. Two-manifest protocol（通用 label separation；候选细节为历史协议）

### Quality manifest

冻结 retriever candidates，可显式注入 positive，注明 conditional reranking。只用于 CE/AUC/NDCG/HR/MRR 和 future-label companion validation。

### Profiler manifest

仅使用 pre-release retriever 与固定 request/cutover probe；不注入 target，不以 future request 决定总体。用于 Current Full–Reuse semantic ground truth、oracle 与 risk-ranker development。

observed-event proxy 只用于外部相关性验证。它在严格下一发布窗口内覆盖的状态会偏向近期高活跃 cohort，因此该验证的结论必须限于 covered observed-event cohort；完整 materialized snapshot 的 cutover label 仍是 ranker 的开发标签。

此前的 panel A/B 是 popularity rank 的两个确定性、互不重叠切片，不是同一 proposal distribution 的独立抽样。所有 single-panel-A frontier 只保留为条件化 development evidence。主候选分布固定为 `Q_main_rank_decay_v1`：从同一、仅依赖 release 前信息的 rank-decay proposal 中独立无放回抽取 32 个 100-way target-free panels，0–15 为 development、16–31 为 held-out。主标签是 development panels 的 mean Top-K regret，CVaR90 是 companion。

在两条 development 边上，development/held-out panel halves 的风险排序 Spearman 为 0.960 / 0.951，Top-10% high-risk recall 为 87.4% / 88.1%，因此 `Q_main` 内的 multi-panel 测量可靠性门通过。严格下一发布窗口内的 observed-event proxy external validation 使用不重叠 panel halves，Spearman 为 0.165 / 0.243、Top-10% 富集为随机的 4.6× / 5.6×；它仅是 covered observed-active cohort 的有信息量但不精确 surrogate，不能被写成 serving-request 预测，亦不能外推至未观测状态。

## 4. Cutover canonical probe

zero-append 不能因模型 next-item readout 缺失而静默跳过。profile endpoint 使用固定 canonical readout token，在 Full 与 Reuse 上采用相同 readout；它只度量 fidelity，不被报告为推荐质量。

对 incremental execution，canonical readout 或首个 current append 的 temporal input 必须保留其相对 cached-prefix 末 token 的真实 time delta；将 append slice 单独编码时把首个 delta 重置为零会改变输入 embedding，违反“Full 与 Reuse 只差 prefix KV version”的 lineage 条件。当前 natural dilution 即使 readout token 类型相同，仍会改变其相对位置、时间间隔与上下文；因此它不能独立证明 state repair。`Current Suffix Only` 只有在与 Full 使用同一 canonical readout 和同一 temporal-input contract 时，才能作为长期 prefix utility 对照。

修复 temporal boundary 后，在两条 development edge 的 complete-case、无 512-token eviction cohort（126 / 128 states）上，固定 query/readout 实验得到：`k=0` 时 Full–Reuse regret 为 0.00122 / 0.00080，Full–Suffix Only 的 long-history contribution 为 0.41493 / 0.11971；加入一个 current-model event 后 long-history contribution 降至 0.02310 / 0.00311，而 Full 相对 latest-append-masked 的变化只有 0.00013 / 0.00001。当前能成立的解释是 **neutral-readout prefix bypass**：suffix-only 路径在一个 event 后已近似 Full，并非一个 token 将 stale KV 修复为 current KV。该观察只覆盖当前 4L/H128/512、one-hop、`Q_main` 和 neutral readout；下一门是 readout/residual 与 layer-position tomography。

需要额外报告 observed-event proxy 的 dilution curve，append count 为 `0,1,2,4,8,16`，并验证 canonical cutover risk 对首个发布后 observed listening event proxy 风险的预测能力。Yambda 不是 serving request log，不能把这一 proxy 写成真实请求。

## 5. Fidelity ground truth

对 Current Full score `s^F` 与 action score `s^a`，计算：

- Current-model Top-K regret（primary）；
- Top-10 overlap loss；
- margin-weighted pairwise disagreement；
- JS divergence；
- normalized score RMS；
- P95/P99 tails。

这些是 control-plane semantic fidelity，而非未来点击标签。future-label metrics 只能作为独立 validation。

## 6. Budgeted oracle

> **当前状态。** 本节以下的旧 oracle/scheduler 文字是 P8 前的技术草案，不构成当前授权，
> 也不得直接复用其数值或启动 controller。P9 先以 P8 的冻结 F release-chain 进行 tomography、
> dependency-closed action 与新的 fidelity–work frontier；只有 P9 的停止门通过才进入 scheduler。

固定 exact-equivalent work budget `b`，扫描：

```text
b ∈ {0, .1, .25, .5, .75, 1}
```

比较 Reuse All、Exact All、version-level uniform gate、Random、Longest-prefix、Activity-first、state-level Oracle。对单条边，version-level policy 只能是 Reuse All 或 Exact All；它不是中间预算下随机 refresh 一部分用户。所有方法在相同预算下比较 mean/P95/P99 fidelity、frontier area 与相对 best-heuristic/oracle gap，而非先选择一个业务 SLO。

诊断 operating points 可以显示 overlap-loss `0.2` 与 `0.5`，但不称为在线 SLO。

## 7. Scheduler development 与 qualification

旧 Q_main metadata ranker 和 frontier 已随协议失效永久降级，不能作为当前 scheduler 证据。当前 controller 尚未授权。

P9 必须先证明：

1. dependency-closed partial action 在同一 exact-equivalent work 下推开 No-op/Exact frontier；
2. state-level near-optimal allocation 明显优于 version-level uniform action；
3. 风险与 action recovery 对用户确有可预测异质性。

只有三项成立，才冻结 P10 controller contract。输入只能来自发布时可得的 prefix length、state age、pre-release activity、history statistics、old-KV cheap sketch、release type、cache-producing parameter delta 和已计费 probe。目标是每个 action 的边际 fidelity benefit/cost，不是 future-label safe/unsafe 分类。

方法、动作、成本和 controller 全部冻结后，才允许新时间边 blind qualification；不得根据 blind edge 调策略。

## 8. Executor 路线

1. no-op/exact baseline；
2. structured Fast Migration；
3. Selective Recompute（层、分片或状态区段）；
4. recursive lineage 与 state-version debt。

Fast/partial action的价值必须在相同 exact-equivalent work 下改善 fidelity frontier，而非通过 post-hoc score mixing 或 target-KV fitting。

## 9. 系统会计

主：token-layer/FLOP equivalent work ratio。

companions：KV read/write、history input、storage-tier transfer、worker-hours、pipeline makespan、rollout completion 与 state-version debt。对实际 GPU timing，要报告 batch、worker、同步方式与测量范围；不可把 batch-1 microtiming 直接外推为生产 P99。

## 10. 当前失效证据

- timestamp 乘五的 v1 日历审计失效；
- right-aligned medium batch 结果失效；
- 将 update-window start 当 release 的两边 screen 失效；
- batch-fixed v3 的旧 Q_main release-fidelity、frontier 与 controller 输出因 temporal-input contract 错误而失效；后来的 P7-P9 结果来自重新冻结的 N/R/F、F release chain 和正确 lineage。
- P7-P9 仍是 development evidence，尚非 paper qualification。

## 11. 禁止项

- 复活 D1/D2/D3 或 KuaiRand 八版本主线；
- 用 post-release served users 定义 migration population；
- 用 future request/append/target 作为主 scheduler feature；
- 将 Current Full 宣称为 future ranking upper bound；
- 将 diagnostic threshold 伪装成真实 production SLA。
