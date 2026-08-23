# EvoKV 阶段性完整结果总结（截至 P11.4）

更新日期：2026-08-22。

本文是当前项目的统一证据总结，覆盖从早期问题清理、Yambda workload
重构，到 P7–P11 的长期状态、跨版本陈旧性、合法迁移动作、预算调度和
真实 recursive lineage 验证。本文中的所有正面数字均属于
**development evidence**；尚未执行 θ3 blind temporal qualification，因此不能直接写成
paper-qualified 最终结论。

## 1. 一句话结论

项目已经建立了下面这条完整的 development 证据链：

```text
长期状态只在部分 workload 中有价值
→ 是否陈旧取决于模型发布语义
→ stale KV 在强发布条件下会损害实际任务质量
→ 陈旧误差具有用户、层和历史位置异质性
→ 依赖合法的 partial action 可以低于 Exact 成本恢复大部分 fidelity
→ 发布时不看未来请求，只用状态特征和 1% 计费 probe 也能分配动作
→ state-level Ridge scheduler 在同成本下稳定优于固定/元数据基线
→ 上述现象在真实跨两个版本的 recursive rolling-cache lineage 中仍成立
```

当前状态可压缩为：

```yaml
development:
  F_long_state_H: established
  cross_version_staleness_S: established
  output_only_noop_control: passed
  stale_state_quality_harm: established_strongest_in_M1_R2
  dependency_closed_partial_actions: implemented_and_validated
  full_population_frontier: established
  minimal_release_time_scheduler: frozen_and_positive
  true_recursive_version_debt: established
  recursive_scheduler_quality: positive_but_metric_and_seed_dependent

not_yet:
  8L_H256_context1024_scale_point: pending
  fixed_count_or_capped_probe_scale_sensitivity: pending
  theta3_blind_temporal_qualification: untouched
  external_workload_validation: pending
  production_storage_network_pipeline: not_claimed
  paper_qualified_conclusion: not_yet
```

## 2. 当前论文问题与系统边界

EvoKV 不负责修复一个训练失败或不应发布的推荐模型。它接收一个已经通过
独立发布质量门的新模型，在发布期后台预算内，为发布时已经物化的全部用户
状态选择演进动作：

\[
\min_{\{a_u\}} \sum_{u\in\mathcal S_t} L_u(a_u)
\quad\text{s.t.}\quad
\sum_{u\in\mathcal S_t} C_u(a_u)\le B_t.
\]

其中：

- \(\mathcal S_t\) 是发布 cutover 时全部已物化状态；
- \(L_u(a_u)\) 是动作相对 Current Full 的执行 fidelity loss；
- \(C_u(a_u)\) 包含 exact-equivalent token-layer work、KV I/O、raw-history read
  和实际执行时间；
- 当前冻结动作集合为 `No-op / Layer0-Recent128 / Layer0-Middle /
  Layer0-Full / Hybrid-Tail128 / Exact-All`。

必须持续区分两道门：

1. **模型发布门**：当前版本自身的 dev 质量是否足以发布；
2. **cache compatibility 门**：对于已经决定发布的模型，旧 persistent KV 是否仍
   兼容，以及需要哪种迁移动作。

Current Full 是当前模型完整执行语义的参考，不是 future recommendation quality
的理论上界。No-op 是一等动作，不是失败。

## 3. 当前实验平台

### 3.1 数据与规模

当前机制与系统开发平台是 Yambda-50M：

- 46,467,212 条 listening events；
- 9,238 个有效用户；
- 877,168 个观测 item；
- 144,441 个观测 artist；
- 时间跨度约 300.93 天；
- 用户历史长度 P50/P90/P99 约为 3,024 / 13,044 / 23,765。

当前模型规模为 `4 layers / hidden 128 / context 512`。该规模足以进行机制、
lineage、动作空间和 scheduler 开发，但尚不足以排除“小模型特例”。

### 3.2 N/R/F workload suite

项目不再要求一个 next-item protocol 同时证明所有命题，而是冻结三个职责不同的
workload：

- **N：Natural Next-Listen**。短期行为和 No-op 负控制；
- **R：3-day Return-to-Familiar**。完整 familiar candidate universe 上的条件排序；
- **F：Explicit Like/Dislike**。candidate-conditioned 显式长期偏好任务。

部署分数统一为：

\[
s_{\mathrm{deploy}}=\operatorname{stopgrad}(s_{\mathrm{base}})+s_{\mathrm{CC}}.
\]

Frozen Base 显式吸收 item/artist count、recency、popularity 和 proposal rank 等
低维合理信号；CC-HSTU residual 只需证明这些统计之外的高维历史增量。

### 3.3 M0 与 M1

- **M0-F**：F workload 的单任务 CC residual 模型，用于隔离显式反馈任务本身；
- **M1**：N/R/F 共享 persistent encoder、通过 query-type embedding 区分任务的
  多任务模型；M1-F 是共享状态 companion。

所有正式 development 条件保留 seed 17/37/71，没有按 H、S、quality 或 scheduler
结果筛选 seed。

### 3.4 发布类型

- **R0 Output-Only**：只更新不产生 persistent KV 的输出路径；
- **R1 Routine Continual Update**：warm-start 的正常全模型增量更新，两条连续边；
- **R2 Periodic Encoder Refresh**：同架构、同目标、累计数据上的完整 encoder refresh。

没有人工旋转、缩放或扰动 K/V，也没有根据 S 调整发布幅度。

## 4. 早期探索与为什么要重构路线

### 4.1 QK/QB 与 KuaiRand

早期探索证明了大 KV 容量和跨版本状态问题具有系统动机，但不少“漂亮 gap”后来被
降级或作废，原因包括：

- 特殊 history score 不是部署 raw score；
- candidate protocol 在 Full/Reuse 间不一致；
- latest item 或 suffix token 可以绕过长期 prefix；
- cache lineage 不合法；
- 模型更新本身降低未来质量；
- popularity prior 提高绝对质量却淹没 sequence residual；
- 逐边选择配置会形成适应性搜索。

这些旧数字不再作为当前论文证据。

### 4.2 Yambda 关键实现修复

Yambda 路线先后修复了四类会改变因果比较的问题：

1. timestamp 实际是秒单位、5 秒精度，而不是“5 秒 bin 编号”；
2. batch 右对齐真实 token，但旧 lengths 路径按左侧位置读取；
3. update-window 起点曾被误当作模型 release cutover；
4. incremental suffix 的首个真实 time delta 曾被错误重置为 0。

修复前的 release-fidelity、oracle、frontier 和 controller 数字永久失效。修复后
Full 与 Reuse 才真正做到只改变 prefix KV version。

### 4.3 Neutral readout bypass

Neutral readout 中，一个 current-model suffix 出现后 `Suffix Only` 几乎贴近 Full。
诊断表明这不是 suffix 修复了旧 KV，而是 readout 绕过了长期 prefix。因此 neutral
readout 只保留为 shortcut/negative control。

### 4.4 Next-listen candidate No-Go

P5/P6 证明 Yambda next-listen target 与 repeat item、artist familiarity、recency、
count 和 retrieval rank 高度纠缠。继续构造第五套 sampled negatives 的科学价值很低，
因此冻结：

```yaml
yambda_next_listen_sampled_reranking:
  long_kv_qualification: no_go
candidate_engineering_on_old_next_item:
  permanently_stopped
yambda_state_evolution_platform:
  retained
```

失败的是该协议实例，不是 Yambda 的时间链、snapshot、lineage 和系统规模能力。

## 5. P7：长期状态 H 资格

P7 完成了 compact manifests、Frozen Base、M0/M1 θ0、三 seed 和封存 qualification。
主比较为同一冻结 checkpoint 的 `Base + Full-512` 与 `Base + Recent-32`。

结果：

- N 未通过长期 H，符合短期/No-op 负控制角色；
- R 未通过长期质量门，Frozen Base 已解释大部分 familiar-return 信号；
- F 第一次通过长期状态资格。

F 的核心数字：

| 模型 | Full-512 相对 Recent-32 的聚合 log-loss 增益 | 95% CI | seed 方向 |
|---|---:|---:|---:|
| M0-F | 0.00204 | [0.00034, 0.00341] | 2/3 正 |
| M1-F | 0.00279 | [0.00139, 0.00466] | 3/3 正 |

P7 支持的命题是：

> 在 Frozen Base 已经吸收 count、recency、popularity 等简单统计后，Explicit
> Feedback 中 Recent-32 之外的历史仍为 CC residual 提供小但可重复的增量价值。

它不支持“所有推荐 workload 都需要长期 KV”。

## 6. P8：跨版本陈旧性 S 与发布语义

P8 在 F 上运行 M0-F/M1、三个 seed 和 R0/R1/R2。所有候选版本均通过既有 model
admission。

### 6.1 Target-free S

| Release | M0-F：Current Full vs Reuse JS | M1-F：Current Full vs Reuse JS | 解释 |
|---|---:|---:|---|
| R0 output-only | ≤ 4.44e-15 | ≤ 4.44e-15 | cache producer 未变，可 No-op |
| R1 edge1 | 9.07e-5 [5.64e-5, 1.53e-4] | 5.38e-5 [2.40e-5, 9.16e-5] | routine update 产生 S |
| R1 edge2 | 3.61e-4 [3.83e-5, 9.95e-4] | 2.67e-5 [1.28e-5, 4.44e-5] | 第二条 routine edge 复现 |
| R2 refresh | 4.02e-4 [1.01e-5, 6.23e-4] | 1.35e-3 [4.79e-5, 3.75e-3] | refresh 风险更强 |

R0 的 `4.44e-15` 是强负控制：lineage、候选和评分路径没有凭空制造差异。

### 6.2 质量影响

最强质量证据位于 M1-F R2。Current Full 相对 Reuse Parent KV：

- log loss 改善 `0.003274`；
- ROC-AUC 提高 `0.01331`；
- dislike PR-AUC 提高 `0.01773`；
- Brier 改善 `0.000999`。

P8 因此建立：

> 长期状态可能有价值，但是否需要迁移取决于 release semantics。Output-only 发布
> 可以安全 No-op；改变 cache producer 的 routine update/refresh 会产生 S；强 refresh
> 下 stale KV 可以伤害实际任务质量。

R1 的 fidelity S 稳定存在，但多数质量 CI 较小或跨零，不能写成“每次 routine update
都稳定损害业务质量”。

## 7. P9：从诊断结构到合法动作和成本 frontier

### 7.1 Tomography 与用户异质性

粗粒度 layer/segment tomography 显示：

- 最佳诊断区域恢复约 `78%–99%` stale error；
- recent-128 在全部非 R0 条件和 seed 上正恢复，约 `45%–97%`；
- recent-32 约 `13%–42%`；
- recent-1 仅约 `1%–2%`；
- 部分单层 splice 会出现负恢复。

非 R0 条件的 Top 10% 用户贡献约 `38.6%–49.1%` 的总 S。二维扫描中，M0-F R2
的 `layer0 × middle` 为 3/3 seed 正，恢复 `92.06%` JS，并同步改善：

- log loss `0.001231`；
- ROC-AUC `0.002963`；
- dislike PR-AUC `0.003396`；
- Brier `0.000310`。

但 M1 和部分 R1 条件具有强 seed 异质性，不能声称存在一个跨模型、跨 seed 的通用
最佳 layer×position 格子。

### 7.2 Dependency closure

任意 exact-KV splice 只是诊断干预。合法动作必须能从真实保存的 state 和 raw history
获得全部输入依赖。最终实现的最小合法动作包括：

- `Layer0-Recent128`：只重投影最近 128 个 layer-0 K/V；
- `Layer0-Middle`；
- `Layer0-Full`；
- `Hybrid-Tail128`：保留旧前缀，使用当前模型因果 replay 完整尾段；
- `Exact-All`。

Layer-0 K/V 只依赖当前位置输入 embedding、normalization 和 K/V projection，因此可
独立刷新；上层任意 segment splice 通常缺少下层 current hidden，不能伪装成可部署动作。

### 7.3 真实 rolling lineage

request-local 重算曾发现约 `86.5%–87.0%` 请求发生 rolling eviction，因此不能代替
release-cutover 物化状态。真实执行器改为：cutover 只物化一次，之后严格逐事件 append，
并在每次 append 前执行 cap-512 eviction。

24-cell rolling validation 中：

- R0 所有动作仍为精确零；
- R2 M0-F 的 Recent128/Middle/Layer0-Full/HybridTail128 恢复
  `80.4% / 95.6% / 99.2% / 83.4%`；
- R2 M1 为 `89.1% / 92.3% / 89.4% / 94.7%`；
- R1 的最佳动作随 model/edge 变化。

### 7.4 全人群 cutover

全人群由发布时信息定义，而不是由未来会发请求的用户定义：

- edge1：8,229 个已物化状态；
- edge2：8,488 个已物化状态；
- 未来 F served users 仅占约 36%。

P9.8 在 24 个 `release × model × seed` cells 上封存了 19,108,800 条
candidate-action raw rows。代表性合法动作结果：

- HybridTail128 在全部非 R0 cells、全部 seed 正恢复；
- R1 聚合恢复约 `42.8%–58.6%`；
- R2 M0-F/M1 分别恢复 `85.5% / 94.2%`；
- 其逻辑工作约为 Exact token-layer work 的 `27.2%`；
- 某些 release-specific Layer0-Full 用约 25% 工作恢复 `98%–99.6%`，但并非所有
  seed 都安全。

全人群 No-op 风险 Gini 为 `0.316–0.803`，Top 10% 状态贡献约 `18.4%–70.5%`
总风险。这证明存在 state-level allocation 机会，但当时的风险排序读取 CurrentExact，
只能作为 offline oracle。

### 7.5 Heldout quality 与成本

P9.9 共完成 469,176 次 heldout request evaluation。M1-R2 的 No-op log-loss harm
为 `+0.001779 / +0.000619 / +0.003486`；HybridTail128 将其降至
`+0.000499 / +0.000308 / +0.000177`，恢复约 83.3% 的 equal-seed aggregate
log-loss harm 和约 94% 的 AUC harm。

逻辑成本：

| Action | Exact-equivalent token-layer work |
|---|---:|
| Layer0-Recent128 | 约 6.48% |
| Layer0-Middle | 约 12.50% |
| Layer0-Full | 约 25.00% |
| HybridTail128 | 约 25.92% |
| Exact-All | 100% |

原型单状态 runtime 约为 `0.042 / 0.040 / 0.040 / 0.185 / 0.474 ms`。全人群
HybridTail128 单 GPU kernel rollout 为约 `2.22–2.35 s`，Exact 为
`3.91–4.09 s`。逻辑 work 与 wall time 不相等，原因包括小 batch、ragged lengths、
cache clone 和 kernel launch overhead。

## 8. P10：未知未来请求时的 profiler、scheduler 与 executor

### 8.1 发布时可用信息

P10 回答了一个关键部署问题：未来哪些用户会请求、请求什么 item 都未知时，能否在
cutover 预先决定迁移动作？

冻结方案只读取：

- effective prefix length；
- state age；
- 发布前 1/7/30 日 activity；
- 7 日 unique items；
- organic ratio 与 repeat ratio；
- 对确定性抽取的 1% 状态执行 target-free 多动作 probe。

预测器固定为 `StandardScaler + Ridge(alpha=1.0)`，预测每个动作相对 No-op 的
边际 fidelity benefit；probe 的所有动作成本从发布预算中扣除。没有 future request、
label、target KV、GBDT 或阈值搜索。

### 8.2 Target-free scheduler frontier

1% probe 的非 R0 equal-seed 平均恢复率：

| Release / Model | 5% budget | 10% budget | 25% budget |
|---|---:|---:|---:|
| R1 edge1 / M0-F | 38.0% | 54.1% | 72.9% |
| R1 edge1 / M1 | 41.6% | 60.8% | 87.9% |
| R1 edge2 / M0-F | 39.7% | 54.9% | 78.5% |
| R1 edge2 / M1 | 41.3% | 67.5% | 95.4% |
| R2 / M0-F | 54.1% | 79.2% | 95.8% |
| R2 / M1 | 38.1% | 58.2% | 83.4% |

R0 通过 release metadata 直接选择全体 No-op、零 probe、零迁移。2% probe 完整报告，
但额外 probe 成本在低预算下通常抵消更多样本的收益，因此冻结 1% 为 development 主配置，
不宣称它是普适生产比例。

### 8.3 封存后质量

1,194,300 条 assignment 在连接 label 前封存。最重要的 M1-R2 中，1% probe 的
No-op→Policy log-loss 改善为：

| Budget | seed 17 | seed 37 | seed 71 | 平均 |
|---|---:|---:|---:|---:|
| 5% | +0.000229 | -0.000146 | +0.003879 | +0.001321 |
| 10% | +0.000957 | -0.000118 | +0.003232 | +0.001357 |
| 25% | +0.001009 | +0.000187 | +0.003332 | +0.001509 |

25% budget 为 3/3 seed 改善 aggregate log loss。

### 8.4 同成本非学习基线

Ridge 与最佳 release-level uniform、可行 uniform partial、random Exact，以及按
prefix length/state age/activity/unique items 排序的 zero-probe Exact 比较。

5% 预算下，六个非 R0 `release × model` 条件中 Ridge 相对每格最强非学习基线的
三-seed平均优势为 `24.9–43.0 pp`。六格均为 3/3 seed 正，且 paired-user
bootstrap CI 为 3/3 正。

这证明 state-level scheduler 的价值不是“比 No-op 好”这么弱，而是在相同总成本下
优于简单发布级或元数据策略。其主要价值位于低预算区；高预算下固定 partial 已经很强。

### 8.5 Mixed executor 与 full-stack freeze

初版 mixed-policy GPU rollout 为 Exact runtime 的 `36.4%–72.2%`，节省
`27.8%–63.6%`。随后只做不改变任何 UID action 的 grouping/clone/buffer 优化：

- 128-state canary 的 per-UID K/V 最大差 `2.62e-6`，低于 `1e-5`；
- 10/10 runtime 条件加速 `1.12×–2.56×`；
- 几何平均加速 `1.60×`；
- operation batch 数减少 `21%–61%`；
- 最终相对 Exact-All runtime 节省 `35.8%–80.4%`。

P10.6 据此冻结 development full stack：1% deterministic probe、固定 Ridge、六动作
allocator、预算计费与 grouped executor 不再调整。

## 9. P11：真实 version debt 与 recursive scheduler

### 9.1 One-hop、Direct Age-2 与 Recursive debt

三种状态不能混写：

- **One-hop**：θ1 在 edge2 当前前缀上生成 KV，由 θ2 读取；
- **Direct Age-2 diagnostic**：θ0 在 edge2 当前前缀上重算 KV，由 θ2 读取；它不是
  可部署 lineage；
- **Recursive mixed**：edge1 只物化一次 θ0 KV，θ1 服务期逐事件 append/evict，最终
  由 θ2 读取；这才是连续 No-op 后的真实状态债务。

M1 seed17、32 用户 canary：

| Lineage | Mean MSE | Mean JS | P95 JS |
|---|---:|---:|---:|
| One-hop θ1 | 0.006707 | 0.0001884 | 0.0004214 |
| Direct θ0 diagnostic | 0.008604 | 0.0002344 | 0.0007027 |
| Recursive θ0→θ1 mixed | 0.007211 | 0.0001950 | 0.0006133 |

Recursive debt 比 one-hop mean JS 高约 3.5%，而伪 direct-age2 高约 24%，说明直接用
θ0 重算 edge2 prefix 会夸大真实债务。

### 9.2 全人群 recursive lineage

P11.1 完成 M0-F/M1 × 三 seed 六格，每格 8,229 用户，并逐格 replay 2,535,994
个中间事件。Recursive Exact 与 Current Full 的最大 logit 差为 0。

合法动作相对 Recursive No-op 的恢复：

| Model | Action | 三 seed平均恢复 | 最低 seed |
|---|---:|---:|---:|
| M0-F | Hybrid-Tail128 | 51.60% | 42.96% |
| M0-F | Layer0-Full | 57.61% | 17.02% |
| M0-F | Layer0-Middle | 51.33% | 17.56% |
| M0-F | Layer0-Recent128 | 33.93% | 6.43% |
| M1 | Hybrid-Tail128 | 55.62% | 52.83% |
| M1 | Layer0-Full | 97.45% | 96.48% |
| M1 | Layer0-Middle | 77.47% | 75.94% |
| M1 | Layer0-Recent128 | 54.13% | 53.82% |

M1 的结构跨 seed 非常稳定。M0-F 更异质：seed17 的 recursive mean JS 为
`0.004498`，远高于其他 seed，并且 layer0-only recovery 较弱；该 seed 被完整保留。
Top 10% 用户贡献 `30.8%–54.6%` 的 recursive No-op JS。

### 9.3 Frozen scheduler 在 recursive lineage 上的同成本结果

P11.2 沿用冻结算法，而不是沿用旧 edge2 的动作 assignment：每次发布仍按同样的
1% deterministic probe、固定 feature set、固定 Ridge 和固定 allocator 重新校准。
assignment 在同成本裁决前封存。

| Model | Budget | Ridge recovery（seed17/37/71） | 相对最强确定性基线的 seed平均优势 |
|---|---:|---:|---:|
| M0-F | 5% | 29.8% / 35.8% / 37.2% | +18.2 pp |
| M0-F | 10% | 41.0% / 56.4% / 57.7% | +11.2 pp |
| M0-F | 25% | 69.4% / 81.7% / 88.6% | +20.2 pp |
| M1 | 5% | 46.3% / 48.5% / 32.9% | +28.1 pp |
| M1 | 10% | 71.2% / 66.8% / 60.3% | +12.0 pp |
| M1 | 25% | 95.3% / 93.3% / 95.4% | +17.3 pp |

M0-F 和 M1 在全部三个预算点均为 3/3 seed 优于最强确定性非学习基线，而且每个
seed 的 paired-user bootstrap CI 均为正。Random Exact 的恢复约等于预算比例，说明
结果并非“随便迁移一些用户”即可得到。

### 9.4 Recursive rolling quality

P11.4 在六格中分别评测 3,015 用户、18,959 条真实显式反馈请求。raw action logits
先封存，再与 P11.2 assignment join；Recursive Exact 与 Current Exact K/V 精确一致。

M1、1% probe 的 No-op→Policy aggregate log-loss 改善：

| Budget | seed 17 | seed 37 | seed 71 | 平均 | 正 seed |
|---|---:|---:|---:|---:|---:|
| 5% | +0.000225 | -0.000048 | +0.000223 | +0.000133 | 2/3 |
| 10% | +0.000209 | +0.000008 | +0.000195 | +0.000137 | 3/3 |
| 25% | +0.000102 | +0.000121 | +0.000673 | +0.000299 | 3/3 |

M1 的 ROC-AUC 在 10%/25% 预算均为 3/3 seed 改善，平均绝对增益分别约
`0.000483 / 0.001641`。M0-F aggregate log-loss 的 seed平均在三个预算均为正，
但单 seed 方向只有 1/3–2/3 正，异质性更强。

Dislike-only log-loss 在两模型的每个主预算中均有 2/3 seed 恶化。Exact-All 本身相对
Recursive No-op 在许多相同 seed 上也同方向恶化，说明很大一部分来自当前模型完整语义
和 rare-class calibration，而不是 scheduler 单独制造。但 EvoKV 仍不能声称 aggregate
fidelity 自动保证每个业务切片质量。

## 10. 当前已经验证了哪些论文假设

| 假设 / 研究问题 | 当前裁决 | 证据范围 |
|---|---|---|
| 某些合理推荐 workload 会使用长期高维状态 | **是** | F；M0-F 聚合正、M1-F 3/3 seed |
| 所有 workload 都需要长期 KV | **否** | N/R 未通过，是合法 No-op/短状态区域 |
| 所有模型发布都会使 KV 过时 | **否** | R0 最大 JS 4.44e-15 |
| 改变 cache producer 会产生 query-visible S | **是** | R1 两边和 R2，三 seed development |
| stale KV 可能伤害任务质量 | **是** | 最强在 M1-F R2；R1 更弱且混合 |
| S 在用户、层和历史位置上有结构 | **是** | tomography、Gini、Top-K risk |
| 诊断局部恢复能变成依赖合法动作 | **是** | Layer0 与 HybridTail executor |
| Partial 能以低于 Exact 的成本恢复 fidelity | **是** | P9 logical/runtime frontier |
| 不知道未来请求时仍能决策 | **是** | cutover features + 1% target-free probe |
| 用户级学习策略优于同成本固定/元数据策略 | **是** | P10 六条件；P11 recursive 两模型，3/3 seed |
| 连续 No-op 后 version debt 仍存在 | **是** | P11 true recursive θ0→θ1→θ2 lineage |
| Recursive scheduler 能恢复实际质量 | **部分成立** | M1 aggregate/AUC 较稳定；M0、PR 与 dislike slice 混合 |
| 方法可跨模型规模和未见时间边泛化 | **尚未证明** | 需 8L scale point 与 θ3 blind edge |
| 当前实现是生产级迁移系统 | **尚未声称** | 当前为 PyTorch prototype 与 development executor |

## 11. 当前最准确的论文叙事

目前可以支持的克制表述是：

> Persistent recommendation state compatibility depends jointly on query
> semantics, model release semantics, and individual state history. Natural
> short-term queries and output-only releases can remain compatible, while
> long-horizon explicit-preference state becomes stale after cache-producing
> updates. EvoKV uses release metadata, sparse target-free probes and
> dependency-closed state actions to allocate a bounded migration budget,
> recovering substantially more Current-Full fidelity than uniform or
> metadata-only policies at the same work.

目前不能写成：

- 所有推荐 workload 都需要迁移长期 KV；
- 每次 routine update 都稳定损害业务质量；
- Current Full 在所有未来 ranking 指标上都是理论最优；
- 1% probe 是所有生产规模的普适比例；
- aggregate fidelity 恢复保证 dislike 等所有切片改善；
- 当前 development 数字已经通过 blind/paper qualification。

## 12. 证据纪律与主要限制

1. **Development vs qualification**：θ0–θ2 已用于方法开发；θ3 仍完全未打开。
2. **Seed 是重复单位**：三个 seed 全部保留，不能用更多 request 代替独立训练重复。
3. **F 是唯一主长期 workload**：N 是负控制，R 未通过；不能事后重新设计旧 next-item
   candidates。
4. **Rare dislike caveat**：必须持续报告 dislike PR-AUC 和 dislike-only log loss。
5. **模型规模**：当前只验证 4L/H128/context512、Yambda-50M。
6. **发布覆盖**：R0/R1/R2 是受控 development release family，不代表所有工业更新。
7. **成本范围**：已测 token-layer work、KV/history bytes proxy 和 batched GPU kernel
   runtime；尚未完成生产存储、网络、容错和多机 rollout。
8. **外部真实性**：RecFlow 等真实多阶段 request/candidate 数据仍是后续外部验证，不是
   当前路线的前置救命数据集。

## 13. 下一阶段

在不修改当前 development 方法的前提下，推荐顺序是：

1. **EvoKV v1 full-stack freeze**：封存六种动作、1% probe、features、
   StandardScaler + Ridge(alpha=1.0)、5%/10%/25% budgets、grouped executor、
   三 seed、合同、checkpoints 和 raw seals，禁止再修改方法。
2. **规模点**：运行预注册的 `8L / H256 / context1024`，只复现 F 的 H、R0/R1/R2
   的 S、冻结 partial/scheduler，不重新搜索动作。
3. **Probe scale sensitivity**：完整报告 1%、fixed 64/128/256 与预冻结 capped-rate，
   避免把开发人口上的 1% 当作无限扩展规则；该实验不重新选择主配置。
4. **θ3 blind contract**：在训练/查看 θ3 前冻结 release recipe、model admission、H/S、
   cost/fidelity gate、quality non-inferiority 与 dislike safety companion。
5. **一次性 θ3 qualification**：不调参运行 Frozen EvoKV；结果揭示后停止并裁决。
6. 通过后再考虑更大数据规模、外部 workload 和论文主表；若 blind edge 不通过，应报告
   泛化边界，而不是回到 θ3 上调 predictor/action。

## 14. 关键文档与产物

- 当前路线：`docs/current_route.md`
- P7 workload：`docs/archive/yambda_stateful_workload_suite.md`
- P8 H/S：`docs/archive/p8_result_summary.md`
- P9 tomography/closure：`docs/archive/p9_2_result_summary.md`、
  `docs/archive/p9_2_closure_result_summary.md`、`docs/archive/p9_3_result_summary.md`
- P9 legal executor/frontier：`docs/archive/p9_5_rolling_validation_result.md`、
  `docs/archive/p9_8_full_population_result.md`、`docs/archive/p9_10_11_frontier_result.md`
- P10 profiler/scheduler：`docs/archive/p10_0_1_profiler_result.md`、
  `docs/archive/p10_3_4_scheduler_freeze.md`、`docs/archive/p10_5_6_full_stack_freeze.md`
- P11 recursive lineage：`docs/archive/p11_0_version_debt_canary.md`、
  `docs/archive/p11_1_recursive_population_result.md`
- P11 recursive scheduler/quality：`docs/archive/p11_2_3_recursive_scheduler_result.md`、
  `docs/archive/p11_4_recursive_policy_quality_result.md`
- 最新 P11 target-free adjudication：
  `results/p11/p11_3_recursive_scheduler_baseline_gate_v1.json`
- 最新 P11 quality adjudication：
  `results/p11/p11_4_recursive_policy_quality_v1.json`

## 15. 最终阶段性判断

当前项目已经不再停留在“是否可能存在 stale KV”的动机阶段。它已经完成了：

```text
可证伪 workload
→ H
→ release-dependent S
→ quality harm
→ localized recoverability
→ dependency-closed executor
→ full-population measured frontier
→ label-free sparse profiler
→ same-cost scheduler gate
→ recursive version-debt and rolling-quality validation
```

因此，EvoKV 的**问题、观察、动作空间和最小 development 方法已经基本闭环**。剩余的
主风险不再是“实验对象是否存在”，而是：

> 冻结方法能否在更大模型和完全未查看的 θ3 时间边上复现，并在 aggregate quality、
> rare-class safety 和真实系统成本之间保持可辩护的 frontier。
