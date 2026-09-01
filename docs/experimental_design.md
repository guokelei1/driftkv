# EvoKV HSTU-Native 路线与论文具体实验设计

更新日期：2026-08-28

本文是当前 HSTU-native 路线 compact 和具体实验设计。它记录仓库当前实际保留的模型、数据、
训练、评测入口，并把后续研究分成三个层次：一次 release refinement、连续多 release state
evolution 和 GPU transformation runtime。已经观察到的数字只写入
[核心 Motivation 与 Observation](motivation_observations.md)；本文不把尚未验证的 continuous
policy、runtime 或阈值写成结论。

## 0. 当前状态与下一阶段核心

当前仓库已经形成可复现的 HSTU-native motivation 链条：

~~~text
HSTU 架构代码
  -> Yambda-500M 时间因果数据与固定人口
  -> foundation / release candidate 训练
  -> Full-only release admission
  -> Current / Parent / Recompute / One-hop Reuse rolling 对照
  -> D14/E14 motivation 与 long-age direct Reuse observation
~~~

当前主实验是 Yambda-500M Small、4L/H128/context512、seed 17 的递归 release chain。D=14/E=14 的五条连续边已经产生稳定 motivation：四条常规正收益边的 One-hop Reuse 侵蚀为 25.5%–47.9%，producer version age 的 direct Reuse 损失严格单调增加。v3→v4 的 Full-only 收益很小，不能把其放大的 418.7% 比例当作典型效果。完整表格和边界以 [核心 Motivation 与 Observation](motivation_observations.md) 为准。

当前 3,000-user recommendation-state observation 将主 Insight 收敛为：persistent HSTU state 是被
candidate bank 重复消费的 user-evidence compatibility field；early version drift 含跨用户共享的
typed item/action coordinate，aggregation 与 U gate 将剩余变化转成 user-context residual；compact
state 必须保留 evidence mass，而 raw item/action equality 不足以定义可合并关系。此前的跨层传播、
CAST/PATCH 互补和 GROUP/SCALE 消融保留为机制证据，并推导
`PLAN -> CAST -> GROUP/PATCH -> SCALE/COMMIT` Design 0；
[`CAST / PATCH / GROUP / SCALE`](typed_state_refinement_algebra.md) 继续作为底层 plan IR。
该 candidate-shared 结构已进一步通过 signed、逐 head、四种 bank width 和真实 exposed candidate
的五边 causal 复核；shared-only 在真实分布的 20/20 edge×width 组合中优于 residual-only。

现有机制证据首先推出一次 `Parent -> Current` Design 0 strong baseline。事前固定的
`CAST384 + GROUP/PATCH 128->64 + SCALE2` 已完成五条 full-population rolling AUC：4/5 edge
改善 Reuse，v4->v5 失败。它证明 Design I 有可执行任务收益，但不准入一个对所有 release 无条件
采用的固定配置。完整计划的保守理论 FLOPs 为 Exact-All 的 48.0%，理论减少 52.0%；该值不预言
GPU runtime。专家建议的第一轮最小机制已在独立 prospective contract 下执行：matched-cost
`CAST signed value measure + Current anchor residual` 在五边 label-free canary 上 0/5 不弱于
Design 0，按合同停止，未进入 formal AUC/log-loss。随后 reader-stage/persistence 证据将主对象收敛为
per-user AV correction；旧 compact-probe score canary 4/5 通过但生成成本仍约 Full 的 40.5%。
最新 lightweight PRO 已取消 per-position translated-prefix 物化：joint map 只进入一次固定 probe，
recent-128 固定 32 个 Parent-conditioned Current carrier。held-out v2 正确性/成本门通过，理论 FLOPs
为 Full 的 9.1%，只写 512 scalar。机制冻结后的五边全人口 quality 已完成：AUC 5/5 正向、
log-loss 3/5 正向，五边均值两项均改善。总体 Design viability 通过；事前严格双门未过，因此
serving admission、额外 seed 和 runtime qualification 仍锁定。
最新 label-free progressive follow-up 也已完成：C32 error decomposition 不支持纯幅值或
segment-decay 解释；held-out C32/C48/C64 frontier 中 C64 对 C32 的 relative L2 为 cutover/rolling
各 5/5 改善，但 rolling direction 为 0/5 达到 0.90 且 C48/C64 非单调。按冻结规则不选择升级，
不在既有五边重读 quality label。

One-Release 通过后，研究重心转向 Continuous：以 estimated compatibility debt 和最大近似深度
限制 `v0 -> v1 -> v2 -> ...` 的误差累积，再用 sampled Current-Exact shadow 检测假设失效并触发
加固或 Rebase。GPU Runtime 最后实现，只在当前阶段预留 typed plan 和 controller 接口。

当前阶段的研究顺序固定为：

~~~text
Background & Motivation
  -> short System Overview
  -> Insight-Driven One-Release Refinement
     -> 3,000-user candidate-broadcast / typed-coordinate observation (complete)
     -> Design 0 strong baseline and per-user AV correction
     -> lightweight PRO correctness/cost and full rolling quality (complete, mixed-positive)
  -> Debt-Bounded Continuous State Evolution
     -> bounded debt / Exact shadow / quality-triggered rebase
  -> GPU Transformation Runtime
     -> plan lowering / batching / I/O pipeline / atomic commit
  -> scale and external validation
~~~

因此，当前只冻结 One-Release mechanism semantics 和 plan-IR 语义。不冻结 `tau_patch`、
carrier-density 安全阈值、profiler、scheduler、multi-release debt/rebase policy 或 GPU executor，
也不把旧 Yambda-50M development 的具体动作当作最终方案。

## 0.1 当前仓库的复现入口

当前主链对应的代码和输入如下：

- HSTU 与 persistent K/V：`src/hstu_kvcache/models/`；
- Yambda 时间窗口、人口、mapping、manifest 和 OOV：`src/hstu_kvcache/data/`；
- foundation training：`src/hstu_kvcache/training/`、`scripts/train_yambda500m_foundation_fsdp.py`；
- 数据处理：`scripts/download_scale_datasets.py`、`prepare_yambda500m_scale_populations.py`、`build_yambda500m_unified_scales.py`、`build_yambda500m_foundation_manifests.py`；
- 当前 manifest：`data/manifests/yambda500m_scale_v1/`、`yambda500m_small_foundation_v1/`、`yambda500m_small_five_version_v1/`、`yambda500m_small_hstu_native_rolling_matrix_fast_v3/`；
- 处理后数据：`data/processed/yambda500m_unified_v1/`；
- 当前 release、rolling、D14 和 long-age 评测脚本：`scripts/run_yambda500m_hstu_native_*.py`；
- 当前结果：`results/yambda500m_small_seed17/`，其中 `train_1d`、`train_4d`、`train_7d` 与 D14/E14 证据按保留策略保存。

`configs/contracts/` 中的合同冻结数据边界、模型 recipe、admission 和评测语义；`tests/` 验证时间因果、manifest、cache lineage、训练和 evaluator contract。旧 archive、legacy、frozen development、旧 manifest 和旧实验控制代码不属于当前复现链。

## 1. 实验总逻辑

实验按以下顺序推进：

~~~text
数据与时间因果正确
  -> HSTU-native foundation 正确
  -> 上游 release recipe 的 Full-only 质量检查
  -> 固定切片上的 Current Full / Parent Full / Reuse 对照
  -> 版本年龄、用户和序列结构 observation
  -> 在 Design I 内观察 candidate-broadcast evidence field 与 typed-coordinate/context residual
  -> 将四阶段固定流水线定位为 Design 0 strong baseline
  -> Design 0 rolling quality (complete, mixed)
  -> reader-stage/persistence -> lightweight PRO correctness/cost (complete)
  -> lightweight PRO rolling quality (complete: AUC 5/5, log-loss 3/5)
  -> progressive PRO label-free decomposition/frontier (complete: no upgrade selected)
  -> label-free admission/calibration on independent development evidence
  -> Debt-bounded Continuous feedback loop
  -> GPU runtime
  -> S/M/L 与外部 workload 验证
~~~

前两步是实验基础，第三步决定如何构造可信模型发布，第四步是当前论文 motivation 的核心。
One-Release 先验证最小可行转换；Continuous 再研究多个转换的生命周期；Runtime 最后优化物理执行。
三者不能由一次 output-fidelity probe 一并宣称完成。

## 2. 当前主实验：Yambda-500M Small

### 2.1 数据

主 workload 使用 Yambda-500M 的 Explicit Feedback：

- 原始事件按真实 timestamp 排序；
- 使用 listens、likes、dislikes 构造用户历史和监督信号；
- item mapping 在 foundation cutoff 前固定，未来新 item 进入稳定 OOV bucket；
- 所有 prefix、request、target 和 label 遵守严格 as-of 时间边界；
- 当前 Small 由统一 Yambda-500M 人口的固定 UID hash 前缀构成，目标是 10,000 用户；
- 其余 Medium/Large 人口只在规模实验阶段使用，不与当前 Small motivation 数字混合。

数据源主体覆盖 `[Day 0, Day 300)`，并保留 day300 已观测尾段。基础模型使用 `[Day 0, Day 217)`
的历史；基础 release matrix 使用 `[Day 217, Day 300)`，v5 扩展的 E14 使用 `[Day 287, Day 301)`。
所有 E14 统一命名，同时在结果中保留精确半开日期范围、请求数与实际 source coverage；事件顺序在
相同 timestamp 下使用稳定 tie-break。

### 2.2 模型

当前 motivation 只使用 HSTU-native 模型：

- 4 layers；
- hidden size 128；
- context length 512；
- item、behavior、time-delta 和 query/readout 均属于同一 HSTU 语义；
- 使用 256 个稳定 OOV bucket；
- 当前正式 motivation 的训练 repeat unit 是 seed 17。

Frozen Base + residual、CC scorer、N/R 多任务模型和旧 8L architecture pilot 不进入当前 motivation 的 deployment score，也不作为新结果的主模型。

## 3. 版本训练与 release recipe

### 3.1 Foundation v0

v0 在 Day 0–217 的基础历史上训练，形成固定 parent checkpoint 和 compact item mapping。v0 训练完成后，先验证：

- Full forward 与 cache materialization 一致；
- time delta、prefix boundary 和 OOV 处理无泄漏；
- checkpoint、manifest、mapping hash 与合同一致；
- 当前模型的 Full-only 质量可被稳定复现。

v0 只是初始模型，不因为训练完成就自动成为论文中的“有效 release”；它是后续版本和 persistent state 的 producer。

### 3.2 Upstream recipe scan

当前先做 label-blind release recipe scan，训练窗口和 observation window 只由事前合同决定：

| update duration D | 包含固定 v0 的版本数 | Full-only observation E |
| ---: | ---: | --- |
| 1 day | 60 | 1 |
| 4 days | 20 | 1, 4 |
| 7 days | 12 | 1, 4, 7 |
| 14 days | 6 | 1, 4, 7, 14 |

D=14 链为 v0→v1→v2→v3→v4→v5，共五条相邻边。每个 update candidate 使用对应时间窗训练；矩阵 recipe 使用 one-pass fresh AdamW、固定 learning rate/weight decay、完整 epoch checkpoint，不按看到的结果选择中间 checkpoint。每个 release 的 Full-only 评测可以与下一 update 的训练窗口在时间上重叠，但窗口定义不能事后移动。

当前 matrix scan 的目的不是测 Reuse，而是回答：

- 哪些固定 update duration 能在多个连续边上产生可解释的 Parent→Current Full 差异；
- 训练终点和数据边界是否严格落在 Day<301；
- candidate recipe、manifest 和 Full-only raw evaluator 是否稳定。

scan 阶段禁止读取 Reuse、KV distance、JS、release debt 或 scheduler 输出，因此不会用 compatibility 结果挑选 release recipe。

### 3.3 Accepted release

后续 release chain 使用两个完全分离的步骤：

1. candidate 训练完成；
2. 独立 Full-only validation 比较 Parent Full 与 Current Full，决定 accepted 或 rejected。

只有 accepted checkpoint 封存后，才允许读取该边的 Reuse 结果。rejected candidate 不成为 cache producer，serving parent 和 cache lineage 保持不变。

## 4. 当前 motivation 对照

在固定 D=14/E=14 切片上，对每条 accepted edge 构造相同的：

- 用户；
- causal history 和 cutover prefix；
- query、target、candidate；
- 当前模型与 readout；
- rolling append/eviction 规则。

然后比较四个视图：

| 视图 | 含义 | 主要用途 |
| --- | --- | --- |
| Parent Full | 旧模型完整重算 | release gain 的旧模型基线 |
| Current Full | 当前模型完整重算 | Full-only release gain |
| Current Exact Rolling | 当前模型在 cutover 后完整重算并真实 append | rolling reference |
| One-hop Reuse Rolling | 父模型 prefix KV，之后当前模型真实 append | 直接 persistent-state mismatch 对照 |

两类差值必须分开：

~~~text
Release gain = Quality(Current Full) - Quality(Parent Full)

Reuse harm = Quality(Current Exact Rolling) - Quality(One-hop Reuse Rolling)
~~~

第一类回答模型是否变好，第二类回答旧 KV 是否阻碍当前模型兑现收益。不能把 Current Full/rolling、Parent/Reuse 或 request-local/rolling 语义混在一个指标中。

## 5. 评价指标

### 5.1 主指标

当前 motivation 重点报告：

- ROC-AUC；
- dislike PR-AUC；
- event log-loss；
- Brier；
- user-equal 与 event-weighted paired difference。

对正的 release gain，额外报告：

~~~text
erosion ratio = Reuse harm / Release gain
retention = 1 - erosion ratio
~~~

当 release gain 很小时，erosion ratio 可能被分母放大，必须同时报告绝对差值，不能只报百分比。

### 5.2 Companion

同时保存：

- Bernoulli JS；
- normalized score RMS；
- probability shift；
- Top-K overlap 和 pairwise/margin disagreement；
- user-level tail；
- append count、old-state remaining fraction 和 cutover dilution；
- exact-equivalent token-layer work、KV read/write bytes 和 makespan。

状态 companion 用于解释为什么产生 mismatch，不用于事后反向选择 checkpoint、edge 或 headline。

## 6. 分阶段研究目的与预期观察

### Phase 0：数据与执行正确性

目的：证明时间因果、mapping、manifest、Full/cache/append 语义正确。

预期结果：相同模型下 Full 与合法 append/rolling 路径一致；未来事件和 label 不进入 prefix 或调度输入。失败时停止，不进入质量结论。

### Phase 1：HSTU-native foundation

目的：确认 HSTU-native 模型能够在固定历史和请求语义下稳定训练、物化和评测。

预期结果：foundation checkpoint、cache producer hash、manifest 和 OOV 结果可复现；不把 bounded canary 当科学结果。

### Phase 2：上游 release recipe

目的：在不读取 Reuse 的情况下构造独立、连续、可接受或可拒绝的模型 release。

预期结果：不同 update duration 可能产生不同 release gain；有些边可能无提升甚至退化，必须完整报告，不能筛掉。只有 Full-only 规则能决定 accepted release。

### Phase 3：Motivation

目的：检验新模型发布收益是否会被父版本 persistent KV 侵蚀。

支持 motivation 的观察模式是：

- Current Full 优于 Parent Full；
- One-hop Reuse 低于 Current Exact Rolling；
- 多条连续边重复出现，而不是单边偶然；
- release gain 和 reuse harm 可以同时出现，说明模型更新收益与状态兼容性是两个问题。

这一步不要求每条 edge 都有 harm，也不证明最终迁移策略。

### Phase 4：Design I — Insight discovery

该阶段已完成 Small/seed17 discovery 和机制裁决；结果见
[核心 Motivation 与 Observation](motivation_observations.md) 和
[Insight-Driven State Refinement Develop Map](insight_develop_map.md)。原始协议是先冻结 observation protocol 和切片定义，再对已保存的 raw
result、state companion、请求属性、历史统计、item/catalog 漂移和 HSTU 中间状态做分析。分析
必须同时保留总体配对结果和 cohort 内结果，不能只报告最显著的用户或请求子集。

第一步先解决 evaluation characterization：比较 cutover 后 Day 1..7、full 7-day、append
count、old-state remaining fraction、eviction 和 Current–Reuse gap 的关系。第一天可以作为
primary candidate，但只有在 gap 确实集中于 cutover 且随 current-model append/state dilution
衰减时才冻结为 primary；不能因为第一天数字更大就事后选择它。

第二步按事前定义的维度观察风险来源：

- 用户：long-history、heavy、repeat-heavy、preference-drifting；
- 历史：recent/old segment、长期兴趣和偏好转换；
- item/embedding：hot、new、快速变化或高 OOV item；
- 模型：layer、head、projection、readout；
- 版本：update direction、representation drift、producer age；
- 执行：cutover gap、append dilution、eviction 和 remaining old-state fraction。

重点不是重复已有的 Transformer hotspot 结论，而是寻找 recommendation-specific 结构，例如：

- preference-drifting heavy user 是否比稳定用户更容易受到旧 state 影响；
- long-term-interest segment 是否比 short-term intent 更容易失效；
- item 热度、embedding 漂移、新 item/OOV 和用户兴趣变化是否共同决定 risk；
- 某个 layer/head/readout 的敏感性是否只有在特定用户、历史区域或 item 类型下才出现；
- release update direction 和 representation drift 是否改变 risk 的空间分布，而不是只改变一个全局 gap；
- old-state remaining fraction 是否解释了 cutover 后 mismatch 的衰减。

每个候选 observation 都必须经过总体重复性、cohort 规模、时间稳定性、edge 重复性和反例检查。
没有足够稳定性的切片只能作为 hypothesis，不能直接变成 action feature。实验上 discovery 与
mechanism qualification 可以分阶段执行；论文中这些结果直接进入第 3.1 节 `Design Insights`，
不是独立 Insight 章节。每条阶段产出必须是：

~~~text
Observation
  -> repeatable mechanism insight
  -> Design implication
  -> corresponding mechanism
~~~

例如，preference-drifting heavy users 的风险稳定集中，才可能支持 user-selective migration；
high-drift item/embedding 与 mismatch 稳定相关，才可能支持 embedding-aware compatibility
analysis；long-term-interest region 更易失效，才可能支持 semantic-region-aware state
evolution。这里的例子是待验证假设，不是当前结论。

当前 Small/seed17 discovery 已将主线收敛为 candidate-broadcast user evidence、typed
coordinate→context residual 和 evidence mass 三条 recommendation-specific Insight。3,000-user
observation 已完成，合同和结果为：

- `configs/contracts/yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml`；
- `results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/`。
- `configs/contracts/yambda500m_small_hstu_native_candidate_shared_causal_v1.yaml` 与
  `results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/`；
- `configs/contracts/yambda500m_small_hstu_native_evidence_measure_basis_v1.yaml` 与
  `results/yambda500m_small_seed17/insight_evidence_measure_basis_v1/`。
- `configs/contracts/yambda500m_small_hstu_native_reader_compatibility_correction_v1.yaml` 与
  `results/yambda500m_small_seed17/insight_reader_compatibility_correction_v1/`。该路线先定位
  query-dependent reader correction 的最早 HSTU 阶段，再检查真实请求间持久性；两门通过前
  不得执行 layerwise broadcast residual 机制 canary。
- `configs/contracts/yambda500m_small_hstu_native_av_broadcast_residual_v1.yaml` 与
  `results/yambda500m_small_seed17/insight_av_broadcast_residual_v1/`。两道门已通过，唯一
  compact-probe AV sidecar 的无标签 score canary 为 4/5；正式 quality 与 action admission 仍锁定。
- `configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v1.yaml` 保留 dimensionful AV
  absolute threshold 导致的失败；`yambda500m_small_hstu_native_pro_lazy_reader_v2.yaml` 在机制不变、
  下一批 32 用户上通过 scale-aware fused equivalence、零 translated-prefix 物化和理论成本门。
  对应结果为 `results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/`；没有 score/quality 授权。

仍保留四个精确缺口：

- layer-0-only 已被否定、Tail-128 已证明有用但不完整；尚未做等宽
  old/middle/recent/random-128 位置对照，所以不冻结“Tail 最敏感”；
- aggregate layerwise CAST 已稳定正恢复；尚未分解 normalized layer bundle 和 token
  quartile 贡献，所以不外推每层/每位置同样可转换；
- GROUP64+SCALE 已有 matched quality 消融；包含 CAST 的完整计划为 Exact 的 48.0% 保守
  causal FLOPs。kernel utilization、KV bandwidth 和 wall time 是 Design III Runtime 问题，
  不作为 Design I Insight 缺口；
- candidate-shared reader correction 的 signed causal structure 与真实 candidate 外部性已通过；
  这不是可物化 history basis 的证明。第一个 matched-cost executable plan canary 为 0/5，未获
  rolling quality 资格。stage localization 与跨请求 persistence 已分别定位 `qK·V/AV` 并以
  5/5 通过；唯一 AV sidecar score canary 为 4/5，但 `v3→v4` 仍失败，尚未授权正式 quality。
  raw same-item/action grouping 已被 3/5 mixed edge 结果否定，也不能作为替代。
- lightweight PRO 的 32-carrier theoretical compute 为 Full 的 9.1%，action 内 translated-prefix
  positions 为 0；新机制 rolling quality 已在五边完成，AUC 5/5、log-loss 3/5、五边均值两项均
  改善。它仍不等价于真实 latency；Parent bandwidth 和 5.0 MiB conservative logical streams/user
  必须在独立 admission/qualification 后单独 profile。

### Phase 5：Design I — Lightweight PRO qualification

这一阶段与 Phase 4 在论文中共同构成第 3 章 `Insight-Driven State Refinement`：Phase 4 提供
Design Insights，Phase 5 将已执行的 CAST、compact contextual repair 和 `(r,c)` Design 0 保留为
strong baseline，再按 stage→persistence→AV sidecar→no-materialized-prefix PRO 的顺序验证主方法。
它只回答一次 `Parent -> Current` 怎样低成本转换，不承担连续版本管理和 GPU runtime。
固定 `r=128,c=64` 的第一条 one-hop 路径已在五条 D14/E14 edge 上完成 rolling AUC，正式请求总数
217,584；它在 4/5 edge 上提高 Reuse，前三条保留既有 Full-only release gain 的 97.2%、117.9% 和
87.3%，但 v4->v5 比 Reuse 低 0.265765 AUC point。对应合同、raw seal 和逐 edge 表为：

- `configs/contracts/yambda500m_small_hstu_native_d14_one_release_refinement_auc_v1.yaml`；
- `results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/d14_one_release_refinement_auc_v1/`。

因此 rolling 任务质量不再是完全未测；结果同时否定了“该固定配置可以无条件用于所有 edge”。
在 512-position state 上，旧 Design 0 理论为 0.301 GFLOPs/user（Full 的 48.0%）。轻量 PRO 的
32-carrier primary 为 0.057 GFLOPs/user（Full 的 9.1%），16-carrier diagnostic 为 5.2%；完全等价
地将 fused map 用于每个 carrier query 的 32-carrier 版本为 40.0%，已在执行前因成本失败拒绝。
Phase 5 下一步不再开放新的 operator/region 探索，也不实现复杂 scheduler：

1. 在新 prospective contract 下验证最小的 label-free 事前安全判断与 Reuse/Exact fallback；
2. lightweight PRO 正确性/成本及全人口 rolling quality 已完成；primary 32、latest-item、mass4
   和 coverage decay 全程未改。其总体 Design viability 通过，但事前严格双门因 log-loss 3/5 未过。
   本五边转为 development evidence；任何 admission/calibration 改动必须在新 seed 或新冻结 edge
   验证，不得拟合 target K/V 或加入 per-candidate Route；
3. 只有独立 quality gate 通过后，才执行额外 training seed 与同机 GPU/runtime/I/O profile。

既有诊断性 region/layer splice 只保留为 Insight 因果证据，不加入已冻结的 scale action set。
安全信号开发不得读取未来 label；新增 training seed 仍需独立合同、资源估算、canary 和明确 launch。
机制和成本对照保留：

- `CAST(stale)`；
- `CAST + typed PATCH(residual scope)`；
- `CAST(prefix) + GROUP -> PATCH -> SCALE + UNION`。
- lightweight PRO：reader-pushed map + Parent-conditioned carrier32 + AV sidecar。

对照为 No-op、Exact-All 和旧固定宏 baseline。Design I 报告 rolling AUC/log-loss、output fidelity、
解析 FLOPs 和结构 token/pair work；raw-history/state I/O、CUDA time、storage、throughput 和 makespan
统一在 Design III Runtime 中实测。

Design I 的完整完成条件是：Design 0 作为 strong baseline 保留；事前固定的 lightweight PRO 与
release-level Exact-shadow admission 在 rolling AUC/log-loss 上报告全部五边且不隐藏反例，并在解析
FLOPs 上低于 Exact-All 20%。理论计算、正确性和五边质量评价对 PRO 已完成；总体可行性通过，但
严格 edge-count gate 未过，故“always-on 固定配置”尚未通过。这里不要求先证明复杂 scheduler；
PRO、edge-level safety contract 和 Reuse/Exact fallback 足以向 Continuous 层交付一跳 transition。理论 FLOPs
不保证实际加速，后者由 Runtime 单独回答。

不得用这五条 qualification label 调整 `r/c` 或选择报告 edge。若以后确实需要 population budget
allocation，再在独立 development contract 下开放 target-free PATCH value estimator、threshold
compiler、random/metadata-only allocation 和 offline oracle 对照。任何 policy 都不得使用 future
label 或 target K/V fitting。

Design I 仍按 typed IR 保留合法重放和 COMMIT 所需的内部 lineage；但 Continuous controller 的
决策接口只暴露三个量，避免把底层状态布局扩展成第二套复杂算法：

~~~text
last_exact_or_rebase_version,
approximation_depth,
estimated_compatibility_debt
~~~

### Phase 6：Design II — Debt-Bounded Continuous State Evolution

这是 One-Release 之后的下一项核心研究，当前先冻结为一个简单闭环：

~~~text
estimate remaining debt for candidate plans p=(r,c)
  -> choose the cheapest plan with debt <= tau
  -> execute incremental refinement
  -> sampled Current-Exact shadow
  -> keep / strengthen repair / Exact Rebase
~~~

第一轮优先复用已封存的 `v0..v5` checkpoint、请求和 state，做 focused inference/evolution probe；
它不授权 theta3、Medium/Large 或新的长训练。

#### 6.1 Debt-bounded plan selection

对每个候选固定计划 `p=(r,c)`，构造发布前、target-free 的剩余误差估计 `D_hat(S,p)`，并选择：

~~~text
p* = argmin_p C(p), subject to D_hat(S,p) <= tau
~~~

候选集保持小而有序：No-op/CAST-only、当前固定 compact repair 的若干加固档，以及 Exact/Rebase。
若低成本计划不满足 `tau`，逐级扩大 repair width、增加 carrier 或降低压缩；若所有近似计划失败，
执行 Exact/Rebase，并将 debt 与 approximation depth 清零。

同时施加硬上限 `approximation_depth <= H`。即使估计器持续判定安全，达到 `H` 后也必须建立新的
Exact anchor。这只能保证近似过程和回退路径有界，不能声称 AUC 数学上永不下降。

第一轮 `D_hat` 不需要立即训练复杂 predictor，可以从 producer/version span、当前计划配置和小规模
Exact-shadow fidelity 的保守组合开始。`tau`、`H` 和特征必须在 prospective development contract 中
固定，不能用 qualification/scale outcome 反向调节。

#### 6.2 Sampled Exact feedback 与分级回退

每个 release 事前固定一小部分 shadow population，同时计算 Continuous plan 和 Current Exact：

- 即时无标签信号：normalized score RMS、Bernoulli JS、Top-K overlap、margin disagreement；
- 延迟任务信号：封存 canary population 上的 AUC/log-loss 等真实质量，只做配置级 safety audit。

控制器只有三个状态：

1. **Normal**：保持当前 `(r,c)`；
2. **Warning**：扩大 `r`、增加 `c` 或降低压缩；
3. **Invalid**：对受影响的 release/lineage Exact Rebase，并暂时禁用该近似配置。

进入阈值与恢复阈值分开，避免在边界附近反复切换。即时计划选择和用户级 migration 必须保持
label-free；延迟行为标签不得回填到同一请求或单个用户的调度，只能按事前合同触发后续 release
的全局/lineage 级保守回退或配置失效。

#### 6.3 核心实验和最低完成条件

Continuous 第一轮只回答三件事：

1. debt estimate 是否能排序不同 `(r,c)` 计划的剩余 fidelity/quality risk；
2. `tau + H` 是否能限制多 release 的最坏 approximation depth 和 Exact gap；
3. sampled shadow 能否及时捕获失效配置，并以低于每版 Exact-All 的摊销成本触发 Rebase。

主要比较保持简洁：每版 Exact-All、连续近似但无反馈、固定周期 Rebase，以及 debt-bounded +
Exact-shadow feedback。报告每个 release 的 fidelity/rolling quality、Normal/Warning/Invalid 次数、
Rebase 比例、shadow 开销、总转换开销和相对每版 Exact-All 的摊销节省。

当前 producer-age 单调 observation 只动机化该闭环；固定 one-release plan 在 v4->v5 的失败说明
feedback/rebase 有实际必要性，但仍没有验证 `D_hat`、`tau`、`H`、shadow rate 或质量回退。
只有多 release 结果同时满足封存质量边界和摊销成本优势，Design II 才准入。

### Phase 7：Design III — GPU Transformation Runtime

Runtime 在 Design I/II 语义稳定后实现。逻辑计划允许每个用户/segment 使用不同 `(r,c)` 和 lineage；
物理执行按以下 signature 量化并组成 GPU micro-batch：

~~~text
(source_version, target_version, cast_type,
 r_bucket, c_bucket, sequence_length_bucket, dtype)
~~~

执行数据面依次包括 metadata scan/plan lowering、state 与 raw-history prefetch、batched CAST、
GROUP/gather、ragged Current PATCH、SCALE/layout、异步 writeback 和 atomic COMMIT。允许不同阶段
在不同 micro-batch 上重叠，但不能牺牲 serving isolation 或覆盖 post-cutover append。

Runtime 至少报告：operator 与 end-to-end CUDA time、GPU utilization、state/history bytes、write
amplification、throughput、migration makespan、tail latency、失败恢复和 serving interference。
在真实实现完成前，结构 token-layer work 不能替代这些系统数字。

### Phase 8：规模与外部验证

S/M/L 使用统一 Yambda-500M 母体，分别扩大模型计算量、context、catalog 和 state population；同一 checkpoint 的不同 cache length 是评测变量，不是重复训练模型。

当前 Medium 的 prospective 执行边界已单独整理在
[`medium_scale_training_plan.md`](medium_scale_training_plan.md)：沿用 day217 的 30k fixed-UID
population、6L/H192/context1024 与 foundation item mapping，先训练一份共享 Full-only v0，再从同一
v0 串行扫描 D7×10（E3/E7）和 D14×4（E3/E7/E14）。该方案只指导代码、合同、canary 和资源准备；
在 Full-only recipe 稳定并完成独立 admission 前，不解锁 Reuse 或 PRO。

RecFlow 是 prospective external validation：先验证 raw request-group、chronological history、multi-positive target 和 serving-space candidate 语义，再考虑受门控的 Medium 训练。它补充 Yambda-F，不替代当前 motivation。

## 7. 实验纪律

- model admission 不读 Reuse 或 compatibility 结果；
- Reuse 只在 accepted release seal 之后执行；
- 不用 future label、selected-edge reporting、target-KV fitting 或人工 K/V 扰动；
- 不按结果移动时间窗、删除坏天、反向挑 checkpoint 或筛用户；
- diagnostic splice 不进入 executable action；
- 所有 seed、edge、horizon 和失败结果完整报告；
- 任何 Medium/Large 长训练都需要独立合同、资源估计、focused canary 和显式启动。

## 8. 主要产物

每个阶段都应产出可独立审计的：

1. immutable data/model/recipe contract；
2. manifest、mapping、checkpoint 和 producer hash；
3. raw-first evaluation table；
4. seal 后的 label adjudication；
5. paired metric、state companion 和 cost summary；
6. 对当前阶段结论、反例和不能外推范围的短报告。
