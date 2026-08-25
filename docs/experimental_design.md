# EvoKV 论文具体实验设计

更新日期：2026-08-25

本文记录论文如何把概念问题变成可执行实验：模型架构、数据、版本训练、对照路径、阶段目的、预期观察和后续扩展。具体结果不写在这里，而写入 [核心 Motivation 与 Observation](motivation_observations.md)。

## 1. 实验总逻辑

实验按以下顺序推进：

~~~text
数据与时间因果正确
  -> HSTU-native foundation 正确
  -> 上游 release recipe 的 Full-only 质量检查
  -> 固定切片上的 Current Full / Parent Full / Reuse 对照
  -> 版本年龄、用户和序列结构 observation
  -> 再决定 EvoKV 的 migration granularity 与 scheduler
  -> S/M/L 与外部 workload 验证
~~~

前两步是实验基础，第三步决定如何构造可信模型发布，第四步是当前论文 motivation 的核心。只有 motivation 和结构 observation 成立，才进入 partial action、budget scheduler 和 executor 的系统实验。

## 2. 当前主实验：Yambda-500M Small

### 2.1 数据

主 workload 使用 Yambda-500M 的 Explicit Feedback：

- 原始事件按真实 timestamp 排序；
- 使用 listens、likes、dislikes 构造用户历史和监督信号；
- item mapping 在 foundation cutoff 前固定，未来新 item 进入稳定 OOV bucket；
- 所有 prefix、request、target 和 label 遵守严格 as-of 时间边界；
- 当前 Small 由统一 Yambda-500M 人口的固定 UID hash 前缀构成，目标是 10,000 用户；
- 其余 Medium/Large 人口只在规模实验阶段使用，不与当前 Small motivation 数字混合。

数据源覆盖约 301 天。基础模型使用 Day 0–217 的历史；后续 release update 和 observation 使用 Day 217–301 的时间线。所有逻辑窗口是半开区间，事件顺序在相同 timestamp 下使用稳定 tie-break。

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

### Phase 4：Recommendation-specific observation

在 motivation 成立后，按事前定义的切片观察风险来源：

- 用户：long-history、heavy、repeat-heavy、preference-drifting；
- 历史：recent/old segment、长期兴趣和偏好转换；
- item/embedding：hot、new、快速变化或高 OOV item；
- 模型：layer、head、projection、readout；
- 版本：update direction、representation drift、producer age；
- 执行：cutover gap、append dilution、eviction 和 remaining old-state fraction。

目的不是挑一个好看的 cohort，而是建立 Observation → System Opportunity → Design Principle 的证据链。

### Phase 5：EvoKV system design

只有前面 observation 说明风险具有稳定结构后，才冻结：

- dependency-closed migration actions；
- profiler 的 target-free features；
- partial/exact cost；
- state scheduler；
- grouped executor；
- budget operating points。

此阶段必须比较 No-op、Exact-All、fixed partial、random、metadata-only、learned target-free policy 和 offline oracle，并在相同预算下报告 quality、fidelity、compute 和 I/O。

### Phase 6：规模与外部验证

S/M/L 使用统一 Yambda-500M 母体，分别扩大模型计算量、context、catalog 和 state population；同一 checkpoint 的不同 cache length 是评测变量，不是重复训练模型。

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

