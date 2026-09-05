# EvoKV Insight 2 与 Design 1 探索计划

日期：2026-09-03  
状态：**v1 KV-only protocol 已完成；Migration Sketch prospective mechanism 已形成；方法实验与最终 admission 待合同**

> 执行注记：第 7 节保留最初的 prospective 顺序，第 10 节保留 2026-09-03 中途检查点，均不按后续
> 结果回写。实际执行已经继续否定 paired native response、defect coordinates、source-certified residual、
> activation topology、causal suffix、head circuit、release algebra 与 post-hoc state ports。当前总裁决见
> 第 11 节、[当前 KV-only 接口裁决](current_kv_only_interface_adjudication.md)与
> [追加式探索日志](exploration_log.md)。专家讨论后的正向 Design 与下一阶段执行计划见第 12 节和
> [统一论文材料](../../../docs/insight2_design1_expert_brief.md)；前述历史段落不回写成方法结果。

## 1. 研究目标

当前已经成立的 Insight 1 是：跨版本误差虽然可以在某些 token、窗口或层的
Exact-state 诊断替换中被部分消除，但 locality 不能稳定转化为 0%–20% 成本内的
dependency-closed 高恢复迁移动作。因此，Design 1 不再以 token/layer locality
作为主要抽象。

Insight 2 要回答一个可被否证的问题：

> 分布在历史 token 与多层 K/V 中的跨版本误差，经过 Transformer reader 的
> query-conditioned interaction、value aggregation 和 residual update 后，是否会在
> 某个最早阶段收敛成紧凑、跨请求稳定且因果充分的功能差异？

若答案为是，Design 1 才迁移该功能状态；若只在更晚的 residual/user
representation 才成立，接口随之后移；若所有阶段都高维且强 query-dependent，
则停止 sidecar 假设，转向少量 dependency-closed recomputation 或模型—系统协同边界。

最初待检验的候选 Insight 是：

> Attention aggregation may be a functional bottleneck: distributed cache error can
> remain non-local in token space while its effect on repeated recommendation queries
> occupies a much smaller response subspace.

该命题的 representation 部分已经成立，但 constructor 部分没有成立。当前阶段性表述已收窄为：
aggregation 后的 compactness 是 reader-conditional 的，不自动形成 query 前可生成的 persistent quotient。
AV、PRO、单向量 offset 和 C8 都只作为 baseline/control；协议继续允许“当前接口无 Design”胜出。

## 2. 当前实验边界

- 只研究单个相邻 release edge，不涉及多版本连续迁移、debt、rebase 或 runtime controller。
- 主资产是 Yambda-500M Medium、seed 17、HSTU-native CC、6L/H192/context1024，
  使用已经封存的 `v0..v5` 六个 D14 checkpoint 和五条相邻 edge。
- 固定人口上限 30,000；本协议复用 Insight 1 的 3,000-user label-free population 与
  64-candidate panels。先运行 32-user canary，再使用 512-user discovery split；只有在
  设计冻结后才能读取剩余 2,488 users 的 confirmation 结果。
- 本轮不训练新模型，不读取 theta3，不启动 RecFlow，不把 TIGER 加成第二套系统。
- 当前六个 checkpoint 训练的是 Yambda explicit-feedback ranking/classification。
  因而本轮能发现 HSTU ranking 上的边界，但不能把同一权重的 candidate panel 冒充
  next-item-trained retrieval 证据。Yambda next-item 或第二 workload 必须另立训练合同后验证。
- 四张 A40 只做独立 UID shard；每个 rank 内 checkpoint 与 edge 串行。任何预计超过
  30 分钟的 formal run 使用 detached session，并先给出 canary 资源估计。

## 3. 三条必须同时检验的性质

### 3.1 空间收敛

比较误差在 token、layer、head 与 hidden dimension 上的分布。低 tensor distance、单个
高能 token 或单层 oracle 恢复都不足以准入；目标是找出误差从 position-indexed state
变成 position-free functional response 的最早阶段。

### 3.2 请求共享性

同一用户面对不同 candidate query 时，correction 必须可由一组事前固定 anchor queries
估计，并因果恢复未参与估计的 held-out queries。直接在全部候选上计算均值再回注同一批
候选只是 oracle ceiling，不是共享性证据。

### 3.3 时间持久性

在 cutover 生成的 correction 要在后续真实请求与 Current append 后仍保持方向和恢复。
每个 request 重新读取 Current Exact 的做法没有 persistent-state 系统价值。

三者共同构成迁移对象门槛：**causally sufficient、cross-request stable、short-horizon persistent**。

## 4. 架构中立的 stage taxonomy

三条成对路径在相同用户历史和相同 query 上比较：

1. `Parent Exact`：Parent reader + Parent state，用于分离 model/readout change；
2. `Current Exact`：Current reader + Current state，作为功能 reference；
3. `Current Reuse`：Current reader + Parent state，隔离 cache compatibility error。

| Stage | 架构中立对象 | HSTU 实例 | 主要问题 |
| --- | --- | --- | --- |
| S0 | input/token representation | item/action/time combined input | producer 差异在 contextualization 前有多大 |
| S1 | Q/K/V projections | per-layer normalized Q/K/V | K、V 或 query path 是否单独形成稳定坐标 |
| S2 | query–key interaction | activated `qK` per head/position | query dependence 是否仍高维、位置分散 |
| S3 | per-position value response | activated `qK` times `V` | 贡献是否局部，或只在求和后收敛 |
| S4 | aggregated context | position-summed head context（HSTU AV） | 最早的 position-free compact boundary 是否出现 |
| S5 | transformed attention/update | output projection、normalization、gate/FFN update | 模型非线性是否进一步压缩或旋转误差 |
| S6 | post-block residual | layer output residual stream | 多头/多分量差异是否在 residual 中对齐 |
| S7 | final query/user representation | final norm 前后 hidden | late-bound user correction 是否更充分 |
| S8 | readout | scalar score/probability/ranking | 只用于任务充分性，不因标量低维而宣称 Insight |

S0–S3 的 Exact splice 都是诊断；S4–S7 才是可能的 functional-state interface。
HSTU 名称只写在 adapter 中，论文定义不把 AV 或 U gate 当通用术语。

## 5. 每个阶段的统一测量

### 5.1 表示结构

- token/head/layer error energy 与 top-k coverage；
- 对 candidate-by-feature correction matrix 同时报告 uncentered 与 candidate-centered
  singular spectrum、rank@90%、rank@95% 和 participation-ratio effective rank；
- 报告 candidate mean energy，但不把它等同于低秩证明；
- 在 discovery users 上学习 global basis 时，必须在 UID-disjoint users 上报告投影误差。

### 5.2 因果充分性

64 个 label-free candidates 按每种 panel mode 内稳定奇偶位分成 32 anchors/32 held-out。
在 anchors 上估计 correction，只在 held-out 上 adjudicate：

- `rank0`: query-independent per-user offset，覆盖现有 AV/PRO 表示假设；
- `rank-r`: correction 在 query-conditioned response subspace 中变化，预注册
  `r in {1, 2, 4, 8}`；
- `full-oracle`: 使用 held-out Exact delta 的上界，只检查 hook 与 intervention 正确性，
  不进入设计比较。

correction 在 S4、S5、S6、S7 注入，随后完整执行剩余 Current reader。每层动态 intervention
必须在共同 upstream hidden 上比较 Exact/Reuse，避免重复计算同一误差。S2/S3 只用于诊断
为何 S4 收敛，不把 per-position tensor 伪装成 compact action。

### 5.3 任务与稳定性指标

主指标均不读 label：

```text
probability-gap recovery
  = 1 - MAE(sigmoid(intervened), sigmoid(Current Exact))
        / MAE(sigmoid(Current Reuse), sigmoid(Current Exact))

logit-gap recovery
  = 1 - MAE(intervened, Current Exact) / MAE(Current Reuse, Current Exact)
```

不裁剪 recovery，并同时报告 JS、top-1、top-10 overlap、within-user rank correlation。
时间实验报告 correction cosine、norm drift、固定 correction recovery 与 append count/horizon 的关系。
正式质量阶段才读取 AUC/log-loss；同 cohort 的 Old/New/Reuse/Design 四路径必须全部报告。

## 6. 证据层级与准入门槛

### Gate A：instrumentation

- Reconstructed Exact/Reuse 与 native path 的最大绝对 logit error `<= 2e-5`；
- full-delta intervention 恢复 Current Exact，误差 `<= 2e-5`；
- 32 users × 5 edges × 全部预注册 stage/config 完整、finite、无 label；
- Parent/Current/cache hashes 与 population/candidate manifest 匹配。

### Gate B：functional boundary observation

一个 stage 只有同时满足以下条件，才可称为候选功能边界：

- anchor-to-heldout 因果 recovery 的 edge-equal mean `>= 0.80`；
- 至少 4/5 edges 达到 `0.80`，或至少 3/5 达到 `0.90` 且五边均值仍 `>= 0.80`；
- compact representation 相对同阶段 full delta 显著减小维度，并优于 S2/S3 的
  position-indexed 表示；
- 不是只在 scalar readout 才成立。

该门只定位“表示边界”，不证明它可低成本生成。

### Gate C：persistent boundary

- correction 由 cutover anchors 生成后固定；
- 在预注册 post-release horizons/append bins 上，方向稳定且 recovery 不出现系统性坍塌；
- 至少 4/5 edge 保持正 recovery，edge-equal mean 目标 `>= 0.70`。

若 correction 需要每请求重估，则 Gate C 失败。

### Gate D：executable Design 1

- estimator 只读取 Parent persistent state、release 前可用 raw evidence 和 Current 参数；
- 不读取 target request label、future event、Current Exact full K/V 或 qualification outcome；
- 完整生成 + 注入的解析理论计算处于 Exact-All 的 `0%–20%`；
- 在 512-user discovery split 达到 Gate B 的 `80%` 门，stretch goal 为 `>=90%`；
- 至少 4/5 edge 正向；允许一条明确报告的失败边，但不能选择性隐藏；
- design 选择冻结后，在 2,488-user confirmation split 和正式 rolling quality 上复核。

只有 Gate D 通过，Design 1 才能从 framework 收敛到具体算法。GPU latency、I/O 和 storage
单列，不用理论 FLOPs 冒充 runtime speedup。

## 7. 迭代路线

### Iteration 0：协议、资产与代码正确性

冻结 stage、candidate split、metrics、cost denominator、用户 discovery/confirmation split；复用
Medium v0..v5、3,000-user manifest，不复制 checkpoint 或展开 manifest。

### Iteration 1：oracle stage localization

先跑 32-user/five-edge canary，再在 512 discovery users 上观察 S0–S8。比较 rank0、rank-r
anchor-to-heldout 干预，找到最早通过 Gate B 的 stage。结果可以是 S4、S5、S6、S7 或 none。

### Iteration 2：合法 estimator

只对 Iteration 1 留下的最多两个相邻边界开发 estimator，避免用五边结果搜索大量机制：

1. parameter-only/current-reader probe；
2. 少量 dependency-closed Current carriers；
3. 二者组合。

预算点事前固定为 `5%, 10%, 15%, 20%`。rank0 是 PRO-compatible baseline；rank-r
检验 query-conditioned correction 是否能以相同预算弥补 rank0 residual。所有点全边报告。

### Iteration 3：持久性与更新

在真实 rolling requests 上冻结 cutover correction，按时间与 append count 测量；只在 Gate C
显示必要时比较低频 refresh。refresh 成本必须摊入同一个 0%–20% 总预算。

### Iteration 4：设计冻结与 confirmation

依据 discovery split 冻结：迁移对象、estimator、rank、预算、注入点和失败回退。随后一次性读取
confirmation split，最后才做 task-label quality。未过门就报告 negative result，不在 confirmation
users/edges 上继续调参。

## 8. Design 1 的开放评估框架（历史协议）

Design 1 当前只保留四个架构无关的评估问题，不冻结算法或 active candidate：

```text
READ(parent persistent state, protocol-permitted release evidence)
  -> ESTIMATE(Current-version functional difference, bounded Current compute)
  -> PERSIST(compact per-user functional state)
  -> INJECT/UPDATE(at the selected reader boundary)
```

这些是未来方法必须回答的接口，不表示 probe、sidecar 或 correction 已被预选。历史落点及裁决关系为：

- S4 通过且 rank0 足够：HSTU AV offset 可作为表示实例；历史 PRO 只作为 Small baseline，不是当前
  active estimator candidate；
- S4 的 rank-r/query-conditioned correction 只通过 representation oracle；合法 constructor 已失败；
- residual/user-representation correction 尚无独立 Current-information source，不能因边界后移而自动准入；
- 当前 sidecar、bounded replay 与 post-hoc co-design candidates 均已审计 NO-GO；重新开启需先有新的
  finite-release causal law。

论文最终统一的是“寻找最早、最紧凑、因果充分的功能迁移边界”，不是所有 Transformer 都使用 AV。

## 9. 记录与停止规则

每轮都在 `exploration_log.md` 追加：假设、冻结输入、命令、资源、raw seal、完整结果、反例、
裁决和下一轮唯一新增变量。不得覆盖已有 evidence，也不得用 qualification labels 回调 candidate、
stage、rank、预算或 edge。

探索在以下任一条件满足时停止：

1. 一个合法 estimator 在 0%–20% 成本内达到 Gate D，且 Insight/Design 可清楚表述；
2. 预注册 S4–S7 均不能通过 Gate B/C，形成可信 negative insight 并转向 recomputation；
3. canary 证明现有模型接口无法做一致 intervention，此时先修 instrumentation，不解释质量。
4. representation boundary 已通过，但合法 generator/interface family 已穷尽且所有新构造均退化为
   mapping、generic compression 或超预算 native replay；此时冻结接口否证，不用数值 baseline 顶替 Design。

## 10. 2026-09-03 历史执行检查点

> 本节冻结的是 Iteration 19 之后的中途判断。原“下一轮唯一科学问题”已在 Iteration 23 执行并否决；
> 不再把它视为待执行计划。最新状态统一见第 11 节。

### 已经冻结的正负边界

- S4/activation-region functional bulk 的表示 recovery 很高，说明分布式 state error 确实会在 reader
  aggregation 后收敛；
- Exact-state sparse carriers、chronological/address coreset 和 recursive closure 均失败，正式冻结
  **functional compactness does not imply token-support sparsity**；
- exact release `Delta[K,V]` 在单 UID 结构诊断中呈现 support-dense、history-mode compact，但
  per-user/跨层 shared low-rank basis 已有 xKV 等直接邻近工作；
- single-arm rank-8 replay 虽有较强非正式 recovery，仍只是 generic compression control；
- equal-resolution Parent/Current rank4/rank4 differential 出现 approximation-error cancellation 线索，
  但其单 UID 平均值 `0.866` 并未超过 single-arm rank-8 的 `0.937`；matrix-free input 已把其
  KV-only 成本从 `21.82%` 降到 `18.33%`，所以现在失败的是 mechanism/control gate，而非成本门；
- coupling-depth 只支持“前三层形成主要 paired gain”的弱结构线索。`d=3` 平均 `0.831`、成本
  `16.96%`；唯一 `Current 4->8` handoff 平均 `0.836`，仍被 single-arm 支配，不能写成 Design。
- paired S4 compiler 虽在 `19.66%` 内达到平均 `0.900`、最差 `.823`，但只在 1/5 edge 胜过
  single-r8 functional control，且没有独立 lineage closure；数值达标不能挽救创新门失败；
- Parent-anchored finite-difference execution 的语义成立，但 KV+RMS source 解码和 mandatory Q/gate
  在实际 delta work 前已达 `25.32%`；当前 interface 下不可执行；
- K/V finite interaction 随 depth 增强却高度 edge-dependent。K-only/V-only、去掉 interaction 的加法和
  common-projection native-response control 均不稳定，不能形成新的 response-control law。

### 当前论文创新门

下一方法必须同时满足以下条件；缺一项就不能靠命名进入 Design 1：

1. 新机制位于两版本 Transformer computation 本身，而不是 SVD、mapper、cluster、sampler、rank tuning
   或已有 cache compression 的拼装；
2. 用 matched-compute ablation 证明恢复来自 release-specific mechanism，而不是更多 rank/更大 sketch；
3. 完整 constructor 在读取、projection、attention、nonlinear boundary、sidecar write 和 injection 全计后
   严格处于 Exact-All `0%–20%`；
4. 有 full-rank/no-truncation correctness limit 或同等清楚的语义 invariant；
5. 在 32-user formal canary 前已经定义明确的 falsification condition，不以结果反调边界。

### 当时冻结、现已完成的唯一科学问题

> 若不再把 mode budget 用于压缩完整 Current absolute state，而在每层分别保持 Parent base 与
> Parent→Current finite-release defect，defect-coordinate recurrence 能否保留 generic single-arm
> 没有的信息？

只做一个固定配置 preflight，不扫 rank：

- Parent base 与 defect 分别压缩；Current block 输入由二者因子和组成，而不是重新对
  `Parent+defect` absolute state 做一次 ordinary Current compression；
- total active mode budget 和 matrix-free/terminal-KV口径事前冻结；同 runner 对比 ordinary
  asymmetric replay、paired r4/r4 与 single Current r8；
- full-rank 时必须还原两版 native trajectories。若数值不胜 matched controls、成本超过 20%，或相关
  工作审计表明它只等价于普通 base-plus-delta low-rank inference，则立即退休。

该实验随后只读 UID `1930` 的已开放五 edge 作非正式淘汰，没有建立合同。结果为
`.554/.690/.549/.821/-.076`，mean `.508`，成本 `18.4567%`；它被两个 single-r8 controls 支配，并与
base-plus-delta prior art 重叠，已经 **NO-GO / RETIRE**。原准入条件是：只有它同时显示
`<20%` 静态成本、至少 4/5 正向且相对 matched single-arm 有机制增益，才允许写下一份 prospective
32-user canary contract；实际没有通过。因此当前 KV-only/reduced-replay family 已停止，没有扩大 rank、
探针或用户数。

### 论文创新硬门

最终方法不能以 mapper、SVD/低秩、矩阵自由 range finder、sampling/clustering、rank allocation 或
现有 cache compression 的组合本身作为 Design。它们只允许作为 executor/component。论文贡献必须
同时给出：Transformer reader 中明确的功能边界；Parent→Current 有限版本差分特有的计算 invariant；
不读 Current Exact 的合法 constructor；以及 matched generic control 无法解释的恢复增益。数值可以在
discovery 内逐步稳定，但缺少这四项时，即使 recovery 进入 80% 也不准入 Design 1。

## 11. 2026-09-03 当前接口总裁决

### 11.1 表示成立，生成闭包不成立

当前最稳健的 Transformer-specific 结论是：

> **finite-query functional compactness is reader-conditional, not generator-closed。** 分布式 cache
> mismatch 在 query-dependent aggregation 后可以很紧凑，但这不推出 Parent K/V 中存在一个能在 query
> 前低成本生成、并对未来请求持续成立的 Current functional quotient。

single Current-r8 在固定 UID 五边达到 `.8610/.9173/.9852/.9473/.9753`，mean `.9372`，完整
matrix-free ledger 为 `17.8953%`。它只说明 generic approximate Current replay 是强数值对照；共享
token basis/low-rank cache 已被直接 prior art 覆盖，不能成为 Design 1。

### 11.2 后续候选的完整裁决

- paired native response 为 `.9012 @ 18.2810%`，仍被 single-r8 支配；
- Parent-base + release-defect coordinates 为 `.508 @ 18.4567%`；
- source-certified finite defect 为 `.662 @ 19.4726%`，且是 DEIM/sampled-residual 骨架；
- producer/reader commutator 的 raw oracle 为 `.8831`，但 centered decision effect 不稳定，并依赖
  Exact Current reverse path 与 score mixing；
- activation branch graph 虽有 `87.32%` endpoint agreement，crossing-only causal recovery 约 `.211`；
  same-region continuous deformation 才是主体；
- migration-ready Parent tape 即使免费保存 source response，五层 Current native QK+AV 仍为
  `42.2367%`，并额外增加 `19.5 MiB/user`；
- natural causal suffix 对任意初始 cache 都满足 append/chunk consistency，只提供 query coverage，
  不提供 Current target-state information；Tail-128 已为 `-.0876`；
- dense `W_O`、gate 和 residual 使 head salience 不能形成跨层独立 circuit；第三个 cache layer 的任意
  single-head exact closure 已为 `35.09%`；
- attention gauge、structured parameter update、native-query quotient、finite moments 与 cached causal
  separator 五类 exact algebra 出口均已关闭；真实 release 的 gauge-invariant mismatch 为
  `5.21%--11.89%`，block `Delta W` numerical rank 为 `180--192`；
- post-hoc causal ports 与 recurrent-memory prior art 相撞，而且 Parent-sufficient port 不自动
  Current-sufficient；当前没有非 regression 的 release homomorphism 或 delete law。

### 11.3 当前执行决定

当前 `v0..v5 + Parent KV-only + no new training + <20%` 条件下没有 paper-worthy active candidate。
停止新的 UID/rank/probe/layer/head/operator 扫描，不为该旧接口建立 32/512-user formal contract，
不读取 `[512,3000)` confirmation。该结论不否定第 12 节主动改变 state-creation contract 的
Migration Sketch。

重新打开算法探索前，候选必须先提供一个 generic Current replay 没有的 Current-information source 或
finite-release causal law，并通过：version-essential matched control、no-target constructor、完整成本、
append/eviction closure 与 prior-art reduction。model--system co-design 也只有在先给出具体、可证伪的
release law 后才值得申请新的 Small training contract；“先加 memory tokens 再训练看看”不准入。

完整论证见[当前 KV-only 接口裁决](current_kv_only_interface_adjudication.md)。

## 12. 专家裁决后的 Design 1 与下一阶段 Plan

### 12.1 冻结方向

Insight 2 的最终 scoped 表述为：在当前 legacy-HSTU ranking workload 中，query-conditioned aggregation
会把分散的跨版本 state mismatch 收缩成紧凑的用户级功能差异；但在已审计 Parent-KV-only constructors
与 20% 预算内，没有找到一份可构造并随 append/eviction 演化的紧凑状态。这不是关于 ordinary KV
信息量的普遍不可能性结论。

Design 1 的 prospective mechanism 是 **Migration Sketch**。它不是“两个 decoder 相减”，而是统一的
state-creation 方法：

1. Parent 在实际 Current 未知时随普通 K/V 写入 stable canonical summary；
2. foundation 用目标 edge 之外的 pseudo releases 最小化
   $\mathcal L_{cm}=\|\epsilon_a-\epsilon_b\|^2$，真实版本再对同一冻结 reference residual 做 anchor；
3. Current query 在 shared HSTU kernel 中读取 Current 与 producer view 的 response difference；
4. migration-aware Full 与 append 共用正式 state writer
   $C_{v,i}^{Full}=G_v(r_i,S_{<i})=C_{v,i}^{append}$，$G_v$ 不读取 mixed-lineage request hidden；
5. 每个 chronological segment 保存 contextual atoms、entry state、raw witness、producer tag 和 native
   views，partial Parent drop 后同时刷新 $D_p/D_c$。

foundation 明确使用 paired pseudo-release compatibility supervision；主张只是目标 edge 上无 per-user
Parent→Current fit，不能写成“从未使用 paired model data”。shared probes 也只是有限训练坐标；kernel、
head coordinate、scale 或 normalization contract 改变时直接判 incompatible。

### 12.2 执行顺序

1. 冻结 `BUILD/MIGRATE/READ/APPEND/DROP` schema、共同 reader contract、reference releases、segment size、
   slot budget 和 cost denominator；
2. 实现 toy/full-capacity conditional limit、zero-release、$G_v$ Full/append identity、partial-drop view
   refresh 和 no-target provenance tests；
3. 在 train users 与 Insight cohort UID-disjoint 的条件下建立 retrospective C32 functional contract；
4. 同时运行 `lambda_cm=lambda_a=0`、Current-only、auxiliary-interpreter、ordinary-trajectory-writer、
   memory-token、mapper、frozen-producer、Tail 与 generic C8 controls；
5. 只有 C32 均值 `>=80%`、4/5 edge `>=80%` 且创新 matched-control 通过，才运行 D512；70% 仅作为
   继续开发而非论文准入的 floor；
6. 配置冻结后测试真实 rolling append/eviction，先 seal logits 再连接 task labels；
7. confirmation `[512,3000)` 最后一次性读取；
8. retrospective 通过后，另立 prospective Small Parent/Current pair 合同，证明 Parent 在 Current 未知时
   真正写入可用 sketch，并比较 migration-aware Full 与 unconstrained Full；任何新训练仍需 focused
   canary、资源估计和用户显式授权。

prospective MS evaluation 前还必须独立封存 Full$^G$ vs Reuse$^G$ compatibility gap；若 gap 低于
预注册可识别阈值，则按 No-op 处理，不能借用 legacy gap 或用近零分母报告高 recovery。

legacy 五边只提供 retrospective functional robustness；单个 prospective pair 只证明 forward provenance、
$G_v$ identity 与 clean endpoint。若最终论文声称完整方法满足 4/5-edge robustness，必须另立 prospective
multi-edge chain；否则两套证据分开呈现。

### 12.3 创新与停止门

slot pooling、projection、distillation、subtraction 和 segment ledger 各自都只是 component。真正的
创新假设是：state creation 时把有限-view residual 训练成跨版本共同项，使 native response difference
相消；同时由 Full/append 同源的 $G_v$ 阻止误差写回。若无 common-mode objective 或无 $G_v$ 的 matched
control 解释了全部收益，则降级为 generic compression；若 `<20%` 内长期低于 70%，判当前
implementation 失败，不回到 rank/probe/token 扫描。

当前没有 Migration Sketch 方法数值。已有 S4 `95.34%/99.46%` 是 oracle ceiling。reference one-view
decode 为 `M=32: 28,606,464 FLOPs = 0.5996% Exact-All`，`M=34: 30,394,368 = 0.6370%`；包含 paired
views、`H+2` slots、every-segment entry states、contextual-atom sums、global writer state 和 boundary
witness 示意下界的 state subtotal 分别为 `641 KiB/user` 与 `681 KiB/user`。partial Parent drop 上界为
$B C_\phi+3,575,808$ FLOPs 加 witness I/O、约 36 KiB view 写；Current boundary 为
$B C_\phi+1,787,904$ 加 I/O、约 18 KiB view 写。request overhead 为 `6.25%/6.64% prefix QK+AV`。
initial BUILD、$C_\phi+C_G(M)$、实际
schema/allocator 与 runtime 全部 pending。百分比暂以 sealed legacy denominator 说明量级；prospective
contract 必须重新封存 $\mathrm{ExactAll}^{G}$。
