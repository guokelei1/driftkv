# EvoKV One-Release State Refinement 与 Typed Plan IR

更新日期：2026-08-26

本文是论文第 3 章 `Insight-Driven State Refinement` 的底层 plan-IR companion。三条 Design
Insights 与对应机制在同一章中直接衔接；本文只展开这些机制的 typed semantics。它替换旧的
`Translate / Rematerialize / Synthesize / Retire / Route` 动作 catalog。旧 catalog 将完整
修复算法、scope 选择、读取策略、组合算子和生命周期操作放在同一层，既不原子，
也不能在同一段状态上渐进组合。

论文主线不再将 `CAST / PATCH / GROUP / SCALE` 写成四条并列 headline insight，也不在 Insight
和机制之间增加独立 Design Principles。面向读者的表述是三条 Insight 直接推导四阶段状态迁移
流水线；本文保留四个 typed operator semantics，
是因为它们在底层分别改变不同状态语义，而不是因为它们必须成为四个系统阶段。

底层 IR 将问题定义为：

> 对同一段 versioned persistent state，沿版本坐标、contextual residual、evidence-to-carrier
> granularity 和 aggregation mass 四个独立语义轴，执行可组合、有类型、有资源边界的渐进细化。

当前结论是 **one-release mechanism-level admission**，不是 scale action 准入。本文中的 residual
splice 与 mixed-state intervention 都是诊断，不改变已冻结的 dependency-closed scale action set。
本文不定义多 release 组合、debt/rebase policy 或 GPU runtime；它只给这些后续层交付 typed state。

## 1. 为什么旧动作抽象必须被替换

旧计划的核心形式是将 state 分区后，对每个区域选择 Keep、Translate、Rematerialize、
Synthesize 或 Retire。它有三个结构问题。

1. Translate 和 Rematerialize 是两套完整算法；exact rematerialization 覆盖同 scope 的
   translation，因此看起来只能二选一或分区执行。
2. weighted Landmark-64 实际同时做了 evidence 选择、coverage grouping、Current 物化和
   mass assignment；`SYNTHESIZE` 因而是宏计划。
3. Retire、Route 与 Reweight 在逻辑读路径上都是 read contribution 的系数，却被写成三类
   headline mechanism；Fuse/Commit 又属于另一个抽象层。

新设计不再追加大动作，而是寻找能够组合出这些宏计划的小语义基底。

## 2. Typed state 和抽象边界

一次转换中的 persistent segment 写成：

~~~text
S = (Omega, Z, producer_version, readable_version, mass, lineage)
~~~

- `Omega`：有序 evidence coverage map，包括 item/action/time、occurrence 顺序及 evidence-to-carrier 对应；
- `Z`：实际的分层 K/V payload；
- `producer_version`：最初生成 payload/evidence state 的模型版本；
- `readable_version`：当前 payload 声明可被哪个 reader 解释；
- `mass`：在 HSTU 非归一化 pointwise aggregation 中代表的 occurrence mass；
- `lineage`：producer、base state、dependency boundary、exact/approx 类型及置信度。

`Omega` 不是无序 item set；`lineage` 也不是可选 metadata。若丢掉时间/顺序、base lineage
或 dependency boundary，后续 PATCH 就无法判定是否合法。

一个 typed operator 语义要求：

1. 输入和输出都是合法 typed state；
2. 只改变一个独立状态轴，或一组无法保留合法中间态的联动语义；
3. 可以独立调度、计算资源和做 leave-one-out 干预；
4. 不能用其他已有叶子语义完整表示。

“语义可分解”不等于“每次调用都便宜”。一个 exact PATCH 的 causal closure 可能接近全量；
系统必须证明编译后的整体 plan 在目标资源上低于 Exact-All，而不能由“操作名很小”
推出成本很小。

## 3. 论文主线：三条 Insight 与四阶段流水线

当前最清楚的三条 Insight 是：

1. **分布式失配与非对称修复**：单 layer-0 修复不足，而 dependency-closed Tail-128
   在五条 edge 都有效但只恢复 19.4%–25.8%。Tail 是当前便宜且有用的 causal boundary，
   尚未证明是等宽区域中最敏感的位置。
2. **共享且可转换的版本变化**：parameter-only joint layerwise K/V CAST 在五条 edge
   恢复 21.6%–64.1%，说明 mismatch 包含无需 raw replay 的共享结构成分。
3. **保持证据质量的紧凑重算**：Current repair 不必逐事件物化，但减少 carrier 后必须
   保留 represented occurrence mass。GROUP 和 SCALE 是同一条论文 Insight 的两个底层语义。

由此推导的执行流水线是：

~~~text
PLAN(repair width r, carrier count c)
  -> CAST(large stable region)
  -> GROUP(repair region r -> c) -> PATCH(c Current carriers)
  -> SCALE(represented mass) -> UNION/COMMIT
~~~

`PLAN` 在物理执行上最先发生，选择 CAST/PATCH 分界和 carrier density；它是未来的
target-free planner，不是状态修复算子。exact PATCH 完全覆盖的 scope 不再先做 CAST。
GROUP 要在 PATCH 之前才能减少主要重算；先 PATCH 再 GROUP 只能节省后续读写和存储。

## 4. 四个底层语义

### 4.1 `CAST`：版本坐标

~~~text
CAST(parent_state, parent -> current)
  -> current-readable approximate state
  preserves: ordered coverage, occurrence identity, represented mass
~~~

`CAST` 只负责 projection/coordinate mismatch，不声称恢复 contextual hidden drift。当前实现是
parameter-only、joint K/V、layerwise 的闭式映射；不从 Current target K/V 拟合 mapper。K/V
必须联合转换，因为先前的 K-only/V-only 干预没有稳定单侧主导。

`CAST(all stale state)` 是宏计划；最小作用对象是带完整 layer bundle 和 lineage 的
dependency-safe typed segment。

### 4.2 `PATCH`：contextual payload residual

~~~text
PATCH(base_state, typed_delta)
  -> base payload + delta in a declared scope
  preserves: coverage and represented mass
~~~

PATCH payload 至少需要声明：

- 它由哪个 base lineage 产生；
- 它属于哪个 target version 坐标；
- 它修改的 token/layer scope 和 dependency closure；
- exact、approximate 或 transient diagnostic 类型。

当前 exact tail replay 被重新解释为一种 PATCH generator：从 raw history 生成该 scope 的
Current payload correction。`Tail-128` 和 `Exact-All` 是 PATCH 的宏计划，不是原语。

需要严格区分两件事：

1. payload 层可以写成 `Z + Delta`；
2. HSTU 对 K/V 的读取包含非线性，所以不能推出质量改善可加或单调。

只有当多个 delta 位于相同坐标、base 和合法 dependency boundary 下，才允许将它们组合。
若 exact PATCH 完全覆盖同 scope 的旧 payload，编译器应删除被覆盖的 CAST；这是死代码
消除，不是否认 CAST/PATCH 在未覆盖 scope 或 approximate residual 上的组合价值。

### 4.3 `GROUP`：evidence-to-carrier granularity

~~~text
GROUP(ordered evidence coverage, coverage_map)
  -> fewer or reorganized carrier scopes
  changes: support/cardinality and coverage map
  does not itself invent valid K/V content
~~~

`GROUP` 是结构原语，回答每个 carrier 代表哪些历史 occurrence。它之后仍需要 CAST
或 PATCH 产生可读 payload，并需要 SCALE 声明 represented mass。当前 evenly spaced landmark
是最小机制 probe，不是已冻结的 grouping policy。

`GROUP(128 -> 64)` 和“选择 recent tail”都只是 scope/layout 决定。如果未来 many-to-few generator
可以生成无法由 `GROUP + PATCH + SCALE` 表示的新状态类型，再增加 `GENERATE`；当前
证据不需要它。

### 4.4 `SCALE`：read contribution / represented mass

~~~text
SCALE(state, alpha)
  -> same payload, version and coverage
  -> read contribution multiplied by alpha
~~~

SCALE 表达 multiplicity、confidence 或 transient read mask。当前 probe 为适配现有 HSTU 接口，
暂时将 mass 乘到 V contribution；最终 runtime 应将其作为 typed metadata 由 reader 消费。

Retire/Route 只在“逻辑读 view”上可以改写成 `SCALE(0)` 或 candidate-dependent `SCALE(g(q,S))`。
这不等于物理删除：持久化回收是不可逆的 COMMIT 决定，不能用一个临时 alpha=0 偷换。
当前 Route/Retire 负结果也不支持将 candidate-dependent SCALE 加入第一版计划。

## 5. 寻址、组合和生命周期

### `SLICE`

只声明作用范围，例如 lineage segment、recent interval 或 dependency boundary。它是 plan
IR 的寻址语义，不单独修复状态。workload insight 应进入 SLICE predicate 或 PATCH value
estimator，而不是不断产生新原语。

### `UNION`

生成一个保留 chronology、coverage、version、mass 和 lineage 的 typed read view。UNION 不修改 logit，
不做 score mixing，也不自动将不兼容 segment 当成同质 K/V。

### `COMMIT`

将通过验证的 view 持久化并原子更新 lineage。Transient repair 和 persistent promotion 在此分界。
COMMIT 是生命周期事务，不计入“几类修复原语”。

## 6. 最小 plan IR

~~~text
scope := SLICE(source, predicate)
       | GROUP(scope, ordered_coverage_map)

state := SOURCE(scope)
       | CAST(state, target_version)
       | PATCH(state, typed_delta)
       | SCALE(state, alpha_or_mass)

view  := UNION(state...)
plan  := COMMIT(view)
~~~

一段状态可以连续接受：

~~~text
S' = SCALE(PATCH(CAST(S), Delta), alpha)
~~~

这是 IR 可表达性，不是要求物理执行对同一 scope 固定跑完四步。实际流水线会先
PLAN 分区：大范围经 CAST，expensive repair region 经 `GROUP -> PATCH -> SCALE`；若 exact
PATCH 覆盖某 scope，该 scope 的 CAST 被编译器删除。

当前允许的编译优化只包括已验证或语义严格成立的规则：

- identity 操作可删除；
- exact PATCH 覆盖的同 scope CAST 可删除；
- 同 state 上的静态 SCALE 可合并为系数乘法；
- 多个 PATCH 只有在 base/version/dependency 合同兼容时才能合并；
- `GROUP -> PATCH` 和 `PATCH -> GROUP` 不是代数上可交换的恒等式，只能作为两个计划点测量。

## 7. 旧动作的新表达

| 旧名称 | Refinement Algebra 表达 |
| --- | --- |
| No-op | identity |
| Translate-All | `SLICE(stale) -> CAST` |
| Tail-128 | `SLICE(last128) -> Closure -> PATCH_exact` |
| Exact-All | `SLICE(all) -> Closure -> PATCH_exact` |
| weighted Landmark-64 | `GROUP(128->64) -> PATCH -> SCALE(2)` |
| Translate + Tail | `CAST(stale) -> PATCH_exact(residual scope)` |
| Synthesize | 当前为 `GROUP + PATCH + SCALE` 宏计划 |
| Reweight | `SCALE(alpha)` |
| Retire read mask | `SCALE(0)`；物理回收仍需 `COMMIT` |
| Route | request-dependent `SLICE + SCALE(g(q,S))`；当前不准入 |
| Fuse | `UNION` |
| Commit | lifecycle transaction |

## 8. 新的机制实验

脚本：

~~~text
scripts/insight/probe_refinement_algebra.py
~~~

结果：

~~~text
results/yambda500m_small_seed17/insight_refinement_algebra_v1/
~~~

固定范围为 Small/seed17 五条 v0->v1 ... v4->v5 edge，每条 edge 最多 256 个 append-free、
512-token、first-request-per-UID 请求，共 1,267 个请求。它不训练模型、不拟合 target K/V、
不用 label 选择 scope、不做 score mixing。Current/Reuse sealed logit 复现最大误差为
`7.2e-7`。

所有 recovery 都是 append-free cohort 上的 output-fidelity recovery：

~~~text
R = 1 - gap(plan, Current Exact) / gap(Reuse, Current Exact)
~~~

它不是完整 rolling AUC/log-loss recovery。单独封存的固定
`CAST384 + GROUP/PATCH 128->64 + SCALE2` 已随后完成 full-population rolling AUC；该结果改善
4/5 edge、在 v4->v5 失败，不改变本节这些 output-fidelity 数字的语义。

### 8.1 CAST 与 PATCH 在同 scope 上确实可组合

| 路径 | 五条 edge recovery | 平均 |
| --- | ---: | ---: |
| `CAST(all)` | 21.6%–64.1% | 43.0% |
| `PATCH_exact(tail128, Parent base)` | 19.4%–25.8% | 23.1% |
| `CAST -> base-conditioned PATCH tail128` | 38.7%–72.6% | 57.0% |
| `CAST + Parent-generated additive tail residual` | 37.7%–79.6% | 61.2% |

Parent base 上由 Current raw-history replay 生成的 tail residual，加到 CAST base 的同一 tail scope 后，
五条 edge 都比 CAST/PATCH 较好的单项额外恢复 10.3–23.5 percentage points，平均额外
18.2 points。它比直接在 CAST prefix 上重放 tail 的 57.0% 路径平均再高 4.1 points，
但 v4->v5 低 1.0 point。

这个干预很重要：先前 Translate+Tail 的互补可能只来自 prefix/tail 分区；新结果说明，
在同 tail scope 上，coordinate base 和 contextual delta 也有稳定的叠加结构。因此 CAST/PATCH
不只是旧 Translate/Rematerialize 的改名。

同时，PATCH 仍必须是 typed/base-aware：当用 `Delta = patched(base) - base` 重构其原 base 时，
K/V 最大误差为 `6.0e-8`；不能由此推出任意 model pair、scope 和 lineage 间的 delta 都可携带。

### 8.2 exact PATCH 支持死代码消除

将 full CAST 后的 tail 再做 exact PATCH，与只 CAST prefix、然后 PATCH tail 的最终 K/V 在所有请求上
完全一致，最大误差为 0。这证明编译器可以安全删除被 exact PATCH 完全覆盖的 CAST
工作，而不是在逻辑 IR 中将两者定义为互斥 action。

### 8.3 GROUP/PATCH 顺序形成计算—精度选择，但不是新的质量机制

比较：

~~~text
GROUP raw tail -> PATCH carriers -> SCALE
PATCH dense tail -> GROUP carriers -> SCALE
~~~

在 8/16/32/64 carriers 上，两个顺序的平均绝对 recovery 差分别只有 0.54、0.57、0.71、0.49
percentage points，方向跨 edge 改变。64-carrier 的单 edge 差为 -0.53 到 +1.32 points。

因此当前不支持将两个顺序写成两种质量机制；它们主要是两个 cost plan。在当前 cohort
上，先 GROUP 只重算 carrier，与先重算 128 个 dense states 后再 GROUP 的 output fidelity 接近，
因而 compute-first 顺序是更合理的 provisional 实现。

### 8.4 carrier density 是可用的渐进细化轴

对两个顺序和五条 edge，carrier 数从 8 -> 16 -> 32 -> 64 -> 128 时共有 40 个相邻
密度增量，39 个非负。唯一反例是 v0->v1 的 `GROUP->PATCH` 从 64 到 128 降低 0.012
percentage point，量级极小。

但低密度不一定安全：8 carriers 在三条 edge 仍是负 recovery，16 carriers 也有两条负值。
所以系统需要将 carrier density 与最大 represented mass 作为显式合同，但不能从单 seed
冻结全局 `rho_min` 或 `mu_max`。

### 8.5 SCALE 是 compact replay 中独立且必要的接口语义

在 carriers=8/16/32/64、两个 GROUP/PATCH 顺序和五条 edge 上，共 40 个非平凡 SCALE
消融，mass-aware 路径全部优于 unscaled 路径。64 carriers 上的改善为 9.0–95.3 percentage
points。carriers=128 时 mass=1，SCALE 退化为 identity。

这不说明单一 scalar 能表示任意异质 evidence；8-state + mass16 仍可失败。它证明的是：
payload、carrier density 和 represented mass 不能被合并成一个隐式状态语义。

### 8.6 完整计划的理论计算

在 4-layer/context512、recent-128 scope 上，dense Tail-128 PATCH 重算 Exact-All 的
25.0% token-layers，causal attention-pair work 为 Exact-All 的 43.7%。先将 128 条 evidence
GROUP 成 64 carriers 再 PATCH/SCALE，两者分别下降到 12.5% 和 20.3%。GROUP 和
SCALE 的附加工作对 carrier 数是线性的，不会在算子量上抵消减少的 causal replay。

按一乘一加为 2 FLOPs、理想 causal kernel 只计算有效 pair 的保守口径，完整 Exact-All 为
0.625 GFLOPs/user；CAST384 为 0.201 GFLOPs，compact GROUP/PATCH/SCALE 为 0.099 GFLOPs，
完整 Our 为 0.301 GFLOPs，即 Exact 的 48.0%、理论减少 52.0%。该主值已经包含 per-user CAST，
但排除跨用户摊销的一次性 CAST-map 构造、memory I/O 和共同 serving work。

当前 dense PyTorch graph 的对应比例为 34.1%，因为 Exact 仍计算 causal mask 上方的矩阵元素；它只
作为实现图 companion，不作为 headline。两种 FLOP 口径都不能证明 CUDA latency、kernel utilization
或实际 KV 带宽改善，这些属于 Design III Runtime。

## 9. Insight 如何推导流水线和底层语义

| 已观察 Insight | 需要的语义 | 逻辑链路 | 证据强度 |
| --- | --- | --- | --- |
| parameter-only joint translation 在五条 edge 稳定恢复 | `CAST` | mismatch 包含不读 raw history 也能修复的版本坐标分量 | 直接干预 |
| contextual blocks 产生 64.6%–92.6% Parent-all gap；K/V 和 early/middle dependency 共同传播 | `PATCH` | CAST 无法恢复上下文协同漂移，需要 base-aware、dependency-typed residual | source intervention + closure intervention |
| CAST 后的 residual repair 稳定互补，同 scope additive residual 再次稳定恢复 | `CAST + PATCH` 组合 | 坐标 base 与 contextual delta 是可叠加而非互斥 action | 五 edge 同 scope 干预 |
| Current-state anchor 而非 eviction 驱动 dilution；carrier density 39/40 步非下降 | `GROUP` | 系统需要独立调整 Current carriers 覆盖和密度 | controlled dilution + density intervention |
| 无 mass 的 compact state 恶化，mass-aware 在 40/40 非平凡对照中更好 | `SCALE` | GROUP 后必须保留 aggregation mass；GROUP+SCALE 共同构成 compact-replay insight | 两顺序、五 edge 消融 |
| raw embedding drift 弱且变号，Route/Retire 多数恶化 | 不增加 workload-specific primitive | embedding/user/candidate feature 只能成为未来 SLICE/PATCH value 输入 | 负结果 |

因此 `Insight -> Design implication -> corresponding mechanism` 链条在 mechanism level 已经初步闭合：

~~~text
cross-release projection drift
  -> CAST

contextual Transformer co-adaptation and dependency propagation
  -> PATCH

small Current-state anchoring plus a density/recovery frontier,
while unnormalised aggregation exposes represented mass
  -> GROUP before PATCH + SCALE

typed mixed-lineage state
  -> UNION -> COMMIT
~~~

Release-benefit targeting 和 novel-to-prefix harm 不产生新原语。它们定义的是质量目标、evaluation
weighting 和 failure analysis；当前没有稳定 target-free request route 能把它们转成读路径策略。

## 10. 抽象评估

相比旧 catalog，四阶段 Pipeline + Refinement IR 在抽象层次上更好，原因是：

1. **正交性更强**：CAST/PATCH/GROUP/SCALE 分别改变 version coordinate、payload residual、
   coverage/cardinality 和 read mass。
2. **可组合**：同一 state 可逐步细化，旧 action 只是编译后的宏计划。
3. **可优化**：exact PATCH 覆盖的 CAST 可消除，GROUP/PATCH 顺序可按计算/I/O 不同编译。
4. **与 workload insight 解耦**：embedding、user、candidate 特征改变 scope/value estimate，不改变
   instruction set。
5. **反例可表达**：低 carrier density 失败、Route 失败或 CAST 无效都只会改变计划，不要求
   再加一类宏原语。

但它尚不是完成的可部署系统：

- PATCH 的 target-free residual generator/estimator 还没有验证；
- payload delta 的可加性不保证质量单调；
- GROUP 必须保留顺序、时间和 dependency coverage，不能退化为无序压缩；
- SCALE 只修复 mass，不能使过度压缩的 heterogeneous evidence 恢复语义；
- `P(B1) subseteq P(B2)` 是理想 compiler 性质，不是当前已证明的 quality 单调定理；
- 43%/57%/61% 仍是 output-fidelity 机制数字；固定计划虽已有完整 rolling AUC，但结果跨 edge
  混合且尚无端到端成本，因此仍不是论文最终 quality-cost frontier。

准确裁决是：

> 三条 Insight 已能初步推导四阶段 Pipeline，四个 typed semantics 足以作为稳定底层 IR；固定计划
> 已证明该 IR 可以产生真实 rolling AUC 收益，也暴露了失败 edge。当前仍不准入 always-on 配置、
> 预测器、阈值、scale scheduler 或已证明的端到端性能。

## 11. One-Release 计划选择与成本边界

每个原语 application 写成：

~~~text
o = (instruction, typed_scope, parameter, dependency_contract)
~~~

成本保留为向量：

~~~text
C(o) = (GPU work, raw-history I/O, state I/O, write bytes, storage, latency)
~~~

第一版无需先实现 learned scheduler。可以事前固定一个 `(repair width r, carrier count c)`，并在
Design I 中验证完整计划的解析 FLOPs 低于 Exact-All；真实 GPU、raw I/O 和 storage 分量由
Design III Runtime 分别实测。未来若人口预算需要异质分配，编译器才使用 held-out 验证过的
target-free marginal value。无论是否优化，都不用任意权重将 GPU、raw I/O 和 storage 压成一个
无法解释的总分。

预算可以调整：

- CAST 的 lineage/segment 范围；
- PATCH 的 scope、closure 和 residual fidelity；
- GROUP 的 carrier density 和 coverage map；
- SCALE 的 represented mass/confidence；
- transient UNION 是否值得 COMMIT。

当前不冻结 `tau_cast`、`tau_patch`、`rho_min`、`mu_max` 或 `tau_persist`，也不要求这些阈值
先于固定计划资格出现。策略不得读取 future label；label 只用于事后 rolling quality 评价。

## 12. 向 Continuous State Evolution 的 typed handoff

底层 typed segment 仍保留合法 CAST/PATCH/COMMIT 所需的 producer、base、coverage 和 mass metadata；
Continuous 不把这些字段各自变成新的策略维度。它只从 lineage 聚合出三个 controller field：

~~~text
last_exact_or_rebase_version,
approximation_depth,
estimated_compatibility_debt
~~~

- `last_exact_or_rebase_version` 定义当前近似链的 anchor；
- `approximation_depth` 受硬上限 `H` 约束；
- `estimated_compatibility_debt` 用来为候选 `(r,c)` 计划预测执行后的剩余误差。

Design II 在候选计划中选择满足 `D_hat(S,p) <= tau` 的最低成本方案；没有方案满足阈值或 depth
达到 `H` 时 Exact/Rebase。sampled Current-Exact shadow 为该 estimator 提供无标签安全反馈，
Normal/Warning/Invalid 分别对应保持、加固和 Rebase。`D_hat`、`tau/H` 与 shadow policy 仍需实验，
不是本 IR 已经证明的性质；Design III 再实现 batching、writeback 和 atomic COMMIT。

## 13. 下一步 One-Release 实验

当前不再枚举新的大动作，也不立即训练 scheduler。下一轮先补三条 Insight 的直接证据，
再做系统资格：

1. **Position/closure**：对 old/middle/recent/random-128 做等宽诊断性 region intervention，
   并将位置敏感性与可执行 causal-closure 成本分开报告。
2. **CAST decomposition**：按 normalized layer bundle 和 token quartile 分解 CAST 贡献，不从
   aggregate CAST 直接外推“每层、每个 token 都有效”。
3. **Full-plan theoretical cost**：已计入 CAST、compact PATCH 和 SCALE；Our 为 Exact 的
   48.0% causal FLOPs。CUDA time、raw-history/state I/O、persistent bytes 和 makespan 延后到
   Design III Runtime，不用理论值代替。
4. **Rolling safety boundary**：固定计划的完整 AUC 已报告且在 v4->v5 失败；下一轮在新合同下
   验证事前安全判断/Exact fallback，不用该 qualification label 调 `r/c`。
5. **外推边界**：在额外 seed/更大模型上验证三条 Insight；不从 Small/seed17 直接
   冻结 repair width、carrier density 或 mass 阈值。

固定计划已证明 IR 可产生 rolling AUC 收益并理论减少 52.0% compute，但 4/5 改善尚未形成稳定
跨 edge 次序。只有补上失败边界，Design I 才能作为受控 transition 向 Continuous 交付；实际性能
由后续 Runtime 单独验证。
target-free residual value、held-out threshold compiler 和 budget allocation 是可选增强，不是
Design II 的前置条件。

当前宏 baseline 仍保留 No-op、Translate-All、Tail-128、weighted Landmark-64、Translate+Tail-128 和
Exact-All，但它们的唯一角色是检验新 IR 是否真的带来渐进 quality-cost frontier。

## 14. 证据与清理边界

当前主结果：

- `results/yambda500m_small_seed17/insight_refinement_algebra_v1/`；
- `scripts/insight/probe_refinement_algebra.py`；
- `results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/d14_one_release_refinement_auc_v1/`；
- `scripts/insight/run_one_release_auc.py`；
- `scripts/insight/summarize_one_release_quality_compute.py`。

历史 capability 结果
`results/yambda500m_small_seed17/insight_state_primitive_discovery_v6/` 作为负结果和原始数字证据保留，
但不再定义当前设计。过时的 `probe_state_primitives.py` 和旧原语设计文档已被新 probe/本文替换。
