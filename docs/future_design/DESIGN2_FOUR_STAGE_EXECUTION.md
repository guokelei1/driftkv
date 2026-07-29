# CohortKV Design 2 四阶段执行方案

日期：2026-07-29

状态：D2 的执行控制方案。系统设计、接口、实验和 gates 仍以
`DESIGN2_FINAL_PLAN.md` 为准；本文只回答“如何分阶段推进而不被错误的早期假设锁死”。
Stage A 已完成并冻结；Stage B 的实现与 W1/W2 证据已完成，但独立 W4 normal/hard-failure
尚待安全 GPU2，因此 Stage B 未冻结。物理 GPU0/GPU1/GPU3 上的 W3 NCCL normal 与
hard-failure 已作为开发诊断通过，C0 小样本接线也已完成。当前双 Gate 保持
development GO、正式 Stage-C evaluation BLOCKED。当前事实见
`DESIGN2_DEVELOPMENT_STATUS.md` 和
`DESIGN2_STAGE_B_HANDOFF.md`；进入 Stage B 的原始回查边界仍见
`DESIGN2_STAGE_A_HANDOFF.md`。W3 上的 full682 fixed-action physical-lowering discovery 与
full-payload validation 已完成并为正式 Stage C 选出了候选机制，但均为
`scientific_result=false`，不替代 W4 或 formal protocol。

---

## 0. 怎么使用这四个 Stage

这四个 Stage 不是四张保证成功的 checklist，也不是做完前一张就永久证明前一层没有问题。

每个 Stage 只冻结：

1. **入口**：开始时必须已经可信的输入；
2. **本质目标**：这一阶段真正要消除的不确定性；
3. **出口**：下一阶段可以依赖的可检查产物；
4. **探索空间**：允许在执行时调整的实现选择；
5. **最大风险**：最可能让“看起来完成”变成假完成的条件；
6. **回退规则**：下一阶段出现什么症状时，应回来修这一层。

阶段完成的含义是：

> 当前证据允许下一阶段暂时依赖它，但保留一份明确的 provisional-assumption ledger。

进入下一阶段前必须重新审查上一阶段，而不是只看一个 `passed=true`。

---

## 1. 四阶段总览

| Stage | 本质 | 论文位置 | 对应 D2 总计划 |
|---|---|---|---|
| A：可信边界 | 证明输入、模型切分和计量口径可信 | enabling/implementation | P0–P1 |
| B：物理稀疏原语 | 证明 owner-local retained、sharded exact/append 与 segmented assembly 能独立成立 | D2 核心机制前半 | P2–P3 |
| C：完整 wave | 在固定 action 下把 logical savings 降低成 integrated physical savings | 完整 D2 主设计 | P4 |
| D：论文证据 | 决定哪些机制和 claims 真正成立 | evaluation/claim selection | P5–P7、E0–E5、Gates |

只有 Stage A 主要是准备。Stage B 已经开始实现 D2 的核心数据平面，Stage C 才形成完整 Design 2，
Stage D 决定它是否足以作为论文贡献。

当前的 C0 是已完成的 Stage-C 前沿 development-only 接线诊断，不表示 Stage C 已完成或正式
开始。它在等待 W4 时消除了 16-record fixture 上的 wave/state-machine 接口风险，但不产生
full-cohort、timing、capacity、正式 epoch-publication 或论文证据。

后续完成的 W3 full682 discovery 进一步证明候选 lowering 有正向开发信号，但仍不改变 formal
Stage C 状态。四阶段的核心因果链统一为：

```text
D1 immutable actions / logical sparsity
  → naive sharded physical execution
  → D2 shape-aware segmented physical sparsity
```

---

## 2. Stage A：建立可信、可执行、可计量的边界

### 2.1 本质目标

不是“把 dataclass 和 helper 写出来”，而是证明：

> frozen action partition、HSTU exact frontend、append 语义和 phase ledger 可以被一个统一接口
> 无歧义地表达，并且不改变现有数值结果。

### 2.2 入口

- frozen H12 artifact 和逐 record lineage；
- Stage 4.5 direct-old-K/V operator；
- Stage 4.9 exact/append helpers；
- Stage 5 preflight、fallback、COW 和 lineage 语义；
- 当前 HSTU public `forward/compute_kv`。

### 2.3 出口

下一阶段只能依赖以下结果：

- canonical `D2ActionPlan`，其 counts、record set 和 upstream hash 可复现；
- `D2WavePlan/D2WaveReport` 的最小单卡 adapter；
- split frontend 与原 exact path 的 K/V、hidden、score、Top-100 等价；
- phase-aware lookup/token/byte ledger，明确区分 retained、natural exact、delta 和 latest；
- action-plane 与 physical-plane 分离的 report schema；record/action fraction 不得代替
  token/byte/padding/rewrite/time fraction；
- H12 step 1 的静态 request/dedup ceiling；
- P2P topology microbenchmark 和 per-process capacity 初值；
- 一份 Stage-A handoff，标出哪些结论是 measured，哪些只是 hypothesis。

### 2.4 允许探索

- exact helper 最终放在哪个 library 模块；
- frontend split 的内部函数边界；
- plan/report 使用 dataclass、typed dict 或轻量 schema；
- instrumentation 用 hook、adapter 还是 wrapper；
- static characterization 的 batching scope。

不允许改变 action partition、模型语义、append 顺序或冻结 artifact。

### 2.5 最大风险

**最大风险是把错误的 phase boundary 包装成一个很漂亮的接口。**

尤其要防：

- 把 compiled record 的 append 误记成 embedding-free；
- 把 target-prefix token 和 post-append final token 混用；
- helper 搬迁后 dropout、padding、dtype 或 mask 发生微小变化；
- logical vector volume 被误记为实际跨卡 bytes；
- `134/682 = 19.6%` exact-route records 被误记为只剩 19.6% compute/lookup/communication；
- 单卡 capacity 估计遗漏未来 one-process-per-GPU 的 CUDA context 和 replica。

### 2.6 Exit gate

- old/new exact path 通过既有 tolerance；
- action/token ledger 能从 lineage 机械重建；
- current Stage 5 adapter 的 coverage、lineage 和 commit/abort 行为不变；
- lookup counter 能按 phase 解释所有调用；
- immutable action constraint 可机械检查：除显式 safety fallback 外，runtime requested action
  与 frozen plan 完全相同；
- 没有依赖多卡才能解释的隐藏字段。

### 2.7 Stage B 何时必须回退到 A

在 B 中出现以下现象，不能继续把它当作“分布式 bug”：

- world-size-1 的新 executor 已经与旧 exact path 不同；
- multi-rank 汇总后 token/byte ledger 无法回到 ActionPlan；
- append 阶段无法从 retained/exact output 统一调用；
- per-rank capacity 与 A 的模型相差一个完整 model/program replica；
- output dtype/layout/length contract 在不同 primitive 间不一致。

这些说明 A 的接口、计量或模型切分仍不完整，应先修 A，再继续 B。

---

## 3. Stage B：证明物理稀疏原语，而不是急着跑完整系统

### 3.1 进入前的反向审查

开始 B 前重新问：

- A 的 plan 是否包含 distributed routing 所需的全部长度和 owner 信息？
- phase counter 在不同进程中能否无歧义聚合？
- exact frontend 是否真的与 process/device 无关？
- capacity 是否包含 per-rank model、program、context 和 transient？
- empty rank、padding-only 和 zero-exact rank 是否在接口层有合法表示？

任何答案不确定，都先补 Stage A handoff。

### 3.2 本质目标

证明三个 physical-sparsity primitive 可以在同一 SPMD contract 下独立成立：

1. `compiled_retained` 在 old-K/V owner 本地执行，不搬 old K/V、不访问 embedding；
2. exact/natural/append 通过 row-sharded embedding 获得正确 item vectors，并在 record owner
   产生正确 current-model K/V；
3. compiled append 只生成 suffix K/V，retained segment 不被重新写入；logical-complete
   segmented manifest 与 contiguous reference 等价。

### 3.3 入口

- Stage-A frozen schemas、adapter 和 equivalence tests；
- phase-aware request ledger；
- declared record/embedding owner rules；
- `torchrun`/NCCL launcher 和明确 timeout/failure propagation。

### 3.4 出口

- 1/2/4-rank owner-compute primitive；
- 1/2/4-rank sharded exact/append primitive；
- suffix-only append 与 segmented destination correctness；
- replicated-vs-sharded exact equivalence；
- deterministic collective ordering，包含 empty ranks；
- actual requested/unique/remote IDs 和 collective bytes；
- per-rank HBM/context/program/model ledger；
- owner-local、P2P steal 和 non-owner output-return 的初步实测；
- world-size-1 pytest 与显式 2/4-rank launcher。

此阶段输出 private fragments 和 ready metadata 即可，不要求完成全局 target epoch。

### 3.5 允许探索

- all-to-all、all-to-all-v 或显式 P2P routing；
- padded counts 或 variable-size exchange；
- record owner 与 embedding owner 的映射；
- batch/bucket size；
- primitive 内是否预排序；
- program replication 和 stream arrangement。
- `(S,R)` compiled grouping 与按 `F` 的 exact physical pool。

先追求正确、可解释的 baseline。dedup/overlap 只有在 A 的 ceiling 和 B 的实际 exposed bytes
支持时才加入，不能为了“计划里写过”而实现。

### 3.6 最大风险

**最大风险是两个 primitive 各自正确，但它们不共享一个可组合的 collective/output contract。**

常见症状：

- compiled-only rank 跳过 exact collective，其他 rank 死锁；
- repeated IDs、empty bucket 或 uneven counts 恢复顺序错误；
- row-sharded lookup 正确，但 fan-out/transient 使 HBM 爆炸；
- suffix append 又物化完整 cache，使 owner-local retained 的收益被 rewrite 吃掉；
- semantic exact reasons 被错误地固化成不同 physical phases；
- dense trunk replica 或独立 CUDA context 消除 capacity 动机；
- microbenchmark 支持 owner-compute，真实 concurrent execution 却相反；
- first-context/startup 时间混入 steady-state。

### 3.7 Exit gate

- 1/2/4 rank 均无 collective-order deadlock；
- exact vectors/K/V 与 replicated baseline 等价；
- compiled retained normal path 的 phase-local invariants 成立；
- append lookup 被正确计入；
- retained destination 只写一次，suffix-only materialization 与 contiguous reference 等价；
- actual bytes 可由 request/owner distribution 重建；
- per-rank capacity 有足够 margin；
- launcher 能可靠传播任一 rank failure。

若 replicated dense trunk 在目标 capacity 配置中无法 admission，不得悄悄把 TP 塞进 B。此时应
明确选择：收缩为 table-sharded scope，或单独触发 D2 P7/TP 的新设计审查。

### 3.8 Stage C 何时必须回退到 B

- 只有 mixed phase ordering 下死锁；
- primitive 单测正确，但跨 phase stream/event ownership 不完整；
- exact output 无法稳定落到 record/publication owner；
- 1/2/4 rank 的 numeric 或 byte ledger 随调用顺序改变；
- transaction staging 暴露出 output lifetime/ownership 不明确。

若 world-size-1 也失败，继续回退到 A；若只在跨 rank/phase 出现，则主要属于 B。

---

## 4. Stage C：闭合一个真实 fixed-action target epoch

### 4.1 进入前的反向审查

Stage C 使用两个不可混淆的入口：

1. **C0 development entry（GO，闭合已完成）：** W1/W2 与物理三卡 W3
   normal/hard-failure 已通过；固定 16-record fixture 上的 W1/W2/W3 normal 与 W3
   pre-commit abort 已连接 heterogeneous routes、private fragments、epoch state machine、
   abort/readback 和 phase order。所有输出均为 `scientific_result=false`、
   `formal_stage_c=false`，且不含 full-cohort/timing/capacity 或正式 epoch-publication claim。
   此后 W3 又完成了 full682 physical-lowering discovery 与 full-payload validation；它们同样
   是 `scientific_result=false`，只冻结候选机制，不满足 formal entry。
2. **Formal evaluation entry（当前 BLOCKED）：** 必须等正式 W4 normal/hard-failure、
   `freeze_cohortkv_design2_stage_b.py` 写入 summary 和 `--check` 全部通过，并在
   `docs/eval_protocol.md` 冻结新的 D2 protocol，才能执行完整 H12 integrated evaluation。

C0 不替代 W4、不关闭 Stage B，也不满足本阶段出口。后续针对性 development 或正式
evaluation 前都重新检查：

- compiled、exact、natural exact 和 append 是否使用同一个 final extent contract？
- 所有 rank 是否按同一 phase order 前进？
- B 的 private fragment lifetime 能否覆盖 COW commit/abort？
- fallback 后是否会改变 token、capacity 和 collective ledger？
- formal evaluation 前，one-shot all-exact 是否已经能在同一 harness 中作为强 baseline？
- segmented target 是否能被 publication consumer 或下一轮维护直接读取，而不在 timer 外
  强制拼回 contiguous cache？

### 4.2 本质目标

不是“把几个模块串起来”，而是证明：

> 一份 frozen H12 ActionPlan 可以在同一个 source-read-to-post-commit boundary 内完成
> compiled + exact + append；相对 naive sharded fixed-action execution，shape-aware segmented
> lowering 将 D1 logical sparsity 兑现为可计量的 physical savings，且只发布一个 coverage
> 完整、lineage 正确的 target epoch。

### 4.3 入口

- Stage-B 两类 distributed primitives；
- H12 step 1 canonical ActionPlan；
- D2 transaction/preflight/fallback contract；
- two-stage 和 one-shot all-exact baseline；
- naive sharded fixed-action mixed 和 current SPMD record-DP mixed baselines。
- 若运行超出明确标记为 `scientific_result=false` 的接线 smoke，已在
  `docs/eval_protocol.md` 冻结新的 D2 protocol、计时边界、baseline 和 artifact schema。

C0 已用前三项开发接口和 development-status gate 完成闭合；two-stage/one-shot baselines、
正式 SPMD record-DP 和新 protocol 仍属于 formal evaluation 入口，不能由 C0 的完成状态代替。

### 4.4 出口

- frozen H12 mixed wave 的 1/2/4 GPU integrated run；
- strict-COW normal commit、abort、fallback 和 readback；
- one output per record、无缺失/重复；
- D1-continuity retained-prefix 与 D2-integrated 双边界；
- all-exact、naive sharded fixed-action mixed、D2 physical-sparse mixed 三条同边界路径；
- faster-of-two-stage/one-shot all-exact 对照；
- fixed-action v1→v5 同 binary/同 protocol physical-lowering ablation；
- complete logical/physical work、phase/communication/movement/capacity ledger；
- repeated timing variation 和第一条 preliminary system conclusion。

### 4.5 允许探索

- compiled/exact phase 的执行顺序；
- `(S,R)` compiled extent、按 `F` exact pool 和 rank-local queue；
- segmented finalization 与 suffix-only consumer；
- safe overlap 和 barrier placement；
- fallback regrouping；
- metadata reduction 和 commit coordination；
- owner-local 与已测 placement alternative。

不能改变 H12 requested actions，不能用结果重新调 scheduler，也不能引用 Table 8 代替新 harness
baseline。

### 4.6 最大风险

**最大风险是系统在 correctness 上闭合，但完整 boundary 的收益被 append、collective、COW 或
transaction overhead 消除。**

这不一定是代码 bug，也可能直接否定当前 D2 claim。其他高风险包括：

- natural exact/fallback 改变 collective 顺序；
- strict COW 的 old+new peak 超出 preflight；
- abort 后某个 rank 提前暴露 target；
- one-shot exact 比人为匹配的 two-stage exact 强很多；
- logical lookup 已下降，但 contiguous rewrite、padding 或 fragmented phases 使 naive mixed
  仍不快；
- component timer 看起来快，但 source-to-reclaim wall time 不快；
- 1/2/4 GPU 使用了不同隐含 placement 或同步点。

### 4.7 Exit gate

Correctness 必须全部通过：

- coverage、lineage、checksum、visible version；
- exact/fallback semantics；
- abort 保持 old epoch；
- post-append final cache；
- phase ledger 闭合。

进入 D 的性能条件不是“必须已经赢”，而是至少满足一个：

1. D2 physical-sparse mixed 相对 naive sharded fixed-action mixed 有可归因收益，并相对强
   all-exact baseline 形成有意义的 integrated point；
2. measured bottleneck 明确、与 D 中一个预先定义的 physical-lowering、movement、
   communication 或 capacity hypothesis 直接对应，并且其理论上限足以改变结论。

若两者都不满足，应停止进入大规模 evaluation。不能期待 Stage D 用大量 ablation 把一个没有
system point 的实现“测成”贡献。

### 4.8 Stage D 何时必须回退

- failure matrix 暴露 mixed visibility 或 extent lifetime 问题：回 C；
- 1/2/4 scaling 随机死锁或 bytes 不稳定：回 B；
- phase totals、token counts 或 exact equivalence 不一致：回 A；
- capacity experiment 与 preflight 模型不一致：先回 C 的 staging，再检查 A 的 accounting；
- baseline 边界不公平：回 C 重冻 harness，不能只改 D 的表格。

---

## 5. Stage D：形成论文证据，并允许结论失败

### 5.1 进入前的反向审查

开始 D 前冻结：

- implementation commit/content hashes；
- action plan 和 workload；
- baseline definitions；
- timer、warmup、repeat 和 capacity boundary；
- Stage C 已知的 bottleneck 与 provisional hypotheses。

不允许一边看最终结果一边改变 action、layout 或主指标。

### 5.2 本质目标

不是“把实验矩阵跑完”，而是判断：

> D2 的哪条系统主张在正确、同边界、可复现的证据下成立；哪些机制应删除或降级？

### 5.3 入口

- Stage-C frozen integrated implementation；
- paired baselines 和 timing protocol；
- 已闭合的 correctness/failure smoke；
- measured bottleneck，而不是预想瓶颈。

### 5.4 出口

- 1/2/4 GPU paired results；
- all-exact → naive fixed-action mixed → D2 physical-sparse mixed 的主因果链；
- segmented、shape-aware、exact-pool 的 fixed-action physical-lowering ablation；
- optional plan-known collective ablation；
- optional synthetic lookup + dense-control contention characterization；
- forced-sharding 与 capacity-admission 分开的结果；
- placement、failure 和 tail matrix；
- checked summary 和 artifact-to-claim binding；
- 最终保留、降级、删除的机制表；
- G10 paper-strength 结论。

### 5.5 允许探索

只探索 Stage C 证据支持的方向：

- dedup ceiling 足够才实现 dedup；
- exposed collective 足够大才做 overlap/stagger；
- unsharded admission 确实失败才形成 capacity claim；
- synthetic lookup 与 dense control 分离才允许形成 supporting resource-attribution claim；
- deployment/capacity 真需要 TP 才触发 P7。

探索改变了方法或主 boundary 时，必须建立新的 protocol，不能继续使用原 final matrix。

### 5.6 最大风险

**最大风险是为了保住“D2 必须是论文贡献”的结论而扩大 scope 或事后选择实验。**

具体包括：

- collective optimization 无收益，却保留复杂机制；
- synthetic capacity 只增加未访问 rows，却被用来支撑性能或质量 claim；它最多证明
  unsharded admission 失败，通信收益仍需由真实 accessed IDs 产生；
- 规模不足时只扩 tensor shape，不从真实数据扩大 item/user 覆盖，也不重新训练
  base + 1–2 个短 streaming versions，却把它写成 D1→D2 系统证据；
- synthetic lookup p99 被写成真实 serving latency，或改善其实来自一般 GPU load；
- 单 seed、单配置结果被写成普遍规律；
- 为了结果更好让 D2 根据通信成本重选 exact/compiled actions；
- TP、storage tiering 或 D3 scheduler 被临时拉进来救结果；
- 多个不同 protocol family 被混成一张更好看的表。

### 5.7 Exit gate

- D2 G0–G9 的适用 gates 通过；
- G10 至少一个非平凡系统结果通过；
- correctness、transaction 和 strong-baseline boundary 不退化；
- 负面结果和删除项被保留；
- claim 严格小于等于证据。

若 G10 失败，正确结论是：

> D2 是 D1 的可用 distributed implementation，但当前不足以成为独立 paper design。

这也是有效的项目结果，不允许通过继续增加组件掩盖。

---

## 6. 跨 Stage 反向诊断表

| 后续观察到的症状 | 首先检查 | 可能需要回退 |
|---|---|---|
| 单 rank exact/append 数值不一致 | frontend、dtype、mask、length contract | A |
| token/byte ledger 无法从 ActionPlan 重建 | phase schema、final-vs-prefix 口径 | A |
| 只在多 rank 下错位或死锁 | routing、empty rank、collective order | B |
| per-rank OOM 比预估多一个 replica/context | capacity model 与 process contract | A/B |
| primitive 正确、mixed wave 才死锁 | phase orchestration、stream lifetime | B/C |
| abort 后 target 部分可见 | COW/commit/extent ownership | C |
| exact-route records 约 20%，但 lookup/bytes 没降到约 20% | natural exact、suffix/latest、record length；先检查分母，不改 action | A/C |
| logical token 已降但 complete boundary 不快 | padding、retained rewrite、segmented consumer、phase fragmentation、strong baseline | B/C，可能直接 no-go |
| 当前配置通信占比很小，但 table 仅约 0.60 GiB、unsharded 可轻松 admission | 先放大真实访问压力找 knee；仍受数据规模限制时，从已接受数据扩大真实 item/user，训练 base + 1–2 个短 streaming versions，并重跑相同 D1→D2 流程 | D；新建 discovery protocol |
| scaling 波动大、bytes 随重复改变 | launcher、同步、allocator、routing | B/C |
| capacity/communication claim 解释不唯一 | control、workload contract、计量 | D，必要时回 C |
| 为得到正结果必须改 action/workload | evaluation 已改变方法 | 停止并新建 protocol |

“先检查”不是预设 bug 归属。它只是避免在 D 里用优化掩盖 A/B/C 的基础错误。

---

## 7. 每次 Stage 交接必须留下什么

交接记录保持短小，只需要：

```text
immutable inputs and hashes
outputs/artifacts
passed gates
falsified hypotheses
remaining risks
provisional assumptions
known unsupported claims
first three diagnostics for the next Stage
```

下一 Stage 开始时先复核这份记录，并根据当前代码和测量写一个 just-in-time 小计划。不要现在就把
B、C、D 的内部实现步骤全部锁死；它们必须允许根据前一阶段的新事实调整。

---

## 8. 总体 stop/go 逻辑

```text
Stage A fails
  → repair semantics/accounting; do not start distributed work

Stage B fails
  → distinguish A-contract failure from distributed/scope failure
  → do not hide dense-model capacity failure with an unplanned TP expansion

Stage C fails correctness
  → backtrack by symptom to A/B/C

Stage C has no plausible system point
  → stop before broad evaluation

Stage D fails paper-strength gate
  → retain implementation/negative results; shrink the paper claim
```

这四个 Stage 的价值不是让执行看起来确定，而是让“不确定在哪里、什么时候必须回头、什么时候应该
停止”变得明确。

---

## 9. 执行后的文档同步

这不是第五个实现 Stage，而是每个 Stage 交接后的维护动作。代码、artifact 与文档状态不一致时，
后续 agent 很容易从旧入口恢复已经淘汰的目标。

每次 A/B/C/D 结束后：

1. 先更新 `docs/08_core_insights_and_roadmap.md` 的当前状态、通过/失败 gate 和下一入口；
2. 只有实验语义或可比边界变化时才更新 `docs/eval_protocol.md`，并使用新 protocol/result
   family，不改写冻结历史；
3. 更新 `docs/README.md` 和 `docs/future_design/README.md` 的 active/historical/future
   分类；
4. 在本文和 `DESIGN2_FINAL_PLAN.md` 中只更新真实发生的接口或阶段裁决，不把探索结果倒写成
   预先计划；
5. 对其余 `docs/` 做一次旧目标扫描。历史对照、dataset audit 和冻结计划默认保留原叙述，只在
   顶部补当前状态或新入口，不重写当时的实验结论；
6. 记录被删除、降级和仍不支持的主张，避免后续从 Git 历史或旧 result path 恢复。

文档同步完成不等于 Stage 的科学 gate 通过；它只保证仓库对“现在知道什么、下一步做什么”给出
唯一且可追溯的答案。
