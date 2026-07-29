# CohortKV Design 2 最终综合计划

日期：2026-07-29

状态：Design 2 当前实施 source of truth。实际推进按
`DESIGN2_FOUR_STAGE_EXECUTION.md` 的 A→B→C→D 控制；历史中间稿已在合并后删除，未冻结的
D3 讨论见 `DESIGN3_FUTURE_DIRECTION.md`。Stage A 已完成并冻结；Stage B 的实现与 W1/W2
证据已完成，三 rank W3 NCCL normal/hard-failure 开发诊断也已通过，但正式 W4
normal/hard-failure 尚未闭合，因此 Stage B 仍未冻结。当前采用双 Gate：
`stage_c_development_entry=go` 下的 C0 小样本接线已经完成，只允许继续针对性开发诊断，
`stage_c_evaluation_entry=blocked` 继续禁止正式 Stage-C evaluation 和论文证据。入口、交接与
实时台账分别见 `DESIGN2_STAGE_A_HANDOFF.md`、`DESIGN2_STAGE_B_HANDOFF.md` 和
`DESIGN2_DEVELOPMENT_STATUS.md`；本计划仍是设计定义，不把 Stage A/B/C0
characterization 升格为论文结果。三卡 W3 上的 192-record 与 682-record mechanism discovery
已经把候选机制收敛为 wave-compiled segmented migration，并得到值得进入正式 Stage C 的
开发信号；这些结果仍是 `scientific_result=false`，不能替代 W4、完整 publication boundary
或论文实验。

---

## 0. 已冻结的前置裁决

本计划已经吸收历史方案中的 action-plan exporter、SPMD process model、row-sharded exact、
transaction、实验矩阵和 gates。后续实现不得重新打开以下七项裁决：

1. H12 的 `action_partition_sha256` 位于
   `steps[i].scheduler.action_partition_sha256`，不在 step 顶层。
2. `scheduled_exact` 与 `natural_exact` 都是 runtime `exact`，区别保存在 `requested_reason`。
3. 总 vocab/table rows 决定容量与是否需要 sharding；collective bytes 主要取决于
   requested/unique IDs、embedding dimension 和 owner distribution，不与总 vocab 直接线性增长。
4. sharded item lookup 不能只替换 `ItemEmbedding.forward`；必须把 HSTU 的 item-embedding
   frontend 与 block forward 拆开，同时保持现有 `forward/compute_kv` 数值不变。
5. “compiled 路径不访问 embedding”只适用于 **retained-prefix compiled transform**；
   同一 record 随后的 target-model delta/latest append 仍访问 embedding。phase、record 和
   full-wave 三种边界必须分开。
6. 新的 one-process-per-GPU SPMD harness 与现有单进程多线程 harness 不是同一计量环境；
   Table 8 只能作为历史动机，所有 D2 baseline 必须在新 harness 内配对重测。
7. D2 接收并执行 immutable action plan。通信成本可以决定 physical batching、placement、
   extent 和执行顺序，但不能在 D2 内把 requested `compiled`/`exact` 重新选择；将通信纳入
   semantic action selection 属于未来 D3。

---

## 1. 冻结的 Design 2

Design 2 定稿为：

> **Wave-Compiled Segmented Migration over Row-Sharded Embeddings.** 对一个已经冻结 action
> partition 的 model-update wave，把 D1 产生的 heterogeneous logical work 降低为
> shape-aware physical extents：compiled retained repair 在 old-K/V owner 本地执行，
> exact/append 只对不可避免的 item IDs 访问 row-sharded embedding，retained 与 suffix 以
> segmented private state 保存，经过全局 coverage/lineage 校验后原子发布 target epoch。

Design 2 的三个研究问题是：

1. 如何确保 D1 的 logical sparsity 不被 old-K/V movement、retained-prefix rewrite 或
   distributed lookup 重新吃掉？
2. 如何把已知的 `(retained R, suffix S, final F)` 和不可避免的 exact/append lookup 编译成
   shape-aligned、可计量、确定性执行的 SPMD extents？
3. 如何让多个 shard 的 segmented heterogeneous outputs 在失败下仍只发布一个完整 target
   version？

### 1.1 统一论文叙事：logical sparsity → physical sparsity

论文的两个 motivation 和两个 design 必须保持以下单向递进：

1. **Motivation 1：semantic–compute dilemma。** stale reuse 便宜但失去当前模型版本一致性；
   all-exact refresh 恢复一致性，却为每条记录重放完整历史。
2. **Motivation 2：logical work 与 physical cost 不等价。** 在 row-sharded embedding 上，
   exact refresh 将完整历史变成与 prefix token 数相关的跨卡 ID/vector movement；简单减少
   exact records 或 arithmetic，不保证 wall time、collective、padding 和 state movement
   同比例下降。
3. **D1：决定 what must be recomputed。** version-cohort compiled repair 与有界 exact reset
   把 uniform all-exact job 变成多数 compiled、少数 exact 的 heterogeneous action plan。
4. **D2：决定 how fixed work moves and executes。** D2 不改变该 action plan，而是将它编译成
   owner-local retained transform、shape-aware exact/append communication、segmented
   publication 和一个原子 target epoch。

Motivation 2 的主图首先只能使用 all-exact 或 design-independent exact-volume scaling
baseline：改变 real-history prefix/exact volume 和 shard 数，报告 logical token/work
fraction、off-diagonal vector bytes、collective/exposed time 和 total wave time。尚未测量前，
不能预写“20% recompute 使用 40% 时间”，也不能写“通信占 recompute 的大部分时间”。只有
matched local/replicated-vs-row-sharded exact breakdown 支持后，才能把“通信占比”写成定量
结论。该 motivation 不需要 request trace、前台推荐 workload 或 serving-latency claim。

D2 的首要系统主张不是新 collective primitive，而是让 D1 的 logical sparsity 兑现为
physical sparsity：

> 对已有 hot old-K/V 的 compiled retained prefix，状态留在 record owner，只摊销一个共享小
> 程序；只有 exact records 和所有 record 的 target-model suffix/latest append 访问
> distributed embedding。D2 再对不可避免的通信和 attention shapes 做 whole-wave lowering。

这里的 physical sparsity 指避免不必要的 lookup、padding、retained rewrite、state movement 和
semantic-only phases，不是使用 sparse tensor kernel。

这只覆盖 retained-prefix transform，不覆盖 natural exact、delta append 或 latest append。
因此论文应证明的是 **physical work/traffic 随 exact residual 与 suffix 缩小，而不是仍随完整
history 缩放**；不能笼统地说整个 migration wave embedding-free，也不能把 record exact
fraction 当作 compute、lookup、communication 或 wall-time fraction。

第二个可研究主张是 wave-level plannability。`scheduled_exact`、`natural_exact`、delta 和 latest
的请求集合在 wave 开始前都由 immutable plan 与 raw-history references 确定，因此可以离线分桶、
排序、合并、去重、错峰和 overlap。对照项应叫 `unplanned/final-length order`，用于量化
whole-wave physical lowering 的价值。不要借此声称真实 serving 无法 batching/dedup；D2 的证据
边界是 model-update maintenance wave。

第三个是 D1 Eq. 9 的 layout characterization：

- record/user、layer 和 retained-token/context 轴上的 compiled transform 可局部执行；
- head/channel 轴存在 dense coupling，除非使用 block-diagonal 或 low-rank 近似；
- 该结论只属于 compiled operator。exact HSTU attention 并不因此成为 context-local。

owner-compute、COW/epoch commit、row-sharded lookup 和 cost-vector planner 是完成系统所必需
的机制，但不单独包装为算法创新。D2 的研究点是把 immutable heterogeneous wave 联合 lowering
为 owner-local state、shape-aligned communication 与 segmented destination，而不是把这些常见
原语逐项包装成贡献。

### 1.2 为什么这个架构由 D1 导出

当前直接程序约 `33.59 MB/edge`，而 hot old K/V 平均约 `52.3 MB/record`。更重要的是，程序由
整个 version cohort 共享并在几百条 record 上摊销，而 K/V 是逐 record 状态。因此默认动作应是
把一个共享程序复制到 owner，原地变换许多本地 K/V，而不是为每条 record 移动 K/V。

Eq. 9 是逐 token、逐 layer 的 affine transform，直接给出了 record/layer/context 三个 compiled
局部轴和 head/channel 非局部轴。它不读 raw history，也不要求 compiled worker 持有 item
embedding 或完整 current model；只有 exact/append service 需要这些资源。

MVP 的严格 scope 是：item embedding table 按 row 跨卡部署，而 replicated dense HSTU trunk
仍能放入单卡。当前 312,145-row 配置只证明 forced-sharding compatibility，不证明 replication
因容量失败；只有 G7 通过后才能提出 table-dominated capacity claim。若 dense trunk 本身也放
不进单卡，必须通过 P7 的 TP gate；在此之前不能声称支持任意大模型。

Design 2 不选择谁 migrate 或 exact。它只执行：

```text
immutable action plan
  → layout/placement plan
  → owner-compute compiled + sharded exact
  → private target extents
  → global validation
  → atomic target epoch
```

---

## 2. 与 Design 3 的边界

### 2.1 单向依赖

```text
Design 3（以后）
  exact budget / depth / deadline / composition / admission
                            │
                            ▼
                      D2ActionPlan
                            │
                            ▼
Design 2（现在）
  placement / execution / fallback / transaction / metrics
                            │
                            ▼
                      D2WaveReport
```

| 责任 | D2 | D3 |
|---|---|---|
| record requested action | 读取和校验 | 生成 |
| exact budget | 不决定 | 决定 |
| deadline/depth policy | 不决定，透明记录 | 决定 |
| program composition | 执行已发布 program | 生成/选择 |
| within-wave placement | 负责 | 不负责 |
| compiled/exact compute | 负责 | 不负责 |
| fixed safety fallback | 负责 | 不覆盖 |
| COW/commit/abort | 负责 | 消费结果 |
| cross-wave backlog | 不负责 | 负责 |
| organic mixed-version policy | 不负责 | 负责 |

### 2.2 当前 D2 输入

使用冻结 artifact：

```text
results/system/cohortkv_single_config_full_chain_v1/
  stage4_9_staggered_renewal_h12_seed0.json
```

该 artifact 已核实包含：

- 11 个 steps；
- `steps[i].lineage` 的逐 record action 和 retained-prefix 信息；
- `steps[i].scheduler.scheduled_exact_ids`；
- `steps[i].scheduler.action_partition_sha256`；
- source/target version；
- last-exact 和 migration-depth lineage。

初始集成 edge 采用 `steps[1]`：

```text
migrate = 548
scheduled_exact = 46
natural_exact = 88
resident_records = 682
```

该 edge 的 exact-route record fraction 是
`(46 + 88) / 682 = 19.6%`。其中只有 46 条是 frozen policy 的 scheduled exact，88 条
natural exact 由 zero-overlap 等输入事实决定。**19.6% 只是 action-count statistic，不是
compute、lookup-token、communication-byte 或 wall-time fraction。**

该 edge 的 lineage 可重建出以下 logical lookup-token 边界：

| phase | paired all-exact | mixed wave | 说明 |
|---|---:|---:|---|
| reusable retained prefix | 637,954 | 50,099 | 587,855 migrate-retained tokens 不查表 |
| natural-exact target prefix | 82,612 | 82,612 | 88 条均为 zero-overlap，不是 scheduler refresh |
| reusable-record delta append | 213,669 | 213,669 | 方法无关的 target-model append |
| latest append | 682 | 682 | 每 record 1 token |
| **完整 post-append cache** | **934,917** | **347,062** | logical lookup tokens，约 **2.69×** |

这里修正一个容易混淆的口径：

- D1-compatible retained-prefix 边界只比较 `637,954 → 50,099`，即 lookup-token 数减少约
  `12.73×`；它排除 natural exact 和方法共同的 append。
- D2 integrated boundary 包含 natural exact、delta/latest append、COW 和 commit，因此用
  `934,917 → 347,062` 这一完整 lookup-token ledger，并直接测 full-wave time。
- `346,380` 只是排除了 682 个 latest tokens 的 target-prefix 数，不能与包含 latest 的
  `934,917` 混比。
- mixed full-wave lookup 是 all-exact 的 `37.1%`，不是 `19.6%`。原因是 exact records 的
  长度不等，且所有 compiled records 的 target-model delta/latest append 仍需 lookup。
- 512 维 FP16 下每个返回 vector 是 1,024 B，但 lookup-token 数乘 1,024 B 只是 logical
  vector volume。真实 collective bytes 还取决于 unique-ID ratio、compute/embedding owner
  重合和 fan-out，不能把所有 vector volume 都记成跨卡流量。

`scheduled_exact` 来自冻结 label-free policy；`natural_exact` 则由 zero-overlap、缺失缓存等
wave 输入事实决定。两者都在执行开始前已知，但不能统称为“policy 选定的 exact 集合”。D2
不把 action selection 计作自己的贡献，也不称其为 quality oracle。

---

## 3. 主部署合同

### 3.1 MVP 布局

| 对象 | 布局 | 所有权 |
|---|---|---|
| item embedding | hash/row-sharded over ranks | exact service |
| behavior/temporal embedding | rank-local replicated | exact service |
| dense HSTU blocks | one replica per rank | exact service |
| old K/V | record/extent-sharded hot HBM | stable record owner |
| direct-old-K/V program | replicated per migration rank | record owner |
| raw IDs | prepared input，可路由 | exact service |
| compiled target K/V | record-owner private `{retained,suffix}` segments | destination |
| exact target K/V | record-owner private full-final extent | destination |
| coordinator | metadata only | rank 0/control plane |

MVP 默认让 exact forward 也在 record/publication owner 上执行：

- embedding rows通过 collective 拉到 record owner；
- dense trunk 在 record owner 的 replica 上运行；
- exact target K/V 不需要再跨卡搬回 publication owner。

“把 exact record 移到 embedding-affinity rank 再搬回 target K/V”只作为 placement baseline，因为
它可能省 embedding bytes，却引入约 52 MB/record 的 target-K/V movement。

### 3.2 扩展布局

| K/V 分片 | compiled data path | 处理 |
|---|---|---|
| record/user | local | 主路径 |
| layer/pipeline | layer-local | compatibility extension |
| sequence/context | compiled token-local | compiled compatibility extension |
| head/channel TP | dense coupling | gated low-rank extension |

layer interval 的算术/程序切分已有基础，但真正的 distributed layer pipeline 仍需 runtime 与
publication integration，不能称为“免费完成”。
该表描述 compiled data path 的 admissibility；row-sharded exact/append 的 attention 与
publication compatibility 必须另行实现和测量。

### 3.3 进程模型

正式 distributed runtime 使用 one process per GPU：

- `torchrun`；
- NCCL process group；
- rank-local CUDA device；
- rank 0 只聚合 metadata/report；
- 所有 rank 按同一顺序进入 exact collectives；
- empty exact rank 仍参加 collective；
- 不在现有单进程 `ThreadPoolExecutor` 中增加 distributed maintenance collectives；
- training DDP 与 D2 maintenance process group 分离。

---

## 4. 两层冻结接口

### 4.1 `D2ActionPlan`：D3/policy → D2

文件：

```text
configs/cohortkv_d2/action_plan_<source>_<target>_<policy>.json
```

协议：

```text
cohortkv_d2_action_plan_v1
```

逻辑 schema：

```json
{
  "protocol": "cohortkv_d2_action_plan_v1",
  "source_version": "theta1",
  "target_version": "theta2",
  "producer": "h12_frozen_lineage",
  "provenance": {
    "artifact": "results/system/cohortkv_single_config_full_chain_v1/stage4_9_staggered_renewal_h12_seed0.json",
    "step_index": 1,
    "action_partition_sha256": "<steps[1].scheduler.action_partition_sha256>"
  },
  "records": [
    {
      "record_id": 0,
      "requested_action": "compiled",
      "requested_reason": "migrate",
      "retained_start": 919,
      "retained_tokens": 1129,
      "delta_start": 1129,
      "delta_tokens": 918,
      "target_prefix_tokens": 2047,
      "latest_tokens": 1,
      "final_tokens": 2048,
      "last_exact_version": "theta1",
      "migration_depth": 0
    },
    {
      "record_id": 1,
      "requested_action": "exact",
      "requested_reason": "scheduled_exact",
      "retained_start": 822,
      "retained_tokens": 1226,
      "delta_start": 1226,
      "delta_tokens": 821,
      "target_prefix_tokens": 2047,
      "latest_tokens": 1,
      "final_tokens": 2048,
      "last_exact_version": "theta0",
      "migration_depth": 1
    }
  ],
  "counts": {
    "compiled": 548,
    "scheduled_exact": 46,
    "natural_exact": 88,
    "records": 682
  }
}
```

映射规则：

| artifact action | D2 runtime action | reason |
|---|---|---|
| `migrate` | `compiled` | `migrate` |
| `scheduled_exact` | `exact` | `scheduled_exact` |
| `natural_exact` | `exact` | `natural_exact` |

要求：

- exporter 不重新运行 scheduler；
- 投影后的 record/count/hash 与 artifact 一致；
- canonical JSON 自身另有 SHA256；
- D2 永不 import scheduler；
- 将来 D3 只需生成同一协议。

### 4.2 `D2WavePlan`：D2 planner → D2 runtime

`D2WavePlan` 在 action plan 上增加系统信息：

```text
protocol
job_id
action_plan_sha256
target_version
serving_layout
cohorts[]
record_owner_map
planned_extents[]
compiled_extent_order
exact_physical_pools
segmented_destination_layout
program descriptors
source manifest
raw-history references
destination
publication mode
capacity margin
```

`D2RecordAction` 至少包含：

```text
record_id
cohort_id
source_version
target_version
requested_action
requested_reason
program_id | null
old_extent_id | null
old_owner_rank
raw_history_ref | null
retained_start
retained_tokens
delta_start
delta_tokens
target_prefix_tokens
latest_tokens
final_tokens
last_exact_version
migration_depth
```

约束：

- target version 在 wave 内唯一；
- record ID 唯一；
- MVP 只支持一个 committed source manifest；
- MVP runtime action 是 `compiled|exact`；
- compiled extents 由 `(suffix S, retained R)` lowering，exact extents 由 final length `F`
  lowering；semantic exact reason 保留在 lineage，但不强制拆成不同 physical operator；
- D2 planner 可以改变 placement、batching、order、stream 和 extent grouping，不得改变
  requested action；
- 只有固定 preflight/fallback 可以把 compiled 改为 exact；
- requested action、final action 和 fallback reason 分开记录。

### 4.3 `D2WaveReport`

至少报告：

```text
job/plan/action hashes
visible version before/after
commit/abort
per-record requested/final action
fallback and lineage
per-rank assignment
phase-tagged compiled/exact/append/lookup/collective/stage/commit times
phase-tagged requested/unique/local/remote IDs and embedding ID/vector bytes
P2P/H2D/D2H bytes
peak source/target/transient HBM
record completion distribution
coverage/checksum/readback
failure reason
```

logical bytes、physical bytes、GPU-event time、collective time 和 CPU wall time必须分开。

---

## 5. HSTU exact-path refactor

### 5.1 现状

`HSTU.embed_inputs` 当前完成：

```text
item_emb(item_ids)
+ behavior_emb(behaviors)
+ temporal_enc(time_deltas)
→ in_proj
→ dropout
```

`HSTU.forward` 随后运行 blocks、mask 和 final norm。row-sharded item embedding 不能直接调用
当前 `forward(item_ids, ...)`，否则又会访问完整本地 `item_emb.weight`。

此外，Stage 4.9 的 `_exact_cache`、`_exact_full`、`_retained_batch` 仍在脚本中，并被 Stage 5
直接 import。

### 5.2 冻结 refactor

在不改变现有 public behavior 的前提下拆成：

```text
lookup_item_embeddings(item_ids)
combine_input_features(item_vectors, behaviors, time_deltas)
forward_embedded(x, lengths, return_kv, return_hidden)
forward(item_ids, ...)
  = lookup_item_embeddings
  → combine_input_features
  → forward_embedded
```

新增：

```text
compute_kv_from_item_embeddings(
  item_vectors,
  behaviors,
  time_deltas,
  lengths
)
```

要求：

- 原 `forward/compute_kv/embed_inputs` 输出不变；
- dropout/eval semantics 不变；
- padding mask 在同一位置应用；
- behavior/temporal/in_proj 不复制逻辑；
- existing checkpoint/state_dict keys 不变；
- exact helper 从脚本提升到 migration/recompute library；
- 旧脚本通过 adapter 调用新 helper，冻结 artifact 不重算。

### 5.3 retained-prefix embedding bypass 证据

优先使用非侵入式 instrumentation：

- `register_forward_hook`；
- exact/migration adapter counter；
- distributed lookup counter。

不为计数修改 `ItemEmbedding.forward` 的算术。

必须观测：

```text
compiled_retained item lookup tokens = 0
compiled_retained embedding collective bytes = 0
exact_retained_or_natural item lookup tokens > 0
compiled_record_append item lookup tokens > 0 when delta/latest is nonempty
```

counter 必须带 `phase`，不能只带最终 record action。否则一条 requested action 为 `compiled` 的
record 会因后续 append 查表而错误地推翻、或错误地满足 “embedding-free” 断言。
world size 为 1、或请求 ID 全部命中 compute owner 时，exact/append 的跨卡 bytes 可以为零；
因此“需要 lookup”和“实际 remote collective bytes”必须是两个字段。

---

## 6. 执行协议

### 6.1 Freeze

- 从 H12 artifact 导出 `D2ActionPlan`；
- 校验 source/target、counts、record set 和 upstream hash；
- 生成 `D2WavePlan`；
- action hash 在 GPU execution 前冻结。

### 6.2 Preflight

校验：

- artifact/program/certificate；
- source/target/layout/dtype/world size；
- old manifest、extent presence 和 owner；
- raw histories；
- semantic canary；
- embedding shards；
- per-rank old + complete-new + transient + margin；
- expected global record set。

固定失败语义：

| 失败 | 行为 |
|---|---|
| artifact/version mismatch | transaction 前 reject |
| capacity failure | transaction 前 reject |
| program/shape/certificate/canary failure | affected cohort → exact |
| missing old K/V | affected cohort → exact |
| required raw history missing | reject |

fallback 后重新执行 capacity 和 lineage preflight。

### 6.3 Begin

- 创建不可见 target transaction；
- old epoch 保持唯一可见；
- 广播 transaction ID、plan hash 和 target version；
- rank-local compiled extents 与 exact batches 固定。

### 6.4 Compiled retained-prefix owner-compute

- old K/V 在 owner HBM 读取；
- program 在 owner 执行；
- retained-prefix private segment 仍在同一 record owner，并在 finalization 中只写一次；
- 不执行 item lookup；
- 不移动 old K/V；
- 不执行 embedding collective。

只对 `compiled_retained` phase 成立的 normal-path invariants：

```text
item_lookup_calls = 0
embedding_collective_count = 0
embedding_collective_bytes = 0
old_kv_p2p_bytes = 0
```

### 6.5 Row-sharded exact retained/natural work

对 record owner 上的 exact batch：

1. padding-aware flatten item IDs；
2. 按 deterministic owner rule 分桶；
3. all-to-all IDs 和恢复位置；
4. shard-local item lookup；
5. all-to-all item vectors 回 record owner；
6. 恢复 `[B,L,H]`；
7. rank-local behavior/temporal/in-projection；
8. `forward_embedded` 完成 current HSTU；
9. exact retained prefix 或 natural-exact target prefix 写入 record-owner private intermediate。

`scheduled_exact` 与 `natural_exact` 的 requested reason、count 和 lineage 必须分别保留，但
两者调用相同 full-final exact operator 时应进入同一个按 `F` 组织的 physical pool。这个合并
只删除 semantic/physical phase mismatch，不改变 exact records、lookup multiset 或输出。

实现顺序：

```text
naive per-batch correctness
→ cohort batching
→ shard sorting
→ optional cross-record dedup
→ communication/lookup/trunk overlap
→ cross-rank staggering
```

每个优化独立开关。dedup/shard-affinity 无实测收益则删除。

### 6.6 方法共同的 target append

- 所有 reusable records 的 delta 和所有 records 的 latest token 都用 target model；
- compiled records 的 append 也经过 row-sharded item embedding，因此它不是 embedding-free；
- mixed 与 paired all-exact 使用同一 append executor、batching 开关和 dtype/length contract；
- 可规划 batching/dedup 可以覆盖 append，但其时间和通信必须在 ledger 中单列；
- compiled append 只生成新的 suffix K/V，不重新展开或重写完整 retained K/V；
- compiled logical target 是 `{retained segment, suffix segment}`；exact logical target 可以是
  one-shot full-final extent；
- 只有 coverage 完整的 post-append logical cache 可以进入 target manifest。若 consumer 或
  下一轮维护要求把 segmented cache 同步拼成 contiguous full cache，该拼接和 retained rewrite
  必须进入 primary timer，不能把成本延迟到计量边界之外。

### 6.7 Validate and commit

- 校验 extent shape/dtype/finite/length/offset；
- 校验 requested/final action 和 lineage；
- 校验 checksum、owner、record coverage；
- 校验 segment ordering、logical length 和 segmented-vs-contiguous materialization；
- coordinator 只聚合 metadata/hashes；
- 所有 rank ready 才 commit；
- commit 后 target version 才可见；
- strict COW 主路径在 commit 后 reclaim old extent；
- pre-commit failure abort target，old epoch 继续可见；
- readback 再校验 manifest 和 fingerprints。

fast reclaim 只作为容量对照，不替代 strict-COW primary。

---

## 7. Planner 与 placement

### 7.1 默认规则

> Keep both compiled and exact output on the record owner. Move programs and embedding rows, not
> old or target K/V.

原因：

- compiled 必须读约 52 MB old K/V；
- exact 产生同量级 target K/V；
- dense trunk 已复制，exact 可以在 record owner 执行；
- 分布式 item vectors 远小于完整多层 K/V。

### 7.2 cost vector

\[
C(e)=(
\text{compiled GPU ms},
\text{exact GPU ms},
\text{embedding ID/vector bytes},
\text{P2P/H2D/D2H bytes},
\text{padded attention work},
\text{retained rewrite bytes},
\text{physical phase/collective count},
\text{source/target/transient HBM}
).
\]

该向量只在 frozen semantic action 内选择 physical realization。它可以改变 owner、extent、
batch/order/stream 和 segment layout，不能因为某条 record 的通信更贵就把 requested exact
改为 compiled，或反向改变 exact budget。

先满足：

1. lineage；
2. owner locality；
3. collective ordering；
4. capacity；
5. atomic coverage。

再优化：

- wave makespan；
- exposed collective；
- padding、retained rewrite 与 redundant phase；
- rank imbalance。

### 7.3 必测 placement baselines

- record-owner compute；
- P2P old-K/V work stealing；
- exact compute on non-owner + target-K/V return；
- embedding-affinity exact placement；
- byte/token LPT；
- injected owner imbalance。

现有 2.1 ms transfer vs 1.36 ms compiled 只是 hypothesis。P2P、NVLink/PCIe、并发和 overlap 必须
实测后再决定是否冻结 no-move rule。

---

## 8. 实施里程碑

### P0：冻结接口与便宜的可证伪证据

#### P0.1 Action exporter

新增：

```text
scripts/export_cohortkv_d2_action_plan.py
```

输入 H12 artifact + step index，输出 canonical `D2ActionPlan`。

通过：

- counts/record IDs 与 upstream 一致；
- upstream scheduler hash 一致；
- canonical action-plan hash 可复现；
- 无 scheduler import。

#### P0.2 Single-GPU WavePlan adapter

- 实现 plan/layout/report dataclasses；
- 包装现有 Stage 5 real-edge smoke；
- 保持 per-record output、manifest、fallback 和 bytes 不变；
- 锁定 commit/abort tests。

#### P0.3 Retained-prefix embedding-bypass instrumentation

- `compiled_retained` phase lookup = 0；
- `exact_retained_or_natural` phase item lookup > 0；
- compiled record 的 delta/latest append lookup > 0；
- world-size-1/local-hit 与 multi-rank remote collective 分开断言；
- 使用 hook/adapter，不改核心算术。

#### P0.4 P2P microbenchmark

- 真实 old-K/V extent size；
- 1/2/4 GPU topology；
- concurrent copies；
- copy/compute overlap；
- balanced/imbalanced owners。

#### P0.5 Exact-request 与 dedup ceiling

从 frozen action plan 和对应 raw histories 导出一个 checked static report：

- 按 `scheduled retained / natural prefix / delta / latest` 分开的 total item IDs；
- global、per-batch、per-record-owner、per-embedding-owner 的 unique IDs；
- fan-out distribution 和 remote-owner fraction；
- 无 dedup 与理想 dedup 的 ID/vector logical-byte 上下界；
- 输入 artifact/raw-history hashes。

不能只用 global `unique/total` 决定实现，因为 collective 只能在实际 coalescing scope 内去重。
当前对 H12 step 1 的只读预检得到
`scheduled retained + natural prefix = 132,711 total / 96,844 unique`，global ratio 约
`0.730`：dedup 不是显然无效，但这还不是 runtime 收益。P5 只保留能在实际 batch/owner scope
中至少减少 10% returned logical vector bytes 的 dedup 方案；否则在进入优化实现前删除。

基础 payload 不对称是每 token `8 B item ID : 1,024 B FP16-H512 vector = 1:128`；routing
position/fan-out metadata 会增大 request leg。故优化重点是 remote unique vectors 和 exposed
return leg，而不是仅减少约 1 MB 的裸 ID 请求。所有比例都用实际 collective payload 复核。

P0 不需要新训练。

### P1：Exact replay 进库与 HSTU frontend split

- `_exact_cache/_exact_full/_retained_batch` 从 Stage 4.9 脚本提升到 library；
- 拆出 `forward_embedded/compute_kv_from_item_embeddings`；
- 保持 4 个现有调用点行为；
- state_dict 和 public forward 不变；
- 单卡 exact K/V/hidden/score/Top-100 regression tests。

### P2：SPMD owner-compute 与 distributed transaction

- one process per GPU；
- deterministic record owner；
- program replication；
- rank-local compiled execution；
- private target extents；
- global metadata coverage；
- atomic commit/abort；
- 1/2/4 GPU correctness smoke。

测试分两层：

- 普通 `pytest` 覆盖 pure plan/transaction logic 和 world-size-1 Gloo path；
- 2/4-rank NCCL collective/abort/deadlock 测试由
  `scripts/run_cohortkv_design2_distributed_tests.py` 在 `torchrun` 下启动；
- 多卡 case 使用注册的 `design2_multigpu` marker 或独立 worker case，不让普通 pytest
  collection 隐式初始化多个 process groups；
- launcher 必须传播任一 rank 的失败、设置 timeout，并在失败后销毁 process group。

冻结入口：

```text
pytest -m "not design2_multigpu"
torchrun --standalone --nproc-per-node=2 scripts/run_cohortkv_design2_distributed_tests.py
torchrun --standalone --nproc-per-node=4 scripts/run_cohortkv_design2_distributed_tests.py
```

### P3：Row-sharded embedding exact baseline

- deterministic hash/row owner；
- padding-aware ID routing；
- two-way all-to-all；
- rank-local lookup；
- `forward_embedded`；
- collective byte/time counters；
- replicated exact equivalence。

必须覆盖：

- empty rank；
- padding-only bucket；
- uneven ID counts；
- repeated IDs；
- zero exact records on one rank；
- all exact records on one owner。

可选增加一个 supporting resource-contention microbenchmark：

- 固定速率 synthetic foreground lookup stressor；
- 分别与 idle、paired all-exact retained maintenance、compiled retained maintenance 并发；
- 报 offered/achieved lookup rate、p50/p99、超时/排队和 maintenance wall time；
- 使用相同 embedding shards、lookup mix、CUDA stream policy 和 retained-prefix workload；
- 加一个 rank-local dense/compute microprobe 控制，区分一般 SM/HBM contention 与
  embedding-tier-specific contention；
- 明确标为 resource stressor，不宣称是真实 serving workload。

该实验只做资源归因，不是 Motivation 2、D2 paper-strength gate 或真实 serving claim。它允许
compiled 因共享 SM/HBM 而干扰 synthetic lookup；不能预写“约 0% 退化”。

### P4：Fixed-action integrated wave

- 导入冻结 `D2ActionPlan`；
- 在同一 row-sharded harness 和 action hash 下执行三条路径：
  1. strong one-shot/two-stage all-exact；
  2. naive sharded fixed-action mixed；
  3. D2 physical-sparse mixed；
- D2 physical lowering 至少包含 `(S,R)` compiled extents、按 `F` 的 merged exact pool 和
  segmented suffix-only publication；
- strict COW；
- source-read-to-post-commit-reclaim boundary；
- 1/2/4 GPU repeated timings；
- complete logical-work/physical-work/movement/communication/capacity ledger。

P4 是第一个完整 Design 2 系统结果。

在正式 W4 与 Stage-B freeze 等待期间，C0 development-only 接线子集已经完成：固定
16-record fixture 上的 W1/W2/W3 normal 和 W3 pre-commit abort 均闭合 heterogeneous routes、
private fragments、development epoch state machine、abort/readback 和 phase order。所有
artifact 均为 `scientific_result=false`、`formal_stage_c=false`，无 full-cohort、timing、
capacity 或正式 epoch-publication claim。C0 通过只维持
`stage_c_development_entry=go`，不算 P4 完成，不产生 Stage-C evaluation 入口，也不允许生成
Stage-B summary。正式 P4 仍要求四张独立 A40 的 W4 normal/hard-failure 通过、Stage-B
checked summary 冻结，并在 `docs/eval_protocol.md` 新建 D2 protocol。

在不改变上述 formal gate 的前提下，W3 mechanism discovery 已完成 v1→v5 因果链：
naive staged mixed、phase fusion、segmented suffix-only、`(S,R)` shape order 和 merged exact
pool。full682/B8 development point 为 mixed `3.633 s`、one-shot all-exact `6.716 s`，full
payload equivalence 通过。它证明候选 physical lowering 值得正式验证，但计时尚未包含正式
publication/commit/reclaim 与 consumer boundary，且 `scientific_result=false`，不能算 P4
formal complete 或 paper evidence。

### P5：Exact collective optimization

先冻结 P4 的 physical lowering，再读取 P0.5 ceiling，逐项加入并消融：

1. cohort batching；
2. shard sorting；
3. dedup；
4. lookup/trunk overlap；
5. exact-wave staggering。

所有行保持 action hash、lookup multiset 和 owner/publication boundary 不变。dedup 未通过
P0.5 的静态 gate 就不实现；其他机制没有改善 wall time/exposed communication 就立即删除。

### P6：Capacity 与 stress

两个配置分开：

1. 当前 312,145 行表 forced hash-row sharding：正确性和通信；
2. 若当前配置不足以暴露目标瓶颈，则新建一个 **real-data D2-discovery
   configuration**：完整表或目标 wave 在 1/2/3/4 GPU 上形成有区分度的
   admission/communication 点。

real-data D2-discovery 配置必须：

- 从此前已接受、具有正确 target semantics 的真实数据中扩大 item 和 user 覆盖，不能靠未访问
  cold rows 制造主结果；
- 按与 D1 相同的数据切分、base-vocabulary、streaming version 和质量协议，训练一个 base
  version，再训练一到两个新增流式版本；
- 复用 D1 的 old-K/V、residual fitting、direct program、certificate 和 exact endpoint 流程，
  然后把该 edge/wave 作为 D2 的 discovery 输入；
- 第一轮只需一个 seed 和足以形成版本差的短流式训练，不要求先重跑完整长期 D1 matrix；
- 让真实 requested IDs、unique ratio、cohort 和 per-rank state 在 1/2/3/4 GPU 上形成可解释
  的规模差异；所有 communication 仍按实际 routed IDs/vectors 计量；
- 明确标记为 D2 mechanism discovery。机制收敛以后，才决定是否把这个大配置补成完整 D1
  replication 和论文主证据。

这是当前小模型信号不足时的**诊断性回退**，不是默认起点。现有 full682 W3 discovery 已经
暴露出明确的 physical-lowering 问题并得到正向机制信号，因此当前不触发该回退。若正式结果
再次显示信号不足，必须先用 phase time、collective time/bytes 和 GPU 利用率判断根因。只有
瓶颈归因明确指向规模不足，才依次：

1. 在不改变当前 frozen ActionPlan 的前提下放大同协议的 cohort、requested tokens 或真实
   ID diversity；任何 exact-fraction sweep 都必须预注册为新的 workload/action-plan protocol，
   不能由 D2 根据运行结果修改，也不能混入 H12 主结论；
2. 若现有数据覆盖本身限制了规模，则按上述规则训练 real-data D2-discovery base + 1–2
   streaming versions，并重跑同一条 D1→D2 流程；
3. 只有作为附加机械 control 时，才允许在保持真实 accessed rows/weights 不变的前提下加入
   cold rows，用于单独定位 unsharded admission boundary；
4. 若真实数据驱动的大配置仍没有可观测瓶颈，则暂停 D2，而不是继续扩大模型
   强行得到正结果。

因此 cold-row expansion 只能支撑“为什么必须分片”的机械 admission control；D2 的主性能
与设计证据必须来自真实用户、真实 accessed IDs、实际训练版本和实际通信。新配置必须建立
独立 protocol，不能把不同规模结果悄悄并入当前 family。

### P7：条件 layout extensions

- compiled layer-local；
- compiled retained-context-local；
- TP dense gather；
- TP block diagonal；
- TP blockdiag + rank-\(r\)。

TP 不阻塞 P0–P6。只有通过 deployment/communication/fidelity gate 才进入主论文。

---

## 9. 建议代码落点

新增：

```text
src/hstu_kvcache/migration/design2_plan.py
src/hstu_kvcache/migration/design2_embedding.py
src/hstu_kvcache/migration/design2_runtime.py
src/hstu_kvcache/migration/design2_transaction.py
src/hstu_kvcache/migration/design2_metrics.py

scripts/export_cohortkv_d2_action_plan.py
scripts/characterize_cohortkv_d2_requests.py
scripts/run_cohortkv_design2_smoke.py
scripts/run_cohortkv_design2_distributed_tests.py
scripts/benchmark_cohortkv_design2_wave.py
scripts/evaluate_cohortkv_design2_layouts.py
scripts/freeze_cohortkv_design2.py

tests/test_design2_plan.py
tests/test_design2_exact_frontend.py
tests/test_design2_sharded_embedding.py
tests/test_design2_owner_compute.py
tests/test_design2_transaction.py
tests/test_design2_faults.py
```

复用：

| 模块 | 内容 |
|---|---|
| `stage45_oldkv.py` | direct operator/program/correctness |
| `stage45_resident.py` | HBM source/extent/accounting |
| `stage45_reclaim.py` | reclaim/P2P baseline |
| `destination.py` | transaction/manifest |
| `stage5_closure.py` | preflight/fallback/lineage/readback |
| `recompute.py` | exact batch/rank-local dense forward |
| `stage4_engine.py` | timing/movement/capacity ledger |
| `streaming/distributed.py` | process-group pattern only |

原则：

- 不修改冻结 result family；
- 新 protocol/result family 独立；
- frozen scripts通过 adapter 复用新 library；
- 不让训练 DDP 模块承担 D2 maintenance runtime；
- plan schema、runtime 和 metrics 分文件。
- 在 `pyproject.toml` 注册 `design2_multigpu` marker；普通 `pytest` 不隐式启动 NCCL 多卡
  case，2/4-rank gate 使用显式 `torchrun` launcher。

---

## 10. 实验矩阵

### E-M2：Design-independent logical-to-physical motivation

Motivation 2 先只刻画 exact maintenance，不使用 D1 的迁移结果作为前提：

- 在同一 real-history workload 上按 token/length 分层构造预注册的
  `20/40/60/80/100%` exact-volume points；
- 固定 current model、row-sharded embedding、dense trunk、record owner、dtype 和 endpoint；
- sweep 1/2/4 GPU 或当前合法的 shard 数；
- 同时报 logical records/tokens、unique/local/remote IDs、returned-vector bytes、
  collective/exposed time、padding、rank wait 和 total wave time；
- 用 matched local/replicated embedding exact 作为 communication attribution control。

这张图回答“减少 logical exact work 是否会自动带来同比例 physical savings”。结果出来前不
预写曲线形状或具体比例。若只证明大流量而未证明时间占比，就只写 communication
amplification，不能写 communication dominates。该实验不需要 serving trace。

### E0：正确性

- replicated exact vs split-frontend exact；
- replicated embedding vs row-sharded embedding；
- current Stage 5 vs D2 adapter；
- 1/2/4 GPU owner-compute output；
- strict COW commit/readback；
- fallback cases。

### E1：Integrated hybrid wave

| 维度 | 设置 |
|---|---|
| GPU | 1/2/4 A40 |
| action | frozen H12 step |
| embedding | replicated / forced row-sharded / capacity-scaled |
| dense trunk | replicated |
| old K/V | record-sharded hot HBM |
| destination | strict-COW HBM |
| methods | two-stage paired all-exact / one-shot all-exact / naive sharded fixed-action mixed / D2 physical-sparse mixed / SPMD record-DP mixed |

`SPMD record-DP mixed` 使用相同 frozen actions，但在每 rank 复制当前可容纳的 embedding/model
并按 record 分片，用于测 D2 communication fabric 的增量；它不进入 capacity-infeasible 点。
`naive sharded fixed-action mixed` 使用与 D2 相同的 row-sharded embedding、action hash、
record owner、COW 和 timer，但采用 contiguous full-cache materialization、naive
final-length/per-phase order 和分离 exact reasons；它是证明 logical sparsity 不自动成为
physical sparsity 的直接 bridge。

所有 baseline 必须移植到同一 SPMD harness、相同 rank lifecycle 和相同 timer 后重测；不得
直接引用 Table 8 的单进程多线程数字。D2 必须同时优于或清楚解释相对 naive mixed 与
faster-of-two-stage/one-shot all-exact 的结果。

### E2：Physical-wave compilation 与 communication

#### E2.1 Planned-request ablation

| 方法 | physical order/layout | exact pool | dedup | overlap |
|---|---|---|---:|---:|
| naive fixed-action mixed | final-length/per-phase + contiguous | split by reason | no | no |
| phase fused | final-length + contiguous | merged | no | no |
| segmented | final-length + `{retained,suffix}` | merged | no | no |
| shape-aware | `(S,R)` + `{retained,suffix}` | merged by `F` | no | no |
| optional coalesced | same | same | yes | optional |

所有行使用同一 action hash、lookup multiset、owner map、endpoint 和 consumer boundary。这个
ablation 量化 immutable wave 的 physical lowering，不涉及在线 serving。

#### E2.2 Optional synthetic lookup contention probe

| concurrent maintenance | foreground input | 必报 |
|---|---|---|
| idle | fixed-rate lookup mix | achieved rate、p50/p99 |
| paired all-exact retained | same | 同上 + maintenance time |
| compiled retained | same | 同上 + maintenance time |

至少 sweep 三个 offered-load 点（低负载、knee 前、knee 附近），并报告 lookup mix、remote
fraction、stream priority、background GPU/HBM utilization，以及 rank-local dense microprobe
的对照退化。若 lookup 与 dense probe 同幅退化，只能声称一般 GPU contention reduction，不能
归因于 embedding-tier isolation。该实验不外推线上 recommendation latency，也不是
Motivation 2 或 D2 paper-complete 的必需条件。

### E3：Placement

- owner-compute；
- old-K/V P2P steal；
- non-owner exact + target return；
- embedding-affinity exact；
- global barrier vs stagger；
- balanced vs injected skew。

### E4：Failures

- program/artifact/layout mismatch；
- missing old extent；
- missing raw history；
- canary failure；
- rank capacity failure；
- mid-compiled/mid-exact failure；
- pre-commit rank failure；
- missing/duplicate record；
- readback checksum mismatch。

### E5：Layout extension

- record；
- compiled layer；
- compiled retained-context；
- optional TP \(r\in\{0,4,8,16,32,64,\mathrm{dense}\}\)。

---

## 11. 计量边界

### 11.1 Primary timer

开始前：

- immutable `D2ActionPlan` 已冻结；
- program/model/embedding shards 已加载；
- old K/V 位于 owner HBM；
- exact raw IDs 位于声明 prepared tier；
- capacity preflight 已完成。

计入：

- 任何没有随已发布 ActionPlan 一起持久化的 wave lowering/materialization；
- source extent access；
- ID routing/collectives/lookup；
- compiled/exact compute；
- append；
- any output/P2P movement；
- private staging；
- validation；
- synchronization；
- global commit；
- post-commit reclaim。

排除但单报：

- training；
- checkpoint loading；
- program fitting/compilation；
- first preload；
- embedding-shard construction；
- capacity-row initialization。

若 `D2WavePlan` 已提前序列化并复用，允许把其编译单报，但必须同时发布 plan-inclusive 单 wave
边界和明确的复用次数/break-even；不能因 Python compiler 慢而默认移出主边界。

### 11.2 双计量边界

必须同时发布两个互不替代的边界：

| boundary | 计入 | 用途 |
|---|---|---|
| D1-continuity retained-prefix | reusable migrate/scheduled-exact retained work | 与冻结 Stage 4.9 `U/E` 的问题定义连续 |
| D2 integrated wave | natural exact、retained work、delta/latest append、staging、validation、commit、reclaim | D2 primary system claim |

D1-continuity 边界继续排除 natural exact、append、state movement 和 publication，且不能把新 SPMD
数值与旧 Stage 4.9 `U/E` 合并成同一 result family。它用于解释 retained-prefix 算法收益是否在
新 runtime 中保留，不是 D2 端到端 speedup。

D2 integrated wave 是 primary timer。同一 logical contract 下至少有三条 physical path：
strong all-exact、naive sharded fixed-action mixed 和 D2 physical-sparse mixed。它还必须把以下
component ledger 分别报告：

```text
reusable retained lookup/compute
natural exact lookup/compute
delta append lookup/compute
latest append lookup/compute
staging + validation + commit + reclaim
```

因此正文可以同时说“retained-prefix lookup-token 数 `637,954 → 50,099`”和“完整 post-append
lookup-token 数 `934,917 → 347,062`”，但不能用前者的通信不变量解释后者的整条 record。

### 11.3 必报

性能：

- wave wall time；
- records/s、tokens/s；
- per-rank busy/idle；
- scaling efficiency、imbalance。

physical work：

- `(R,S,F)` extent distribution；
- padded attention proxy/actual elements；
- retained K/V read/write/rewrite bytes；
- suffix-only/segment bytes；
- physical exact pool 与 phase counts；
- segmented consumer/materialization time。

通信：

- 按 phase 的 requested/unique/local/remote ID counts；
- 按 phase 的 routed ID bytes；
- 按 phase 的 returned embedding-vector bytes；
- fan-out bytes/metadata；
- all-to-all count/time；
- P2P old/target-KV bytes；
- H2D/D2H logical/physical；
- overlapped/exposed communication。

容量：

- embedding/model/program；
- old K/V；
- complete target；
- transient/staging；
- old+new peak；
- allocator margin。

正确性：

- K/V fidelity；
- hidden/score cosine；
- Top-100；
- task delta；
- coverage/lineage；
- checksum/readback；
- visible-version correctness。

### 11.4 Paired baselines

必须使用：

- 同一 sharded embedding；
- 同一 HSTU dtype；
- 同一 histories/lengths；
- 同一 record owner/publication layout；
- 同一 COW/commit/reclaim；
- 同一同步点；
- two-stage branch 使用同一 append executor 和 collective-optimization 开关。

old K/V 与 raw IDs 的 source footprint天然不同，必须分别报告，不能伪装等资源输入。

all-exact 分两条：

1. `two-stage paired all-exact`：exact retained prefix 后走与 mixed 相同的 delta/latest append，
   是 D1-continuity 的配对 denominator；
2. `one-shot all-exact`：从完整 target raw history 直接生成 post-append target K/V，再进入相同
   COW/commit/reclaim，是 integrated D2 的强 baseline。

两条都在 E1 报告；integrated speedup 对每个 GPU 点使用两者中更快的 measured baseline，避免
靠强制两阶段 append 人为削弱 exact。若 one-shot 与 mixed 的内部 phase 不同，只要求起点、
终点、placement、dtype 和 publication boundary 相同，不伪造 phase 对齐。

此外必须有 `naive sharded fixed-action mixed`。它与 D2 使用相同 ActionPlan、row-sharded
embedding、record owner、lookup multiset、COW 和 endpoint，但保留 contiguous finalization、
naive shape/order 与 semantic exact phase fragmentation。它用于隔离 D2 physical lowering 的
增量；replicated record-DP 不能替代该 baseline。

Table 8 来自现有单 Python 进程、per-device thread/stream worker 的 Stage 4.5 harness；通用
`MultiGPUCohortExecutor` 还存在共享 operator 实例，但不能把两条实现细节混为同一个基线。
D2 的 `torchrun` SPMD 会改变进程启动、CUDA context、allocator、per-rank program/model
replica 和同步开销，因此：

- Table 8 只能作为 architecture motivation；
- SPMD record-DP、all-exact 和 owner-compute mixed 必须在新 harness 内从头配对重测；
- process startup/first context creation 排除并单报，steady-state context/HBM overhead 计入 capacity；
- 不允许把 Table 8 wall time 直接放进 D2 E1 的 baseline 列。

---

## 12. Gates

当前执行状态把开发入口与评测入口分开：

- W3 NCCL normal/hard-failure 通过后，`stage_c_development_entry=go`；C0
  development-only 接线现已完成，只允许继续针对性开发诊断；
- W3 full682 physical-lowering discovery 与 full-payload validation 也已完成，但均为
  `scientific_result=false`，不改变 formal gate；
- 正式 W4 和 checked Stage-B summary 尚缺，因此
  `stage_c_evaluation_entry=blocked`；
- W3/C0 均不能满足下面的 G4/G5，也不能替代 G2/G3 在正式 W4 上的闭合。

### G0：Action/plan protocol

- upstream counts/hash/record set 一致；
- action-plan hash 可复现；
- WavePlan 不重选 actions；
- requested/final/fallback 分开。

### G1：Mechanical refactor equivalence

- old/new `forward/compute_kv` 输出一致；
- exact helper 搬迁不改变 K/V/hidden/score；
- state_dict keys 不变；
- frozen scripts tests 通过。

### G2：Distributed exact equivalence

- sharded item vectors 等于 replicated；
- exact K/V 通过既有 tolerance；
- empty/padding/uneven routing 无死锁；
- actual collective bytes 可重建。

### G3：Compiled communication invariants

```text
compiled_retained.lookup = 0
compiled_retained.embedding_collective = 0
compiled_retained.old-K/V_P2P = 0
compiled_record_append.lookup_tokens = planned compiled-record delta + latest tokens
compiled_retained.write_count = 1
compiled_append.output = suffix-only
compiled_finalization.retained_rewrite_bytes = 0
```

前三项只适用于 owner-compute normal `compiled_retained` phase。最后一项明确阻止把 phase-local
不变量扩张成 whole-record 不变量。normal H12 step 1 的 phase token ledger 必须与
`637,954 → 50,099` retained-prefix 和 `934,917 → 347,062` full-wave 账本一致；发生 fallback 时
另报 requested/final action 后重建账本。

这里的 `retained_rewrite_bytes = 0` 只约束 destination finalization。suffix attention 当前
仍可能临时整理/提升 retained K/V；其 temporary bytes、padding 和时间必须另报，不能称整个
retained path zero-copy。

### G4：Transaction

- one output per record；
- coverage 无缺失/重复；
- abort 保持 old visible；
- commit 后 readback/lineage 一致；
- rank failure 不产生 mixed visible epoch。

### G5：Integrated advantage

paired source-read-to-post-commit-reclaim boundary 下：

- 1/2/4 GPU 报绝对值和 repeats；
- 同时报告 D1-continuity retained-prefix secondary boundary；
- fixed-action D2 相对 naive sharded mixed 有可归因的 physical-work 或 wall-time 改善；
- integrated claim 对 faster-of-two-stage/one-shot all-exact baseline；
- mixed wave 优势超过 timing variation；
- movement/append/commit/reclaim 全计入；
- HBM margin 通过；
- fidelity/transaction gates 通过。

若只剩 arithmetic advantage，不得写系统 speedup claim。

### G6：Fixed-action physical lowering

segmented destination、shape-aware extent 或 exact-pool lowering 的联合/独立消融必须：

- 在 action、lookup multiset、owner 和 endpoint 不变时减少 padding、retained rewrite、
  fragmented phases、exposed collective 或 wave time；
- 在至少两个点复现；
- 无 correctness/transaction regression；
- 有独立消融。
- 与 naive sharded fixed-action mixed 配对。

dedup/overlap/stagger 是可选通信增强项，不是 G6 必需条件。若核心 lowering 失败，D2 收缩为：

> embedding-free retained-prefix owner-compute plane + sharded-exact/append compatibility +
> atomic publication.

### G7：Capacity claim

- unsharded admission 失败；
- sharded admission 通过；
- original accessed outputs 不变；
- per-rank/aggregate HBM 和 imbalance完整。

失败只影响 large-model-capacity claim，不否定 forced-sharding compatibility。

### G8：Owner placement

P2P/affinity baseline 实测后：

- owner 不差：冻结 no-move default；
- 某些 workload work stealing 更好：记录触发条件并允许受控移动；
- 所有移动进入主 ledger。

### G9：Optional TP

只有同时满足：

- deployment/capacity 需要 channel TP；
- full gather 是 measured bottleneck；
- low-rank 形成通信—fidelity前沿；
- 至少一点评估通过 fidelity 且改善 wall time；

才进入主 D2。否则留 extension 或删除。

### G10：Paper-strength gate

D2 要作为独立的主设计进入系统 A 会论文，除 correctness/transaction 外，核心 physical-sparsity
gate 必须通过：

1. **核心 gate：G5 + G6。** 在完整相同边界上，D2 相对 naive sharded fixed-action mixed
   可归因地把 D1 logical sparsity 兑现为更少 physical work/time，并对
   faster-of-two-stage/one-shot all-exact 保持有意义的 integrated advantage。
2. **增强 gate（至少一个最好成立，但不替代核心 gate）：**
   plan-known optional communication optimization、真实 sharding capacity admission，或明确的
   cross-GPU scaling/imbalance improvement。

Eq. 9 layout characterization 是 generality 证据，不替代上述系统结果。若 G7、optional
communication optimization 和 E2.2 contention probe 等增强项失败，只删除对应的 optional
claim；synthetic contention probe 不再是独立
paper-strength 通道。若 G5/G6 核心 gate 失败，D2 仍是正确的多卡实现，但不足以单独包装为
paper design；应把它收缩为 D1 的 distributed implementation，而不是用 owner-compute、COW
或 planner 术语堆出创新性。

---

## 13. 两级 Definition of done

### 13.1 D2 implementation MVP

完成 P0–P4：

- frozen ActionPlan/WavePlan；
- exact helper/frontend split；
- SPMD owner runtime；
- world-size-1 pytest + 2/4-rank launcher；
- row-sharded exact；
- phase-aware lookup/communication ledger；
- mixed fixed-action wave；
- atomic commit；
- complete ledger；
- correctness/failure smoke。

这时可以开始形成主实验，但不能宣称 capacity scaling 或 collective optimization。

### 13.2 D2 paper complete

完成：

- P0–P6；
- 1/2/4 GPU paired results；
- new-harness record-DP/all-exact baselines；
- D1-continuity 与 D2-integrated 双边界；
- naive-sharded-to-physical-sparse causal ablation；
- optional collective ablation 或诚实删除对应 claim；
- capacity gate 或诚实删除 capacity claim；
- owner/P2P结论；
- failure matrix；
- checked freeze summary；
- optional TP 明确 go/no-go；
- G10 paper-strength gate 通过。

此后再启动 Design 3。

---

## 14. 最终实施顺序

```text
P0  export action + schema/adapter + phase counter + request ceiling + P2P microbench
 ↓
P1  exact helper library + HSTU embedded-input frontend
 ↓
P2  SPMD owner-compute + distributed transaction
 ↓
P3  row-sharded exact equivalence
 ↓
P4  all-exact → naive fixed-action mixed → physical-sparse mixed
 ↓
P5  optional collective planning and ablation
 ↓
P6  capacity and stress
 ↓
P7  layout extensions; TP only after gate
 ↓
freeze D2
 ↓
start D3
```

P0–P7 是机制依赖，不是要求一次完成的开发工单。执行时将它们映射为
`DESIGN2_FOUR_STAGE_EXECUTION.md` 的四个 Stage：A=P0–P1、B=P2–P3、C=P4、D=P5–P7 与
evaluation/gates。第一批代码只进入 Stage A/P0，不实现 scheduler、composition 或 TP。每个
Stage 的产物只被视为 provisional dependency；下一 Stage 暴露接口、计量或容量错误时，必须按
四阶段文档的反向诊断规则回退，不能用后续优化掩盖前序问题。
