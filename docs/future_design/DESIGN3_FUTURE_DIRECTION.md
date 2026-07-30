# EvoKV Design 3 方向：Hierarchical Out-of-Core Pipeline

日期：2026-07-30

状态：**route-aware ResidencyPlan、grouped development E0 和 contribution diagnostics
已完成；正式重复、协议和论文结果尚不存在**。
本文档描述当前最可能的 D3 形态，不是阻塞机制发现的接口合同，也不能作为 paper claim 的
证据。D1/D2 的已有证据仍按各自冻结协议解释，但新的 D3 development 可以从轻量 adapter
开始，并允许根据 DRAM/HBM 实测形成新的跨层 `stack_revision`。

具体的两卡 foundation benchmark、H12 semantic canary、真实 QK 物理 out-of-core
workload、分阶段实现和回退条件见
[DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md)。

## 0. 方向裁决

EvoKV 当前按同一迁移任务分成三个层次：

1. **D1 决定算什么。** 它输出当前 `ActionPlan`，记录每条记录的
   `compiled|progressive|exact` 语义动作和有效 extent。
2. **D2 决定当前两卡怎样执行。** 它给出 owner、operator、physical compatibility、
   collectives 和 target layout。
3. **D3 研究物理 extent 何时进入和离开 HBM。** 它把超显存工作集切成 capacity-safe
   micro-waves，并调度 DRAM→GPU→DRAM 流水。

这三层是当前叙事骨架，不是永久接口。探索分成两条轨道：

- **isolation track：** 在一次 revision 中保持 D1/D2 工作不变，只研究搬运和调度；
- **co-design track：** 允许容量与通信结果反向调整 D1 action granularity、D2 owner/pool/
  layout，再为整个新 stack 重跑 baselines。

不允许的是把旧 baseline 与修改后的 candidate 直接比较；允许的是在运行前重新生成一个
完整、可记录的新 revision。

新的 D3 核心问题是：

> 当完整 source 加 private target K/V 超过每卡可用 HBM 时，怎样在 GPU0/GPU1 上组织
> DRAM↔HBM 与分布式计算流水，并在必要时联合调整上游执行组织，得到超过通用
> fixed-FIFO fine-grained pipeline 的实际收益？

按显存容量顺序分组只是基础 baseline，不是贡献。普通 prefetch 或 double buffering 也不是
贡献。D3 必须证明：在相同 source-byte multiset 和 DRAM endpoint 下，它相对强
action-oblivious fixed-FIFO segmented pipeline 仍有可归因收益，而不只是把 whole-group
buffer 切成 microbatches。

当前不主动恢复 organic mixed-version program graph 或复杂 renewal controller。但若简单
out-of-core profile 表明 action/owner/granularity 必须协同，允许先做小规模跨层候选，再根据
最终机制重新划分论文中的 D2/D3 边界。

## 1. 系统边界

目标数据路径为：

```text
ordinary host DRAM source manifest
  → bounded pinned input staging
  → H2D
  → D2 owner/shard execution
  → D2H
  → bounded pinned output staging
  → ordinary host DRAM private target
  → validation + atomic target-manifest publication
```

边界内包括：

- CPU source→pinned 和 pinned→target copies；
- H2D/D2H；
- `ResidencyPlan` construction 与 qualification（当前单独计量、在 primary timer 外）；
- D2 lookup、collective、compiled/exact/append compute；
- private target writeback、validation、commit、staging reclaim。

边界外包括：

- SSD、database、object store、remote storage 到 host DRAM；
- serving trace、hotness、online admission、foreground SLO；
- host-DRAM oversubscription；
- model-parallel dense trunk；
- durable cross-host transaction。

初始设计假设 ordinary host DRAM 可同时保存 committed source 和 complete private target，直到
commit。两卡 M0 的最小终点可以先使用 private target + coverage/checksum；atomic publication
和 failure closure 在机制稳定后补齐。

## 2. 当前工作快照

### 2.1 D1 `ActionPlan`

D1 在一个 `stack_revision` 内记录 source/target version、policy、provenance、counts 和
content hash。H12 保留为 semantic canary；当前物理 M1 在 QK 上固定 2,048 records，其中
410 exact、1,638 compiled。每条 record 固定：

- record id、`compiled|exact` action 与 reason；
- old/retained/delta/latest/target-prefix/final extents；
- previous-cache presence、last-exact version 和 migration depth；
- history 与 extent-identity hashes。

该 M1 使用 24L/H1536、16.364-GiB global FP32 embedding 和一个短
`theta0→theta1` edge。完整 committed old K/V 为 144 GiB，complete private target 也为
144 GiB；两 rank 各持有 1,024 records 和 72 GiB old/target payload。17 个 route-pure
capacity groups 是当前 S0/S1/D3 共同的固定物理边界。

Compiled program 是由 D2 adapter 绑定的独立 D1 artifact，不是 action-record field。若未来引入
`progressive` action，必须先版本化 schema 并显式声明 auxiliary state；D3 不能自行推断。

Isolation track 不重新选择 action。Co-design track 可以在运行前生成新的 D1 plan，但必须为
新 revision 重跑 baseline。Recommendation labels、per-user drift 和预测 task gain 仍不能
作为 D3 routing signal。

### 2.2 从最小 `WorkManifest` 到正式 constraints

最初两卡 benchmark 只需从当前 D2 runtime 导出：

- record/action/owner；
- source/target extents 与 byte counts；
- operator/pool key；
- source/target locators。

正式 isolation evaluation 若仍需要 capacity-independent constraints，再补充：

- logical GPU owner 和 embedding routing；
- operator；
- `(S,R)` compiled shape-bin membership；
- `F`-keyed exact-pool membership；
- collective dependency/order constraints；
- segmented target layout；
- publication coverage/lineage contract。

正式接口可以是 global constraint plan，而不是 capacity-specific batch schedule。当前
development 不必先证明它是最终接口；若 scheduler 需要改变 owner/operator/pool，则形成新的
co-design revision。

当前仓库尚未物化这个工件。现有 `cohortkv_d2_wave_plan_v1` 是 Stage-A 单 rank adapter；
W3 `D2IntegratedExtent` 是先按 owner-local `(S,R,F,record_id)` 排序、再按固定
`extent_size` 切分的 resident development schedule。它们共同提供 exporter 的输入与
parity reference，但都不是独立序列化、容量无关且带 content hash 的 D3 contract。

### 2.3 D3 schedule

D3 只决定：

- legal bin/pool slices；
- per-rank capacity admission；
- route-specific input/compute/output segmentation；
- 保持各 route 内部顺序的跨 route stable interleave；
- one-lookahead/one-drain prefetch/execute/writeback launch order；
- pinned-buffer credits、backpressure 和 NUMA/PCIe staging placement。

Isolation track 的 scheduler/baseline 共享同一 `WorkManifest`。Co-design track 可以生成新的
ActionPlan/owner/pool/layout，但必须记录新 revision 并重跑对应 baselines。

## 3. 公平的 source contract

Isolation-track mixed baselines 使用相同的 action-required source-byte multiset：

- compiled 只读 valid retained old K/V；
- exact 读 raw history IDs，不读无用 old K/V；
- append 只读 suffix IDs；
- progressive 若存在，显式读取并计量 BF16 hidden suffix；
- target 按 D2 segmented layout 分配和写回。

选择性读取是 isolation track 的共同执行行为，不是 proposed scheduler 相对 strong
double-buffer baseline 的创新来源。Co-design track 若改变 action/source bytes，必须报告变化
并在新 revision 下重跑 baseline。

All-exact 必然有不同的 ActionPlan、physical constraint plan 和 raw-history source bytes。它只与
mixed 方法共享：records、target model、ordinary-host source tier、target
dtype/layout/durability、GPU topology、per-rank HBM budget、timer 和 manifest endpoint。必须
单独报告实际 source bytes。

## 4. 候选机制

### 4.1 Capacity groups are the outer boundary

先从 D2 compatibility membership 建立 route-pure logical pools，再做 capacity-safe cuts。
若一个 pool 超过 HBM，D3 必须切开；它不能声称保留一个物理 global launch。当前 M1 使用
17 个 group-64 cuts。顺序执行是 S0，whole-group lookahead-1 是 strong S1；二者共享完全
相同的 D1/D2 work。

### 4.2 Active mechanism: rate-matched route-aware pipeline

Strong S1 的外层流水为：

```text
prefetch group i+1 || execute group i || drain group i-1
```

它已经把 fair S0 从 48.238 秒降到 32.703 秒，但 whole-group input 仍留下
6.20--6.79 秒 boundary wait。当前 D3 在每个 capacity group 内再切 microbatches：

```text
pageable pack j+1 || H2D j
unchanged D2 compute/collective
D2H j+1           || ordinary-DRAM publish j
```

这两级流水构成固定顺序的 precursor。输入和输出各交替复用两个已有 pinned components；
CUDA events 在 CPU 读取或覆盖 pinned storage 前建立生命周期边界。外层仍只有一个
prefetched group、一个 outstanding drain 和一个 main-thread collective issuer，因此收益
不是来自第三个 resident group、无界队列或改变 D1/D2 work。

Input-only segmentation 是必要的因果 probe：它把 makespan 降到 31.096 秒，同时将瓶颈转移
为 3.71--4.87 秒 output-credit wait。历史 v1 双向 segmentation 再将 makespan 降到
28.885 秒，相对 S1 为 1.133x；full outputs 与 S1 逐字节一致。由于 runner identity
不同，28.885 秒不能作为当前 plan 的 order-only control。

当前 `ResidencyPlan` 在这个 precursor 上再加入两个互相依赖的决定：

1. compiled/exact 各自独立的 input-segment、compute-batch、output-segment；
2. compiled 与 exact 两条内部有序序列之间的 stable interleave。

Planner 只组合来自同一个 no-plan source 的 compiled/exact profile；每个 stage 先取最慢
rank，尾组按离散 segment 数缩放，再使用与运行时一致的 one-lookahead/one-drain
bounded-flow recurrence。小搜索空间穷举所有保持两条 route 内部顺序的 interleave，大空间
使用 Pareto-beam DP。预测全局最优点 3% 内再按 HBM、pinned memory 和 segment 数选择，
该 tie 区间锚定全局最小值，不依赖遍历顺序。

当前选择仍为两 route `(8,8,8)`，但 launch order 变为
`[13,0,1,2,3,4,5,6,7,8,9,10,11,14,12,15,16]`。在同一 exact stack/hash 下，route-major
control 为 28.514442098 秒，selected order 为 28.147194647 秒，即 1.013047x、wall time
降低 1.2879%；相对 S1/fair S0 为 1.16186x/1.71379x。预测 29.244944224 秒，比实测高
3.90%。两 rank 各 77,309,939,712-byte target 均与 S1 逐字节一致，coverage
complete/exactly-once。

Plan 内嵌 profile，并绑定 compiler/program/source code、Torch/CUDA、GPU UUID/PCI、store
tier、groups、checkpoints、HBM 和 pinned limit；两 rank 在 target creation 和 collective
前统一执行 capacity preflight。Plan/profile construction 在 primary timer 外。当前 triple
仍对称，不能声称 route-asymmetric granularity 收益已验证；input-16/output-4 仅在各自观察点
没有提升。相邻 identity-only revision 的 29.7169→28.0497（5.61%）只作为波动诊断。

### 4.3 Per-rank concurrent-buffer admission

对 micro-wave `i`，每个 rank 都必须满足：

```text
fixed(model + embedding shard + program + context)
+ input(i+1)
+ execution transient(i)
+ output(i-1)
<= physical HBM - allocator/safety margin
```

容量以每卡可用 HBM 为硬约束，不能只比较 aggregate bytes。主 sweep 使用

```text
rho = max_r(work_bytes_r / usable_HBM_r)
```

当前真实主点已经是 \(\rho>1\)：144-GiB old 与 144-GiB private target 无法驻留两张 A40。
group-64/microbatch-8 的 peak allocated HBM 为 29.27/29.09 GiB，reserved 为
39.42 GiB。microbatch-16 更慢且把 reserved 提高到 41.54 GiB，所以不选。
当前 exact-stack pair 中，route-major/selected 的 observed reserved 为
39.42/35.25 GiB；这是一条开发态伴随观察，尚不能独立宣称为普遍的调度 memory saving。

### 4.4 Fallback, not the active mechanism

若 independently tuned/repeated E0 或 sensitivity 推翻当前 development-positive
crossover，才恢复 resource-complementary action packing、rank-aware byte balance、
collective-arrival-aware prefetch 或 D1/D2/D3 co-design。任何改变 actions、owners、pools
或 layout 的版本都必须获得新 `stack_revision` 并重跑其 S0/S1。

## 5. Motivation 与实验

### 5.1 Development characterization

H12/W2 M0 已完成最小接口验证，但真实物理结论来自 QK M1。相同 2,048 records、固定
D1/D2 work、17 groups 和 ordinary-DRAM endpoints 下，当前链路是：

```text
fair sequential S0        48.238 s
strong whole-group S1     32.703 s
input-only causal probe   31.096 s
historical v1 segmented   28.885 s
v3 route-major control    28.514442098 s
v3 route-aware plan       28.147194647 s
```

这条链同时报告 CPU staging、H2D/D2H、GPU compute、collective、pipeline waits、peak HBM
和 total completion time。旧 normalized-capsule path 仍只能作为风险警示，不能与
direct-old-K/V M1 pooling。

### 5.2 D3 baselines

当前主 baseline/candidate：

1. fair sequential capacity groups；
2. strong action-oblivious whole-group pipeline；
3. generic fixed-FIFO bidirectionally segmented pipeline：两条 route 共用一个独立调优的
   global `(I,C,O)` triple，不读 route profile；
4. route-aware ResidencyPlan；
5. independently tuned same-boundary all-exact E0，其候选同时包含 grouped 与 bounded
   fine-grained exact pipeline。

E0 现有开发态顺序/双 slot 时间为 44.639/33.549 秒。补充的 owner-local、naive-staged
D1-only 路径为 57.597 秒，当前 binary 的 sequential D1+D2 rerun 为 49.753 秒。它们说明
D1 的逻辑稀疏性在超显存场景下不会自动转化为 wall-time 收益；D2 能减少连续重写和
collective fragmentation，但最终相对 strong all-exact 的明确 crossover 仍依赖 D3。
这些均为单次 development diagnostics，不是 formal waterfall；D1-only 也不是
placement-oblivious owner-compute ablation。

Isolation-track variants 共享 source-byte multiset 和 `WorkManifest`。Co-design variants 记录
不同的 action/owner/source bytes，并在同一新 revision 下重跑 baselines。所有 speedup 只在
ordinary-DRAM→GPU→ordinary-DRAM 同 endpoint 内形成。

机制因果链中，route-specific control 为 compiled/exact 分别选择一个 `(I,C,O)` triple，
但保持 route-major order；selected-order row 必须复用这两个 triples，只增加 stable
interleave。headline ResidencyPlan 可以独立联合调优，但不能拿它与不同 cuts/resources 的
control 做机制归因。

### 5.3 Metrics

- absolute wall time、records/tokens per second；
- CPU staging/H2D/D2H bytes、bandwidth 和 exposed time；
- GPU/collective busy、overlap 和 bubbles；
- compatibility fragmentation、launch/collective counts；
- peak per-rank fixed/input/execute/output HBM；
- private target、commit、abort、reclaim；
- full-payload K/V、hidden、score、Top-k、padding、lineage、manifest correctness。

## 6. 论文条件，不是开发前置 gate

D3 最终进入论文结果时应满足：

1. full-payload correctness 与 bounded-memory admission；
2. ordinary-DRAM same-boundary sequential、whole-group double-buffer、generic segmented、
   D3 和 all-exact；
3. 以 GPU0/GPU1 为主点，并在机制稳定后补必要的 GPU-count evidence；
4. 相对 independently tuned generic fixed-FIFO segmented baseline 的可归因收益；
5. 至少一个相对 fastest same-boundary all-exact 有意义的 operating point；
6. report positive region 和 exact-preferred crossover；
7. 不把 development timing 写成 formal result，也不扩张到 serving、SSD 或 remote tier。

若只胜过 sequential loop 或 whole-group double buffer，不胜过 generic segmented
baseline，D3 只是 implementation path。若 unavoidable old-K/V input lower bound 已慢于
same-boundary exact，继续调 scheduler 没有意义，应停止并重新研究 source
representation，而不是无限调参。

## 7. 当前实现与正式化缺口

当前两卡 adapter、M1 store、独立 route-specific segmentation、stable interleave、
plan/stack hash 和双向流水已经存在。它们依赖的上游事实是：

- immutable H12 `ActionPlan` 及其 hash；
- deterministic owner/embedding rules；
- Stage-A single-rank wave adapter；
- W3 owner-local `(S,R,F,record_id)` ordering、fixed-size resident extents 和 merged-exact
  execution membership；
- segmented destination implementation；
- D2 private-target、coverage、lineage、commit/abort semantics；
- real-history source extents 和现有 D1 exact/compiled operators。

正式 isolation evaluation 仍需要：

- 一个 normalized、capacity-independent 的 D3-facing constraint schema；
- owner/operator/program/membership/collective/layout/transaction 的完整序列化；
- 对当前 runtime 的 record-coverage/parity 检查；
- held-out calibration/evaluation boundary 或第二个 action/capacity mix；
- independently tuned and repeated formal E0 与 transaction closure。

此外，正式 evaluation 不能依赖：

- W3 timing 是正式 D2 结果；
- resident D2 schedule 能放入 HBM；
- segmented target 已被 serving 或下一轮直接消费；
- direct-old-K/V ordinary-DRAM development timing 等同于 formal evidence；
- Python plan/materialization、CPU copy、pinned staging、commit 或 reclaim 免费；
- destination-v4 normalized capsule 能代表本 D3 source contract。

这些正式工件没有阻塞 M0/M1 mechanism discovery。当前 artifact 记录
`scientific_result=false`、`formal_design3=false`、`stack_revision`、per-rank capacity ledger
和 timer components。当前 plan/profile 自身已有 stable content hash，但它不是正式 D2
constraint exporter。当前开发态 E0 已完成，接下来进入独立调优、重复和最小资格验证；
通过后才决定最终 exporter/interface，并在 `docs/eval_protocol.md` 冻结正式 family。

## 8. 实施顺序

1. 已从 H12/W2 导出最小 `WorkManifest` 并完成 M0；
2. 已构造 QK M1 ordinary-DRAM source/target 和真实物理 oversubscription；
3. 已完成 fair S0 与 strong S1；
4. 已构造并验证 generic fixed-FIFO bidirectionally segmented S2；
5. 已实现 route-aware ResidencyPlan，并在同一 exact stack/hash 上完成 route-major 与
   selected-order 配对、full byte parity 和 capacity preflight；
6. 已完成 same-boundary grouped E0 和 owner-local D1-only contribution diagnostics；
7. 下一步完成 independently tuned formal E0/generic S2、held-out qualification、正式
   重复和最小 capacity/action-mix qualification；
8. 若通过，再补 normalized exporter、transaction、formal protocol 和扩展矩阵；
9. GPU0/GPU1 的正式结论清楚前不跑 3/4 GPU。

## 9. 更远期反馈层

Organic mixed-version program graph、program composition、communication-aware semantic selection
和 cross-wave bounded renewal 保留为 D1+D2+D3 之后的 future controller，不属于当前 D3。

该控制器若恢复，仍必须满足：

- 不使用 recommendation labels 或 task-quality admission oracle；
- exact renewal/depth/deadline 不可被 composition 取消；
- abort/reject 不推进 lineage；
- 不发布 stale-as-current 或 mixed target epoch；
- 需要独立 organic workload contract 和新的 protocol family。

当前不实现该反馈层，也不把它计入 EvoKV 的第三个 design。
