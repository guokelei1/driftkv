# EvoKV Design 3 方向：Action-Aware Out-of-Core Pipeline

日期：2026-07-30

状态：**核心问题与两卡主方向已确定，内部接口和机制保持可调整；协议和结果尚不存在**。
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
> double-buffer pipeline 的实际收益？

按显存容量顺序分组只是基础 baseline，不是贡献。普通 prefetch 或 double buffering 也不是
贡献。D3 必须证明：在相同 source-byte multiset 和 DRAM endpoint 下，它相对强
action-oblivious double buffer 仍有可归因收益。

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
- `ResidencyPlan` construction；
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
content hash。当前
H12 v1 在 record level 固定：

- record id、`compiled|exact` action 与 reason；
- old/retained/delta/latest/target-prefix/final extents；
- previous-cache presence、last-exact version 和 migration depth；
- history 与 extent-identity hashes。

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
- micro-wave packing；
- prefetch/execute/writeback launch order；
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

### 4.1 Global compatibility, bounded slices

先从已导出的 D2 compatibility membership 在全 cohort 建立逻辑 pools，再在 pool 内做
byte-bounded cuts。若一个 pool 自身超过 HBM，D3 必须切开；它只能最小化 fragmentation，
不能声称保留一个物理 global launch。W3 的固定 `extent_size` resident schedule 不是这里的
输入。

### 4.2 Per-rank concurrent-buffer admission

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

第一轮只需找到最小的真实 `rho > 1` 主点；`rho = 0.5, 1, 2, 4` 留到机制稳定后的扩展。

### 4.3 Resource-complementary packing

D1 actions 提供不同资源 profile：

- compiled：old-K/V source bytes 大、GPU compute 较轻、retained repair 不访问 embedding；
- exact：raw IDs 小、dense compute 和 row-sharded embedding collective 重；
- append：suffix IDs 与 incremental compute；
- progressive：显式 auxiliary bytes 和部分模型 replay。

D3 的核心假设是：在 shape/owner/collective constraints 内，把 transfer-heavy 和
compute/collective-heavy slices 组合，可以减少 PCIe、GPU 或 collective 暴露空洞。该假设必须
通过 independent ablation 证伪或保留。

### 4.4 Ordinary-DRAM pipeline

候选 pipeline 同时覆盖：

```text
CPU fill pinned input for i+1
H2D i+1
GPU execute i
D2H i-1
CPU drain pinned output for i-1
```

需要独立 CUDA streams、bounded pinned pools、memory credits、backpressure 和
NUMA/PCIe-local placement。所有 rank 必须遵守 D2 collective dependencies，包括 empty
participation。所有 target extents 在全局 coverage、lineage 和 checksum 通过前保持 private。

## 5. Motivation 与实验

### 5.1 D3-independent characterization

先从 H12/W2 导出最小 `WorkManifest`，在 GPU0/GPU1 上建立 M0，比较：

- no-I/O chunk characterization：每个 capacity-safe chunk 在 timer 外 preload，只提供
  optimistic ceiling，不是 same-endpoint baseline；
- sequential capacity groups；
- action-oblivious double buffer。

报告 CPU staging、H2D/D2H、GPU compute、collective、bin fragmentation、pipeline bubbles、
peak per-rank HBM 和 total completion time。旧 normalized-capsule source path 只能作为风险
警示，不能冒充 direct-old-K/V ordinary-DRAM 证据。

### 5.2 D3 baselines

主 baseline：

1. same-boundary all-exact；
2. sequential capacity groups；
3. action-oblivious double buffer；
4. global-compatibility-aware cuts；
5. `+` resource-complementary packing；
6. `+` topology-aware full pipeline。

Isolation-track variants 共享 source-byte multiset 和 `WorkManifest`。Co-design variants 记录
不同的 action/owner/source bytes，并在同一新 revision 下重跑 baselines。所有 speedup 只在
ordinary-DRAM→GPU→ordinary-DRAM 同 endpoint 内形成。

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
2. ordinary-DRAM same-boundary sequential、double-buffer、D3 和 all-exact；
3. 以 GPU0/GPU1 为主点，并在机制稳定后补必要的 GPU-count evidence；
4. 相对 action-oblivious double buffer 的可归因收益；
5. 至少一个相对 fastest same-boundary all-exact 有意义的 operating point；
6. report positive region 和 exact-preferred crossover；
7. no serving、SSD 或 direct-old-K/V DRAM 已验证等越界 claim。

若只胜过 sequential loop，不胜过 strong double buffer，D3 只是 implementation path。若
unavoidable old-K/V input lower bound 已慢于 same-boundary exact，继续调 scheduler 没有意义，
应停止并重新研究 source representation，而不是无限调参。

## 7. D3-ready 入口

当前已经存在、可用于第一步两卡 adapter 的上游事实是：

- immutable H12 `ActionPlan` 及其 hash；
- deterministic owner/embedding rules；
- Stage-A single-rank wave adapter；
- W3 owner-local `(S,R,F,record_id)` ordering、fixed-size resident extents 和 merged-exact
  execution membership；
- segmented destination implementation；
- D2 private-target、coverage、lineage、commit/abort semantics；
- real-history source extents 和现有 D1 exact/compiled operators。

正式 isolation evaluation 仍可能需要：

- 一个 normalized、capacity-independent 的 D3-facing constraint schema；
- owner/operator/program/membership/collective/layout/transaction 的完整序列化；
- 对当前 runtime 的 record-coverage/parity 检查；
- stable content hash，供所有 D3 variants 共同绑定。

此外，当前不能依赖：

- W3 timing 是正式 D2 结果；
- resident D2 schedule 能放入 HBM；
- segmented target 已被 serving 或下一轮直接消费；
- direct-old-K/V ordinary-DRAM source 已有性能结果；
- Python plan/materialization、CPU copy、pinned staging、commit 或 reclaim 免费；
- destination-v4 normalized capsule 能代表本 D3 source contract。

这些正式工件不阻塞 M0。第一轮使用最小 `WorkManifest`、private target 和
coverage/checksum 进入 mechanism discovery。所有新 artifact 记录
`scientific_result=false`、`formal_design3=false`、`stack_revision`、per-rank capacity ledger
和 timer components。只有 mechanism 变得清楚后，才决定最终 exporter/interface，并在
`docs/eval_protocol.md` 冻结正式 family。

## 8. 实施顺序

1. 固定 GPU0/GPU1，从 H12/W2 导出最小 `WorkManifest`；
2. 实现 ordinary-DRAM source/target、capacity groups 和 two-rank sequential path；
3. 加 basic double buffer、event timing 与 bounded buffers；
4. 同时审计 QK，构造最小真实两卡物理 out-of-core M1；
5. 在 M1 上建立 S0/S1/E0，并按 profile 探索 isolated D3 与 cross-layer candidates；
6. 机制明确后再补 normalized exporter、transaction、protocol 和扩展矩阵；
7. 在 GPU0/GPU1 结论清楚前不跑 3/4 GPU。

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
