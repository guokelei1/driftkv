# EvoKV Design 3 方向：Action-Aware Out-of-Core Pipeline

日期：2026-07-30

状态：**问题与探索合同已冻结，机制、协议和结果尚不存在**。本文档是进入 D3 初步实现的
唯一设计入口，不包含新实验结果，也不能作为 paper claim 的证据。D1 已冻结；D2 的
conceptual constraint contract 已足以定义 D3，但可执行、可哈希的 D2→D3 约束工件尚未
导出。D3 可以先做 schema/exporter/source-byte ledger 等准备；不同 scheduler 的比较必须等到
该工件冻结后。D2 仍须独立完成正式 W4、1/2/4-GPU same-boundary protocol、segmented
consumer 和 publication/commit/reclaim closure。开始 D3 不会自动解除这些 D2 gate。

## 0. 方向裁决

EvoKV 的三个设计按同一迁移任务逐层展开：

1. **D1 决定算什么。** 它输出 immutable `ActionPlan`，固定每条记录的
   `compiled|progressive|exact` 语义动作和有效 extent。
2. **D2 决定在哪里、以什么分布式约束执行。** 它必须导出 capacity-independent
   `WavePlan` constraints，固定 owner、operator、physical compatibility、collective
   dependencies 和 segmented target layout，但不冻结 capacity cuts 或 launch order。
3. **D3 决定物理 extent 何时进入和离开 HBM。** 在约束工件冻结后，它生成
   `ResidencyPlan`，把超显存工作集切成 capacity-safe micro-waves，并调度
   DRAM→GPU→DRAM 流水。

新的 D3 核心问题是：

> 当完整 source 加 private target K/V 超过每卡可用 HBM 时，能否在不改变 D1 actions、
> D2 owner/operator/compatibility/dependency/layout 的前提下，通过
> global-compatibility-aware capacity cuts、resource-complementary packing 和
> topology-safe overlap，保留 D1+D2 的实际系统收益？

按显存容量顺序分组只是基础 baseline，不是贡献。普通 prefetch 或 double buffering 也不是
贡献。D3 必须证明：在相同 source-byte multiset 和 DRAM endpoint 下，它相对强
action-oblivious double buffer 仍有可归因收益。

此前的 organic mixed-version program graph、communication-aware semantic selection 和
cross-wave bounded-renewal controller 不再叫 D3。它们被降为三个当前 design 之后的未来反馈
层；D3 不得根据 I/O 压力重新选择 D1 action。

## 1. 系统边界

主边界固定为：

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
commit。不得通过提前回收 source 削弱 abort 后旧 manifest 的可读性。

## 2. 三层不可变接口

### 2.1 D1 `ActionPlan`

D1 在 plan level 固定 source/target version、policy、provenance、counts 和 content hash。当前
H12 v1 在 record level 固定：

- record id、`compiled|exact` action 与 reason；
- old/retained/delta/latest/target-prefix/final extents；
- previous-cache presence、last-exact version 和 migration depth；
- history 与 extent-identity hashes。

Compiled program 是由 D2 adapter 绑定的独立 D1 artifact，不是 action-record field。若未来引入
`progressive` action，必须先版本化 schema 并显式声明 auxiliary state；D3 不能自行推断。

D2/D3 都不能重新选择 action。Recommendation labels、per-user drift 和预测 task gain 都不能
作为 D3 routing signal。

### 2.2 D2 所需的 D3-facing `WavePlan` constraints

该容量无关工件必须固定：

- logical GPU owner 和 embedding routing；
- operator；
- `(S,R)` compiled shape-bin membership；
- `F`-keyed exact-pool membership；
- collective dependency/order constraints；
- segmented target layout；
- publication coverage/lineage contract。

这个接口是 global constraint plan，不是 capacity-specific batch schedule。D3 可以在同一个
compatible pool 内切片，但不能跨 incompatible membership、改变 owner/operator、移动 record
到另一个 semantic action，或破坏 collective dependencies。

当前仓库尚未物化这个工件。现有 `cohortkv_d2_wave_plan_v1` 是 Stage-A 单 rank adapter；
W3 `D2IntegratedExtent` 是先按 owner-local `(S,R,F,record_id)` 排序、再按固定
`extent_size` 切分的 resident development schedule。它们共同提供 exporter 的输入与
parity reference，但都不是独立序列化、容量无关且带 content hash 的 D3 contract。

### 2.3 D3 `ResidencyPlan`

D3 只决定：

- legal bin/pool slices；
- per-rank capacity admission；
- micro-wave packing；
- prefetch/execute/writeback launch order；
- pinned-buffer credits、backpressure 和 NUMA/PCIe staging placement。

约束 exporter 冻结后，不同 D3 scheduler/baseline 必须共享 `ActionPlan` hash 和 D3-facing
constraint hash，但生成不同 `ResidencyPlan`。

## 3. 公平的 source contract

所有 mixed out-of-core baselines 必须使用相同的 action-required source-byte multiset：

- compiled 只读 valid retained old K/V；
- exact 读 raw history IDs，不读无用 old K/V；
- append 只读 suffix IDs；
- progressive 若存在，显式读取并计量 BF16 hidden suffix；
- target 按 D2 segmented layout 分配和写回。

选择性读取是共同执行契约，不是 D3 相对 strong double-buffer baseline 的创新来源。可以保留
uniform-record-image reader 作为弱诊断，但不能用它支撑主 speedup。

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

并覆盖 `rho = 0.5, 1, 2, 4`。

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

在 exporter parity 关闭后，冻结一个上游 `ActionPlan` 和 D3-facing constraint hash，不运行
D3 scheduler。对 real-history working set 扫 `rho=0.5/1/2/4`，比较：

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

Mixed variants 共享 source-byte multiset、ActionPlan hash 和 D3-facing constraint hash。所有
speedup 只在 ordinary-DRAM→GPU→ordinary-DRAM 同 endpoint 内形成。

### 5.3 Metrics

- absolute wall time、records/tokens per second；
- CPU staging/H2D/D2H bytes、bandwidth 和 exposed time；
- GPU/collective busy、overlap 和 bubbles；
- compatibility fragmentation、launch/collective counts；
- peak per-rank fixed/input/execute/output HBM；
- private target、commit、abort、reclaim；
- full-payload K/V、hidden、score、Top-k、padding、lineage、manifest correctness。

## 6. Go/no-go

D3 进入论文结果必须同时满足：

1. full-payload correctness 与 bounded-memory admission；
2. ordinary-DRAM same-boundary sequential、double-buffer、D3 和 all-exact；
3. paired 1/2/4-GPU evidence；
4. 相对 action-oblivious double buffer 的可归因收益；
5. 至少一个相对 fastest same-boundary all-exact 有意义的 operating point；
6. report positive region 和 exact-preferred crossover；
7. no serving、SSD 或 direct-old-K/V DRAM 已验证等越界 claim。

若只胜过 sequential loop，不胜过 strong double buffer，D3 只是 implementation path。若
unavoidable old-K/V input lower bound 已慢于 same-boundary exact，继续调 scheduler 没有意义，
应停止并重新研究 source representation，而不是无限调参。

## 7. D3-ready 入口

当前已经存在、可用于第一步 exporter/schema 工作的上游事实是：

- immutable H12 `ActionPlan` 及其 hash；
- deterministic owner/embedding rules；
- Stage-A single-rank wave adapter；
- W3 owner-local `(S,R,F,record_id)` ordering、fixed-size resident extents 和 merged-exact
  execution membership；
- segmented destination implementation；
- D2 private-target、coverage、lineage、commit/abort semantics；
- real-history source extents 和现有 D1 exact/compiled operators。

进入 scheduler 对比前仍缺少：

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

因此第一轮先关闭 D2 constraint exporter，再进入 mechanism discovery。所有新 artifact 必须记录
`scientific_result=false`、`formal_design3=false`、source/action/wave hashes、per-rank capacity
ledger 和完整 timer components。只有 mechanism 改变 Pareto frontier 后，才在
`docs/eval_protocol.md` 冻结正式 family。

## 8. 实施顺序

1. 实现并冻结 capacity-independent D2 constraint exporter、parity checker 和 content hash；
2. 冻结 ordinary-host source/target extent schema 和 action-required byte ledger；
3. 实现 sequential capacity baseline 与 full-payload readback；
4. 实现 action-oblivious double buffer；
5. 实现 exported constraints→`ResidencyPlan` compiler 和 per-rank admission；
6. 加 resource-complementary/topology-aware scheduling；
7. 先做一个 real-history、单 seed、1-GPU structural screen；
8. 只有 mechanism 改变 Pareto frontier 后才冻结 D3 protocol，再扩展到 2/4 GPU 和第二
   model stream。

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
