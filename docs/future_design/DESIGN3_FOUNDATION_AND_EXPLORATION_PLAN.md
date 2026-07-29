# EvoKV Design 3 Foundation Benchmark and Exploration Plan

日期：2026-07-30

状态：**面向快速机制探索的工作计划，不是冻结接口、正式 protocol 或论文结果**。本文档只
固定 D3 的主方向和第一套两卡 benchmark 路径。D1/D2/D3 的具体接口、责任边界和候选机制都
可以在实测后调整。

当前 checkpoint：Milestone A 已完成开发态 M0/S0。H12/W2 的 682 records 被划为 26 个
logical-payload-bounded groups；GPU0/GPU1 使用一个可复用 pinned slot，跑通 ordinary
DRAM→HBM→真实 D2 compiled/exact→ordinary DRAM。full run 将约 30.64 GB private target
exactly-once 写回普通 DRAM。最新 makespan 17.73 秒。两 rank 包含 embedding collective 与
rank wait 的 D2 execution 为 7.02–7.79 秒，其中 lookup collective 为 0.72–1.64 秒；四段
host/device movement 与 writeback 合计 9.93–10.01 秒。该单次 profile 仅说明 D3 数据通路已成为
同量级瓶颈，仍是
`scientific_result=false` 的容量模拟，不是 speedup 或物理 out-of-core 证据。

当本文档与 [../08_core_insights_and_roadmap.md](../08_core_insights_and_roadmap.md) 或
[../eval_protocol.md](../eval_protocol.md) 冲突时，以后二者记录的已有证据边界为准；尚未形成
证据的 D3 实现路线以本文档为当前工作入口。

## 0. 核心原则：先把两卡 DRAM↔HBM benchmark 跑起来

D3 当前最需要弄清楚的不是一个完美接口，而是：

> 当大量旧 K/V 位于 host DRAM、两张 GPU 无法同时容纳完整 source 和 target 时，怎样把
> DRAM→HBM、两卡 D2 计算、HBM→DRAM 组织成高效流水？

第一优先级固定为 GPU0+GPU1：

- 两张 A40；
- one process per GPU；
- NCCL；
- 同一 NVLink/NUMA0 island；
- 暂不做 3/4-GPU、跨 island 或多节点。

两张卡已经足以暴露分布式 embedding、collective、rank imbalance、PCIe 搬运和流水同步问题。
在两卡上没有得到清楚机制前，扩到更多卡只会增加调试变量。

### 0.1 什么现在需要固定

一次 baseline/candidate 对比中，只固定：

- 相同 model versions 和 records；
- 相同 ordinary-DRAM source 与 target endpoint；
- 相同的 D1/D2 工作快照，或明确记录两者发生了什么变化；
- 相同可用 HBM、buffer budget 和计时边界；
- 输出语义正确，不能漏记录或重复记录。

这足以让开发结果可解释。

### 0.2 什么现在不需要固定

以下都不是进入 D3 benchmark 的前置条件：

- 完整、通用、capacity-independent 的 D2→D3 schema；
- 永久不变的 D1/D2/D3 边界；
- 正式 content-hash protocol；
- 完整 atomic publication/failure matrix；
- segmented serving consumer；
- 1/2/4-GPU 正式矩阵；
- 多 seed、第二数据集或第二 model-update edge。

这些能力在候选设计稳定后再补。早期 benchmark 可以使用轻量 `WorkManifest`、private target
和 coverage/checksum 结束，不必等完整论文级事务实现。

### 0.3 允许跨层回头调整

当前三层分工是一个有用的起点，不是不可穿越的墙：

```text
D1：给出当前 planning 决策
D2：给出当前两卡执行方法
D3：组织 DRAM↔HBM residency 与流水
```

第一轮会尽量保持 D1/D2 不变，以便看清纯数据搬运问题。但如果实验发现：

- D1 的 action 粒度天然应该与 capacity group 协同；
- D2 owner 或 pool 划分导致严重额外搬运；
- exact/compiled 的执行顺序必须配合 prefetch；
- segmented layout 不适合 host writeback；

那么可以修改 D1/D2，形成新的联合设计。唯一要求是为它记录新的 `stack_revision`，并在同一
revision 下重新跑对应 baseline。不能把修改前的 baseline 与修改后的 candidate 直接比较。

最终论文可以重新决定某个机制属于 D1、D2 还是 D3。当前优先理解问题和找到有效机制，而
不是先证明接口划分永远正确。

## 1. 两层 benchmark

### 1.1 M0：H12 两卡快速开发 benchmark

先复用现有真实 KuaiRand H12 工件：

- `theta1→theta2`、16L/H512、FP16 K/V；
- 682 records；
- 548 compiled、46 scheduled exact、88 natural exact；
- 当前 W2 `strict_cow_lpt` owner snapshot；
- 当前 row-sharded embedding 与 D2 operators。

H12 的完整 old K/V 与 private target 合计约 54.97 GiB，实际能够放入两张 A40。因此当前
M0 使用 1 GiB/rank 的 logical-payload group bound 强制产生多个 groups。它不是 analytical
HBM admission；执行器另外报告实测 peak HBM 和 pinned-slot footprint。M0 用于：

1. 写出 pageable DRAM source/target；
2. 跑通两卡 sequential groups；
3. 加入 double buffering；
4. 测量搬运、计算、collective 和空洞；
5. 快速尝试 scheduler。

M0 必须标记：

```text
capacity_emulation = true
physical_oversubscription = false
scientific_result = false
formal_design3 = false
```

M0 是可用的开发 benchmark，但不能成为“真实两卡放不下”的论文证据。不得通过复制记录、
dummy K/V、冷 embedding rows 或人为 owner 倾斜制造容量。

### 1.2 M1：真实物理 out-of-core benchmark

M0 跑通后，使用真实更大 workload 替换其 payload。当前首选 Tenrec QK，因为它具有大量真实
长历史和 item identities。QK 的 39,615 个 raw length≥512 用户只是候选池；先做 label-free
capacity audit，再决定最终 cohort。

候选构造路线：

1. 优先使用 old window 512、滑动 32 tokens、target window 512；
2. 若完整 histories 不足，尝试 old 480 + append 32；
3. vocabulary 只由 base period 拟合；
4. 根据 vocabulary filtering 后的真实 histories 做 stable-hash 用户选择；
5. calibration/program-fit users 与 benchmark users 分离。

对于 16L/H512 FP16 K/V，old 512 + target 512 约为 32 MiB/record。先审计嵌套 cohort：

| Records | 候选 old + private target K/V |
|---:|---:|
| 2,048 | 64 GiB |
| 4,096 | 128 GiB |
| 8,192 | 256 GiB |

这些只是构造目标。实际配置由真实 extents、owner balance、model/embedding footprint 和实测
usable HBM 决定。选择一个两卡完整 resident 明确放不下、但任一 group 可以安全执行的主点。

### 1.3 M1 最小模型版本

只为 D3 mechanism discovery 构造一个 model-update edge：

- 一个训练 seed；
- 初始模型目标 16L/H512；
- 一个 base model `theta0`；
- 一个 short streaming update `theta1`；
- `training_sequences=all_chunks`，记录 effective targets；
- base/update 各自只使用对应 ordinal interval 的训练 targets；
- source 是 `theta0` 对 old window 生成的 exact K/V；
- D1 在 `theta0→theta1` 上产生一份当前 action plan；
- D2 在两卡上产生一份当前执行快照。

第二个 update、recursive migrated source、多 seed 和完整质量复现都推迟到候选 D3 机制出现
以后。若 QK 配置审计失败，可以调整真实 window、catalog 或 record count；不把某一个预设
配置当作必须完成的接口。

## 2. 最小两卡执行骨架

### 2.1 轻量 `WorkManifest`

先不实现通用 D2 constraint exporter。最小 adapter 只需从现有 ActionPlan 与 D2 runtime
导出当前 benchmark 真正会用到的字段：

- record id；
- 当前 action；
- 当前 owner；
- retained/suffix/full extents；
- source locator 与 byte count；
- 当前 operator/pool key；
- target locator。

`WorkManifest` 可以先绑定当前 H12/W2 实现，后续再泛化。它的作用是让 sequential、double
buffer 和 proposed scheduler 读取同一份工作清单，而不是定义永久架构。

每次 D1/D2 adapter 发生变化，更新：

```text
stack_revision
work_manifest_hash
change_reason
```

然后在新 revision 下重跑 baselines。正式 capacity-independent exporter 和完整 parity checker
在最终机制稳定后再决定是否必要。

### 2.2 DRAM source/target

主数据路径是：

```text
ordinary pageable DRAM source
  → bounded pinned input
  → H2D
  → two-GPU D2 compute / collective
  → D2H
  → bounded pinned output
  → ordinary pageable DRAM target
```

使用 in-process pageable tensors 或 `/dev/shm/evokv_d3/<run-id>/` 放临时大 payload，仓库
只保存 compact manifest 和结果。GPU0/GPU1 位于同一 NUMA0 island；正式 profile 再显式
绑定 CPU workers。source pages 可以预先 materialize/prefault，但完整 source 不能预先 pin
住。

M0/M1 都为每条记录保留完整 committed old K/V。执行器可以根据当前 action 只搬需要的
ranges；同时报告：

- `allocated_source_bytes`；
- `source_bytes_read`；
- `H2D_bytes`；
- `D2H_bytes`。

当前 M0 为缩短机制调试，只实际物化 compiled action 会读取的 retained old-K/V，并用 manifest
记录完整 committed store 的容量；这不构成物理容量证据。M1 必须实际物化完整 old-K/V store。

### 2.3 容量控制

最初只需要一个保守的 per-rank admission：

```text
fixed model/embedding/runtime
+ active input buffers
+ current execution transient
+ active output buffers
<= configured usable HBM
```

Admission 可以先由实测 peak 加安全余量完成，不需要一开始就得到完美 analytical estimator。
若运行 OOM，就记录哪一部分估计错误，缩小 group 并修正 ledger。

### 2.4 早期正确性终点

M0 的第一版结束条件是：

- 682 records exactly once；
- valid K/V 与当前 resident/reference path 对齐；
- padding/length 正确；
- target 写回普通 DRAM；
- 每个 group 的 coverage/checksum 可审计；
- peak HBM 不越过配置预算。

第一版不要求完整 atomic epoch switch 和所有 fault injection。private target 不对外发布即可。
在 D3 候选机制确定后，再补 global commit/abort/reclaim，避免事务实现挡住数据流水探索。

## 3. 基线按最短路径建立

### 3.1 S0：sequential groups

```text
for each group:
    DRAM → pinned
    H2D
    D2 execute
    D2H
    pinned → DRAM
```

group 之间不 overlap。S0 首先证明两卡、分组、真实 payload 和计时骨架正确。

### 3.2 S1：strong double buffer

在 S0 上增加：

```text
prefetch i+1
execute i
drain i-1
```

使用独立 CUDA streams/events、bounded pinned slots 和简单 backpressure。S0/S1 使用相同
`stack_revision`、WorkManifest、capacity budget 和 source/target endpoint。

第一版只搜索少量 group-size/buffer-depth 组合，目的是得到可信通用流水基线，不做大规模
tuning。

### 3.3 E0：same-boundary all-exact

E0 对最终系统叙事重要，但不阻塞第一版 M0/S0/S1。M0 流水稳定后，再让 all-exact 使用相同
两卡、DRAM endpoint、target layout、capacity budget 和 timer。E0 可以独立选择适合 exact 的
group size/order。

### 3.4 初始 timer

目标 wall boundary 从第一个 ordinary→pinned copy 前开始，到最后一个 target extent 写回
普通 DRAM 为止。当前 S0 在 timer 内做 sampled finite/metadata 检查，在 timer 后做 global
exactly-once；full checksum/numerical parity 是 S1 对比前的补充检查。记录：

- ordinary→pinned；
- H2D；
- D2 compute；
- embedding collective；
- D2H；
- pinned→ordinary；
- rank wait 和 pipeline bubble；
- peak HBM/pinned memory。

Plan construction、atomic commit 和 reclaim 先单独计量；最终论文 protocol 再决定哪些合入
primary timer。开发阶段优先保证各变体使用同一计时方式。

## 4. D3 探索方向，而不是预先指定唯一算法

S0/S1 跑通后，先看 profile 再决定候选。优先探索：

### 4.1 Capacity-aware grouping

- group 太小导致 launch/collective 碎片；
- group 太大挤压双缓冲；
- 两 rank 字节不平衡导致 collective wait；
- exact/compiled shape 混合导致 padding。

目标是找到合理 micro-wave 单位，而不是先追求通用最优 planner。

### 4.2 Action-aware overlap

当前假设：

- compiled 更偏 old-K/V transfer；
- exact/append 更偏 embedding collective 与 compute。

可以尝试把 transfer-heavy 与 compute-heavy work 交错，使：

```text
compiled H2D
overlap with
exact compute / collective
```

如果 profile 不支持这种互补，就放弃该假设，不强行保留 C2 名称。

### 4.3 Cross-rank coordinated pipeline

两卡独立 prefetch 容易使一张卡提前、另一张卡阻塞 collective。可以探索：

- global group order；
- rank-aware byte balance；
- collective-arrival-aware prefetch；
- shared credits/backpressure；
- 限制同时 H2D/D2H 对 PCIe 的争抢。

### 4.4 允许的联合调整

若单纯 D3 scheduling 不够，可以探索：

- D2 owner placement 同时考虑 source bytes 与 target bytes；
- D2 pool/granularity 更适合 capacity slicing；
- D1 planner 输出更适合 streaming execution 的 action groups；
- D1 budget 与 D3 capacity pressure 的联合 sensitivity；
- segmented target layout 为 host writeback 做调整。

这些变体不再叫“纯 D3 ablation”，而叫新的 `stack_revision`。最终哪部分进入 D2、哪部分进入
D3，由实验结果和论文叙事共同决定。

### 4.5 不急着做的内容

- 3/4 GPU；
- topology sweep；
- SSD/database ingress；
- serving trace；
- multi-update lifecycle；
- formal failure matrix；
- exhaustive parameter sweep。

在 GPU0/GPU1 上 proposed method 尚未清楚胜过 S1 前，不进入这些方向。

## 5. 五个灵活 milestone

Milestone 只定义要看到的输入和输出，不锁死中间实现。遇到问题可以跨 milestone 回头调整。

### A：两卡 M0 sequential benchmark

**状态：完成开发态 foundation。**

已完成 exactly-once、metadata、有限值和 ordinary-DRAM endpoint 检查；D2 算子本身沿用已通过
full-payload development validation 的路径。独立的端到端数值 parity 可在 S1 对比前补，不
阻塞流水骨架实现。

**目标。** 尽快得到第一个真实 DRAM→GPU0/GPU1→DRAM 完整 run。

**输入。** H12 ActionPlan、W2 owner/runtime、现有 checkpoints/program。

**输出。**

- H12 `WorkManifest`；
- pageable DRAM source/target；
- capacity grouping；
- two-rank S0；
- correctness、bytes、phase time、peak HBM。

**最大风险。** 现有 W3 路径硬编码三卡并预载全部 GPU payload。处理方式是新建轻量两卡
D3 executor，复用算子而不是直接改造 W3 benchmark。

### B：两卡 M0 strong pipeline

**状态：当前下一步。**

**目标。** 得到 S1，并知道普通流水已经能隐藏多少搬运。

**输入。** A 的 S0。

**输出。**

- double-buffer streams/events；
- bounded pinned slots；
- basic backpressure；
- S0/S1 同 revision 对比；
- 暴露 H2D/D2H、rank wait 和 bubbles。

**最大风险。** 当前 hot path 的全局 synchronize 破坏 overlap。若出现这种情况，先换成
event-based timing，不要求先重构全部 D2 runtime。

### C：真实 QK 物理容量 benchmark

**目标。** 用真实用户、真实历史和真实 model edge 替换 M0 的软件容量 cap。

**输入。** QK capacity audit 和已经跑通的 A/B executor。

**输出。**

- 冻结一个可执行的 QK cohort；
- `theta0→theta1`；
- D1/D2 当前工作快照；
- 完整 ordinary-DRAM old K/V；
- 物理超出两卡 HBM 的 M1；
- M1 上的 S0/S1，随后补 E0。

**最大风险。** 长 histories 经 vocabulary filtering 后不足，或 source materialization 太慢。
可以调整真实 catalog/window/count，或优化增量 writer；不必维持预设的 4,096 records。

### D：D3 与跨层机制探索

**目标。** 在 M1 上找到比 S1 更好的、收益可解释的机制。

**输入。** C 的两卡 S0/S1/E0 与 phase profile。

**输出。**

- 若干小而清楚的候选；
- 每个候选对应的 `stack_revision`；
- 同 revision baseline；
- wall-time 与 bytes/wait/bubble 归因；
- 保留一个最小有效机制，或得到明确 no-go。

**最大风险。** strong S1 已经接近 DRAM/PCIe 上限。此时可以探索 D1/D2/D3 联合调整，但不能
通过弱化 S1 制造贡献。

### E：机制稳定后再正式化

**目标。** 把开发 benchmark 上的机制变成论文设计。

**届时再补。**

- 最终 D1/D2/D3 责任边界；
- 通用 schema/exporter/hash；
- atomic publication、abort、reclaim；
- exact crossover；
- capacity/action-mix sensitivity；
- 1/2/4-GPU；
- 第二 model edge 和正式 replication；
- frozen protocol。

**最大风险。** 收益只存在于一个 group size 或 M0 capacity cap。若在真实 M1 上消失，则回到
D，而不是用更多形式化工作包装失败候选。

## 6. 开发判断标准

当前不设置复杂 formal gates，只连续回答四个问题：

1. **能否运行？** 两卡 bounded-memory DRAM→GPU→DRAM 是否正确完成？
2. **瓶颈是什么？** 搬运、compute、collective、rank wait 还是 writeback？
3. **通用流水有多强？** S1 相对 S0 隐藏了多少？
4. **场景特异机制是否有额外收益？** proposed 是否稳定胜过 S1，且能由 profile 解释？

如果 proposed 只胜 S0、不胜 S1，它暂时只是实现路径。如果 action-aware scheduling 没有收益，
允许回到 owner、pool、group 或 D1 plan 粒度继续探索。这里的回退不是违反接口，而是设计
过程本身。

论文阶段仍需要严格可比性、correctness、physical capacity、all-exact、failure 和扩展实验；
但这些不阻塞当前两卡机制发现。

## 7. 状态持久化

开发产物放在 `configs/evokv_d3/development/`。当前文件和后续位置是：

```text
h12_w2_m0_work_manifest.json
h12_w2_m0_group_plan.json
h12_w2_m0_foundation.json
h12_w2_m0_s0_canary.json
h12_w2_m0_s0_full.json
h12_w2_m0_s1_*.json
h12_w2_m0_candidate_*.json
```

每个 run 至少记录：

- `stack_revision`；
- model/action/owner identities；
- GPU0/GPU1 UUID 与 topology；
- configured/observed HBM；
- source/target bytes；
- group/buffer configuration；
- phase times；
- correctness；
- 当前结论、失败原因和下一步。

大 K/V payload 只存在进程内 pageable DRAM 或 `/dev/shm` 等临时路径，不进入 Git。

## 8. 立即执行顺序

接下来严格限制在 GPU0/GPU1，但实现内部保持灵活：

1. 已从 H12/W2 导出最小 `WorkManifest`；
2. 已新建 pageable-DRAM source/target 与 byte-bounded grouping；
3. 已跑通 two-rank S0 canary 和 full682；
4. 当前加入 S1 double buffer、rank-wait/bubble metrics，并重跑同 revision 对比；
5. 同时完成 QK capacity audit；
6. A/B 稳定后训练一个 QK base + 一个 short update，并物化 M1；
7. 在 M1 profile 上决定第一个真正的 D3 候选。

这个顺序的目标是尽快得到可反复使用的两卡 benchmark，而不是先完成一套可能随后被设计
推翻的正式接口。
