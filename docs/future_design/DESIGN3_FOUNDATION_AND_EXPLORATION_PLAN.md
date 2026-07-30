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

M1 已推进到真实物理 out-of-core 的 sequential S0，并冻结为当前 D3 开发边界。QK 全体用户
的 first-64 raw exposures 产生 2,859,835 个 base-active item entities；top-250k 是
prediction rows，其余 2,609,835 行是 lossless context entities，stream-only item 只 hash
到已有 context rows。含 padding 的 H1536 FP32 embedding 有 2,859,836 行、16.364 GiB。
2,560-user 输入包含 512 个 fit/calibration users 和一个 2,048-user benchmark pool。
双卡 24L/H1536 sharded trainer 已完成 `theta0→theta1`；独立
`window_1=[544,576)` 上，NDCG@10 从 0.371468 升至 0.380294，Hit@10 从 0.520259 升至
0.547073，得到进入系统机制探索所需的正推荐信号。

该 edge 的 D1 snapshot 固定 410 exact、1,638 compiled，即 20.0195% exact。D2
characterizer 已在相同 records 上完成：mixed/all-exact request tokens 为
262,336/1,048,576，off-rank FP32 return-vector bytes 为
805,380,096/3,216,408,576。完整 exact old K/V 已实际物化到 ordinary DRAM，共 144 GiB。
随后 GPU0/GPU1 full S0 以九个 sequential groups 处理全部 2,048 records，并写出完整
144-GiB private target；old+target 为 288 GiB。S0 makespan 为 53.497 秒。决定 makespan 的
rank 0 phase sum 为 50.017 秒，其中 pageable→pinned、H2D、D2H、publish 合计
26.397 秒，占 52.8%。因此“顺序分组存在强 DRAM staging/publication 瓶颈”已经测到，下一步
不再是继续构造 M1，而是实现同 revision 的 S1 double buffer。

为给需要两个 resident slots 的 S1 建立同容量 baseline，同一 revision 又完成了
`group_records_per_rank=64` 的 paired S0：17 groups、makespan 54.577 秒，比 group-128
慢 2.019%。rank 0 movement/publication 为 29.000/52.619 秒（55.1%），rank 1 为
29.368/52.637 秒（55.8%），两 rank peak allocated HBM 都是 20.146 GiB。缩组没有消除
瓶颈，反而增加了 movement 与 fragmentation；后续 S1 必须与这个 group-64 S0 配对，同时把
group-128 保留为顺序执行的较快 characterization 点。

以上 training、D1/D2 snapshot、materialization 和 S0 全部仍是
`scientific_result=false`、`formal_design3=false` 的单次开发工件；它们冻结 benchmark
边界，不冻结论文 protocol，也不构成 speedup 或最终 D3 机制。

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

M0 跑通后，使用 Tenrec QK 的真实大 workload 替换软件容量 cap。当前开发态 foundation 已
固定并物化：

1. 每个用户的 first-64 raw exposures 拟合完整 base entity address space；
2. base-frequency top-250k 为 prediction rows，其余 base-seen items 一一对应 context rows；
3. 仅在 base 后首次出现的 item 通过 SplitMix64 映射到已有 context rows，不扩展冷行；
4. 512 个 fit/calibration users 与一个 stable-hash 2,048-user benchmark pool 分离，benchmark
   还保留 512/1,024/2,048 nested prefixes；
5. old history 是 `[0,512)`，`theta1` update 是 `[512,544)`，target history 是 `[32,544)`，
   独立 held-out evaluation 是 `[544,576)`。

该表有 2,859,835 个语义行和一个 padding row。按 H1536 FP32，它占
17,570,832,384 bytes（16.364 GiB）。24L/H1536 的一条 512-token FP16 K/V、单版本为
72 MiB；2,048 records 的 complete old 与 private target 合计 288 GiB。这个几何现在已由
实际 ordinary-DRAM store 兑现：144-GiB complete old store 已完整物化，S0 又写出
144-GiB private target。旧 H512 QK 小 canary 继续只用于接口 smoke test，不能替代 M1。

### 1.3 M1 最小模型版本

只为 D3 mechanism discovery 构造一个 model-update edge：

- 一个训练 seed；
- 固定 24L/H1536、24 heads、每 head 64 维；
- 一个 base model `theta0`；
- 只用 `window_0=[512,544)` 产生一个 short-update `theta1`；
- `window_1=[544,576)` 只做 held-out `theta0`/`theta1` 推荐检查；
- `training_sequences=all_chunks`，记录 effective targets；
- base/update 各自只使用对应 ordinal interval 的训练 targets；
- source 是 `theta0` 对 old window 生成的 exact K/V；
- D1 在 `theta0→theta1` 上产生一份当前 action plan；
- D2 在两卡上产生一份当前执行快照。

双卡 row-sharded trainer 已完成 seed-0 的 base/update 各一轮训练并写出分片 checkpoint。
固定 held-out window 含 13,426 个 positive targets；`theta1` 相对 `theta0` 的
NDCG@10 为 0.380294 对 0.371468，Hit@10 为 0.547073 对 0.520259，sampled cross entropy
为 3.653369 对 3.707804。该正信号只用于确认这个短 edge 没有退化，并冻结为当前 D3
development boundary；它不是多 seed 的算法质量结论。

edge-specific direct-old-K/V program、20.0195%-exact D1 action snapshot 和 D2 request
characterization 都已完成并绑定相同 checkpoint/data identity。第二个 update、recursive
migrated source、多 seed 和完整质量复现继续推迟到候选 D3 机制出现以后。

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
这一 M1 要求现在已经完成：两 rank 各持有 72 GiB、1,024 records 的 complete exact
`theta0` old store，总计 144 GiB，coverage 无 partial/missing。S0 的 source
materialization 独立于 primary timer，随后复用绑定后的 old store。

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

### 2.5 大 extent 的索引边界

M1 的第一批大 compiled groups 暴露了小 benchmark 看不到的 Triton 指针索引错误。每 rank
的一个 full group 含 128 records，每条 retained prefix 为 480 tokens，因此一个 extent 有
61,440 tokens；在 H1536 下，layer 23 的 flattened source start 是
2,170,552,320 elements，已经超过 signed int32 的 \(2^{31}\)。

直接 old-K/V affine kernel 现已在地址计算前将 program ID、token count、token/output/reduction
offset 全部提升为 int64。修复后，cold-cache 路径首次执行同一个跨 \(2^{31}\) 大组并完成 full
S0。该事件应保留为“大规模 extent 需要 64-bit indexing”的实现教训和 correctness
validation；不能把 bug 修复本身包装成 D3 性能设计。

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

M1 S0 已完成这一目标。固定 D1/D2 work 被划为九个 route-pure groups：七个 compiled
groups 和两个 exact groups。两 rank 各处理 1,024 records，最终 2,048 records exactly
once，complete target coverage 无 partial/missing。完整 ordinary-DRAM old/private-target
footprint 为 288 GiB，makespan 为 53.497 秒。

rank 0 决定 makespan。其 phase sum 为 50.017 秒：

- pageable→pinned：10.861 秒；
- H2D：2.648 秒；
- D2H：3.244 秒；
- pinned/pageable target publication：9.643 秒。

四项合计 26.397 秒，占 phase sum 的 52.8%。这不是把 52.8% 都称为 PCIe；它包含两端
ordinary-memory copy。当前可以下的结论只有：顺序分组使 DRAM staging/publication 成为强
瓶颈，值得马上验证 S1 是否能把它与 D2 compute/collective 重叠。

S1 的双 slots 不能直接与 group-128 单 slot 比容量，因此已经补跑 group-64 paired S0：

- 2,048 records、17 groups、makespan 54.577 秒；
- rank 0 movement/publication 29.000/52.619 秒（55.1%）；
- rank 1 movement/publication 29.368/52.637 秒（55.8%）；
- 两 rank peak allocated HBM 均为 20.146 GiB。

它比 group-128 S0 慢 2.019%，且 movement time/fraction 都更高。这排除了“只把 group
缩小就能解决搬运瓶颈”的简单解释，也冻结了第一版 group-64 S1 的直接 S0 control。

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

当前下一步直接在 M1 的同一 work/source revision 和 group-64 capacity setting 上实现 S1，
以 group-64 paired S0 为直接 control。group-128 S0 继续报告为 sequential
characterization；M0 S1 可以作为小型调试入口，但不再是启动真实 S1 的前置条件。

### 3.3 E0：same-boundary all-exact

E0 对最终系统叙事重要，但不阻塞第一版 M1 S1。S1 流水稳定后，再让 all-exact 使用相同
两卡、DRAM endpoint、target layout、capacity budget 和 timer。E0 可以独立选择适合 exact
的 group size/order。

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

当前 M1 S0 还将 complete old-store materialization、model/program load、operator warmup 和
target prefault 放在 primary timer 外。53.497 秒是 `run_s0` 的两 rank makespan；50.017 秒
是 makespan rank 的分项 phase sum。52.8% 搬运比例以 phase sum 为分母，不能与 wall time
混写。group-64 S0 也使用相同边界，其 54.577 秒与未来 group-64 S1 才是直接配对。
S1 必须保持这一 revision 和边界，新增的 overlap/wait/bubble 指标再单独说明。

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

### B：真实 QK M1 边界与 sequential foundation

**状态：完成开发态 foundation。**

**目标。** 用真实用户、真实历史、真实 model edge 和物理超 HBM 的 K/V 替换 M0 软件容量
cap，并先测出顺序分组的问题。

**已完成输出。**

- 已冻结的 512 fit/calibration + 2,048 benchmark QK cohort；
- 双卡 24L/H1536 `theta0→theta1` 及独立 `window_1` 正推荐信号；
- 410 exact + 1,638 compiled 的 D1 development snapshot；
- D2 embedding request/traffic characterizer 和 edge-specific direct-old-K/V program；
- 144-GiB complete ordinary-DRAM old K/V；
- 288-GiB old/private-target、九组 group-128、2,048-record M1 S0；
- 同 revision 的 17-group group-64 paired S0；
- makespan 53.497 秒，以及 rank-0 movement/publication 26.397/50.017 秒的瓶颈归因。

**实际遇到的最大风险。** 大 extent 使 Triton 的 int32 flattened index 跨越 \(2^{31}\)。
该问题已改为 64-bit pointer arithmetic，并通过 cold-cache 大组执行。这个回看说明：后续若
S1 在大组上异常，先检查 A/M0 从未覆盖的规模边界，再判断是流水逻辑错误。

### C：两卡 M1 strong pipeline

**状态：当前下一步。**

**目标。** 在 B 的同一 development revision 上得到 S1，并知道普通 double buffering 能隐藏
多少已经测到的 52.8% DRAM staging/publication 开销。

**输入。** B 的固定 checkpoints、D1/D2 work snapshot、完整 144-GiB old store、相同
ordinary-DRAM private-target endpoint、S0 timer 和 group-64 capacity setting。

**输出。**

- bounded input/output double buffers；
- CUDA streams/events 与 basic backpressure；
- group-64 S0/S1 同 revision、同 capacity 对比；
- H2D/D2H、ordinary-copy、rank wait 和 bubbles；
- S1 稳定后补 same-boundary E0。

**最大风险。** 当前 hot path 的全局 synchronize 可能破坏 overlap；双缓冲本身也会占用更多
pinned/HBM credits，迫使 group 变小并增加 collective/launch 碎片。若 S1 迟迟没有收益，先
分别检查 event 依赖、CPU copy 并发、group capacity 和双 rank arrival，而不是立刻否定 B
测到的瓶颈。允许在同一 work snapshot 内调整物理 group size，但每个 S0/S1 pair 必须共享
相同设置。

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

M1 数据身份单独由 `configs/evokv_d3/m1/qk_entity_manifest.json` 和
`configs/evokv_d3/m1/qk_entity_cohorts.json` 绑定；大 NPZ、checkpoints 和 K/V payload
保持 local/ignored。当前 development snapshot 还包括：

```text
configs/evokv_d3/m1/qk_entity_adjacent_action_snapshot.json
configs/evokv_d3/m1/qk_entity_request_characterization.json
results/system/evokv_design3_m1/qk_entity_h1536_sharded_two_version_training_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_adjacent_compiler_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_materialize_full_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_s0_full_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_s0_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_s1_group64_dry_run_seed0.json
```

materialize、group-128 S0 和 group-64 paired S0 分别证明 144-GiB old store 与两种
288-GiB sequential boundaries 已执行；S1 文件当前只是 dry-run schedule，不是 S1 timing。
这些工件仍不证明正式 protocol、speedup 或最终 D3 机制。

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

大 K/V payload 只存在进程内 pageable DRAM 或 `/dev/shm` 等临时路径，不进入 Git。当前
M1 old store 可以在同一 run-id 下通过完整 coverage 和 model/data/plan/owner binding
复用；失败后的 partial target 不能自动覆盖，进入 S1 前应使用独立 target identity 或显式
reset，不能误把残留 coverage 当成新 run。

## 8. 立即执行顺序

接下来严格限制在 GPU0/GPU1，但实现内部保持灵活：

1. 已从 H12/W2 导出最小 `WorkManifest`；
2. 已新建 pageable-DRAM source/target 与 byte-bounded grouping；
3. 已跑通 two-rank S0 canary 和 full682；
4. 已完成 QK base-entity audit、512+2,048 cohort 和 two-window input 物化；
5. 已完成 H1536/24L 两卡 `theta0→theta1` 与 held-out 正推荐检查；
6. 已生成 20.0195%-exact D1 snapshot、D2 characterizer 和 direct-old-K/V program；
7. 已物化 144-GiB complete old K/V，并完成 2,048-record、九组、288-GiB group-128 M1
   S0；
8. 已补 group-64、17-group paired S0；当前实现同 setting 的 M1 S1 double buffer、
   rank-wait/bubble metrics；
9. S1 稳定后补 E0，再根据 M1 profile 决定第一个真正的 D3 候选。

这个顺序的目标是尽快得到可反复使用的两卡 benchmark，而不是先完成一套可能随后被设计
推翻的正式接口。
