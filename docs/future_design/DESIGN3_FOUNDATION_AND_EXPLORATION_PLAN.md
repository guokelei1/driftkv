# EvoKV Design 3 Foundation Benchmark and Exploration Plan

日期：2026-07-30

状态：**面向快速机制探索的工作计划，不是冻结接口、正式 protocol 或论文结果**。本文档只
固定 D3 的主方向和第一套两卡 benchmark 路径。D1/D2/D3 的具体接口、责任边界和候选机制都
可以在实测后调整。

当前 checkpoint：Milestone D 已完成 route-aware ResidencyPlan 的开发态实现、同一
stack/hash 下的 route-major/selected paired full 运行，以及完整 payload 校验。早期
Milestone A 的 H12/W2 682 records 被划为 26 个
logical-payload-bounded groups；GPU0/GPU1 使用一个可复用 pinned slot，跑通 ordinary
DRAM→HBM→真实 D2 compiled/exact→ordinary DRAM。full run 将约 30.64 GB private target
exactly-once 写回普通 DRAM。最新 makespan 17.73 秒。两 rank 包含 embedding collective 与
rank wait 的 D2 execution 为 7.02–7.79 秒，其中 lookup collective 为 0.72–1.64 秒；四段
host/device movement 与 writeback 合计 9.93–10.01 秒。该单次 profile 仅说明 D3 数据通路已成为
同量级瓶颈，仍是
`scientific_result=false` 的容量模拟，不是 speedup 或物理 out-of-core 证据。

M1 已推进到真实物理 out-of-core 的首个 D3 候选，并冻结为当前 D3 开发边界。QK 全体用户
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
144-GiB private target；old+target 为 288 GiB。原始 S0 makespan 为 53.497 秒。决定
makespan 的 rank 0 phase sum 为 50.017 秒，其中 pageable→pinned、H2D、D2H、publish
合计 26.397 秒，占 52.8%，直接暴露了 DRAM staging/publication 瓶颈。

为给需要两个 resident slots 的 S1 建立同容量 baseline，同一 revision 又完成了
`group_records_per_rank=64` 的 paired S0。最初的 54.577 秒包含每组
`gc.collect()`/`torch.cuda.empty_cache()`，只保留为 fragmentation 诊断。去掉这两个
非算法性维护动作后，公平 S0 为 48.238 秒，仍是 17 groups、相同 records/actions/endpoints，
两 rank peak allocated HBM 都是 20.146 GiB。

Strong S1 使用两组 bounded slots，把整组的 prefetch \(i+1\)、D2 execute \(i\) 和 drain
\(i-1\) 重叠，完成时间为 32.703 秒，相对公平 S0 为 1.475x。但它仍暴露
6.795/6.195 秒的 rank-0/1 input-boundary wait。只对输入做 microbatch ping-pong 后，
该等待降到 0.275/0.243 秒，makespan 为 31.096 秒；同时 output-credit wait 升到
4.865/3.706 秒，说明瓶颈从输入明确转移到串行输出。

历史 v1 precursor 把同样的微段流水扩展到两个方向：CPU packing \(j+1\) 与 H2D \(j\) 重叠，
D2H \(j+1\) 与 ordinary-DRAM publish \(j\) 重叠。它不增加 group-level lookahead、HBM
resident group、pinned slot 或 drain credit，也不改变 D1 action、D2 collective order 和
target layout。group-64/microbatch-8 完成时间为 28.885 秒，相对 S1 为 1.133x、相对公平
S0 为 1.670x；两个 77.3-GB/rank target 均与 S1 逐字节一致。microbatch-16 为
29.337 秒且占用更多 reserved HBM。由于它使用旧 runner，28.885 秒不能作为当前 planner
的 order-only control。

在该 precursor 上，当前 planner 已将 input segment、D2 compute batch 和 output segment
真正解耦，并允许 compiled/exact 使用不同配置。它使用 max-rank 三段 service 与实际
one-lookahead/one-drain recurrence，在保持各 route 内部次序和两 rank 相同 collective
launch order 的前提下选择 stable interleave。Profile 必须来自同一次 joint run，尾组按
离散 segment 数缩放；小空间穷举，大空间使用 Pareto-beam DP。当前 hashed plan 保留两
route `(8,8,8)`，但将 group 13 放在第一位，并在 compiled tail 附近交错其余 exact
groups。同一 exact stack/hash 下，route-major 为 28.514442098 秒，selected order 为
28.147194647 秒，即 1.013047x、wall time 降低 1.2879%。两 rank 各
77,309,939,712-byte target 均与 S1 逐字节一致。Compiled input-16 与 output-4 只是在各自
开发观察点没有提升，不能解释为一般性拒绝；当前也没有验证非对称 triple 的收益。

同一 QK M1 容量边界现已补齐 contribution diagnostics。16 个 group-64 的 sequential
all-exact 为 44.639 秒，强双 slot all-exact 为 33.549 秒；17 个 group-64 的
owner-local naive-staged D1-only 为 57.597 秒，当前 binary 的 sequential D1+D2 为
49.753 秒。D1-only 与 D2 都请求 262,336 个 embedding tokens，但前者每 rank 发起 852
次 collective，后者为 387 次。该结果不支持“D1 单独带来系统加速”；它支持的链条是：
D1 先制造逻辑稀疏性，D2 回收连续重写和 collective fragmentation，D3 再把仍然暴露的
out-of-core movement 组织成流水。相对 strong all-exact，D1+D2 S1 的收益只有 2.52%，
selected D3 的开发态收益为 16.10%。

当前 selected D3 已接近 fixed D1/D2 的 compute-bound 区域。关键 rank 的 affine、
non-lookup compute、lookup 和 assemble 合计约 26.59 秒，剩余 exposed pipeline wait
约 1.49 秒；仅继续消除 bubble 的上限约 5.5%。下一轮不应优先增加 buffer 或继续穷举
segment size，而应先测 DMA 与 affine/embedding collective 的 pairwise interference，
再探索 phase-aware DMA pacing、rank/collective-arrival-aware planning，以及独立的
compiled compute-batch 轴。

以上 training、D1/D2 snapshot、materialization、S0、S1 和 D3 candidate 全部仍是
`scientific_result=false`、`formal_design3=false` 的开发工件；它们冻结 benchmark
边界和当前机制选择，不冻结论文 protocol，也不是论文 performance claim。

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
migrated source、多 seed 和完整质量复现继续推迟到 formal E0、重复与当前候选的最小
资格验证以后。

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
在 D3 候选通过 formal E0、重复和最小 sensitivity 并最终选定后，再补
global commit/abort/reclaim，避免事务实现挡住机制资格验证。

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
ordinary-memory copy。该历史 checkpoint 得出的结论是：顺序分组使 DRAM
staging/publication 成为强瓶颈，因此进入了后续 S1 与 D3 验证。

S1 的双 slots 不能直接与 group-128 单 slot 比容量，因此已经补跑 group-64 paired S0。
第一遍执行包含每组 GC/allocator flush：

- 2,048 records、17 groups、makespan 54.577 秒；
- rank 0 movement/publication 29.000/52.619 秒（55.1%）；
- rank 1 movement/publication 29.368/52.637 秒（55.8%）；
- 两 rank peak allocated HBM 均为 20.146 GiB。

它比 group-128 S0 慢 2.019%，且 movement time/fraction 都更高。这排除了“只把 group
缩小就能解决搬运瓶颈”的简单解释。当前公平 control 进一步删除了每组
`gc.collect()`/`torch.cuda.empty_cache()`，其余边界不变，makespan 为 48.238 秒。后续
S1/D3 speedup 只与这个 no-flush S0 比较；54.577 秒只保留为旧 harness characterization。

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

M1 strong S1 已完成。它保持 group-64、同一 work/source revision、相同 ordinary-DRAM
endpoints 和 primary timer，使用两个非别名 slots、lookahead-1 和 single-drain credit，
makespan 为 32.703 秒，相对公平 S0 为 1.475x。两 rank 的 input-boundary wait 仍为
6.795/6.195 秒，因此 S1 是可信的强基线，但不是 out-of-core 流水的终点。

### 3.3 S2：generic fixed-FIFO bidirectionally segmented I/O

这个强通用 baseline 在 S1 的 bounded group pipeline 内再加入一级 microbatch pipeline：

```text
input:  pageable pack j+1  || H2D j
compute: unchanged D1/D2 group and collective order
output: D2H j+1            || pageable publish j
```

正式 S2 独立调优一个 global `(I,C,O)` triple，并让 compiled/exact 两条 route 共用，
保持 fixed FIFO；它不读取 route profile。后续 route-specific causal row 才允许两条 route
各用一个 triple但仍保持 route-major order；selected-order causal row 完全复用这两个
triples，只增加 stable interleave。

输入和输出各交替复用两个现有 pinned components。CUDA event 在 pinned storage 被 CPU
读取或覆盖前建立生命周期边界；外层仍只有一个 prefetched group 和一个 outstanding drain。
因此该机制不靠增加 resident group 或无界队列获取收益。

只分段输入的中间版本为 31.096 秒，并把 input wait 转移成 4.865/3.706 秒 output-credit
wait。双向版本进一步降到 28.885 秒，output-credit wait 降至 1.735/0.738 秒。相对 strong
S1 的额外收益为 1.133x。full S1/D3 两 rank 的 target 均逐字节一致，ledger 为
complete/no-partial/no-missing/exactly-once。当前最佳为 microbatch-8；microbatch-16
29.337 秒且 reserved HBM 更高。

### 3.4 D3：route-aware ResidencyPlan

固定 `(8,8,8)` 只在 capacity group 内优化 I/O；它仍按 compiled-then-exact 执行。当前
planner 将 D1/D2 route 看成不同资源画像：compiled 是 I/O-heavy，exact 是
compute-heavy。它独立记录每条 route 的 input/compute/output 粒度，通过 bounded-flow
模型选择两条内部有序序列的 stable interleave，并把 group/stack/profile/capacity/launch
哈希为一个可 replay 的 plan。

当前 plan 只组合来自同一次 no-plan run 的 route profiles，并内嵌所选 profile、HBM 总量、
pinned limit、compiler/program、相关源码、Torch/CUDA、GPU UUID/PCI、store tier、
checkpoints 和 group identity。计划预测 29.244944224 秒；同一 exact stack/hash 上，
route-major control 为 28.514442098 秒，selected order 为 28.147194647 秒，即
1.013047x、wall time 降低 1.2879%。相对 S1/fair S0 为 1.16186x/1.71379x。计划开始执行前，
两 rank 已完成 hash agreement 与统一 capacity preflight；执行仍只有一个 prefetch、一个
drain credit 和一个 collective issuer。

相邻的 identity-only revision 曾观察 29.7169→28.0497 秒，但该 5.61% 只记录运行波动，
不能作为冻结收益。Compiled input-16 和 output-4 只是在各自开发点未提升；selected triple
仍是两 route `(8,8,8)`，因此尚未证明 route-asymmetric granularity 的收益。

### 3.5 E0：same-boundary all-exact

开发态 E0 已完成。它让 all-exact 使用相同两卡、DRAM endpoint、target layout、capacity
budget 和 timer；16 个 group-64 的 sequential/S1 时间为 44.639/33.549 秒，均写出完整
144-GiB target，old-K/V read 为零，coverage 与 exactly-once 通过。正式 E0 仍需独立调优
group/microbatch、重复运行并冻结 source hash，不能把这两个单次 development artifact
直接当作论文分母。

### 3.6 初始 timer

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
混写。fair group-64 S0、S1 和当前 D3 pair 使用相同 runtime 边界。跨历史 runner 的开发链
为 48.238→32.703→28.885 秒；当前 exact-stack order-only pair 为
28.514442098→28.147194647 秒。54.577 秒的旧 S0 含每组 runtime maintenance，不再作为
speedup 分母；28.885 秒也不能作为当前 plan 的 order-only 分母。overlap/wait/bubble
指标用于因果分析，其中
`estimated_hidden_{input,output}_seconds` 只是保守估计，不能替代 wall/credit/boundary wait。

## 4. Profile 收敛出的 D3

### 4.1 当前选择

机制不是预先指定的。公平 S0 先证明 whole-group sequential path 的搬运瓶颈；strong S1
证明普通 double buffering 能隐藏大部分开销，但留下 input-boundary stall；input-only
candidate 消除该 stall 后又暴露 output-credit stall。由这条因果链得到的最小机制是：

- 外层保持 capacity-safe group 的 `prefetch i+1 / execute i / drain i-1`；
- 内层把每组切成 bounded microbatches；
- 输入端交替复用两个 pinned components，使 CPU pack 与 H2D 重叠；
- 输出端交替复用两个 pinned components，使 D2H 与 ordinary-DRAM publication 重叠；
- 保持固定 D1 actions、D2 owner/collective order、route-pure groups 和单 drain credit。

这比继续增加 whole-group lookahead 更符合容量边界：它不额外驻留第三组 old K/V，也不建立
无界 output queue。固定顺序 28.885 秒是 active precursor；当前 active candidate 是在其上
增加 route-aware rate matching 与 stable interleave 的 hashed ResidencyPlan。

### 4.2 接下来只做资格验证

当前不再泛化搜索 scheduler，也不做完整粒度笛卡尔积。下一步先完成：

- independently tuned and repeated formal E0；
- 至少一个 action/capacity mix sensitivity；
- 最小重复执行或第二 edge；
- publication/transaction 计时边界；
- 判断收益是否稳定跨越 exact-preferred crossover。

当前 capacity group、三段粒度、双向 credits 和 launch order 已经固化为可 replay 的
development `ResidencyPlan`。若资格检查通过，再冻结正式 schema/protocol。Coupled
microbatch-16、compiled input-16 和 compiled output-4 在各自开发观察点没有提升；这只支持
停止当前 exhaustive tuning，不构成对这些粒度的一般性拒绝。

### 4.3 只在资格验证失败时回退

若 formal E0 或 sensitivity 否定当前机制，再考虑 rank-aware byte balance、collective-arrival-aware
prefetch、D2 owner/pool granularity 或 D1/D3 budget co-design。任何改变 actions、owners、
pools 或 layout 的候选都是新的 `stack_revision`，必须重跑自己的 S0/S1，而不是与本 revision
直接比较。

### 4.4 不急着做的内容

- 3/4 GPU；
- topology sweep；
- SSD/database ingress；
- serving trace；
- multi-update lifecycle；
- formal failure matrix；
- exhaustive parameter sweep。

当前 proposed 已胜过 S1 和单次 development E0，但相对同 stack route-major control 只有
1.2879% 的单次收益；它尚未通过 independently tuned generic S2、formal E0 和最小资格
验证，因此不能据此冻结 paper claim。

## 5. 五个灵活 milestone

Milestone 只定义要看到的输入和输出，不锁死中间实现。遇到问题可以跨 milestone 回头调整。

### A：两卡 M0 sequential benchmark

**状态：完成开发态 foundation。**

已完成 exactly-once、metadata、有限值和 ordinary-DRAM endpoint 检查；D2 算子本身沿用已通过
full-payload development validation 的路径。后续 M1 已进一步完成 S1/D3 full-target
逐字节 parity；M0 不再承担独立 reference 资格验证。

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
- 同 revision 的 17-group group-64 old-harness diagnostic 与 48.238-second fair S0；
- makespan 53.497 秒，以及 rank-0 movement/publication 26.397/50.017 秒的瓶颈归因。

**实际遇到的最大风险。** 大 extent 使 Triton 的 int32 flattened index 跨越 \(2^{31}\)。
该问题已改为 64-bit pointer arithmetic，并通过 cold-cache 大组执行。这个回看说明：后续
formal E0、sensitivity 或 replication 若在大 extent 上异常，先检查 A/M0 从未覆盖的规模
边界，再判断是机制问题。

### C：两卡 M1 strong pipeline

**状态：完成开发态 strong baseline。**

**目标。** 在 B 的同一 development revision 上得到 S1，并知道普通 double buffering 能隐藏
多少已经测到的 52.8% DRAM staging/publication 开销。

**输入。** B 的固定 checkpoints、D1/D2 work snapshot、完整 144-GiB old store、相同
ordinary-DRAM private-target endpoint、S0 timer 和 group-64 capacity setting。

**已完成输出。**

- bounded input/output double buffers；
- CUDA streams/events 与 basic backpressure；
- fair group-64 S0/S1 同 revision、同 capacity 对比：48.238→32.703 秒；
- H2D/D2H、ordinary-copy、rank wait 和 bubbles；
- 完整 target ledger、exactly-once 和 full-payload parity。

**实际风险与回看。** S1 获得 1.475x，但 whole-group prefetch 仍有 6.20--6.79 秒输入
边界等待。这个结果没有否定 B 的瓶颈；它把下一层问题定位为 capacity group 内部的串行
staging。S1 保留为 action-oblivious strong baseline。

### D：D3 与跨层机制探索

**状态：route-aware isolation candidate 已完成 exact-stack paired 开发执行。**

**目标。** 在 M1 上找到比 S1 更好的、收益可解释的机制。

**输入。** C 的两卡 S0/S1 与 phase profile；第一轮机制发现当时不由 E0 阻塞，现已补齐
development E0。

**已完成输出。**

- input-only causal probe：31.096 秒，证明瓶颈转移到 output credit；
- historical v1 bidirectionally segmented I/O：28.885 秒；
- 当前 exact-stack route-major/selected pair：28.514442098/28.147194647 秒；
- selected order 相对同 stack route-major/S1/fair S0 为
  1.013047x/1.16186x/1.71379x；
- unchanged D1/D2 work revision、full byte parity 和 bounded-memory ledger；
- same-source profile、可扩展 stable-interleave search、嵌入 profile 和完整开发身份绑定；
- microbatch-16、compiled input-16、compiled output-4 是未改善观察点，不是一般性 rejection。

**当前最大风险。** 当前只有一个 exact-stack paired control/candidate，仍共享一个训练
seed、action mix，且 profile selection 与 evaluation 尚未形成正式 held-out boundary。
Development E0 已产生 positive crossover，但尚未独立调优或重复。相邻 identity-only
revision 的收益幅度更大，表明系统波动不可忽略。下一步应补 formal E0、正式重复和最小
held-out sensitivity，而不是继续堆叠 buffer 或弱化 S1。

### E：机制稳定后再正式化

**状态：当前下一阶段。**

**目标。** Development E0 已给出 positive crossover；现在用 independent tuning/repeats
和最小 sensitivity 判断当前机制能否变成论文设计，再正式化。

**届时再补。**

- 最终 D1/D2/D3 责任边界；
- 通用 schema/exporter/hash；
- atomic publication、abort、reclaim；
- exact crossover；
- capacity/action-mix sensitivity；
- 1/2/4-GPU；
- 第二 model edge 和正式 replication；
- frozen protocol。

**最大风险。** 当前 development crossover 无法在 independently tuned/repeated E0、
capacity/action sensitivity 或 replication 中保持。若资格验证失败，则回到 D 重新定位
机制，而不是用更多形式化工作包装失败候选。

## 6. 开发判断标准

当前不设置复杂 formal gates，只连续回答四个问题：

1. **能否运行？** 两卡 bounded-memory DRAM→GPU→DRAM 是否正确完成？
2. **瓶颈是什么？** 搬运、compute、collective、rank wait 还是 writeback？
3. **通用流水有多强？** S1 相对 S0 隐藏了多少，generic fixed-FIFO S2 又能隐藏多少？
4. **场景特异机制是否有额外收益？** proposed 是否稳定胜过 independently tuned S2，
   且能由 profile 解释？

当前四个问题在这一个 M1 point 上的回答分别是：能；双向 staging/writeback 加 route
resource imbalance；S1 为 1.475x，fixed-FIFO segmentation 已经贡献大部分后续收益；
selected D3 相对同 stack route-major control 只有 1.013047x。它已经超过“只胜
sequential”的最低开发门槛并在该单次 point 上回答了 E0 crossover，但尚未证明相对强
generic S2 的稳定 paper-level 增益；held-out generality、正式重复和论文证据仍未完成。

论文阶段仍需要严格可比性、correctness、physical capacity、independently tuned/repeated
all-exact、failure 和扩展实验；但这些不阻塞当前两卡机制发现。

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
results/system/evokv_design3_m1/qk_entity_h1536_s0_group64_noflush_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_e0_s0_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_e0_s1_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d1_only_s0_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d1_d2_s0_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_s1_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_v1_group64_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_v1_group64_mb16_seed0.json
configs/evokv_d3/m1/qk_entity_residency_plan_development_v2.json
configs/evokv_d3/m1/residency_profiles_development_v2/
results/system/evokv_design3_m1/qk_entity_h1536_d3_v3b_route_major_control_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_v3b_residency_selected_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_v3b_same_runner_validation_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_profile_compiled_i16_seed0.json
results/system/evokv_design3_m1/qk_entity_h1536_d3_profile_compiled_o4_seed0.json
```

这些文件记录从物理 materialization、顺序瓶颈、strong S1、input-only 归因、双向分段到
route-aware stable interleave 的完整开发链，并建立单次 development E0 crossover。它们仍
不证明正式 protocol、independent replication、generality 或 paper speedup。

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
复用；失败后的 partial target 不能自动覆盖。每次 candidate run 前应使用独立 target
identity 或显式 reset，不能误把残留 coverage 当成新 run。

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
8. 已补 fair group-64、17-group S0 和同 setting 的 M1 strong S1；
9. 已完成 input-only causal probe 和 generic fixed-FIFO bidirectionally segmented S2；
10. 已实现 route-specific 三段解耦、bounded-flow planner、stable interleave、plan/stack
    hash，并以 28.514442098/28.147194647 秒完成同一 exact-stack route-major/selected pair
    与 full byte parity；
11. 已补 grouped E0 与 owner-local D1-only contribution diagnostics；
12. 下一步先补 independently tuned formal E0 与 generic S2、正式重复与最小 held-out
    sensitivity，再决定
    是否冻结正式 D3。

这个顺序的目标是尽快得到可反复使用的两卡 benchmark，而不是先完成一套可能随后被设计
推翻的正式接口。
