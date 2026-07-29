# EvoKV Design 3 Foundation Benchmark and Exploration Plan

日期：2026-07-30

状态：**可执行计划；尚无 D3 实现、冻结协议或论文结果**。本文档把
[DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md) 中的问题合同落实为两卡
foundation benchmark、可信基线和机制探索路径。所有初期产物必须标记
`scientific_result=false` 和 `formal_design3=false`；只有候选机制改变 Pareto frontier 后，
才在 `docs/eval_protocol.md` 中另行冻结正式 D3 protocol。

当本文档与
[../08_core_insights_and_roadmap.md](../08_core_insights_and_roadmap.md) 或
[../eval_protocol.md](../eval_protocol.md) 冲突时，以后二者为准。

## 0. 本计划要得到什么

D3 不是重新运行一遍 D1 和 D2，也不是把 K/V 按显存容量机械分组。它要回答：

> 当一个固定模型更新产生的 committed source K/V 与 complete private target K/V
> 无法同时驻留 HBM 时，EvoKV 能否在不改变 D1 语义动作和 D2 分布式执行约束的前提下，
> 比一个调优后的通用 double-buffer pipeline 更有效地组织 DRAM↔HBM 搬运与 GPU 执行？

三个设计的边界固定为：

```text
D1 ActionPlan
  决定哪些记录 compiled migration，哪些 exact recomputation

D2 global distributed constraints
  决定 owner、embedding shard、operator、compatible pool、
  collective dependency、segmented layout 和 transaction

D3 ResidencyPlan
  决定 legal slice、capacity admission、micro-wave packing、
  prefetch / execute / writeback 次序、buffer credit 和 backpressure
```

容量分组只是 D3 的执行切片。一个 group 不能重新生成 D1 action，不能重算 D2 owner，不能
改变 exact/compiled membership，也不能单独发布 target epoch。完整 cohort 始终对应一个
全局 `ActionPlan`、一个 D2 constraint hash、多个 capacity slices 和一次最终 atomic commit。

本轮工作的结束状态不是“所有正式实验完成”，而是：

1. 得到一个真实、可重复、物理上超出两张 A40 HBM 的固定 benchmark；
2. 得到正确且计时边界完整的 sequential 与 strong double-buffer baselines；
3. 在完全相同的 D1/D2 工作上探索若干 D3 机制；
4. 选出一个能稳定胜过 strong double buffer、且收益可归因的候选，或诚实得到 no-go；
5. 将选中的机制、适用区间和正式扩展实验写成可冻结的 D3 protocol 草案。

## 1. 两层 foundation，而不是一个伪造的大 workload

### 1.1 F0：现有 KuaiRand H12 语义 canary

先复用已经冻结的真实 H12 工件：

- `theta1→theta2`、16L/H512、FP16 K/V；
- 682 records：548 compiled、46 scheduled exact、88 natural exact；
- immutable `ActionPlan` content hash
  `c4bc383d28f3558fdd11be8788799aaa6f66e80f778a4670f781eb9295f0027e`；
- W2 `strict_cow_lpt` owner map hash
  `0cfa3fb4553c02b21c89a7b73599ffe0c7fae2c8dc07c71de5b238360584bd86`。

它的 old K/V 为 28,383,969,280 B，complete target K/V 为 30,635,360,256 B，严格
copy-on-write 合计约 54.97 GiB。两卡 owner 后每 rank 约 27.48 GiB K/V，因此它实际上可以
放进两张 A40，**不能证明物理 out-of-core 问题**。

F0 只承担三个任务：

1. 关闭 D2→D3 constraint exporter/hash；
2. 验证 ordinary-DRAM source、capacity slices、segmented target 和全局事务；
3. 用例如 24 GiB/rank 的显式开发 cap 迫使多 wave，快速检查 bounded-memory executor。

所有 F0 capacity-cap 结果必须显式记录：

```text
capacity_emulation = true
physical_oversubscription = false
scientific_result = false
formal_design3 = false
```

不得复制 682 条记录、添加无用 embedding rows、制造 owner 倾斜或填充 dummy K/V 来把 F0
伪装成论文 benchmark。

### 1.2 F1 候选：真实 Tenrec QK 两卡物理 out-of-core benchmark

机制选择必须使用真实扩大后的 F1。QK 是当前首选候选：它是已接受的 ordered-exposure
dataset，有 39,615 个 raw history length ≥512 的用户候选池，但该数字尚未经过 vocabulary
filtering、完整 extent materialization 和 owner mapping，不能直接当作可用 F1 容量。只有
Stage B 的 capacity audit、真实 byte ledger 和 G1 全部通过后，QK 配置才被冻结为 F1。QK
只有用户内 ordinal order，论文和产物都不能将其描述为 calendar-time update。

F1 的数据选择不得查看 feedback label、模型输出、D1 action 或 D3 性能。先做一次
capacity audit，按以下顺序选择第一个能保留至少 8,192 个完整真实 history 的设置：

1. 512-token old window，向前滑动 32 tokens，target 仍为 512；
2. 若完整 history 不足，则使用 480-token old window 加 32-token append；
3. 若 catalog filtering 是瓶颈，扩大 base-fitted catalog，直到真实 cohort 达标；
4. 若仍不足，重新审计真实 context/record 组合，但不复制用户或 K/V。

用户按 stable hash 排序后形成一个 master cohort；2,048、4,096、8,192 是其嵌套前缀，而不是
三次重新选择。对于 16L/H512 FP16 K/V，512 tokens 的单版本 K/V 是 16 MiB；old 512 +
target 512 是 32 MiB/record。因此候选 footprint 约为：

| Records | Old + private target K/V | 预期作用 |
|---:|---:|---|
| 2,048 | 64 GiB | resident/crossover 邻近点 |
| 4,096 | 128 GiB | 两卡主探索点 |
| 8,192 | 256 GiB | deeper out-of-core 压力点 |

表中是候选字节目标，不是提前宣称的实际 `rho`。Stage B 必须从真实 extent 和冻结 owner map
重新计算每 rank 字节，并以实测 fixed HBM 与 safety margin 确定：

$$
\rho=\max_r\frac{\text{complete-work bytes}_r}{\text{usable HBM}_r}.
$$

主点默认选择最接近 `rho≈2` 的嵌套 cohort；相邻 `rho≈1` 和 `rho≈4` 只在主机制选定后用于
趋势检查。若 4,096 records 的真实 owner-local footprint 没有物理超出可用 HBM，则扩大到
8,192 或回到 dataset audit，不能用软件 cap 冒充正式容量点。

### 1.3 固定训练与版本边界

F1 只为 D3 mechanism discovery 构造一个真实 model-update edge，不做完整论文复现：

- 训练 seed 固定为一个；
- 模型保留 HSTU 的 defining structure，初始目标为 16L/H512；
- vocabulary 只在 base period 拟合；
- 训练使用 `training_sequences=all_chunks` 并记录 effective target counts；
- base 只使用 base interval targets，update 只使用当前 ordinal update interval targets；
- 训练一个 base model `theta0` 和一个短 update `theta1`；
- committed source 是 `theta0` 对 old window 生成的 exact source-version K/V，因此第一轮是
  one-hop foundation；它不能冒充 recursive lifecycle 中的 previous-actual migrated cache；
- 可保留第二个短 update `theta2` 作为候选机制选定后的稳定性检查，不进入第一轮搜索；
- calibration/program-fit users 与 benchmark cache users 必须分离；
- 推荐任务 labels 不参与 cohort selection、D1 routing 或 D3 scheduling。

若 16L/H512 的训练或 embedding footprint 在两卡上不可执行，可以调整 batch、梯度累积和
row-sharded embedding 实现；不得为了制造 D3 容量问题扩大不会被 workload 访问的冷参数。

### 1.4 固定 D1 与 D2 工作

在 `theta0→theta1` 上只运行一次 D1。主配置使用一个预先声明的约 20% exact budget，使
benchmark 同时具有 transfer-heavy compiled work 与 compute/collective-heavy exact work。
这是 D1 的上游预算配置，不是 D3 根据 I/O 重新选动作。冻结后必须报告实际：

- record action fraction；
- retained/exact/append token fraction；
- action-required source bytes；
- exact reasons、age/depth/lineage；
- plan/program/checkpoint hashes。

不得为了让 D3 更好看而在看到调度结果后修改 action。10%/30% exact 只能在 D3 候选冻结后
作为 sensitivity，且必须生成新的 D1 plan hash。

D2 随后在完整 cohort 上只运行一次 global planning，冻结：

- `strict_cow_lpt` logical owner；
- `item_id % world_size` row-sharded embedding rule；
- compiled/exact/append operator binding；
- `(suffix, retained)` compiled-bin membership；
- merged exact-pool membership；
- collective templates 与 empty-rank participation；
- segmented retained/suffix target layout；
- coverage、lineage 和一次全局 atomic publication。

D3 group 只是这些 global pools 的 deterministic slice。**不能逐 group 重跑 D1 或 D2
planning。**

## 2. Foundation 的物理与存储合同

### 2.1 固定硬件

第一轮固定：

- GPU0/GPU1，两张 A40；
- one process per GPU，NCCL；
- 两卡均位于 NV4/NUMA0 island；
- CPU source/target pages 与 worker first-touch 绑定 NUMA0；
- 每次运行记录 GPU UUID、driver/CUDA/PyTorch、topology 和空闲显存；
- 其他 GPU 不参与主计时。

后续复制到 GPU2/GPU3 只用于检查另一个等价 island；跨 island 和 1/4-GPU 扩展不阻塞最初
候选搜索。

### 2.2 Ordinary-DRAM endpoint

主边界固定为：

```text
ordinary pageable DRAM committed source
  → bounded pinned input
  → H2D
  → D2 GPU execution / collectives
  → D2H
  → bounded pinned output
  → ordinary pageable DRAM private target
  → validation
  → one atomic manifest publication
```

大 payload 使用 `/dev/shm/evokv_d3/<run-id>/` 作为临时 ordinary-DRAM backing，并通过
NUMA first-touch 物化页面；仓库只保存 compact manifest、byte ledger、configuration 和 hash。
不得把 100–300 GiB payload 写入 Git 或在空间紧张的 `/data` 下长期复制。`/dev/shm`
payload 是可重建的临时运行状态，不是正式 artifact archive。

source model forward 与 source cache generation 在 benchmark timer 外；benchmark 开始时
ordinary source pages 已存在。ordinary→pinned CPU copy、H2D、D2H、pinned→ordinary copy
全部在 timer 内。完整 source/target 不得预先 pin 住，只有 bounded slots 可以是 pinned
memory。

### 2.3 Action-required source contract

所有 mixed variants 逐字节共享同一个 source multiset：

- compiled 只读 valid retained old K/V；
- exact 只读 target raw history IDs，不读无用 old K/V；
- append 只读 suffix IDs；
- target 直接写 D2 segmented retained/suffix layout。

F1 在 ordinary DRAM 中为每条记录物化完整 committed old K/V；这才是超显存缓存的真实
source 和 abort 后仍可读的旧 epoch。执行时的物理 reader 只取 action 所需 ranges：exact
记录的旧 K/V 仍存在但不会被无意义地搬到 GPU。结果同时报告 `allocated_source_bytes` 与
`action_required_source_bytes_read`。选择性读取是所有 mixed baselines 的共同合同，不是
proposed D3 相对 double buffer 的贡献。

第一轮只支持当前 `compiled|exact` 加 append 的 action schema。Progressive repair 要等其
auxiliary BF16 hidden state 被显式版本化后再扩展，不能提前宣称支持。

### 2.4 Transaction 与 lifetime

每个 micro-wave 写入 private target extents。处理完一个 group 只能释放 GPU/pinned staging，
不能发布该 group，也不能使对应 source 不可读。仅当完整 cohort 满足 coverage、lineage、
shape、checksum 和 full-payload validation 后，所有 ranks 才共同生成 commit certificate 并
一次切换 target manifest。

任一 rank 在 source copy、H2D、compute、D2H、host drain 或 pre-commit 阶段失败，都必须：

1. 全局 abort；
2. 不暴露 partial target；
3. 保持旧 source manifest 可读；
4. 回收全部 private target 与 bounded staging。

## 3. 必须先导出的 D2→D3 工件

现有 W3 schedule 按 resident `extent_size` 切分并预载全部 GPU inputs，不能作为 D3 输入。
先新增一个 normalized、capacity-independent、content-hashed constraint artifact。它至少
包含：

- ActionPlan、checkpoint、program、history manifest hashes；
- world size、logical/physical rank、owner map 与 embedding shard rule；
- per-record action、reason、`R/S/F` extents 与 source identities；
- operator/program binding；
- canonical compatible pool membership 与稳定次序；
- collective dependency template、ordinal 和 empty participation；
- segmented target schema；
- global coverage/lineage/transaction contract；
- per-record logical/physical source/output/transient byte estimates。

它明确**不包含**：

- group size 或 HBM capacity cut；
- launch order；
- pinned slot 数量；
- prefetch distance；
- action-aware packing 决策。

Exporter 必须证明它与当前 D2 runtime 在 record coverage、owner、operator、pool membership、
lookup-token multiset、collective dependencies 和 target layout 上一致。Planner 只能扫描
上述 metadata；记录：

```text
kv_payload_bytes_read_during_planning = 0
```

若 exporter/parity 没有关闭，任何 scheduler 对比都不能宣称执行了相同 D2 工作。

## 4. Foundation baseline

### 4.1 R0：resident/no-I/O characterization

每个 capacity-safe chunk 在 timer 外 preload，只测 D2 GPU kernel/collective。它只给出
optimistic compute ceiling 和 phase profile，不是 D3 speedup denominator，也不能与完整
ordinary-DRAM path 直接形成主结论。

### 4.2 S0：可信 sequential capacity groups

S0 从全局 D2 constraints 生成稳定、byte-bounded、per-rank safe 的 legal slices。它只有一套
bounded input/output staging，严格执行：

```text
plan one global ResidencyPlan
for each micro-wave in global order:
    ordinary source → pinned input
    pinned input → HBM
    execute unchanged D2 operators and collectives
    HBM → pinned segmented output
    pinned output → ordinary private target
validate complete target
atomic commit once
```

下一 wave 必须等上一 wave 完整 drain，wave 之间没有 overlap。S0 不是弱 strawman；它需要：

- 只读取 action-required source；
- 使用全局 owner/pool/collective order；
- 计入 planning、所有 copies、validation、commit 和 reclaim；
- 输出 full-payload correctness 与实际 per-rank HBM peak；
- 不在 timer 外 concat segmented target；
- 不预扫描完整 K/V payload。

### 4.3 S1：strong action-oblivious double buffer

S1 与 S0 共享：

- ActionPlan hash；
- D2 constraint hash；
- source/target manifests；
- source-byte multiset；
- legal slices 与 capacity contract；
- transaction endpoint。

它只加入通用流水：

```text
CPU fill / H2D for i+1
GPU execute i
D2H / CPU drain for i-1
```

实现需要独立 H2D/compute/D2H streams、CUDA events、CPU workers、bounded input/output credits
和 backpressure。所有 ranks 仍执行同一 collective ordinal，不能各自独立推进。

S0 与 S1 必须共同调优，不允许只给 S1 或 proposed 方法更大的 group/buffer。最初在一个小的
预声明集合中搜索 slice cap 与 slot depth；选定后同时重跑 S0。若 S1 不能稳定胜过 S0，先修复
timer、同步、CPU staging 或 pipeline，而不是进入复杂 D3 scheduler。

### 4.4 E0：same-boundary all-exact

All-exact 使用同一 records、target model、ordinary-DRAM tier、target layout/dtype/durability、
两卡 topology、HBM budget、timer 和 atomic publication endpoint。它从 raw target histories
执行完整重计算，报告自己不同的 source bytes，不能假装与 mixed method 共享 ActionPlan。
它可以在相同容量合同内独立调优自己的 legal cuts 和 record ordering，主表使用最快的可信
all-exact，而不是故意沿用 mixed schedule。

R0、S0、S1、E0 共同构成 foundation。只跑一个按容量分组的 S0，不足以开始宣传 D3。

## 5. D3 候选机制

候选按由弱到强的顺序加入，便于知道收益来自哪里。所有变体只允许改变
`ResidencyPlan` hash。

### 5.1 C1：constraint-preserving global cuts

从 D2 已冻结的 canonical compatible pools 内选择 per-rank byte-safe cut points，减少：

- padding 和 transient waste；
- empty-rank imbalance；
- compatible pool 被容量边界过度切碎后产生的 launch/collective fragmentation。

C1 是 D2 与 out-of-core execution 之间必要的 capacity compiler。若它只带来小幅收益，应作为
实现基础，而不是独立 contribution。

### 5.2 C2：collective-slack complementary packing

这是第一优先级的创新候选。D1 固定动作天然产生不同资源 profile：

- compiled slices：retained old-K/V H2D 较重，owner-local affine compute 较轻；
- exact/append slices：raw IDs 较小，embedding collective 与 dense replay 较重。

C2 在不合并 incompatible operator pools 的前提下，把多个 legal pool slices 放入一个
micro-wave，并优化两个目标：

1. 平衡两 rank 到下一个 collective epoch 的到达时间；
2. 用 exact/append compute 或 collective 的空档搬运下一批 compiled old K/V，反之亦然。

“packing”只组合 residency 与发射次序，不改变 D2 的 kernel membership、owner、lookup tokens
或 collective semantics。这种 resource-complementary scheduling 是 EvoKV action mixture
特有的机会，也是它必须相对通用 double buffer 证明的核心。

### 5.3 C3：phase-aware credits and backpressure

在 C2 上按当前瓶颈动态发放 bounded credits：

- H2D 暴露时限制 writeback 争抢；
- collective/compute 空档允许提前 prefetch；
- output drain 落后时阻止 input 无限积压；
- 任一 rank 落后时保持全局 collective epoch 一致。

C3 不是无限自适应控制器。输入只允许使用可在运行时直接测得的 queue occupancy、event
completion、bytes 和 phase 状态，不读取推荐 labels，也不改变 D1 action。

### 5.4 次要探索项

在 C2/C3 已有正向结果后，才考虑：

- NUMA-local CPU worker 与 pinned-pool placement；
- micro-wave size 对 copy/collective/launch amortization 的影响；
- segmented writeback coalescing；
- topology-aware placement 扩展到 4 GPUs。

这些内容单独很难成为 D3；它们应服务于主机制，而不是掩盖主机制不胜 strong baseline。

## 6. 实验逻辑

### 6.0 主计时边界

计时开始前允许：

- process group、target model、program 和 row-sharded embedding 已加载；
- bounded pinned pools 和 CUDA streams 已建立；
- ActionPlan、D2 constraints 与 source manifest metadata 已在 host memory；
- ordinary source pages 已物化、prefault 且保持 pageable；
- 一次不计入结果的小 canary warmup。

coordinator primary timer 从 `ResidencyPlan` construction 前开始，覆盖 plan/local lowering、
ordinary→pinned、H2D、lookup/collective/compiled/exact/append、D2H、
pinned→ordinary、private-target writes、完整 validation、atomic publication 和 staging
reclaim。终点是完整 post-append target manifest 对外可见、旧 source 仍可读、临时
GPU/pinned references 已回收。主结果是 all-rank makespan；另报 execution-only 时间只能用于
归因，不能取代 plan-inclusive 结果。

### 6.1 第一轮：只回答机制是否存在

Stage B/G1 通过后，固定由真实 byte ledger 选出的 F1 主点、一个 seed、GPU0/1，按顺序运行：

1. R0 no-I/O characterization；
2. E0 same-boundary all-exact；
3. S0 sequential；
4. S1 tuned action-oblivious double buffer；
5. C1 compatibility-aware cuts；
6. C1+C2 complementary packing；
7. C1+C2+C3 full candidate。

这一轮不 sweep dataset、model、GPU count、seed 和 exact budget。其目标是找到机制，而不是
提前制造完整论文矩阵。

### 6.2 归因指标

主结果使用 coordinator monotonic wall-clock 的 all-rank makespan。CUDA events 只做 phase
归因。每个完整 run 至少记录：

- plan-inclusive 与 execution-only wall time、records/s、tokens/s；
- ordinary→pinned、H2D、D2H、pinned→ordinary 的 logical/physical bytes、时间与 exposed
  time；
- compiled、exact、append compute；
- embedding lookup/collective bytes、calls、time；
- per-rank arrival wait、collective wait、pipeline bubbles；
- padding、fragmentation、transient 和 repeated-write bytes；
- per-rank fixed/input/current-output/peak HBM；
- pinned pool peak、credit wait 和 CPU drain backlog；
- target validation、commit、abort 和 reclaim；
- ActionPlan、constraint、ResidencyPlan、source 和 target hashes。

必须验证完整 valid K/V、hidden/score/top-k、padding、segmented readback、coverage 与 lineage。
小 canary 可以 warm up；每次完整 timing 都要重新建立 private target，并在结束后验证旧 source
仍可读。

### 6.3 机制选定后的最小扩展

只有 full candidate 胜过 S1 后，才按以下顺序扩展：

1. `rho≈1/2/4` 容量点；
2. exact budget 10%/20%/30% sensitivity；
3. GPU2/GPU3 等价 island 重复；
4. 1/2/4-GPU paired evidence；
5. 第二个 model-update edge；
6. 正式多 seed/artifact replication。

正式矩阵必须在 `eval_protocol.md` 中冻结后再运行。F0、初始 F1 搜索和所有失败候选继续保留
为 development evidence，不能与正式 family 混合。

## 7. Go/no-go 与回退条件

### 7.1 Foundation gates

| Gate | 通过条件 | 失败时回退 |
|---|---|---|
| G0：上游不变 | Action/owner/operator/pool/lookup/layout hashes 在所有 mixed variants 相同 | 回到 exporter/parity，禁止调 scheduler |
| G1：真实容量 | 候选 F1 每 rank resident source+target+fixed state 超过物理 usable HBM，单 record 可 admission；通过后才称为 F1 主点 | 扩大真实 QK cohort 或重新审计 context；不造 dummy |
| G2：可信 S0 | full payload、bounded HBM、完整 timer、global commit/abort 全通过 | 修 source/transaction/executor |
| G3：强 S1 | tuned S1 稳定快于共同调优的 S0 | 检查同步、CPU copies、stream/event 和 slice size |

### 7.2 Design gates

| Gate | 通过条件 | 含义 |
|---|---|---|
| G4：超越通用流水 | full candidate 在主 `rho≈2` 点稳定胜过 tuned S1；以约 10% 作为值得继续扩展的目标，而非事后显著性门槛 | 否则只有 implementation path |
| G5：可归因 | exposed transfer、rank wait 或 bubbles 的下降能解释 wall-time 收益，且 source bytes/action/owner 未改变 | 否则结果不可解释 |
| G6：保留 D2 | D3 capacity cuts 没有通过 fragmentation/collective inflation 吞掉 D2 的 physical-sparsity 收益 | 否则回到 global cuts/constraint interface |
| G7：相对 exact 有区域 | 至少一个真实容量点优于 fastest same-boundary E0，并报告 exact-preferred crossover | 否则不能形成完整系统收益 |
| G8：事务正确 | full target 一次发布，任意中途错误均不暴露 partial epoch | 否则不能进入论文 |

如果 unavoidable compiled old-K/V DRAM input lower bound 已经慢于 same-boundary all-exact，
停止无限调 scheduler，回到 source representation 问题。如果 C2/C3 只胜 S0、不胜 S1，就把
当前 pipeline 保留为 implementation，不包装成第三个 design。

以下行为说明 D3 越界，实验应立即作废：

- 根据容量或通信重新选择 compiled/exact；
- 逐 group 重算 owner 或 D2 pools；
- 改变 lookup token multiset 或 operator；
- 将 incompatible pools 合并成新的 D2 kernel；
- 逐 group 发布或提前回收 committed source；
- 将完整 payload pre-read、pre-pin 或 pre-copy 隐藏在 timer 外。

## 8. 五个执行阶段

这些 stage 定义输入、结束状态和回退点，不预先锁死中间实现。每进入下一 stage，都要重新
检查上一 stage 的假设是否仍成立。

### Stage A：冻结 foundation contract，并用 H12 打通接口

**本质。** 把一个全局 D1/D2 计划转换成 D3 能消费、但尚未做 capacity scheduling 的稳定
接口。

**输入。** 现有 H12 ActionPlan、program/checkpoints、W2 owner、D2 integrated runtime 和
GPU0/1 topology。

**必须输出。**

- capacity-independent D2 constraint schema/exporter/hash；
- exporter↔runtime parity checker；
- source/target extent schema 与 action-required byte ledger；
- 简单 capacity-safe `ResidencyPlan` compiler；
- H12 F0 多-wave semantic canary。

**最大风险。** 现有 W3 extent schedule 已混入 capacity cut，或 segmented/transaction
接口仍要求把完整 target 拼回连续 GPU tensor。

**何时回头。** 若 Stage B 发现需要逐组重算 owner、collective order 无法全局展开，或 planner
必须读 K/V payload，说明 Stage A 的 constraint interface 错了，先修接口而不是给 Stage B
打补丁。

### Stage B：构造真实 QK model edge 与物理 out-of-core artifacts

**本质。** 用真实用户、真实历史和真实模型更新得到一个不会被“你只是软件限额”推翻的固定
benchmark。

**输入。** QK raw order、capacity audit、16L/H512 training pipeline、Stage A schemas。

**必须输出。**

- label-free master cohort 与 2,048/4,096/8,192 nested manifests；
- base-fitted vocabulary、`theta0→theta1` checkpoints 与训练记录；
- disjoint calibration users、compiled program/certificate；
- immutable D1 ActionPlan；
- W2 D2 constraint artifact；
- ordinary-DRAM committed source、empty private-target manifest；
- actual per-rank byte/capacity ledger 和一个物理 `rho≈2` 主点。

**最大风险。** QK filtering 后长 history 不够、D1 program/certificate 失败、action mixture
失去 compiled/exact 异质性，或生成 100+ GiB source 的时间/空间不可控。

**何时回头。**

- history 不够：回到无标签 dataset capacity audit，扩大 base vocabulary 或调整真实
  window，不复制记录；
- certificate 失败：回到 D1 calibration/model edge，不让 D3 掩盖 D1 失效；
- 两卡仍可 resident：扩大真实 nested cohort；
- source materialization 过慢：优化增量生成与 ordinary-DRAM writer，但不改语义工件。

### Stage C：关闭可信 baselines

**本质。** 在设计 scheduler 前，先知道真实 sequential、通用流水和 all-exact 分别有多强。

**输入。** Stage B 固定 benchmark 和同一个 D1/D2 contract。

**必须输出。**

- R0、S0、S1、E0；
- event-based asynchronous hot path；
- bounded pinned/HBM admission；
- full-payload segmented readback；
- global commit/abort/reclaim；
- phase/byte/wait metrics。

**最大风险。** target writeback 主导所有路径；全局 synchronize 破坏 overlap；per-rank 局部
切片导致 collective deadlock；S1 已经吃满 DRAM/PCIe，几乎没有调度空间。

**何时回头。** correctness/owner/hash 错误回 Stage A；真实 payload/capacity 不成立回 Stage B；
S1 不胜 S0 先修 Stage C 自身。只有 S0/S1 正确且可信，才进入 Stage D。

### Stage D：探索并选择 D3 机制

**本质。** 在同一 benchmark 上依次验证 C1、C2、C3，找到相对 strong generic pipeline 的
场景特异收益。

**输入。** Stage C 冻结的 workload、baselines、timer 和 metrics。

**必须输出。**

- 每个 candidate 的独立 `ResidencyPlan` 与 ablation；
- 相同 actions/source bytes/D2 constraints 的证明；
- wall time 与 transfer/collective/bubble 归因；
- 一个通过 G4–G8 的候选，或明确 no-go/pivot。

**最大风险。** C1 只是小工程优化，C2 没有可利用的资源互补，C3 的动态控制开销超过收益，
或 candidate 暗中改变了 D2 工作。

**何时回头。** collective fragmentation 吞掉收益回 Stage A 的 constraint/cut interface；
资源 profile 不够异质先检查 Stage B 的真实 D1 plan，而不是事后改 action；S1 measurement
不稳定回 Stage C。若 strong S1 已是上限，接受 no-go，不降低 baseline。

### Stage E：冻结候选并准备论文证据

**本质。** 把“一个开发点跑得快”变成边界清楚、可复现、能扩展的 D3 design。

**输入。** Stage D 选中的最小机制。

**必须输出。**

- mechanism spec 与伪代码；
- frozen artifact/protocol schema；
- `rho`、exact-budget、GPU-count、update-edge 的最小扩展矩阵；
- correctness/failure/crossover experiments；
- paper-ready motivation、design、evaluation claim boundaries。

**最大风险。** 收益只存在于一个 slice size、一个容量点或一个 update edge；扩展后 D2/D3
边界变得含糊。

**何时回头。** 只在归因失败时回 Stage D，语义/owner/transaction 失败则直接回 Stage A/B/C
对应层，不通过增加实验数量掩盖机制问题。

## 9. 首批应持久化的产物

建议新 artifact namespace 使用 `configs/evokv_d3/development/`，至少包括：

```text
foundation_spec.json
dataset_capacity_audit.json
cohort_manifest.json
model_edge_manifest.json
d1_action_plan.json
d2_d3_constraint_plan.json
source_manifest.json
source_byte_ledger.json
residency_plan_<variant>.json
run_<variant>.json
foundation_summary.json
```

结果文件记录：

- exact file/content hashes；
- `scientific_result=false`、`formal_design3=false`；
- D1/D2/D3 plan identities；
- hardware UUID/topology/NUMA；
- actual capacity and buffer contract；
- full command/environment；
- timer components 与 correctness；
- `payload_path_ephemeral=true`；
- 当前通过的 gate、失败原因和下一步。

大 K/V payload 不进入仓库。任何等待硬件、重新生成 source、失败候选、回退理由和下一条安全
命令都写入 `foundation_summary.json` 与后续 D3 status ledger，避免状态只留在对话中。

## 10. 紧接着执行什么

第一步不是立即训练 8,192-user QK，也不是开始写复杂 scheduler，而是：

1. 实现 D2 capacity-independent constraint exporter/hash；
2. 用 H12 验证 exporter parity、zero-payload planning 和 global capacity slicing；
3. 同时完成 QK ≥544-history 的 label-free capacity audit；
4. 审计通过后冻结 F1 cohort/model spec，再启动一个 base + 一个 short-update 训练；
5. 物化 ordinary-DRAM source 后关闭 S0/S1；
6. 只有 strong baseline 可信后，开始 C1→C2→C3 探索。

这条顺序保留了快速反馈，但机制选择只可发生在通过 G1 的真实两卡物理 out-of-core
benchmark 上；当前首选构造路线是 QK，而不是预先宣称 QK 配置已经成立。
