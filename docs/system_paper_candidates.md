# 从 cohort-tiered migration 到系统论文：候选方向与当前主线

> 状态：2026-07-26 第五轮收敛文档，属于设计探索，不是新的研究事实来源，也不改变
> `08_core_insights_and_roadmap.md` 与 `eval_protocol.md` 的优先级。
>
> 目标：回答“我们做的是一个什么系统”，并给出若干可独立成文的系统中心、共同组件、
> 评测要求和停止条件。本文中的系统名均为占位名。
>
> 最新决策：第 0 节是当前优先推进的收敛方案。后面的 A–G 仍保留为设计空间与失败备选，
> 但不再表示论文要同时实现这些系统，也不再以“模型发布系统”为当前定位。
> 进一步收敛：系统不预测某个 version “能不能 reuse”。每个 stale cohort 都必须发布一个
> 通过无标签 current-model semantic contract 的同步方案；version 用于编译、认证和合批。
> 当前数据不提供真实请求 arrival、热度、routing 或训练/推理共置关系，所以在线 lifecycle、
> foreground SLO 和自动冷热 placement 已从主线移除。第三层现在是显式 destination 下的
> out-of-core 批量更新与事务式版本发布。
> Update Coordinator 仅解析 job spec 并调用三层接口，是必要控制面胶水，不构成第四项
> 创新，也不恢复 admission、在线调度或自动 destination selection。

## 0. 先给结论

### 0.1 当前系统定义与边界

当前最可信、最聚焦的定位是：

> **一个面向持续更新生成式推荐的、编译式流式 KV cache 迁移系统。**

暂用名 **StreamKV**。它接收一个新模型版本、旧版本的持久化 cache/capsule、固定更新 cohort
和调用方指定的 destination；输出一组完整的 target-version K/V objects 与一个已提交
manifest。它不负责训练模型、不决定何时发布 checkpoint，也不负责分发或回滚 weights。
模型更新只是触发 KV 更新作业的外部事件。

论文的一句话 thesis 是：

> **模型更新会使一个 version cohort 的持久化 KV 整体失效。StreamKV 将这次全局失效转换成
> 一个可编译、可单次执行、可跨存储与 GPU 流式处理的 cohort-level state transformation。**

这不是“算法、算子、多 GPU/SSD”三个独立优化点的拼接。三者由同一个性质串起来：一个
version pair 共享同一迁移程序，而 cohort 内每条 cache record 在执行时彼此独立。因此程序
可以只编译一次，records 可以重排并组成大 batch，大 batch 又可以按顺序从存储读出、在多张
GPU 上变换并写回。

### 0.2 唯一的主流水线

```text
old-version cache/capsule records
              |
              v
 [1. Cohort Migration Compiler]
 version pair -> shared migration program
              |
              v
 [2. One-pass Capsule-to-KV Operator]
 one batch: old capsule -> final target-layout K/V
              |
              v
 [3. Cohort-streaming Migration Engine]
 bounded waves | transform | destination publish
              |
              v
committed target-version K/V manifest
```

它们的依赖关系是：

1. compiler 把模型差异变成 version-cohort 共享、执行期无状态的 migration program；
2. 无状态 program 使每条 cache 可独立变换，从而允许按版本和长度重排、合批；
3. 重排后的 cohort 可组织成大块顺序访问，算子可直接生成最终 K/V layout；
4. 顺序访问和直接写出使 read–transform–write 能够重叠，并可把 extents 分给多张 GPU；
5. 因而已有的 kernel-level 低成本才有机会转化成固定 destination 下的端到端更新时间、
   峰值工作内存和数据移动降低。

### 0.3 三个且仅有三个技术贡献

#### Contribution 1：Cohort Migration Compiler

它回答“一个 old→current cohort 到底计算什么”。

- 输入是 old/current model pair、少量 cohort calibration samples 和目标 fidelity/cost；
- compiler 先生成 projection、compiled affine、structural replay 与 exact 候选，再在与
  fit/调参用户隔离的无标签 probe 上验证 K/V、fresh-score 与 top-100 semantic recovery；
- 主 fast path 在大模型上拟合 attention-use-weighted full-affine
  `fresh - cheap` K/V residual，并把它折叠进旧 `Norm(x)` 到目标 K/V 的单个 affine
  projection；rank 只影响离线统计容量，编译后不再被误当成在线 cost knob；
- 输出不是一批迁移结果，而是该 version pair 共享的、可缓存的 migration program，包括
  投影参数、输入/输出表示、合法长度范围、证书和有序 fallback tier；
- compiled repair 与 full recompute 构成同一 compiler 接口下的基本 fidelity ladder；
  residual-delta replay 只有在目标 fidelity 上形成新的实测 Pareto 点时才进入 program，
  不分别包装成论文贡献；reuse 仅是零维护 baseline。

这里的关键系统价值是把 per-cache 的模型执行问题提升成一次编译、多 cache 执行的问题。
现有 cohort-tiered 算法就是系统的计算语义核心，而不是被 runtime 替换掉的一个插件。

#### Contribution 2：One-pass Capsule-to-KV Operator

它回答“一个 migration batch 如何接近硬件上限地执行”。

基线 PyTorch path 会 stack 逐层 `Norm(x)`、转 FP32、执行 batched projection、加
bias、生成 padding mask、回转原 dtype，再切分 K/V。当前 Triton path 已实现下述目标：

> 读取一次旧 capsule，在一个逻辑 pass 中完成 projection 与 epilogue，并直接写成 runtime
> 消费的最终 K/V layout，避免 materialize 完整的中间 `projected` tensor 和二次 layout
> conversion。

算子内部只围绕这一目标包含两个相连机制：

1. projection 与 bias、padding/length handling、cast 和 K/V split/pack 的 fused epilogue；
2. 对 ragged records 做长度分桶或 grouped launch，并直接写入目标连续 extent 或 paged slots。

FP16/BF16、Tensor Core、CUDA Graph、Triton/CUDA/cuBLASLt 等只是实现选择和 ablation，不单列
成贡献。是否采用 low-rank factorized execution，也由测得的算力—带宽平衡决定，不能预设为
额外 novelty。

#### Contribution 3：Destination-oriented Out-of-core Migration Engine

它回答“如何在有限工作内存下，把整个持久化 cohort 更新并完整发布到指定目的地”。

engine 只主打两个互相依赖的机制：

1. **Cohort-oriented organization**：保留 `user -> object` 逻辑索引，同时按
   `migration_anchor/version pair + representation + length bucket` 形成可顺序扫描的
   logical extents；
2. **Destination-specific Read–Transform–Publish pipeline**：以有界 wave 读取 capsule、
   在 GPU 上变换，并通过有界 publication queue 写入目标 backend；只有 complete、
   duplicate-free record coverage 才提交 target-version manifest。

多 GPU 不是第四个贡献，而是该流式引擎的自然扩展：以 extent 为工作单元分配给 GPU worker，
每个 worker 运行相同 migration program 和同一三阶段 pipeline。当前实现按预估 work bytes
做静态 LPT；只有后续 profile 证明队列失衡时才引入动态 rebalance。destination 决定执行
范式：HBM 由目的 GPU 直接写入，DRAM/本地文件/远端对象采用 host staging 与 manifest
commit。论文首先在现有四张 A40、HBM 与 host DRAM 上闭环；filesystem/remote 先实现接口，
只有真实硬件和独立 protocol 才形成 SSD 或网络结果。

### 0.4 为什么这三层合起来像一篇 system paper

单独看，affine migration 可能只是算法，fusion 可能只是 kernel engineering，异步 I/O
可能只是常规 pipeline。合起来的系统问题却是明确的：

> 如何把模型更新触发的海量、持久、版本相关的 KV invalidation，转成一种能一次生成、
> 大批执行并跨设备持续流动的 state transformation？

论文可以围绕三个逐层递进的假设组织：

- **H1，compile**：共享 version-cohort program 在目标质量下的执行成本显著低于 full
  recomputation；
- **H2，execute**：one-pass operator 将算法优势转化成 GPU time 和 memory-traffic 优势；
- **H3，stream**：有界 wave、destination-specific publication 与 manifest commit 将单卡
  kernel 优势转化成固定 endpoint 下的全 cohort completion time 和可控峰值内存。

最终 contribution list 应严格保持为：

1. 一个 cohort migration compiler；
2. 一个 one-pass capsule-to-KV operator；
3. 一个 destination-oriented out-of-core multi-GPU engine。

问题定义与 motivation 是这三项贡献的研究依据，不额外扩成一个系统层。Coordinator 只把
job specification、已发布 program、capsule shards、devices 与 destination 交给现有 runtime，
也不单列为 contribution。

### 0.5 每一层的同位替换项

下面的备选只在对应主机制不 work 时占据同一个“槽位”，不会与主方案同时扩张成更多
contributions。

| 位置 | 首选机制 | 何时判定不 work | 可同位替换，但不改变整体流水线 |
|---|---|---|---|
| Compiler | compiled affine `old Norm(x) -> target K/V` | 某些 cohort fidelity 不足，或 calibration 无法 amortize | compiler 对困难 layer/cohort 选择 residual-delta replay；小 cohort 直接 reuse/full；若误差集中在少数层，则生成 affine + selective replay 的单一混合 program |
| Operator | projection + fused epilogue + direct final-layout write | projection 完全支配耗时，full fusion 收益很小；或 paged random write 破坏合并访问 | 保留高效 library GEMM，只融合 epilogue；或者先写连续 staging extent，再异步 scatter 到 pages。两者仍是一个 capsule-to-KV operator |
| Engine organization | 物理或准物理的 cohort extents | 重排成本高于顺序更新收益 | 不移动主对象，只维护 version/length secondary index，按索引生成 sorted gather plan；流水线仍消费 cohort batch |
| Engine pipeline | 固定 read–transform–write 重叠 | 三阶段长期失衡，固定 batch/buffer 造成空转或拥塞 | 用同一 engine 内的自适应 batch sizing、双/三缓冲和 backpressure；若数据已 resident，则退化成多 GPU extent sharding，而不硬凑 SSD 故事 |

这组替换有两个约束。第一，接口不变：compiler 仍产生 program，operator 仍完成
capsule-to-KV，engine 仍执行 cohort stream。第二，每层最终只保留一个经 profiling 选出的
实现，论文中其他方案只作为 baseline、ablation 或负结果。

### 0.6 明确降级或排除的内容

为了避免再次变成组件罗列，下列内容不进入当前主 claim：

- checkpoint 何时发布、weights 分发、canary、事务式 rollout 与 rollback；
- 通用 cache coherence 协议和复杂版本图；
- 独立的 admission paper，以及任何声称能预测某版本 task-quality reuse 安全性的 classifier；
- 独立的多级存储或多 GPU scheduler：它们只是 streaming engine backend 和 extent assignment；
- 把 residual replay、full recompute、FP16、ragged batching、CUDA Graph 分别算成贡献；
- 在没有 profile 与硬件条件前承诺跨节点 RDMA 或 GPU-native SSD。

后文 A–G 提供了这些方向的详细设计空间。如果主流水线的某个 gate 失败，可以从中抽取一个
同位机制替换；在当前方案成立时，不应把它们全部装入一篇论文。

### 0.7 2026-07-26 实现状态

第一个三层 prototype vertical slice 已经落地：

- `MigrationCapsuleBatch` 把旧 `Norm(x)` 组织成连续的
  `[layer, batch, sequence, hidden]` payload，并携带 record IDs 与
  `migration_anchor_version`；
- `MigrationProgram` 将 source/target version pair 与现有 compiled affine adapter
  绑定，执行前检查 anchor、depth、width 和 device；
- `MigratedKVBatch` 同时保留旧 `migration_anchor_version` 与新的
  `served_kv_target`，防止把只生成 K/V 的近似迁移误当成可链式迁移的新 anchor；
- `CohortStreamingExecutor` 已提供 CPU reference path，以及在单张 CUDA GPU 上使用独立
  H2D、compute、D2H streams 的有界 inflight pipeline；
- `PackedMigrationOperator` 已实现 FP16 packed execution、`baddbmm` bias fusion 和
  in-place padding mask，并有逐阶段 CUDA-event profile；
- `CohortBatchPlan` 已实现 contiguous 与 length-bucketed logical extents，物理重排后仍可按
  record ID 恢复逻辑顺序；
- `MultiGPUCohortExecutor` 已实现基于 extent work bytes 的贪心分配与 1/2/4 GPU 并发执行。

独立的 `streamkv_system_prototype_v1` diagnostic 显示：packed operator 在 batch 8–128 上
相对 FP32 reference 为 1.95–2.90x，FP16 relative error 约 3.28e-4；length bucketing 将
payload 降至 0.81–0.83x；两 GPU 的完整 host→GPU→host 路径相对同步 FP32 reference 达到
3.56–4.45x。四 GPU 未稳定优于两 GPU，暴露出共享 host movement/NUMA/topology 瓶颈。
这些是 synthetic 初步系统结果，不替代真实数据上的算法证据。

真实 4+12 checkpoint/capsule 的第一条受限 vertical slice 也已跑通。固定 theta11/D16，
theta0→theta11 rank-32 repair 在 16 个 held-out diagnostic users 上为 0.061x exact resident
compute，并恢复 61.0% K/V fidelity；p8 为 0.550x/84.6%。真实 capsule 的 packed FP16 HBM
operator 相对 FP32 reference 为 3.97x，relative error 3.53e-4。完整 pinned-host→GPU→host
路径在 12 records 上从 1 GPU 的 363.5 records/s 扩展到 2/3 GPU 的 623.5/759.5 records/s；
相同 pinned-host 输入/输出位置的同步 full baseline 为 24.8 records/s。
这些均是单 seed、小用户数、三张空闲 GPU 的 implementation diagnostic，不是论文结果。

随后完成的两 GPU、全用户算法评测使用 40 fit、60 label-free probe 和 582 test users。
attention-use-weighted full-affine program 在 age 11/7/1 上只需约 0.064x exact，就恢复
88.6%/89.1%/93.6% K/V fidelity；相对原 rank-32 fast path，三档平均 MeanRank/AUC/NDCG@100
绝对偏差分别下降 44.5%/44.5%/47.3%。原 p8 约需 0.549x exact，却在三个 endpoint 上都没有
更高 K/V fidelity，因此不再自动作为主 middle tier。该程序是在同一 seed/test 上迭代得到的
exploratory candidate，下一步必须冻结复现，不能把当前 582 users 继续用于新的算法搜索。
详细记录在 `experiments/migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md`。

在此基础上，verified compiler 已实现“生成—认证—发布”闭环。用户角色进一步固定为
40 fit、60 早期 program selection、60 独立 label-free certificate 和 522 final test。
冻结 contract 同时要求 cache/score/top-100 三种误差至少恢复 70%，one-sided 90% recovery
lower bound 不低于 70%，one-sided 90% coverage lower bound 不低于 80%，primary cost
不超过 0.30x exact。age 11/7/1 都发布 compiled full affine，certificate cost 约
0.063x，最差 recovery lower bound 为 0.853/0.837/0.923；p8 只在 age 11/1 成为超预算
fallback，age 7 直接 fallback 到 exact。522-user final test 上，三个 endpoint 成本约
0.064x、K/V recovery 为 88.6%/89.1%/93.6%；最关键 age 11 的 MeanRank/AUC、NDCG@100、
Hit@100 signed recovery 分别为 98.8%、90.3%、88.9%。详细记录在
`experiments/migration/VERIFIED_COHORT_COMPILER_V1.md`。它仍是 adaptive seed-0 evidence，
下一步是冻结后在新 seed 或数据集复现，而不是继续搜索当前用户。

两 GPU 的真实 checkpoint 系统闭环现已完成。它不是把三层重新拆成更多贡献，而是让主流水线
首次在同一 boundary 下闭合：

- compiler 层直接读取 theta0/theta4/theta10 的 verified full-affine programs，不在系统用户上
  重新选算法，也没有 version reuse admission；
- operator 层用 Triton 融合 affine、bias、length mask 与 K/V split，并直接写两个连续的
  K/V 目标 tensor；在 length 2,047 的真实 capsule 上相对 packed FP16 快 1.19x；
- engine 层支持多个 source cohort 共存、三个 program 每卡常驻、pinned-host
  H2D/compute/D2H 三流、持久化目标 extent 和两卡 LPT；最终 64-user trace 达到
  903.7 records/s 和 1.951x 的 1→2 GPU scaling；
- 强基线也已补齐：current model 在两卡上复制，raw histories 可跨旧 version 合批，使用独立
  调优的 BF16/FP32 三流 full recompute 并发布同样的 FP16 host K/V。最终 compiled path
  相对更快的 BF16 baseline 为 11.22x。

这轮也得到两个有用的负/边界结果。第一，persistent destination 在 warm steady state
只有 0.9998x，不应包装成吞吐优化；它的价值是固定地址与消除首次 pinned registration
抖动。第二，在细粒度 batch-1 trace 上，LPT 只比 round robin 快 1.011x；调度器的主价值是
有界负载与处理不规则 extent，而不是在当前均衡 trace 上制造大收益。额外 `Norm(x)` capsule
仍是逻辑 FP16 K/V 的 50% 状态开销，这轮明确接受，不把压缩硬塞进当前贡献。

详细协议和结果在 `experiments/system/TWO_GPU_MIGRATION_SYSTEM_V2.md`。该结果闭合了受控
host boundary，但没有覆盖全 cohort、有限工作内存、destination backend 或 durable commit。

后续已经按“必须来自迁移场景”的标准实现了 cohort-jagged/page-compacted operator，而不是
继续叠加通用 tile 调参。同一 version program 下的有效 token 可以跨用户拼接；最终实现还把
记录拆成 256-token K/V pages，并压成最多 2,048-token tiles，直接发布到 pinned host 或
HBM extent。全部 1,609,760,768 个有效 FP16 K/V 元素与 dense fused 完全一致。

但这个候选没有通过独立 operator-contribution gate。64-user final trace 上，page packing
把 batch 64→50、padding 0.509%→0，却只让两卡 host boundary 快 1.019x；在 direct-HBM
boundary 上反而是 one-record path 的 0.984x。resident fusion 仍有 1.182x，但 HBM
end-to-end fused/packed 只有 1.005x。32-user search 上约 5% 的 compaction 收益没有泛化。
因此 page metadata 与 direct publication 可以保留为 engine 机制，但不能把
“cohort mega-batching/去 padding”写成主要 novelty。详细记录在
`experiments/system/COHORT_JAGGED_SYSTEM_V3.md`。

冻结 v3 final trace 的四卡 follow-up 也已完成，不重新搜索 layout 或 program。host-staged
compiled migration、direct-HBM compiled migration 和同 host boundary 的 BF16 exact
分别达到 3.275x/3.331x/3.592x 的 1→4 scaling；四卡效率为
81.9%/83.3%/89.8%，LPT assigned-work imbalance 不超过 0.30%。compiled migration 在
四卡 host boundary 仍比 BF16 exact 快 10.39x。这个结果证明当前 extent sharding 能从
单卡扩展到四卡，也显示 migration 在任务变短后比 arithmetic-heavy exact 更早受到固定
调度/传输开销影响。它不替代 full-cohort destination-v4 gate。详细记录在
`experiments/system/FOUR_GPU_SCALING_V1.md`。

这也收紧了主线：当前 compiler 仍然强，host-backed end-to-end 优势仍然成立；operator 是
可信的 enabling implementation，而不是已经成立的独立大贡献。

最新 destination-v4 vertical slice 已实现统一的 HBM、DRAM、POSIX filesystem 与 remote
object contract。HBM 走 direct-device publication；其他 backend 走有界 host-staged wave
和异步 publication queue。每个 job 预声明完整 record set，只有所有 extent 都完成后才提交
`streamkv_destination_manifest_v1`；中断作业不暴露半成品版本。filesystem 与 remote 当前
只是正确性实现，不是 SSD/网络性能结果。代码和边界记录在
`experiments/system/DESTINATION_OUT_OF_CORE_V4.md`。

薄层 `scripts/run_streamkv_update_coordinator.py` 已把 job spec、program/capsule artifacts、
devices、destination 与 v4 engine 串起来；默认只输出 plan。它不生成 program、不物化
capsule、不选择 backend，也没有恢复/重试机制。当前 host-staged engine 只对 transform 输出
与 publication backlog 做有界 wave，完整 CPU source capsules 仍由调用方准备；HBM direct
路径保留完整 target K/V。这些是后续 full-cohort gate 的实现边界，不是新的贡献。

> **历史附录边界：** 第 1–11 节冻结了早期候选与转向原因。其中关于 request trace、
> admission、foreground SLO、模型发布和训练/推理共置的“缺口”不再是当前 StreamKV 的需求，
> 也不能覆盖第 0 节与第 12–16 节的主线。保留它们仅为了记录同位替换和 pivot 空间。

## 1. 为什么“再加几个系统组件”还不够

本节以下到第 11 节保留较宽的历史候选空间。其中依赖请求 arrival、热度、routing、前台
SLO 或训练/推理共置的 A/C/F 等方案不再是当前主线，因为现有数据不能验证这些前提。当前
论文闭环以第 0 节和第 12–16 节的确定性 destination update job 为准。

系统论文通常不是算法外面包一层 scheduler，而是同时具备以下闭环：

1. **新的系统对象或抽象。** 这里可以是“带模型 lineage 的持久派生状态”和
   “bounded semantic staleness”，而不只是 tensor。
2. **真实的资源冲突。** 前台增量推理、后台 migration、adapter calibration、状态读写会争用
   GPU、PCIe/NVLink、CPU/SSD 和容量。
3. **跨时间的控制问题。** 每次模型更新都会产生新 target，旧迁移可能尚未结束，不能把每个
   theta-0→theta-11 实验视为互不相干的一次 kernel 调用。
4. **明确的系统 contract。** 系统必须说明一个请求究竟消费了哪个模型、哪个 cache
   lineage、哪一级近似，以及迁移失败时会发生什么。
5. **端到端实现与 workload。** 需要自然 mixed-version trace、请求 arrival、模型版本事件、
   分层状态容量和并发执行，而不只是 GPU-resident batch。
6. **SLO 下的整体收益。** 要报告 goodput、P99、更新完成时间、质量 deficit、数据移动和成本，
   而不只是某个算子的相对 kernel time。

因此，下面几件事单独做都不够构成当前项目的系统主张：

- 把 cached state 从 GPU 放到 CPU/SSD；
- 把同一个 migration batch 切到四张 GPU；
- 把 projection 写成一个更快的 CUDA kernel；
- 再搜索一次 suffix、interval 或 recent-token rectangle；
- 只展示 adapter fitting 能在大 cohort 上 amortize。

它们可以是重要组件或优化，但必须服务于“可编译、可单次执行、可流式运行的 cohort
状态转移”这一中心。

## 2. 当前已经有的基础，以及系统缺口

### 2.1 已经具备的核心资产

| 已有资产 | 对系统论文的价值 |
|---|---|
| 明确的版本失效问题 | 区分于普通 token append、固定模型 prefix cache 和 cache compression |
| fixed-endpoint cache-version matrix | 说明 age 只是粗排序，不是稳定的质量控制信号 |
| version-cohort calibration | 把一次适配成本从 per-user 热路径移到 version pair |
| compiled affine projection | 在线仍是一组规则、GPU 友好的 layer-batched projection |
| residual structural tier 与 exact endpoint | 为不同 fidelity SLO 提供离散 action library |
| 27 个 replication seeds | 证明算子的 kernel cost 与 K/V fidelity 跨容量较稳定 |
| full endpoint 并非处处有正 task gain | 直接产生 admission 这一独立系统问题 |
| calibration break-even 估计 | 已知 cohort size 是系统是否成立的必要变量 |

当前 27-seed 结果的正确系统含义是：当某个 cohort 确实存在维护价值时，compiled tier 往往
能以约 `0.121x` full kernel cost 恢复约 `0.587` 的 K/V gap；但 6/9 的严格 task gate
说明“是否维护”和“如何维护”必须分开。

### 2.2 当前实现仍是离线、单 cohort、resident-GPU primitive

当前代码和协议尚未提供：

- 持久化 cache object、version directory 或 page/chunk allocator；
- 自然请求触发的 cache version 分布；
- 并发 serving 与后台 migration；
- host/SSD 读写、DMA、network 或 allocator 时间；
- 多 GPU placement、work stealing 或 topology-aware scheduling；
- migration-program registry、失效和垃圾回收；
- cache 状态机、幂等 migration 与原子 pointer switch；
- 版本更新快于迁移时的 cancellation、retargeting 和 write coalescing；
- end-to-end throughput、goodput、P99 和 update completion time。

### 2.3 一个必须正面解决的新问题：migration 不是天然可链式执行的

当前 compiled operator 的输入是旧版本逐层 `Norm(x)`，输出是目标版本的近似 K/V。它不会
同时生成目标版本的逐层 hidden 或 `Norm(x)`。因此不能把一次 approximate migration 的输出
简单标记成“完整 theta-t 状态”，再无条件作为 theta-t→theta-(t+1) 的输入。

一个长期运行系统至少要区分两个版本字段：

- `served_kv_target`：当前 K/V 是为哪个 serving model 生成或近似生成的；
- `migration_anchor`：系统仍持有哪个版本的 normalized/hidden capsule，可用于未来迁移。

这带来一组真正的系统决策：

- 继续从老 anchor 直接编译到最新版本；
- 在某个时刻做 exact refresh，更新 anchor；
- 为少数 anchor 保留 adapter，合并或回收小 cohort；
- 在新版本到达时取消通往中间版本的写入，直接 retarget 到最新版本；
- 防止多轮近似造成不可控的误差累积和状态语义混乱。

这个“serving version 与 migration anchor 分离”的问题，目前比再优化一个 layer kernel 更有
潜力成为本项目特有的系统贡献。

## 3. 更广的候选方向总览（探索存档）

本节及后续 A–G 早于第 0 节的收敛决策。它们用于记录相邻问题和失败后的转向空间，不表示
当前论文仍以 A+C 为中心，也不表示需要实现一个模型发布系统。

| 候选 | 一句话系统定义 | 当前算法是否居中 | 系统广度 | 新颖性风险 | 建议 |
|---|---|---:|---:|---:|---|
| A. CohortKV | 持续训练 GR 的版本感知 cache coherence 系统 | 很高 | 很高 | 中 | 早期首选，现为备选 |
| B. RollKV | 模型与派生 cache 的事务式发布/回滚系统 | 高 | 高 | 中 | 并入 A，或强化后独立 |
| C. MigraSched | mixed-version、SLO-aware 后台迁移调度系统 | 很高 | 很高 | 中 | **与 A 组合** |
| D. TierState | 面向 migration capsule 的多表示分层状态库 | 中 | 高 | 高 | 作为组件，不宜单独主打 |
| E. VersionGraph | 长版本链上的 anchor、adapter 与迁移路径管理系统 | 高 | 中高 | 中 | 高风险候选 |
| F. FreshTrain | 联合决定训练 checkpoint 发布与 cache 维护的系统 | 高 | 很高 | 中低 | 高回报、高工作量 |
| G. CohortExecutor | 异构 cohort 的 grouped projection 与多 GPU 执行器 | 高 | 中 | 中高 | 作为性能组件 |

“新颖性风险”越高，表示越容易被已有系统覆盖，而不是表示研究价值越高。

## 4. 候选 A：CohortKV——版本感知的派生状态一致性系统

### 4.1 系统定义

> CohortKV 是一个面向持续训练生成式推荐的 cache coherence 系统。它不要求模型更新后所有
> 用户状态立刻 byte-identical，而是在质量、延迟和资源 SLO 下，为每个 version cohort
> 提供可测、可追踪、最终可收敛的 semantic consistency。

传统 cache coherence 关心副本是否等于某个内存值；这里的状态是模型计算的派生结果，
近似 migration 还形成多个 fidelity tier。因此更合适的 contract 是：

- 每个 cache 必须具有明确的 model lineage 和 migration anchor；
- 系统不得把错误 adapter 或错误模型的 cache 静默混用；
- 请求允许消费经过 admission 的 reuse/approximate 状态；
- 系统记录其 fidelity class、目标版本和 freshness deadline；
- 在资源允许或 deadline 到达时，状态收敛到指定 tier 或 exact current；
- rollback 能回到一个已知健康的 model-cache 组合。

### 4.2 核心 cache object

建议把系统对象定义为：

```text
CacheObject {
  user_or_prefix_id
  history_epoch
  kv_target_version
  migration_anchor_version
  representation: KV | NORM_CAPSULE | HIDDEN_CHECKPOINT | RAW_HISTORY
  fidelity_class
  physical_location
  adapter_id
  state: READY | STALE | PLANNED | MIGRATING | COMMITTED | RETIRED
}
```

cohort key 不应只有 `(old_version, current_version)`，而应至少包含：

```text
(migration_anchor, target_version, representation, length_bucket, location_tier)
```

这样 planner 才能同时推断 action 可行性、I/O 字节数、batch shape 和 adapter。

### 4.3 系统组件

1. **Update Observer**
   - 接收 streaming trainer 的新 checkpoint；
   - 记录 model lineage、更新时间、参数/模块更新摘要；
   - 枚举当前仍存活的 migration anchors。

2. **Cohort Profiler**
   - 从每个 active anchor 抽取少量、互斥的 fit/probe cache；
   - 测量 reuse、fresh、compiled 和 structural tier 的 fidelity、成本与必要 task signal；
   - 输出带置信度的 cohort profile，而不是 per-user predictor。

3. **Admission Controller**
   - 先判断 `reuse / maintain / exact-only / retire`；
   - 对 full maintenance endpoint 近零或方向不稳的 cohort 不启动大规模 migration；
   - 保持 admission signal 与 tier-selection signal 分离。

4. **Migration Compiler**
   - 复用现有低秩 residual fit 和 affine folding；
   - 生成带 source/target/version hash 的不可混用 adapter；
   - 注册 compiled projection、residual replay 和 full action 的 profile。

5. **Versioned State Directory**
   - 管理每个 user cache 的 lineage、表示、位置、状态和原子指针；
   - 支持幂等 migration、失败重试、取消和垃圾回收；
   - 防止 serving 看到一半写入的新 K/V。

6. **Global Migration Scheduler**
   - 在 cohort 之间分配 GPU、I/O 和 deadline；
   - 决定 proactive、lazy-on-read、drop-on-read 或 exact refresh；
   - 新 target 到达时跳过无意义的中间写入。

7. **Local Executors**
   - 按 version cohort、length bucket 和 representation 聚合；
   - 执行 read → migrate/recompute → write → atomic commit；
   - 与前台 serving 使用可控的 stream、优先级和显存预算。

8. **Rollout and Health Manager**
   - 执行 model activation、canary、drain、commit 与 rollback；
   - 监测 latency、quality proxy、fidelity drift 和 backlog；
   - rollback 时恢复一个合法 model-cache 组合，而不只恢复 weights。

### 4.4 一次模型更新的完整控制流

```text
stream trainer
      |
      v
new checkpoint ----> Update Observer
                           |
                    enumerate anchors
                           |
                           v
                  Cohort Profiler
                    /           \
             admission        tier profiles
                    \           /
                     v         v
                 Global Planner ----> Adapter Registry
                       |
             proactive / lazy tasks
                       |
       +---------------+----------------+
       v                                v
Versioned State Store <----> Local Executors <----> Serving path
       |                                |
       +-------- atomic metadata -------+
```

### 4.5 关键新机制

#### A1. Bounded semantic coherence

不是用一个 TTL 表示“新/旧”，而是给每个 cohort 记录：

- 当前 reuse deficit；
- 被 admission 的最低 fidelity class；
- migration deadline；
- 已完成比例与暴露流量；
- rollback anchor。

系统 SLO 可以写成“95% 请求在发布后 D 分钟内消费至少 tier-k 的状态，且 P99 不超过 L”，
而不是不现实地要求发布瞬间全量重算。

#### A2. Anchor-aware lifecycle

把 approximate serving K/V 与可迁移 capsule 分开管理。planner 决定何时：

- 保留老 anchor 并 direct-to-latest；
- exact refresh 以重建新 anchor；
- 将小 cohort 合并到一个 anchor family；
- 回收无法 amortize 的 adapter；
- 限制 anchor age 或 direct migration distance。

#### A3. Moving-target coalescing

如果 theta-12 到达时 theta-11 migration 尚未结束：

- 未开始任务直接 retarget 到 theta-12；
- 正在读但未写的任务可取消；
- 已产生 theta-11 K/V、但 anchor 仍是 theta-8 的任务无需再读写一次，可从 theta-8
  capsule 直接生成 theta-12；
- 只有 exact refresh 成功后才更新 anchor。

这类似 write coalescing，但正确性单位是 model lineage。

#### A4. Hybrid eager/lazy convergence

- 高频访问 cohort 或临近 deadline 的状态 proactive migrate；
- 冷用户保留 durable capsule，首次访问时 lazy migrate；
- 极小 cohort 若低于 calibration break-even，可直接 lazy full；
- endpoint 不值得维护的 cohort 继续 reuse，直到自然 refresh 或 eviction。

这里可以使用用户访问概率做资源调度，但不能重新把用户级 K/V drift 当成质量预测器。

### 4.6 为什么现有算法在这个系统中不可替代

如果没有 compiled cohort operator，系统只有两个动作：

- reuse：零后台成本，但质量 deficit 不可控；
- full：每次版本发布造成全量 history rewrite。

现有算法提供了低成本、共享编译、可 cohort-batch 的中间动作，使 coherence controller
第一次有机会在后台预算内推进状态收敛。反过来，没有 admission、anchor lifecycle 和
mixed-version scheduler，当前算法仍只是一组离线 kernel 点。二者天然形成系统—算法闭环。

### 4.7 最接近工作与边界

- [DroidSpeak](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_liu-yuhan_prepub.pdf)
  已经研究不同 fine-tuned model 之间的 KV reuse、关键层重算以及计算—传输流水化，是最接近
  的跨模型 KV 工作。我们的区别必须落在 persistent per-user state、持续发布产生的全局
  invalidation、mixed anchor 生命周期、cohort admission 和 moving-target convergence，而
  不能只声称“跨版本部分重算”。
- [Ekko](https://www.usenix.org/conference/osdi22/presentation/sima) 与
  [QuickUpdate](https://www.usenix.org/conference/nsdi24/presentation/matam) 管理推荐模型
  参数的低延迟发布、优先级、relaxed consistency 或 rollback。我们的对象是 weights 发布后
  仍遗留的大规模派生用户状态；最强叙事是补上它们没有处理的 derived-state transition。
- [MTServe](https://arxiv.org/pdf/2604.22881) 这一 2026 年 preprint 已经直接覆盖生成式
  推荐的 GPU/CPU hierarchical KV、Page-Chunk layout、异步搬运与 LRU。因此 A 的主张不能退化成
  “给 HSTU 建一个分层 cache”。

### 4.8 主要风险

- admission 的 label-free proxy 可能无法稳定判断 task endpoint；
- 当前 public workload 的用户量可能不足以展示 rollout backlog；
- anchor state 的额外容量可能吞掉 compute savings；
- 若自然流量中 active version 数很少，复杂 lifecycle 的价值会下降；
- 若系统最终退化为 lazy full on read，compiled tier 不再是核心。

## 5. 候选 B：RollKV——事务式模型与 cache 联合发布

### 5.1 系统定义

> RollKV 把一次模型发布视为 weights 与持久化派生 cache 的联合状态转移，为近似迁移、
> 渐进 activation、失败恢复和 rollback 提供明确协议。

这个候选更强调 consistency、availability 与 failure handling，弱化全局最优调度。

### 5.2 状态机

一次 rollout 可以定义为：

1. **PREPARE**
   - 加载新模型；
   - 枚举 active anchors；
   - fit/probe、admission、compile adapter；
   - 预留必要 GPU/host capacity。

2. **CANARY**
   - 小流量同时运行 fresh、新 tier 和旧 reuse；
   - 验证 parity、latency 和 health；
   - adapter 尚未通过时不能进入 active registry。

3. **ACTIVATE**
   - 新请求使用 theta-t model；
   - 每个旧 cache 根据 directory 中的 cohort plan 走 reuse、lazy migration 或 full；
   - 所有结果携带 lineage。

4. **DRAIN**
   - 后台推进被 admission 的 cohort；
   - 旧版本状态仍可读，但写入新对象并在完成后原子切换。

5. **COMMIT**
   - 满足 coverage/deadline 后回收旧 served K/V；
   - 只保留 planner 选择的 migration anchors 和 rollback snapshot。

6. **ABORT/ROLLBACK**
   - 停止新 migration；
   - 将请求重定向到已知健康 model-cache pair；
   - 清理未 commit 对象，保留幂等日志。

### 5.3 可形成系统贡献的点

- model version 与 cache lineage 的联合 version vector；
- approximate cache 的原子发布协议；
- migration task 的 exactly-once effect，而不强求 exactly-once execution；
- 新模型 rollout 与前一个 rollout 重叠时的 supersede 规则；
- adapter hash、source capsule hash 和 target checkpoint hash 的强校验；
- rollback 时避免把“新 cache + 旧 model”或“旧 cache + 错 adapter”组成非法状态；
- failure injection 下的 bounded downtime 和 zero silent corruption。

### 5.4 何时可以独立成文

如果能展示以下现象，B 可以成为独立系统中心：

- 大规模 rollout 中失败、重启、慢节点和版本交叠是常态；
- naive eager/lazy update 会出现明显的 partial-state corruption、长时间双倍容量或停机；
- 新协议在不牺牲 P99 的前提下显著缩短 activation/rollback；
- consistency contract 能推广到 embedding cache、feature-derived state 等其他模型状态。

否则，B 更适合作为候选 A 必须具备的一章，而不是单独论文。

## 6. 候选 C：MigraSched——mixed-version、SLO-aware 后台迁移调度

### 6.1 系统定义

> MigraSched 是一个把 version-cohort migration 当作持续后台 workload 的 GPU 集群调度器；
> 它联合选择 admission、fidelity tier、eager/lazy 时机和执行位置，在前台 serving SLO 下
> 最小化累计质量 deficit 与状态移动。

### 6.2 为什么它不是普通 batch scheduler

每个 cohort 都具有不同的：

- cache 数量与请求到达率；
- anchor/target version；
- history length 和物理位置；
- calibration fixed cost；
- action-specific compute、read bytes、write bytes；
- fidelity recovery 与可辨识的 task endpoint；
- deadline 和被后续模型 supersede 的概率。

单纯按 FIFO、age 或 cache 数量排序都会忽略这些差异。

### 6.3 一个可实现的优化目标

对 cohort \(c\) 和 action \(a\)，定义：

- \(\lambda_c\)：未来访问率；
- \(N_c\)：未完成 cache 数；
- \(q_{c,a}\)：probe 估计的 serving deficit；
- \(g_{c,a}\)：GPU 时间；
- \(r_{c,a}, w_{c,a}\)：读写字节；
- \(f_c\)：calibration/compile fixed cost；
- \(d_c\)：freshness deadline。

系统不是只解一次静态 knapsack，而是周期性最小化：

\[
\sum_c \lambda_c \cdot q_{c,a_c}
+ \alpha \sum_c g_{c,a_c}
+ \beta \sum_c (r_{c,a_c}+w_{c,a_c})
+ \gamma \sum_c \text{deadline\_violation}_c
\]

并满足前台 P99、GPU memory、PCIe/SSD bandwidth 和 migration worker 数量约束。

论文不一定需要复杂求解器。更重要的是用 workload observation 得到一个稳定、可解释、
在线开销低的 policy，并与 oracle、age-only 和 periodic baseline 比较。

### 6.4 调度层次

#### 全局层

- 为 GPU/node 分配 cohort；
- 在数据 locality 与 batch aggregation 之间权衡；
- 控制 serving/migration 的总体资源份额；
- 新模型到达时进行 retarget 与 backlog coalescing。

#### 本地层

- 按 adapter、length bucket 和 representation 组成 microbatch；
- 使用 deadline-aware batching，避免等待大 batch 造成请求 miss；
- 在 projection、structural replay、full 与 I/O stream 之间 overlap；
- 允许短任务抢占或在 cache-object boundary 让出 GPU；
- 只有在预估节省超过跨 GPU 搬运时才 work steal。

### 6.5 多 GPU 设计空间

四张 A40 足以先做单机多 GPU 原型：

- **replicated-current-model + user-state sharding**：每张 GPU 有当前 weights，按 state
  locality 处理用户；
- **dedicated migration GPU**：容易实现，但低流量时浪费且可能成为单点瓶颈；
- **elastic worker pool**：根据前台 queue depth 动态借用 GPU stream 或整卡；
- **cohort-affine placement**：adapter 很小，可复制；大状态不轻易跨 GPU；
- **grouped heterogeneous projection**：小 cohort 使用 grouped GEMM，避免版本碎片导致
  utilization 下降；
- **topology-aware stealing**：优先 NVLink/同 NUMA node，再考虑 PCIe/host bounce。

多 GPU 的论文价值来自“状态 locality、版本 batchability 与 serving interference 的三方
权衡”，而不是线性扩展曲线本身。

### 6.6 必须比较的 baseline

- 永久 reuse；
- 每次发布立即全量 full；
- next-access lazy full；
- 固定周期 full；
- age threshold；
- 只有 cohort-tier selection、无全局 scheduler；
- 只有 popularity/LRU、无 version signal；
- equal-cost 与 equal-quality 下的最优离线 oracle。

### 6.7 主要风险

- public trace 没有真实在线 QPS，需要明确区分真实 event order 与合成 arrival rate；
- 当前模型较小，前台 interference 可能不够强；
- scheduler 若主要收益来自简单 popularity，版本方法会被边缘化；
- cohort fragmentation 可能使 compiled projection 的 kernel 优势消失；
- 若 calibration fixed cost 占主导，调度问题会退化为 cohort-size threshold。

## 7. 候选 D：TierState——面向 migration capsule 的多表示分层状态库

### 7.1 系统定义

> TierState 不是只存 K/V 的 cache，而是同时管理 serving K/V、migration capsule 和 raw
> history 的多表示状态库，并根据版本生命周期选择表示与放置。

### 7.2 为什么当前项目确实需要一个状态层

fast tier 需要逐层 old `Norm(x)`；structural tier 还需要部分 hidden checkpoint；full
需要 raw history 或可重建输入。它们的访问模式不同：

| 表示 | 主要用途 | 访问频率 | 更新方式 |
|---|---|---:|---|
| serving K/V | 每次在线增量推理 | 高 | append 或 migration 替换 |
| normalized capsule | compiled projection | 每次 model update 或 lazy miss | exact refresh 时重建 |
| hidden checkpoint | structural tier | 少数高 fidelity cohort | exact refresh 时重建 |
| raw history | full endpoint/灾难恢复 | 低但必须可靠 | 行为流 append |

当前六层设置中，保存所有 normalized states 的额外容量约为 K/V 的 50%；这不是可以在系统
评测中忽略的小 metadata。

### 7.3 设计空间

#### 多表示 admission

- 热用户：HBM 中保留 K/V，host 中保留 capsule；
- 温用户：host 中同时保留 K/V 与 capsule；
- 冷用户：只保留 compressed capsule 或 raw history；
- 小 cohort：不保留 specialized hidden checkpoint，next access 直接 full；
- anchor 即将回收时：先决定 exact refresh 还是丢弃 capsule。

#### Version-major physical layout

普通 user-major layout 适合单请求读取；后台 migration 更希望：

- 同一 `(anchor, target, layer, length bucket)` 连续；
- 大块读取 normalized states；
- 一次 projection 后顺序写目标 K/V；
- adapter 和 metadata 只加载一次。

可以研究 user-major 与 cohort-major 的 hybrid index，或后台 compaction 是否值得。

#### Tier-aware action selection

planner 比较的不能只有 kernel cost：

\[
\text{end-to-end action cost}
= \text{read source}
+ \text{compute}
+ \text{write target}
+ \text{metadata/fragmentation}.
\]

例如 normalized state 位于 SSD 时，full recompute 若 raw history 已在 host，可能反而更快；
这会改变 action frontier。

#### Read-migrate-write pipeline

- CPU pinned double buffer；
- layer/chunk 级 read 与 projection overlap；
- scatter/gather 或直接写 paged destination；
- write-behind 与 atomic metadata commit；
- 对被新版本 supersede 的输出做 write cancellation；
- I/O slack 内运行 migration，避免与前台 attention 争用。

### 7.4 为什么不建议把它单独作为首选主线

这个方向已经非常拥挤：

- [MTServe](https://arxiv.org/pdf/2604.22881) 这一 2026 年 preprint 已经针对生成式推荐
  提出 GPU/CPU Page-Chunk hierarchy、双缓冲 DMA、layer-wise overlap 与 LRU；
- [Mooncake](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/)
  把 CPU DRAM、SSD 与 NIC 组成分布式 KV cache，并围绕 SLO 调度；
- [LMCache](https://arxiv.org/abs/2510.09665) 的 2025 年 preprint 已提供跨
  GPU/CPU/storage/network 的 KV movement、batching、pipeline 和 control API；
- [Tutti](https://arxiv.org/abs/2605.03375) 的 2026 年 preprint 进一步把 SSD-backed KV
  的 I/O control path 移到 GPU，并做 slack-aware scheduling。

因此“SSD + GPU RAM 调度”本身不再是足够强的差异。TierState 必须围绕
**versioned multi-representation state、anchor lifecycle 和 migration/recompute 联合选择**
来做，最好作为 A/C 的数据层。

## 8. 候选 E：VersionGraph——长期版本链上的 anchor 与 adapter 图

### 8.1 系统定义

> VersionGraph 管理长期运行服务中的 model-version DAG、migration anchors、direct adapters
> 和 exact refresh，避免每次更新都对所有用户重写，也避免 adapter 与 anchor 数量无界增长。

### 8.2 核心问题

假设系统存在 active anchors \(v_1,\ldots,v_k\)，当前模型为 \(t\)。直接做法需要为每个
\((v_i,t)\) fit 一个 adapter。随着版本持续发布：

- adapter 数量与 active anchors 成正比；
- 小 cohort 可能无法 amortize calibration；
- approximate K/V 不会自动生成新 anchor；
- rollback 会让 version lineage 从线性链变成 DAG；
- 每次 direct-to-latest 都从很老 capsule 出发，fidelity 可能下降。

### 8.3 候选机制

1. **Sparse anchor policy**
   - 只允许少数版本成为 durable migration anchor；
   - 其余版本只产生 serving K/V，不产生长期 capsule；
   - anchor 数量由容量、adapter fixed cost 和 error distance 决定。

2. **Direct vs. exact renewal**
   - planner 比较从旧 anchor direct migration 与一次 exact refresh；
   - exact 不只是高 fidelity action，还会“续租”未来迁移能力。

3. **Adapter family compression**
   - 研究相邻更新 residual 是否共享低秩子空间；
   - 共享 basis、每版本只保存小 coefficient；
   - adapter memory、compile latency 和 probe sample 同时下降。

4. **Path planning**
   - 如果设计出可组合的中间 canonical state，可选择 direct 或 multi-hop；
   - 路径 cost 同时计 error accumulation、I/O 和 fixed compile cost；
   - 若当前 K/V-only 输出不可安全组合，则明确禁止 multi-hop，并以 exact anchor renewal
     维持语义。

5. **Adapter and state GC**
   - 当 cohort size、未来访问概率或 rollback window 低于阈值时一起回收；
   - 防止 adapter 被删除但 capsule metadata 仍指向它。

### 8.4 学术潜力与风险

这个方向能把当前 one-pair algorithm 推到真正的长期系统，而且“exact refresh 既是
fidelity endpoint，也是重建 future migratability 的投资”是一个有意思的新视角。

风险也很高：

- 若 residual 无法跨版本共享，adapter compression 不成立；
- multi-hop canonical state 可能需要新的算法设计；
- public version chain 仍较短；
- 很容易重新变成一篇 adapter/composition 方法论文，而非系统论文。

建议先做一个最小诊断：测量不同 old/current pair 的 residual subspace overlap、direct
fidelity 随 anchor distance 的变化，以及 exact renewal 对后续两次迁移的价值，再决定是否
升格为主线。

## 9. 候选 F：FreshTrain——cache-aware 的持续训练与发布系统

### 9.1 系统定义

> FreshTrain 联合决定“何时发布一个流式训练 checkpoint”与“发布后如何维护派生 cache”，
> 最大化 serving 端实际获得的 freshness value，而不是只追求 weights 尽快更新。

### 9.2 动机

持续训练系统通常把 checkpoint 发布视为收益：模型越新越好。但对 persistent KV 的生成式
推荐，每次发布同时产生：

- 新模型的潜在质量收益；
- 所有旧 cache 的 compatibility debt；
- calibration 与 adapter fixed cost；
- 后台 read/compute/write；
- 版本碎片和 rollout 风险。

如果模型更新很小、cache maintenance endpoint 也很小，立即发布可能制造大量系统工作却
几乎不改善请求质量；相反，某些更新会产生局部 staleness jump，应更快发布并优先迁移。

### 9.3 联合控制器

控制器可选择：

- 立即发布 checkpoint；
- 合并多个小 update 后发布；
- 只发布部分参数/模块；
- 发布 weights，但暂不 admission 某些 old-cache cohorts；
- 为新 checkpoint 选择 migration budget 和 deadline；
- 当 backlog 过大时降低发布频率或执行 exact anchor renewal。

目标是最大化：

\[
\text{fresh-model gain}
- \text{stale-cache exposure}
- \text{migration resource cost}
- \text{rollout risk}.
\]

### 9.4 与现有推荐更新系统的关系

[Ekko](https://www.usenix.org/conference/osdi22/presentation/sima) 研究更新传播优先级和
rollback；[QuickUpdate](https://www.usenix.org/conference/nsdi24/presentation/matam)
研究 prioritized parameter updates、间歇 full update、model transformation 与 relaxed
consistency。FreshTrain 的差异应是把 **derived per-user state debt** 纳入 checkpoint
publication，而不是只优化参数传输。

### 9.5 为什么它可能很强

- 把训练与 serving 的目标真正闭环；
- 当前 fixed-endpoint 跳变与 6/9 admission 边界都直接支持这个问题；
- 能解释“更新越频繁不一定越新鲜”，因为 cache 可能长期追不上；
- 现有 migration primitive 是提高可发布频率的核心 enabler。

### 9.6 为什么它工作量最大

- 会改变当前“训练版本外生给定”的实验边界，需要新 protocol；
- 需要一个可信的 checkpoint benefit 与 cache debt estimator；
- 训练、发布和 serving 三类时间尺度要在同一 trace 中回放；
- 很难仅靠当前三份 public data 模拟真实生产发布策略；
- 若没有 production-like update/traffic trace，系统结论容易显得人为。

建议把它作为 A/C 完成后的扩展，或在 admission v2 出现很强 signal 时升级为主线。

## 10. 候选 G：CohortExecutor——异构 cohort 的多 GPU 执行器

### 10.1 系统定义

> CohortExecutor 在多个 version pairs、length buckets 和 state locations 并存时，保持
> compiled projection 的高 GPU 利用率，并把状态搬运与 migration/recompute pipeline 融合。

### 10.2 可做的机制

- per-adapter grouped GEMM，支持一个 launch 中的多个小 cohort；
- ragged/length-bucket input，避免 padding 把 I/O 与 projection 都放大；
- adapter weights 常驻或按热度缓存，状态按 locality 放置；
- CUDA Graph/cache，降低短 projection 的 launch overhead；
- read/projection/write 三阶段 double buffering；
- structural/full 长任务 chunking，允许前台请求插入；
- 跨 GPU work stealing 时同时考虑状态字节与剩余 deadline；
- adapter broadcast 与 state sharding 分离。

### 10.3 何时足以独立成文

需要先 profile 证明：

- mixed-version fragmentation 让现有 projection 明显掉出高效 GEMM 区间；
- grouped executor 不只改善 microbenchmark，也提升 end-to-end goodput/P99；
- 同一个 abstraction 能覆盖 compiled、structural 和 full 三类异构 action；
- 至少有两项互相依赖的新机制，而不是调用已有 grouped GEMM library。

否则 G 应是 C 的 local executor 章节。

## 11. 早期组合方案（已降级为备选）

这一节记录第 0 节收敛之前的 A+C 组合。它现在不是第一推荐，只在后续 profiling 证明主要
难点落在长期 mixed-version coherence、而非批量迁移执行时，才考虑重新升格。其原建议范围是：

> **A（versioned coherence） + C（mixed-version scheduler） + B 的最小事务语义
> + D/G 的必要数据面。**

### 11.1 早期架构

#### Control plane

- Model Update Observer
- Cohort Profiler
- Admission Controller
- Fidelity/Cost Planner
- Adapter Registry
- Global Scheduler
- Rollout/Health Manager

#### Data plane

- Versioned State Directory
- HBM/host 两级 State Store，SSD 作为后续可选 tier
- Per-GPU Local Scheduler
- Compiled/Residual/Full Executor
- Async Transfer Pipeline
- Atomic Commit and GC

第一版不必立即实现跨机 RDMA 或 GPU-native SSD。先在四张 A40 上把 mixed-version、state
movement、foreground interference 和 lineage correctness 做完整；只有 profile 证明 host
capacity 或 PCIe 是决定性瓶颈后，再增加 SSD/GDS。这样 SSD 是被问题驱动的设计，不是为了
让架构图更复杂。

### 11.2 早期的一句话 paper claim

> 持续训练不仅发布新 weights，还会异步地使海量持久用户状态失效。CohortKV 把这一过程
> 建模为版本 cohort 上的 bounded semantic-coherence problem，通过 update-level admission、
> compiled cache translation、anchor-aware background scheduling 和安全 rollout，在前台
> serving SLO 下避免每个版本的全量历史重算。

### 11.3 早期的贡献结构

1. **Problem/characterization**
   - 首次系统刻画持续训练下 persistent GR cache 的 mixed-version invalidation；
   - 展示 age、cohort size、history length、state tier 和 update interval 的联合分布。

2. **Abstraction/contract**
   - versioned derived-state object；
   - serving target 与 migration anchor 分离；
   - bounded semantic coherence 与合法状态机。

3. **Control plane**
   - cohort admission；
   - fidelity/cost tier selection；
   - moving-target、eager/lazy、deadline-aware scheduler。

4. **Data plane**
   - 现有 compiled affine migration；
   - representation-aware read/migrate/write；
   - mixed-cohort batching 和多 GPU execution。

5. **Evaluation**
   - organic mixed-version replay；
   - equal-quality/equal-cost baselines；
   - end-to-end goodput、P99、freshness convergence、bytes 和 failure recovery。

## 12. 最新主线的系统评测

### 12.1 独立 protocol

当前架构与发布语义使用 `streamkv_destination_out_of_core_v4`。后续真实 HBM/DRAM
full-cohort 性能结果必须在该边界上建立独立、版本化的 performance protocol，不能写回现有
resident-kernel result family。每次实验至少固定：

- old/current model pair、compiler program 和 calibration/test split；
- cohort size、真实 length 分布、capsule 与目标 K/V 的 dtype/layout；
- source/destination tier、容量、实际读写 bytes 和是否包含 cold-cache I/O；
- GPU 数、worker 数、wave 大小、publication queue depth 和 manifest commit 语义；
- compile、read、transform、write、layout conversion 是否计入端到端时间。

质量仍遵守现有协议：full 是 cache-fidelity reference，而不是 ranking quality 的必然上界；
training seed 仍是统计复现单位。按真实分布复制 cache records 只能扩大 systems workload，
不能被计作额外质量重复。

### 12.2 三条与贡献一一对应的证据链

| 层 | 核心问题 | 主要 baseline | 必须报告 |
|---|---|---|---|
| Compiler | 能否以一次 version-pair 编译替代 per-cache recomputation | reuse、cheap projection、residual replay、full | K/V fidelity、paired task difference、compile time、含 calibration 的 cohort break-even |
| Operator | 能否把 program 做成真正的 one-pass capsule-to-KV | 当前 PyTorch prototype、library GEMM + unfused epilogue、full kernel | batch latency、records/tokens per second、kernel launches、读写 bytes、临时显存、不同 shape/length |
| Engine | 能否把单批优势转成全 cohort、不同 destination 与多 GPU 优势 | 同步 read→compute→write、无 cohort organization、single GPU、同 endpoint pipelined full | end-to-end completion time、peak HBM/host staging、bytes、manifest commit、1/2/4 GPU scaling |

端到端比较必须给 full recompute 与 compiled migration 相同的输入位置、输出位置和计时边界，
不能让一方只计 GPU kernel、另一方计 host/SSD movement。

### 12.3 Workload 与硬件范围

- KuaiRand 使用真实日历顺序；QB/QK 只称 ordinal replay，不伪装成 wall-clock trace；
- 真实数据提供 cache shape、length、version-pair 和质量样本，系统规模可按这些分布扩展；
- 首轮覆盖 target-GPU HBM 与 pinned-host DRAM destination，分别建立 direct-device 与
  host-staged 边界；
- 多 GPU 扫 1/2/4 张 A40，报告共享 PCIe/CPU 路径是否成为瓶颈；
- POSIX filesystem 与 remote object 先冻结接口和原子发布语义；
- SSD 只有在记录物理设备、mount、缓存状态和 durability 选项后才加入；
- 跨节点只有在获得真实网络硬件和可复现 topology 后才进入主结果。

### 12.4 最小必要 ablation

只围绕三项贡献做消融：

1. compiler：去掉 fitted residual，或把困难 cohort 替换为 residual/full program；
2. operator：unfused epilogue、无 length grouping、staging write 与 direct final-layout write；
3. engine：无 cohort ordering、无 pipeline overlap、single GPU 与 multi-GPU extent sharding。

admission、量化 capsule、自适应 batch/buffer 等只有在最终系统实际采用时才消融。rollout、
rollback、通用 eviction policy 和跨节点 failure protocol 不属于当前 evaluation checklist。

### 12.5 整篇论文必须同时成立的结果

- compiled program 在足够大的真实 cohort 上仍能 amortize，且质量结论不弱于现有证据；
- one-pass operator 相对当前 prototype 带来可解释的 launch、memory traffic 或 latency 收益；
- host-resident end-to-end 路径仍显著优于同边界的 full recompute，而不只是在 HBM kernel 上赢；
- cohort pipeline 在 1→2→4 GPU 上有可解释的 scaling，并能保持有界峰值工作内存；
- 相对 DroidSpeak、MTServe 和通用 KV movement 系统，收益确实来自 version-cohort
  compile–execute–stream 共设计，而不是把三个已有技巧顺序连接。

## 13. 分阶段研究 gates

### Gate S0：冻结 compiler contract

沿用当前 verified compiler，定义 migration program 的输入、输出和 metadata，并在新
validation split 上冻结 compiled/residual/full action library。系统不依据未观测请求或
task-quality oracle 做 per-user admission。

当前状态：接口和 adaptive seed-0 certificate 已完成；冻结的新 seed/数据复现仍待完成。

### Gate S1：operator byte/roofline profile

对当前 PyTorch prototype 分解 stack、cast、BMM、bias、mask、cast-back、K/V split 的时间、
launch 和 bytes，确认瓶颈究竟在 GEMM、memory traffic 还是 framework overhead。这个 profile
决定采用 direct-write fused kernel，还是 library GEMM + fused epilogue。

当前状态：v1/v2 stage profile 已完成，支持 fused epilogue/direct-write；不能由此推出
cohort compaction 必然有效。

### Gate S2：one-pass operator

先实现 contiguous final-layout output，再比较 direct paged write 与 staging + scatter。
只有相对当前 prototype 在代表性 batch/length 上形成稳定收益，才把 operator 保留为独立
contribution。

当前状态：contiguous direct-write 已实现，并在 v2 resident 与 host boundary 分别获得
1.19x/1.176x；v3 paged compaction 在独立 final trace 上只有约 1% host 改善、HBM 略退化。
因此 one-pass direct write 保留，page compaction 只是条件式布局，operator 作为独立论文
贡献仍取决于后续同边界证据。

### Gate S3：单 GPU end-to-end stream

实现 pinned host read → transform → host write 的有界双/三缓冲路径，同时测 compiled 与
full recompute。若 movement 抹平计算收益，优先替换 capsule representation、output path 或
pipeline balance，不进入复杂 scheduler。

当前状态：受控 64-record host-DRAM boundary 已通过，并有 independently pipelined exact
baseline；尚不能替代 full-cohort destination-v4 结果。

### Gate S4：cohort organization 与四 GPU

比较物理/准物理 extent 和 secondary-index sorted gather，选出一个最终 organization；
再以 extent sharding 扫 1/2/4 GPU。这里解决的是 stream partition、bounded wave、
publication backpressure 和共享 I/O 瓶颈，不扩展成在线请求 scheduler。

当前状态：controlled mixed-version trace 的 LPT 与 1→2→4 scaling 已完成；page
compaction 为负结果。完整 source stream、全 cohort 以及把相同四卡 measurement 纳入
destination-v4 transaction 的工作仍待完成。

### Gate S5：destination backend 扩展

HBM、DRAM、POSIX filesystem 和 remote-object contract 已实现 reference vertical slice。
下一步先闭合 HBM/DRAM 的全量真实数据结果。只有记录真实 SSD/GDS 或网络硬件后，才把对应
backend 升级为性能结果；它们增强 engine 的适用范围，但不改变论文贡献数量。

当前状态：transaction/backends 与薄层 Coordinator 接口已完成正确性 vertical slice；
source-side lazy reader、同 transaction exact baseline 和真实 backend 性能仍待完成。

## 14. 停止或转向条件

先在第 0.5 节做同位替换。经过有界优化后若仍出现以下情况，应缩小或放弃 StreamKV 主线：

1. 固定更新 cohort 普遍小于含 calibration 的端到端 break-even；
2. capsule 的持久化与移动成本使 compiled path 不优于 raw history + full recompute；
3. fused/direct-write 与 library baseline 相比没有稳定收益，无法支撑 operator contribution；
4. cohort ordering 和 read–transform–write overlap 不优于普通 batched loader + GEMM；
5. 1→2→4 GPU scaling 被共享 I/O 完全吞噬，且 extent partition/backpressure 无法改善；
6. compile、operator、engine 三层中有两层必须被删除，因果链不再成立；
7. 与 DroidSpeak、MTServe 的完整对比后，剩余内容只是已有跨模型重算与通用 I/O pipeline
   的直接组合；
8. public workload 无法支撑 cohort 规模、状态移动或 out-of-core working-set 压力的可信
   结论。

## 15. 相关系统论文给出的“布局启发”

这些工作不等价于本项目，但可以借鉴它们如何把一个 kernel/algorithm 扩成系统：

| 工作 | 可借鉴的布局 | 本项目不能重复的主张 |
|---|---|---|
| [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | 新状态抽象、memory manager、scheduler、distributed engine、端到端 throughput | 仅做普通 paged KV |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | 先刻画阶段干扰，再做资源分离、placement 与 SLO goodput | 机械拆分两类 GPU worker |
| [Llumnix](https://www.usenix.org/system/files/osdi24-sun-biao.pdf) | global/local scheduler、live state migration、load abstraction、tail/SLO | 仅把 cache 从 GPU A 搬到 GPU B |
| [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu) | tier-optimized format/loading、locality-aware scheduling、live migration | 只实现多级加载 |
| [Mooncake](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/) | KV-centric global cache、分层资源、全局 scheduler、overload/SLO | 普通分布式 KV pool |
| [CacheGen](https://www.microsoft.com/en-us/research/publication/cachegen-fast-context-loading-for-language-model-applications-via-kv-cache-streaming/) | 传输/重算联合选择、带宽自适应、端到端 context-loading delay | 只做 tensor compression |
| [LMCache（2025 preprint）](https://arxiv.org/abs/2510.09665) | modular connector、batched movement、compute-I/O pipeline、control API | 通用 KV movement layer |
| [MTServe（2026 preprint）](https://arxiv.org/pdf/2604.22881) | GR-specific persistent state、GPU/CPU Page-Chunk、DMA overlap、LRU | “首个 GR hierarchical cache” |
| [Tutti（2026 preprint）](https://arxiv.org/abs/2605.03375) | GPU-native object/I/O abstraction、slack-aware I/O scheduling | 普通 SSD-backed KV |
| [DroidSpeak](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_liu-yuhan_prepub.pdf) | 跨模型 KV、离线 profile、选择性重算、传输—计算 pipeline | “首个跨模型/版本 KV reuse” |
| [Ekko](https://www.usenix.org/conference/osdi22/presentation/sima) | update priority、状态监测、低延迟 rollback、生产 update trace | 只优化 weights dissemination |
| [QuickUpdate](https://www.usenix.org/conference/nsdi24/presentation/matam) | prioritized/partial update、intermittent full、relaxed consistency | 只控制参数发布带宽 |

对最新主线，最重要的 related-work 判断是：

> 我们不能再以“层重算”“分层 KV”或“模型低延迟更新”中的任意一个作为单独 novelty。
> 需要形成差异的是一个端到端闭环：**把持续更新模型造成的持久化 per-user KV 全局失效，
> 编译成 cohort 共享程序，用 one-pass operator 执行，并作为 state stream 跨存储与多 GPU
> 完成转换。**

这仍是初步判断，正式 novelty claim 前必须继续完成 primary-source audit。

## 16. 最终优先级建议

### 第一优先：StreamKV 三段式主线

严格保持第 0 节的三层闭环：

1. cohort migration compiler；
2. one-pass capsule-to-KV operator；
3. destination-oriented out-of-core multi-GPU migration engine。

论文定位是“compiled destination-oriented KV update system”，不是 model publishing、rollout、
通用 coherence、通用 hierarchical cache 或通用 scheduler system。现有算法位于 compiler
中心，算子和 runtime 分别证明它能在单批与全 cohort 尺度转化成真实系统收益。
Coordinator 只是将这三层接到一个 job entry point，不改变贡献数量。

### 第二优先：同接口替换，不扩展贡献数量

若某一层 gate 失败，先使用第 0.5 节的同位替换：

- compiler 在 affine、selective residual replay、reuse/full 之间换 program；
- operator 在 direct paged write 与 contiguous staging + scatter 之间切换；
- engine 在物理 cohort extent 与 secondary-index gather plan 之间切换。

替换后仍维持 compile–execute–stream 三个接口，不顺势增加新的 control-plane 子系统。

### 第三优先：只有主流水线失败才转向旧候选

- 若 end-to-end 主要矛盾是多个版本长期共存与前台 SLO，再考虑 A+C 的 coherence/scheduler；
- 若主要矛盾是长期 anchor 不可链式迁移，再考虑 E 的 version graph；
- 若真实 update trace 显示发布频率本身决定成本，再考虑 F 的 training/publishing controller。

这些都是转向条件，不是当前 StreamKV 要同时覆盖的内容。
