# EvoKV Design 3 Foundation Benchmark and Exploration Plan

最后更新：2026-08-03

状态：**历史 M0/M1 机制发现账本 + successor benchmark 的灵活执行入口，不是冻结
protocol 或论文结果**。本文档完整保留 fixed-512、GPU0/GPU1、complete-private-target
开发链，但它不再定义正式 benchmark endpoint。未来 formal matrix、资源边界和执行顺序以
[../10_paper_experiment_blueprint.md](../10_paper_experiment_blueprint.md) 为准；promotion
前的检查登记在 [../11_benchmark_qualification.md](../11_benchmark_qualification.md)。
Qualification 不阻止 workload、runner、baselines 或机制设计，只阻止未合格结果晋升为
paper evidence。

当前 checkpoint 搜索已经结束。下一轮直接使用 registry 绑定的 QK theta0--theta4
primary 与 QB theta0--theta3 secondary；旧 M1 专用 checkpoint payload 已回收，历史结果
仍按原 hash 解释。完整路径与重训入口见
[../13_cross_dataset_stream_checkpoint_plan.md](../13_cross_dataset_stream_checkpoint_plan.md)。

## Successor benchmark boundary

下一轮 foundation 从一开始采用同一个现实场景，而不是在 D3 才临时引入异质性：

1. `X-QK-HET` 是 D1→D2→D3 的 primary workload。它保留自然
   old/retained/evicted/append/target 长度，容量、分组和吞吐按 valid K/V bytes 定义，
   owner 使用冻结 stable hash 并保留自然 rank imbalance。
2. `X-QK-HOM` 与 HET 共享模型、edge、base-only entity table、用户选择 salt、record
   IDs、valid histories 和 D1 action 语义，只将同一记录 masked-pad 到 512-slot physical
   layout 作 matched causal control。不得根据最终 speedup 在 HET/HOM 中择优，也不为
   HOM 复制全部实验矩阵。同一 nominal cohort 由 HET valid bytes 冻结，但 HET/HOM 的
   micro-wave admission 分别使用实际 input+shadow+workspace allocation，HOM padding
   完整计入。
3. 集成主路径只使用 `compiled|exact`。Progressive residual replay 是 D1-only supporting
   extension；若扩入完整 stack，必须建立独立 source/action contract 和重跑 baselines。
4. XP 固定为 2,859,835 个 base-period semantic rows 加一个 padding row，即
   2,859,836×4,096 physical FP32 table（43.638 GiB）、
   owner-side E4096→H1536 projection 和 24L/H1536 core。硬件 HBM cap 先独立冻结，
   qualification 再验证单卡不可复制、2/4-rank admission 和短 edge；不能查看 EvoKV
   speedup 后换 geometry。只有真实 optimizer-updated active rows 计入强制分片；
   两个 formal edges 上 all-exact 与所有冻结 fixed-action exact/append/fallback 的
   semantic-request union 必须全部 active 并记录 hash，且 active embedding 加
   dense/projection bytes 自身超过单卡可分配量。
5. 正式 runner 从实现上参数化为 1/2/4 rank。XP 以 capacity-admitted 2/4 rank 为
   headline；X2/R-KR 提供 1-rank path 和 rank-count sanity。三卡不属于预声明矩阵。
6. D3 维护一份 live cache 与 bounded group shadow/staging。每组执行
   `stage → compute → writeback → validate → group commit → old-group reclaim`；host
   peak 不再包含完整第二版本。
7. Baseline-first 顺序为 tuned exact、S0、S1、independently tuned fixed-FIFO S2 和
   profile-aware work-conserving generic scheduler，再运行 D3。Profile-aware generic
   把 group 当作 opaque jobs，可以读取相同 stage profile/capacity/credits 和 bounded-flow
   recurrence，但不能读取 compiled/exact 标签、route-specific parameter sharing 或
   EvoKV stable-interleave objective。

首批 foundation 输出不是固定接口，而是 HET/HOM manifests、XP capacity ledger、
rank-parameterized runner、rolling group store、共同 timer/correctness ledger 和上述强
baselines。中间实现允许根据 profile 调整；只要 `stack_revision` 改变，就为新 revision
重跑自己的 baselines。

## Foundation Review 0：successor 已开始实际构建

2026-07-31 的首轮实际构建已经完成 workload、两卡 embedding 物理 canary 和 rolling
lifecycle canary。它回答“successor foundation 能否建立”，不回答 D3 候选是否有效，也不
晋升任何 timing 为论文证据。

QK Pass A 完整扫描 493,458,970 行、5,022,750 个用户。按新的 stable salt，1,715,916 个
用户具有至少 96 条 exposure，2,219 个用户具有至少 1,024 条。已冻结互斥 post-base
roles：2,048 个 `theta1→theta2` 长用户、2,560 个 `theta0→theta1` 用户、各 512 个
fit/profile/qualification 用户和 65,536 个 final records。Pass B 为 final records
物化 12,257,432 个实际历史 token；压缩工件约 37.7 MB。角色、长度、owner、capacity 和
历史身份位于：

```text
configs/evokv_foundation/foundation_review_development_v0.json
configs/evokv_foundation/qk_post_base_roles.json
configs/evokv_foundation/qk_foundation_summary.json
data/processed/evokv_foundation/qk_full_user_lengths.npz
data/processed/evokv_foundation/x_qk_het_foundation.npz
```

自然 HET target 长度的 min/median/p95/max 为 96/153/404/512，只有 2.1835% records
达到 512。完整 65,536-record universe 的 old/target valid K/V 分别为
1,498,194,247,680/1,801,465,380,864 bytes；同记录 HOM 的单版本 allocated K/V 为
4,947,802,324,992 bytes。由 HET target valid bytes 冻结的
36/72/144/288/576/720-GiB points 分别需要
1,416/2,815/5,625/11,272/22,544/28,192 records。720-GiB point 的 HET old live
K/V 为 642,565,177,344 bytes，仍可在一份 live cache 加 bounded shadow 的边界内进入
后续 host qualification；同 nominal point 的 HOM live allocation 超出当前主机，因此
HOM 只运行 capacity-admitted matched controls，不能强行复制完整矩阵。

all-exact valid targets 在两个预声明 edge 上各包含 12,216,969 个 token，union 为
929,554 个真实 mapped rows，hash 已写入 summary；这些 rows 全部属于 base catalog，但
catalog frequency 不是 optimizer activity。XP 的 2,859,836×4,096 FP32 物理表已在
GPU0/GPU1 实际按 modulo 分片：全局 46,855,553,024 bytes、每 rank
23,427,776,512 bytes，两个 rank 的实测 peak allocated 均为 23,478,130,688 bytes。
16 requests/rank 的 canary 中每 rank 有 8 个 remote requests；owner 在 FP32
E4096 上查表并投影后，只返回 H1536 FP16，数值 oracle 通过。该 canary 只证明容量和
通信实现，不证明 checkpoint 已训练。

XP 的 dense core（含 4,096×1,536 projection）为 291,863,040 个 FP32 parameters；
global embedding 加 dense 为 48,023,005,184 bytes，超过本机单卡 Torch allocatable
47,699,722,240 bytes。Forced sharding 只能按真正 optimizer-updated rows 计算；精确门槛
为 2,840,105 个 semantic rows，即 99.3101%。当前 selected XP checkpoint 的 bitmap
包含 2,859,736 个 active semantic rows，因此 byte gate 已通过。尚未闭合的是两个 formal
edges 及所有 fixed-action exact/append/fallback 路径的 request-union-to-active membership
join；冷分配和仅有 929,554-row request-union count 都不能代替这项检查。

真实 HET/HOM 工件上的四组 short/mid/long/saturated full-payload lifecycle canary 均
完成 validate-before-commit、old reclaim、一次故障不发布、一次幂等 replay 和
exactly-once coverage。它当前使用 deterministic full-extent payload 验证容量/事务语义，
明确 `executes_d1_d2_numeric=false`。历史 fixed-512 X2 双卡 D1/D2 回归另以 8 条
compiled/exact records 跑通，makespan 0.3271 秒、exactly-once 通过、单 rank peak
allocated HBM 11.157 GB；它只证明旧数值栈和 collective 环境可用。

因此本轮结论是：**workload、真实 XP 两卡分片投影、active-row checkpoint 和 rolling
事务骨架已经可运行；正式 request-union membership、ActionPlan overlay、byte-bounded
D1/D2 rolling runner 和 36/72-GiB problem-existence baselines 尚未闭合。** 下一次系统
review 应在 request-union gate 与第一条真实 HET `compiled|exact` rolling canary 闭合后
进行，而不是在当前组件 canary 上判断 D3 机制成败。上述结果全部保持
`scientific_result=false`、`formal_design3=false`。

以下 checkpoint 是**历史 fixed-512/full-private-target M0/M1 开发账本**。Milestone D
已完成 route-aware ResidencyPlan 的开发态实现、同一
stack/hash 下的 route-major/selected paired full 运行，以及完整 payload 校验。早期
Milestone A 的 H12/W2 682 records 被划为 26 个
logical-payload-bounded groups；GPU0/GPU1 使用一个可复用 pinned slot，跑通 ordinary
DRAM→HBM→真实 D2 compiled/exact→ordinary DRAM。full run 将约 30.64 GB private target
exactly-once 写回普通 DRAM。最新 makespan 17.73 秒。两 rank 包含 embedding collective 与
rank wait 的 D2 execution 为 7.02–7.79 秒，其中 lookup collective 为 0.72–1.64 秒；四段
host/device movement 与 writeback 合计 9.93–10.01 秒。该单次 profile 仅说明 D3 数据通路已成为
同量级瓶颈，仍是
`scientific_result=false` 的容量模拟，不是 speedup 或物理 out-of-core 证据。

M1 已推进到真实物理 out-of-core 的首个 D3 候选，并冻结为历史 D3 开发边界。QK 全体用户
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

## 0. 历史 M0/M1 原则：先把两卡 DRAM↔HBM benchmark 跑起来

这一原则描述已经完成的 M0/M1 mechanism-discovery 阶段。当时最需要弄清楚的不是一个
完美接口，而是：

> 当大量旧 K/V 位于 host DRAM、两张 GPU 无法同时容纳完整 source 和 target 时，怎样把
> DRAM→HBM、两卡 D2 计算、HBM→DRAM 组织成高效流水？

该阶段第一优先级固定为 GPU0+GPU1：

- 两张 A40；
- one process per GPU；
- NCCL；
- 同一 NVLink/NUMA0 island；
- 当时暂不做 3/4-GPU、跨 island 或多节点。

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

这种延期只适用于已经完成的 M0/M1 机制发现。Successor paper-core 现在明确要求
1/2/4-rank runner、rolling group lifecycle、segmented consumer、reproducible identity 和
formal protocol；早期 benchmark 使用轻量 `WorkManifest`、private target 和
coverage/checksum 的做法只能继续作为历史回归。

### 0.3 允许跨层回头调整

当前三层分工是一个有用的起点，不是不可穿越的墙：

```text
D1：给出当前 planning 决策
D2：给出当前分布式执行方法
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

## 1. 历史两层 benchmark

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

> 这里的 M1 是冻结的 fixed-512 development workload，不是 formal X-QK role split。
> 它的 users、action snapshot、model edge、complete-private-target endpoint 和 timings
> 都不能直接晋升或汇入 successor paper evidence。

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
为 3.653369 对 3.707804。该正信号只用于确认这个短 edge 没有退化，并冻结为历史 D3
development boundary；它不是多 seed 的算法质量结论。

edge-specific direct-old-K/V program、20.0195%-exact D1 action snapshot 和 D2 request
characterization 都已完成并绑定相同 checkpoint/data identity。第二个 update、recursive
migrated source、多 seed 和完整质量复现当时被推迟；successor 现在按实验蓝图重新建立
独立 role split、held edge 和 formal replication。

## 2. 历史最小两卡执行骨架

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

`WorkManifest` 当时可以先绑定 H12/W2 实现。它的作用是让 sequential、double buffer 和
proposed scheduler 读取同一份工作清单，而不是定义永久架构。

每次 D1/D2 adapter 发生变化，更新：

```text
stack_revision
work_manifest_hash
change_reason
```

然后在新 revision 下重跑 baselines。正式 capacity-independent exporter、完整 stack
identity 和 parity checker 不是历史 M1 mechanism discovery 的前置条件，但在任何 formal
D2/D3 paper claim 前都是必需项。

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

历史第一版不要求完整 atomic epoch switch 和所有 fault injection；private target 不对外
发布即可。Successor 不再补 global epoch COW，而是要求每个 rolling group 在 timed path
中完成 writeback、validation、versioned commit 和 old-group reclaim，并验证幂等恢复。

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

### 3.3 S2 specification 与历史 bidirectional precursor

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

只分段输入的历史中间版本为 31.096 秒，并把 input wait 转移成 4.865/3.706 秒
output-credit wait。旧 runner 的双向 precursor 进一步降到 28.885 秒，output-credit
wait 降至 1.735/0.738 秒，相对 strong S1 的诊断比值为 1.133x。full S1/D3 两 rank 的
target 均逐字节一致，ledger 为 complete/no-partial/no-missing/exactly-once；microbatch-16
为 29.337 秒且 reserved HBM 更高。由于 28.885 秒不属于 current exact-stack runner，
它不是已完成的 independently tuned formal S2，也不能作为当前 planner 的 order-only
分母。

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

### 3.6 历史 M1 timer 与 successor timer

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

历史 M1 将 plan construction、atomic commit 和 reclaim 单独计量。Successor D2/D3
mechanism 图使用 execution-only timer：从首个 ordinary-DRAM stage 开始，到最后一个
group 完成 writeback、validation、commit 和 old-group reclaim 为止；plan/profile
construction 单列，同时报告 plan-inclusive single-wave cost 与 break-even。E1 不沿用这个
execution-only 边界：其 end-to-end primary 从 model checkpoint publication 后开始，计入
该 edge 的 D1 fit/compile、D2 lowering/routing、D3 profile/plan construction 和完整
rolling execution。

当前 M1 S0 还将 complete old-store materialization、model/program load、operator warmup 和
target prefault 放在 primary timer 外。53.497 秒是 `run_s0` 的两 rank makespan；50.017 秒
是 makespan rank 的分项 phase sum。52.8% 搬运比例以 phase sum 为分母，不能与 wall time
混写。fair group-64 S0、S1 和当前 D3 pair 使用相同 runtime 边界。跨历史 runner 的开发链
为 48.238→32.703→28.885 秒；当前 exact-stack order-only pair 为
28.514442098→28.147194647 秒。54.577 秒的旧 S0 含每组 runtime maintenance，不再作为
speedup 分母；28.885 秒也不能作为当前 plan 的 order-only 分母。overlap/wait/bubble
指标用于因果分析，其中
`estimated_hidden_{input,output}_seconds` 只是保守估计，不能替代 wall/credit/boundary wait。

## 4. 历史 M1 profile 收敛出的 D3

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

### 4.2 当时的资格验证计划

历史 M1 在这一节点停止泛化 scheduler 和完整粒度笛卡尔积，当时计划完成：

- independently tuned and repeated formal E0；
- 至少一个 action/capacity mix sensitivity；
- 最小重复执行或第二 edge；
- publication/reclaim 计时边界；
- 判断收益是否稳定跨越 exact-preferred crossover。

该 capacity group、三段粒度、双向 credits 和 launch order 已经固化为可 replay 的
development `ResidencyPlan`。Coupled
microbatch-16、compiled input-16 和 compiled output-4 在各自开发观察点没有提升；这只支持
停止该历史 revision 的 exhaustive tuning，不构成对这些粒度的一般性拒绝。当前 successor
不会等待这些检查才开始 HET/HOM、XP、rolling runner 和 strong-baseline foundation。

### 4.3 历史 revision 的回退规则

若 formal E0 或 sensitivity 否定该历史机制，再考虑 rank-aware byte balance、collective-arrival-aware
prefetch、D2 owner/pool granularity 或 D1/D3 budget co-design。任何改变 actions、owners、
pools 或 layout 的候选都是新的 `stack_revision`，必须重跑自己的 S0/S1，而不是与本 revision
直接比较。

### 4.4 历史 M1 当时延后的内容

- 3/4 GPU；
- topology sweep；
- SSD/database ingress；
- serving trace；
- multi-update lifecycle；
- formal failure matrix；
- exhaustive parameter sweep。

这些延期不适用于当前 paper-core：1/2/4-rank、第二 edge、rolling failure recovery 和
formal protocol 已进入 successor 蓝图。历史 candidate 相对同 stack route-major control
只有 1.2879% 的单次收益；邻接但不同 runner 的 E0 diagnostics 只能提示可能存在 crossover，
不能建立 independently tuned/repeated E0→D3 结论。

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
邻接的 development diagnostics 提示可能存在 crossover，但它们不是 same-binary、
independently tuned and repeated E0→D3 对比，因而尚未建立 crossover。相邻
identity-only revision 的收益幅度更大，表明系统波动不可忽略。

### E：机制稳定后再正式化

**状态：历史过渡点；当前执行入口已由本文顶部 successor boundary 和实验蓝图取代。**

**当时目标。** 用 independent tuning/repeats 和最小 sensitivity 判断 development
candidate 能否变成论文设计；历史结果只提示可能的 crossover，并未证明它。

**届时再补。**

- 最终 D1/D2/D3 责任边界；
- 通用 schema/exporter/hash；
- rolling group validation、commit、abort、reclaim；
- exact crossover；
- capacity/action-mix sensitivity；
- 1/2/4-GPU；
- 第二 model edge 和正式 replication；
- frozen protocol。

**最大风险。** Historical development indication 无法在 independently tuned/repeated
E0、strongest generic baseline、capacity/action sensitivity 或 replication 中保持。若该
候选失败，successor 仍在已经固定的 benchmark foundation 上重新定位机制。

## 6. 历史 M1 开发判断标准

该阶段不设置复杂 formal gates，只连续回答四个问题：

1. **能否运行？** 两卡 bounded-memory DRAM→GPU→DRAM 是否正确完成？
2. **瓶颈是什么？** 搬运、compute、collective、rank wait 还是 writeback？
3. **通用流水有多强？** S1 相对 S0 隐藏了多少，generic fixed-FIFO S2 又能隐藏多少？
4. **场景特异机制是否有额外收益？** proposed 是否稳定胜过 independently tuned S2 与
   profile-aware generic 中的强者，且能由 profile 解释？

历史四个问题在这一个 M1 point 上的回答分别是：能；双向 staging/writeback 加 route
resource imbalance；S1 为 1.475x，fixed-FIFO segmentation 已经贡献大部分后续收益；
selected D3 相对同 stack route-major control 只有 1.013047x。它已经超过“只胜
sequential”的最低开发门槛，但尚未完成同 current stack 的 independently tuned S2、
profile-aware generic 或 E0→D3 comparison，因此没有回答 formal crossover；held-out
generality、正式重复和论文证据仍未完成。

Successor 仍需要严格可比性、correctness、physical capacity、independently tuned/repeated
all-exact、strongest generic、failure 和扩展实验；这些不抹去历史两卡机制发现，也不阻塞
当前 HET/XP/rolling foundation 实现。

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
route-aware stable interleave 的完整开发链。相邻但不同 runner 的 E0 与 D3 数字只提示
可能的 crossover；它们不建立 same-binary formal waterfall，也不证明正式 protocol、
independent replication、generality 或 paper speedup。

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

大 K/V payload 只存在进程内 pageable DRAM 或临时物理 arena，不进入 Git。历史 M1 old
store 曾完整物化；只有在 arena 仍存在且 coverage/hash/model/data/plan/owner 全部通过时
才能复用，否则必须 rematerialize。Successor 的 576/720-GiB 等大点使用 qualification
后的 NUMA-aware anonymous ordinary DRAM，不把 `/dev/shm` mount 当作唯一 backend。失败后
的 partial replacement 不能自动覆盖；恢复必须按 group version/lineage 幂等继续，不能把
残留 coverage 当成新 run。

## 8. 历史 M0/M1 执行账本

以下是已经完成或停留在旧 revision 的 GPU0/GPU1 路径，不是当前 formal 执行顺序：

1. 已从 H12/W2 导出最小 `WorkManifest`；
2. 已新建 pageable-DRAM source/target 与 byte-bounded grouping；
3. 已跑通 two-rank S0 canary 和 full682；
4. 已完成 QK base-entity audit、512+2,048 cohort 和 two-window input 物化；
5. 已完成 H1536/24L 两卡 `theta0→theta1` 与 held-out 正推荐检查；
6. 已生成 20.0195%-exact D1 snapshot、D2 characterizer 和 direct-old-K/V program；
7. 已物化 144-GiB complete old K/V，并完成 2,048-record、九组、288-GiB group-128 M1
   S0；
8. 已补 fair group-64、17-group S0 和同 setting 的 M1 strong S1；
9. 已完成 input-only causal probe 和 historical v1 bidirectionally segmented
   precursor；同 current stack、独立调优的 formal S2 未完成；
10. 已实现 route-specific 三段解耦、bounded-flow planner、stable interleave、plan/stack
    hash，并以 28.514442098/28.147194647 秒完成同一 exact-stack route-major/selected pair
    与 full byte parity；
11. 已补 grouped E0 与 owner-local D1-only contribution diagnostics；
12. 该 revision 停在 independently tuned formal E0/generic S2、正式重复与最小 held-out
    sensitivity 之前。

这条历史顺序的目标是尽快得到可反复使用的两卡 benchmark，而不是先完成一套可能随后被
设计推翻的正式接口。

## 9. 当前 successor 执行顺序

当前不再按“先把旧两卡 candidate 做完，再决定是否扩展”串行推进，而按 baseline-first
foundation 建设：

1. 从同一 QK role split 生成 HET primary 与 HOM matched-control manifests，冻结长度、
   valid-byte、owner-imbalance 和身份统计；
2. 先冻结与方法无关的硬件 HBM cap，再 qualification 已预声明的固定 XP
   （2,859,835 semantic rows + 1 padding row 的 2,859,836×4,096 physical FP32 table、
   E4096→H1536、24L/H1536），冻结 1/2/4-way
   placement ledger 和 model-update edge；若 qualification 失败，必须在任何 timing 前
   发布新的 benchmark identity，不能在 ladder 中按 EvoKV 结果择优；
3. 把现有 executor 参数化为 1/2/4 rank，并实现 live arena、bounded group shadow、
   validation、versioned group commit、old-group reclaim 和幂等 resume；
4. 在同一 stack 上先跑 tuned exact、S0、S1、current-stack S2 和 profile-aware generic，
   完成 timer、correctness、capacity 与 baseline qualification；
5. 在 strongest generic winner 上重新 profile 和探索 D3；isolation track 固定 D1/D2，
   co-design track 允许形成新 `stack_revision`，但必须重跑它自己的全部 baselines；
6. 只有结果准备晋升为 formal/paper evidence 时，才执行
   [Benchmark Qualification Registry](../11_benchmark_qualification.md) 中登记的具体检查，
   冻结 protocol，并开展 repeats、capacity/action/model/held-edge matrix。

这些步骤只固定输入、输出和公平边界，不锁死中间接口或 scheduler 实现。任何阶段若暴露
上游 action、owner、group 或 store 假设错误，可以回退并生成新 revision；不能用旧
baseline 解释新 stack。
