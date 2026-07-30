# EvoKV 论文实验总蓝图

日期：2026-07-30

状态：**论文级 benchmark 与 evaluation 设计，尚不构成新的结果 protocol 或论文证据**。
已有结果的有效性仍由 [eval_protocol.md](eval_protocol.md) 决定；D1/D2/D3 的当前事实与
边界仍以 [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md) 为准。本文档
冻结的是“论文要回答哪些问题、每个问题用什么规模和对照回答、总共需要多少实验”的实施
版图。正式运行前，仍需把本文中的新实验族逐一写成 versioned protocol 和 machine-readable
config。

## 1. 结论先行

论文实验不应做成

```text
所有数据集 × 所有模型 × 所有 GPU 数 × 所有机制 × 所有容量
```

的笛卡尔积。这样既不可执行，也会让每个图失去明确问题。更合适的 portfolio 是：

1. **Q-SEM：真实语义与跨数据集证据。** KuaiRand、Tenrec QB、Tenrec QK 各用三个
   coupled data-model tiers 和四个独立训练 seed，回答 reuse–recompute 矛盾与 D1 的
   质量—成本 trade-off。该部分复用已经完成且协议有效的 3×3×4 证据。
2. **R-KR：长上下文与异质 shape 证据。** KuaiRand H12 的 682-record、16L/H512
   workload 用于 D1 生命周期、source representation、D2 extent/shape ablation 和
   resident strong scaling。它不是大 embedding workload。
3. **X-QK：大 embedding 与 out-of-core 系统证据。** 已物化的 X2 与计划训练的 X1
   使用 Tenrec QK 真实用户，分别具有 16.364/10.909 GiB 全局 FP32 embedding；resident
   communication characterization 使用 9 GiB，out-of-core paper core 覆盖
   144–720 GiB 真实 old/new K/V footprint，承担 Motivation 2、Motivation 3、D2 多卡、
   D3 DRAM↔HBM 和端到端结果。

主论文新增 **66 个去重后的 formal timed cells**。每个 cell 做一次完整 correctness、
一次 untimed warmup 和五次 measured complete job，因此是 330 次 measured jobs、
462 次完整 executions。D1 已有的 36 条模型版本链与冻结系统结果不重跑；旧 W3/D3
development 数字只用于选机制和估算预算，不进入论文结果表。

主系统配置固定为 X2-QK：24L/H1536、2,859,836 行 entity embedding、T=512。两卡
2,048 records 是约 300 GiB 的主点，四卡 4,096/5,120 records 是约 600/720 GiB 的
大容量点。formal builder 必须为 5,120 条记录选择互不相同的真实 QK 用户，无需复制
trace；当前已物化的 development cohort 仍只有 2,048 条。

当前 COW/atomic-publication 边界要求 committed old 与完整 private target 同时存在。
因此论文可以诚实地宣称物理处理了最高 360 GiB 的单版本 cache、720 GiB 的事务
footprint；不能把 720 GiB 写成“720 GiB cache”，也不能把两个独立 wave 写成一个
1 TiB transaction。

## 2. 论文主张与研究问题

实验按下面六个 research questions 组织，而不是按代码模块组织。

| RQ | 要回答的问题 | 主要证据 |
|---|---|---|
| RQ1 | 模型更新后，reuse 与 exact recomputation 之间是否存在真实且可重复的质量—计算矛盾？ | M1、D1-A、D1-B |
| RQ2 | 减少逻辑 exact work 是否会自动按比例减少多卡上的物理工作？ | M2、D2-A |
| RQ3 | D2 的 owner-local execution、合并 exact pool、shape-aware lowering 和 segmented target 是否把逻辑稀疏性真正转成物理稀疏性？ | D2-A、D2-B |
| RQ4 | 当完整 cache 超过可用 HBM 后，简单顺序分组和 whole-group double buffering 还暴露什么瓶颈？ | M3、D3-A |
| RQ5 | D3 的双向分段流水与 ResidencyPlan 是否在相同 D1/D2 工作下优于强通用 baseline？ | D3-A、E1 |
| RQ6 | EvoKV 在模型规模、GPU 数、K/V 容量、exact mix 和另一个更新边上是否保持收益与正确性？ | D3-B、C1 |

三层设计的实验边界保持清楚：

- D1 评价“做什么”及其 fidelity/cost；
- D2 固定 D1 action，评价“在哪里、以什么物理形态执行”；
- D3 固定一个 stack revision，评价“超出 HBM 后如何搬运和流水”；
- 若 D3 探索迫使 D1/D2 发生实质变化，则产生新的 `stack_revision`，并在该 revision
  下重跑其 own baseline，不能跨 revision 相除。

## 3. Benchmark portfolio

### 3.1 数据集职责

| 数据集 | 原始规模与时间语义 | 论文使用量 | 证据职责 | 明确不承担的职责 |
|---|---|---|---|---|
| KuaiRand-1K | 11,713,045 rows、1,000 users、真实毫秒时间戳和 31 个日期 | Q-SEM 使用 250/500/980 users；R-KR 使用 945-user preparation 中的 682-record H12 job | 真实日历更新、D1 质量、长上下文、重复更新、shape 异质性 | 大 embedding 的 D2/D3 headline |
| Tenrec QB | 2,442,299 rows、34,240 users、130,637 raw items；只有 user 内 ordinal order | Q-SEM 使用 1,000/3,000/5,000 users | 紧凑的跨 workload 质量复现 | 大规模 D2/D3；calendar-time drift |
| Tenrec QK | 493,458,970 rows、5,022,750 users、3,753,436 raw items；只有 user 内 ordinal order | Q-SEM 使用 1,000/3,000/5,000 users；X-QK 从 25,770 个满足 576-event 边界的用户中抽取真实互异 records | 大 entity embedding、多卡 lookup、物理 out-of-core、容量与模型敏感性 | 将 ordinal window 写成“按天更新” |

QB/QK 是 Tenrec 同一 collection 的两个相关表，论文必须如实说明，不能把它们包装成两个
完全独立的数据来源。ZhihuRec 和 Taobao 的既有 negative boundary 不进入主矩阵。
R-KR 使用的是 4+12 preparation 的 945-user artifact；8+8 preparation 的 965 users
属于另一条冻结协议，不能混入这一 workload。

### 3.2 Q-SEM：质量矩阵

这一矩阵已经完成，不重训。每个 cell 有 seed 0–3 四条独立模型版本链。

| Dataset | Tier | Users | Catalog | Model | FP32 embedding | Parameters | Base one-pass input tokens / targets | Sum of one-pass update input tokens / targets |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| KuaiRand | S | 250 | 50k | 3L/H64/4×16, T=128 | 12.21 MiB | 3.269M | 732,517 / 318,090 | 298,548 / 36,204 |
| KuaiRand | M | 500 | 50k | 6L/H96/4×24, T=128 | 18.31 MiB | 5.090M | 1,062,974 / 492,711 | 575,564 / 67,956 |
| KuaiRand | L | 980 | 50k | 9L/H128/4×32, T=128 | 24.41 MiB | 7.160M | 1,286,263 / 620,958 | 932,542 / 107,863 |
| QB | S | 1,000 | 50k | 3L/H64/4×16, T=128 | 12.21 MiB | 3.268M | 64,000 / 37,494 | 960,235 / 29,596 |
| QB | M | 3,000 | 50k | 6L/H96/4×24, T=128 | 18.31 MiB | 5.090M | 190,939 / 126,610 | 2,861,489 / 94,182 |
| QB | L | 5,000 | 50k | 9L/H128/4×32, T=128 | 24.41 MiB | 7.160M | 311,572 / 187,711 | 4,651,250 / 138,235 |
| QK | S | 1,000 | 5k | 3L/H64/4×16, T=128 | 1.22 MiB | 0.388M | 58,298 / 40,074 | 514,166 / 12,767 |
| QK | M | 3,000 | 5k | 6L/H96/4×24, T=128 | 1.83 MiB | 0.770M | 169,441 / 108,580 | 1,459,626 / 33,062 |
| QK | L | 5,000 | 5k | 9L/H128/4×32, T=128 | 2.44 MiB | 1.400M | 277,409 / 170,800 | 2,376,059 / 51,617 |

表中 base tokens 是一个 base epoch 的输入 coverage，没有乘六；stream tokens 是各个
增长历史 update job 的 one-pass input coverage 之和，不是原始 stream rows，也没有乘两
个 update epochs。共享训练设置是 batch 32、theta0 六个 epochs（`3e-4`）、每次 update 两个 epochs
（`1e-4`）。KuaiRand 使用 14 个真实 base dates、11 个日更新和一个 unseen endpoint；
QB/QK 使用每用户 64-exposure base、11 个 four-exposure updates 和一个 final ordinal
window。每个 dataset 内的 tier users 是 base-activity-only nested cohorts；catalog 只由
base period 拟合。

Q-SEM 的 embedding 小于 25 MiB。它用于语义与统计复现，不用于证明 embedding
communication 或 out-of-core capacity。

### 3.3 R-KR：长上下文系统切片

R-KR 固定为已有 KuaiRand 4+12/H12 workload：

- 16 layers、hidden/K/V width 512、8×64 heads、maximum history 2,048；
- 181,082,112 total parameters，其中 entity embedding 为 312,144 个 non-padding
  entities 加一个 padding row，即 312,145×512 FP32，约 0.595 GiB；dense core
  21,263,872 parameters，约 0.079 GiB FP32；
- 682 real records、1,087,785 prefix tokens；
- 单版本 logical FP16 K/V tensor 为 35,644,538,880 bytes，约 33.1966 GiB；当前
  serialized physical extent payload 为 35,644,555,248 bytes，二者分开报告；
- frozen H12 action plan 为 548 compiled、46 scheduled exact、88 natural exact；
- complete wave 的 all-exact/mixed lookup tokens 为 934,917/347,062。

它的优点是长度和 `(suffix, retained)` shape 丰富，并且已经有 direct-old-K/V、
1/2/4-GPU resident 与 11-update lifecycle 证据。缺点是 embedding 只有 0.595 GiB，
所以 D2 的大表 headline 必须来自 X-QK。

### 3.4 X-QK：系统模型

所有 X-QK 配置使用同一 base-only entity address space：

- 2,859,835 个 base-active semantic rows，加一个 padding row；
- top-250k rows 可作为 prediction targets；
- 其余 2,609,835 rows 是 lossless context entities；
- stream-only item identity 映射到已有 context rows，不扩张 future-fitted vocabulary；
- old history 为 512 tokens；主更新 append 32 个新 exposures，target 仍为 512 tokens；
- embedding 按 row modulo 分片，dense core 在 rank 间复制。

| ID | Model | Dense parameters, excluding embedding | FP32 dense | Global FP32 embedding | FP16 K/V per record per version | Role |
|---|---:|---:|---:|---:|---:|---|
| X1 | 16L/H1024, 16×64, T=512 | 84,990,976 | 0.317 GiB | 10.909 GiB | 32 MiB | 第二个真实系统模型；model sensitivity |
| X2 | 24L/H1536, 24×64, T=512 | 285,571,584 | 1.064 GiB | 16.364 GiB | 72 MiB | 所有 headline system experiments |
| X3，可选 | 32L/H2048, 32×64, T=512 | 675,428,352 | 2.516 GiB | 21.819 GiB | 128 MiB | 核心结果稳定后的三点 model stress |

X2 已有一个 development `theta0→theta1` edge。它使用 225,737 个 base eligible targets、
13,492 个 update targets，并在 13,426 个 held-out positive targets 上得到正常的正更新
信号。正式论文不直接复用它的 development timing；它只说明这一配置是可训练、可更新、
可执行的。X1 需要训练一个短 `theta0→theta1→theta2` 链，X2 需要补一个独立
`theta1→theta2` edge。系统 scale 不要求重新训练多 seed，也不把这一个 seed 当作 D1
质量泛化证据。

大 address space 不能靠未访问的冷行制造。当前 development 2,048-record characterizer
中，all-exact/mixed 分别触达 309,467/126,588 个 unique rows；formal table 必须同时报告
address-space rows、unique touched rows/coverage、request tokens 和 off-rank response
bytes。16.364 GiB 是真实物化表容量，不自动等价于每个 wave 都扫描整张表。

formal X-QK 用户角色在第一次运行前用一个 base-only stable-hash salt 冻结，之后不能按
结果更换：

| Role | Users | Use |
|---|---:|---|
| `theta0→theta1` edge training | 2,560 | 一次 base epoch、一次 32-exposure update epoch；记录 effective targets |
| `theta1→theta2` long-user subset | 2,048 | 从 raw length≥1,024 的 2,219-user pool 预选；第二个 32-exposure update，并用 `[576,608)` 检查正更新信号 |
| D1 program fit/calibration | 512 | exact K/V target collection、shared program fitting；不进入 final timing |
| D3 route profile | 512 | baseline/profile tuning 与 service-rate estimation |
| Held-out qualification | 512 | 冻结 plan 前的合法性、prediction 与 capacity qualification |
| Final system benchmark universe | 5,632 | 取 nested prefixes 512/1,024/1,536/2,048/2,560/4,096/5,120/5,632 |

D1 fit、D3 profile、qualification 和 final benchmark 四个评价角色彼此不重叠。模型训练
用户是否与这些系统角色重叠不构成统计复现，但必须记录；优先选择完全互斥的用户，因为 QK
容量足够。25,770 只证明 576-event boundary；第二个 edge 的 `[576,608)` 推荐检查必须
单独审计上述 length≥1,024 long-user subset 在 entity mapping 后仍满足 608-event boundary。
system benchmark records 只要求其 cache history 覆盖到 target version。salt、实际 raw
user IDs、各 window 的 retained rows/targets 和交集审计都写入 formal manifest。

### 3.5 X2 K/V 容量档

| Real records | One committed cache epoch | Old + complete private target | Intended use |
|---:|---:|---:|---|
| 512 | 36 GiB | 72 GiB | D3 resident/out-of-core crossover |
| 1,024 | 72 GiB | 144 GiB | 双卡 capacity sensitivity |
| 1,536 | 108 GiB | 216 GiB | 双卡中间档 |
| 2,048 | 144 GiB | 288 GiB | 双卡 headline、1/2/4 strong scaling |
| 2,560 | 180 GiB | 360 GiB | 双卡 NUMA-local stress |
| 4,096 | 288 GiB | 576 GiB | 四卡大点 |
| 5,120 | 360 GiB | 720 GiB | 四卡最大 paper-core 点 |
| 5,632，可选 | 396 GiB | 792 GiB | 独占机器下的最大 physical stress |

formal cohort 从 base-only stable-hash order 中抽取，先排除 fit/profile/qualification users，
再取 nested prefixes。formal builder 必须为直到 5,632 records 的各档选择真实且互不相同
的 QK 用户；当前 machine-readable cohort 只物化到 2,048 benchmark users。第一轮
paper-core 禁止复制用户；因此也不存在跨副本 dedup 人为放大收益的问题。

若未来真的需要 trace multiplication，每个 replica 必须有独立 cache ID、owner 和物理
extent，完整支付 source/compute/target bytes，禁止跨 replica dedup，并标为 synthetic
capacity workload。它不能进入质量统计，也不能把 replica 数写成真实 dataset users。

## 4. 硬件与统一资源边界

### 4.1 GPU 与拓扑

- 4×NVIDIA A40；`nvidia-smi` 标称 46,068 MiB，即 44.988 GiB，当前 Torch/CUDA
  allocatable total 为 47,699,722,240 bytes，即约 44.424 GiB；
- formal admission cap 统一为 **38 GiB/rank**，相对 CUDA-visible total 保留约
  6.42 GiB 给 CUDA context、NCCL、allocator fragmentation 和不可见 workspace；
- GPU0/1 在 NUMA 0，彼此为 NV4；
- GPU2/3 在 NUMA 1，彼此为 NV4；
- 两个 pair 之间为 `SYS`；
- 两卡主实验固定 GPU0/1；四卡 rank 0/1 first-touch NUMA 0，rank 2/3
  first-touch NUMA 1；
- formal 四卡运行只在四卡均空闲、机器独占时启动；不能终止或挤占外部 GPU 进程来完成
  matrix。

X2 direct-old-K/V program 约 0.422 GiB。按当前 EvoKV 实现，X2 的 HBM 账本为：

| GPUs | Embedding shard / rank | Dense + program + embedding fixed / rank | Dynamic budget under 38 GiB cap |
|---:|---:|---:|---:|
| 1 | 16.364 GiB | 17.850 GiB | 20.150 GiB |
| 2 | 8.182 GiB | 9.668 GiB | 28.332 GiB |
| 4 | 4.091 GiB | 5.577 GiB | 32.423 GiB |

当前 X2 development checkpoint 只有 world-size-2 的两个 modulo shards。正式 1/2/4-GPU
矩阵开始前，必须从同一 canonical global embedding 生成并验证 1/2/4-shard artifacts，
记录相同 global-table digest，并检查 reshard 前后的 lookup 输出一致；不能把现有两 shard
文件直接当成一卡或四卡 checkpoint。

R-KR/2-GPU strict-COW resident 点是一个必须先通过的硬 preflight：old+private target
本身每 rank 约占 33.2 GiB，加上 fixed state 后只剩约 4.4 GiB transient headroom。
preflight 必须运行完整 payload 与峰值 workspace，而不是只做静态 byte arithmetic。
当前 66-cell 预算以它通过为前提；若 exact 或 complete D2 任一侧超过 38 GiB/rank，
则只能在冻结正式协议前删除这两个双卡 R-KR cells、重算总数并保留四卡点，不能回退成
非 strict-COW 路径后仍把它画在同一 resident panel。

表中的 fixed bytes 包含 EvoKV program；exact baseline 不被迫加载无用 program，而是报告
自己的 fixed bytes。统一报告 per-rank 物理压力：

$$
\rho_{\mathrm{KV}} =
\max_r
\frac{B_{\mathrm{old},r}+B_{\mathrm{private\ target},r}}
{38\ \mathrm{GiB}-B_{\mathrm{fixed},r}}.
$$

X2/2,048 records 的 1/2/4-GPU \(\rho_{\mathrm{KV}}\) 分别约为
14.29/5.08/2.22；X2/5,120 records 的四卡压力约为 5.55。这样比仅报告“分了多少组”
更能解释 D3 operating region。这些数值是等长 records 与理想均衡 owner 下的近似；
formal artifact 使用实际 per-rank bytes 计算 max-rank value。该 \(\rho_{\mathrm{KV}}\)
只描述 mixed COW pressure，exact 另报 raw-history source 与 target-working-set pressure。

### 4.2 Host DRAM

机器有两个约 504 GiB 的 NUMA node。formal run 必须：

- 在独占机器上检查 total `MemAvailable` 与 per-node free memory；
- source 与 target 按 rank 的 GPU NUMA first-touch；
- primary timer 前 reserve backing，并用 rank-specific non-payload sentinel 对 committed
  source 与 private-target 的每个物理页做 dirty write-touch；只 read-touch、
  `MAP_POPULATE` 或 `mincore` 不能排除 Linux shared-zero-page/overcommit 假象；
- source old K/V 在 timer 前完整物化并保持 immutable；private target 只能预写 sentinel，
  真实 target payload 必须在 timer 内完整覆盖；
- 记录 total/per-node RSS、`numa_maps`、major faults 和 swap-in/out；formal timer 内
  major-fault 与 swap 增量必须全部为零；
- 报告普通 DRAM、pinned DRAM、HBM allocated/reserved 的 standing 与 peak bytes。

当前 `/dev/shm` 只有约 504 GiB，因此 576/720/792 GiB 点不能继续使用现有单一
`/dev/shm` file backend。它们需要 NUMA-aware anonymous pageable-DRAM arena，或明确扩容
且按 NUMA 放置的 tmpfs。不能用 `/data` 上的 NVMe 文件冒充 ordinary-DRAM result。
materialization 在 primary timer 外，但真实 physical pages、coverage、bytes 和 checksum
必须存在并被记录。

### 4.3 1 TiB 边界

在当前事务语义下：

- 720 GiB 是 360 GiB old cache 与 360 GiB private target 的共存 footprint；
- 792 GiB 是 396+396 GiB，已经是当前机器的最大合理 physical stress；
- 一个 800 GiB 的单版本 cache 需要约 1.6 TiB DRAM 才能维持当前 COW boundary；
- 把容量拆成两个 wave 并在中间释放，只能称为 across-two-transactions 的 cumulative
  source-plus-private-target footprint，不能称为一个 1 TiB cache、transaction 或实际
  transfer volume；
- generator、sparse file 或未物化 logical payload 不能作为 DRAM capacity evidence。

若以后要越过这个边界，只能取得更大 DRAM 机器，或者把 D3 改成可验证的
incremental reclaim/segmented commit。后者改变事务机制，必须重新定义 protocol 和 baseline，
不能作为一个隐藏的实验技巧。

## 5. 统一计时、统计和公平性规则

### 5.1 正式 cell

每个新的 timed cell 固定：

1. 一次 complete-workload correctness pass；
2. 一次 untimed complete-workload warmup；
3. 五次 measured complete jobs。

五个 timing samples 用于描述 runtime variation，不是五个模型样本。报告 median、五个原始
样本或 min–max，并对同一 revision 的方法做 paired comparison。质量统计仍以 training
seed 为 replication unit。

每个 cell bundle 只物化一份 immutable old epoch 和一块可复用 private-target arena。
correctness、warmup、五次 measured job 以及 paired methods 都从同一个 old manifest 和
相同 source checksum 开始。每次 timer 内的 commit 发布 target，并把 old extents
**逻辑地**归还给 arena allocator；timer 结束后，benchmark harness 在 timer 外恢复 old
manifest、逻辑释放 candidate target，再让下一次 job 完整覆盖同一 target pages。
因此即使在 720 GiB 点也不需要第二份 old cache。

这一规则把 hot-path reclaim 明确定义为 manifest/allocator transition，而不是 OS
`madvise`/page decommit。source pages 为了可重复输入而继续物理驻留，但不能被 candidate
修改。reset 时间必须单独记录且对 paired methods 相同。如果某个实现选择真的 decommit
old pages，就必须在每个 job 前重新物化相同 old epoch，并把 decommit/reset/
rematerialization 计入另报的 cold-inclusive 成本；两种 reset 语义不能混在同一 panel。

### 5.2 Primary timers

resident 与 out-of-core 使用两个明确、不可混算的 timer。

**Resident timer。** HBM-resident old K/V、raw IDs/history、model、embedding shards 和
compiled program 已就绪。timer 包含当前 wave 尚未持久化而必须现场完成的 lowering、
compute/collective、target construction、consumer-ready setup、coverage/lineage validation、
commit 和 reclaim。只有已经合法序列化、绑定当前 stack 且可复用的 WavePlan 才可把
lowering 移到 execution timer 外。

**Out-of-core timer。** ordinary-DRAM source 到 consumer-ready ordinary-DRAM target，包含：

- job 内的逻辑 extent allocation、ownership 和 ledger setup；
- source extent 读取到 bounded pinned staging；
- pageable→pinned packing、H2D；
- D2 compute 与 embedding collective；
- D2H、pinned→ordinary-DRAM publication；
- segmented/contiguous consumer-ready setup；
- coverage/lineage validation、manifest publication、commit 与 reclaim。

为保证 formal timer 内没有 page faults，物理 backing arena 的 reservation 与逐页 dirty
write-touch 对所有方法对称地放在 timer 外；真实 target payload 的写入必须完整发生在
timer 内。

以下单独报告，不塞入 execution-only timer：

- 模型训练和 checkpoint materialization；
- source old-K/V 的第一次生成；
- D1 fit 与 program compilation；
- 已持久化 WavePlan 的一次性生成；
- D3 route profile 与 ResidencyPlan construction。

同时必须给出 execution-only、plan-inclusive single-wave time 和对应 break-even/reuse
count，防止 offline cost 被完全隐藏。HBM resident 与 ordinary-DRAM endpoint 分成两个
panel，不能把二者的 raw time 串成一个虚假 waterfall。

`cold-allocation-inclusive` 数字定义为 arena reservation、dirty first-touch 与第一次
execution 的和；source old-K/V 的首次生成仍作为独立 preparation cost 报告。重复实验的
timer-external manifest reset 也单列，不能藏进 warmup。这样同时保留生产 hot path、
冷启动成本和 benchmark 可重复性三个不同边界。

### 5.3 Baseline tuning

每个 baseline 在相同 38 GiB/rank cap 下独立调优，不能让 EvoKV 使用更大的 group 或更多
buffer。调优只用独立 profile users：

- 先做 legality/correctness 与 capacity preflight；
- 候选空间按机制最多保留约 12 个有意义点，不做无边界 Cartesian product；
- 每点一次 screen，top-3 做三次复测；
- 根据 median 选定，tie 时优先低 HBM、低 pinned bytes、低 padding；
- 选定后冻结到 qualification 和 final benchmark。

具体而言，D2 的 exact denominator 在 one-shot/two-stage candidates 中独立选最快者；
D3 的 exact denominator 在合法 one-shot/S0/S1 中独立选最快者，action-oblivious mixed
baseline 也在 S0/S1 中独立选择。M3 为解释机制而保留 exact S0 与 S1 两行，但 E1 使用
二者更快值。不能因为 development 中 S1 曾获胜，就在所有容量上预设 double buffering
一定最快。

baseline 必须使用同一 source/target tier、GPU topology 和 timer。不同算法天然读取不同
source representation 时，分别报告其 logical/physical bytes；不能通过假装 source bytes
相同制造公平，也不能忽略多出来的状态。

### 5.4 Append accounting

所有 D2/M2 结果同时报告：

1. retained-prefix maintenance，不含 method-common append；
2. complete wave，包含 append。

compiled retained repair 可以是 zero lookup/zero embedding collective；compiled records
的新 token append 仍然必须查 embedding。任何“embedding-free”措辞都只能限定在
retained-prefix phase。

## 6. 具体实验矩阵

### M1：Reuse–Recompute dilemma

**状态：复用已有证据，不新增 timed cells。**

| Item | Configuration |
|---|---|
| RQ | streaming update 有价值时，stale reuse 是否仍留下可恢复的 version-consistency gap？ |
| Data/model | Q-SEM 的 3 datasets × 3 tiers × 4 seeds，共 36 条模型版本链 |
| Baselines | frozen model、current-model stale reuse、current-model exact recomputation |
| Metrics | BestRank/MeanRank、rank utility、NDCG/Hit、streaming value、reuse value、maintenance gap、staleness tax |
| Output | 一张 3×3 grouped heatmap；完整 per-seed 数值放 appendix |

它支持“问题存在且与 workload/regime 有关”。它不支持“模型越大 gap 必然越大”、
“exact 是 ranking quality 上界”或“用 task labels 路由 cache”。

### M2：Logical sparsity is not physical sparsity

**状态：新必跑，14 个 timed cells。**

固定 X2-QK 的 64 个真实、互异 records。这个规模的 mixed old+target 为 9 GiB，在
replicated/local embedding control 下也能通过 38 GiB/rank resident preflight。这个实验
在介绍 D1 前构造一个预声明的
partial-exact workload，不使用 D1 program 或 D1 输出挑选 exact records：被选中的记录
exact replay retained prefix，未选中的记录保留原 K/V；两者都执行相同的 32-token append。
因为每条 retained length 相同，用 stable hash 加 token strata 固定
目标 0/20/50/100% retained-exact fractions，对应 0/13/32/64 exact records；图表使用
实际 0/20.3125/50/100% 数值。
所有 M2 rows 使用 resident timer，不经过 ordinary-DRAM capacity grouping，避免把 D3
prefetch/writeback 混入 Motivation 2。

| Placement/control | GPUs | Exact fractions | Cells |
|---|---:|---:|---:|
| Row-sharded embedding | 2, 4 | 0%, 20%, 50%, 100% | 8 |
| Local embedding control | 1 | 20%, 100% | 2 |
| Replicated embedding control | 2 | 20%, 100% | 2 |
| Replicated embedding control | 4 | 20%, 100% | 2 |
| Total |  |  | **14** |

必报指标：

- exact/non-exact records、retained/append/complete lookup tokens；
- address-space rows、requested/unique/local/remote IDs、unique-row coverage；
- routed-ID bytes 与 returned-vector bytes；
- collective calls/time、padding、rank wait、GPU busy time；
- retained-only time 和 complete-wave makespan。

主图用两 panel：

1. logical exact fraction 对 physical lookup/remote-vector bytes；
2. logical exact fraction 对 complete-wave time 与 phase breakdown。

这张图只证明“减少 exact work 不会自动线性变成多卡物理收益”。它不测 D1 speedup，也不在
没有 breakdown 支持时声称 communication dominates。

### M3：HBM boundary 与顺序分组瓶颈

**状态：新增两个 exact S0 cells；相同规模的 exact S1 cells 计入 D3 capacity matrix。**

| Model/data | GPUs | Records | Host COW / exact target payload | Methods |
|---|---:|---:|---:|---|
| X2-QK | 2 | 1,024 | 144 / 72 GiB | exact S0、exact S1 |
| X2-QK | 2 | 2,048 | 288 / 144 GiB | exact S0、exact S1 |

顺序 baseline 是完整的 capacity group：

```text
read/pack group → H2D → distributed exact → D2H → publish → next group
```

报告 ordinary-DRAM pack/publish、H2D、D2H、GPU、collective、input-boundary wait、
output-credit wait 和 peak HBM。GPU/collective-only phase sum只是诊断下界，不是同 endpoint
speedup denominator。

这个实验用于说明 out-of-core execution 是独立且真实的问题，以及 whole-group buffering
之后仍可能有 residual bubble；它不预先证明 D3 有效。
在这两个规模上，S0/S1 都是 formal rows，E1 denominator 使用二者中更快者；不能预设
double buffering 必胜。其他 capacity/GPU 点的 exact baseline 也要独立选择合法
one-shot/S0/S1 中的最快实现。

### E1：端到端性能

端到端主图使用两个资源边界清楚的 panel。

#### Resident panel

1. X2-QK 每 rank 64 records 的 weak scaling，1/2/4 GPU：
   fastest exact、naive fixed-action mixed、complete D2；九个点与 D2-B 共用。
2. R-KR 682 records 的 resident comparison，2/4 GPU：
   exact 与 complete D2；四个点与 D2-A/B 共用。完整 old 与 private target 在
   38 GiB cap 下不能形成 1-GPU strict-COW resident point，因此不伪造该点。

#### Out-of-core panel

X2/2,048 records/2 GPU 的四行来自 D3-A：

1. independently tuned all-exact，在合法 S0/S1 中取更快者；
2. fixed D1 actions + naive physical lowering，在 action-oblivious S0/S1 中取更快者；
3. D1+D2，在 action-oblivious S0/S1 中取更快者；
4. D1+D2+D3。

X1/4,096 records/2 GPU 的三行来自 D3-B model sensitivity：

1. independently tuned exact；
2. tuned action-oblivious D1+D2；
3. full EvoKV。

报告 absolute makespan、records/s、valid tokens/s、speedup、peak HBM、pinned/DRAM bytes、
phase breakdown、validation/commit/reclaim。主张是每层在自己的资源边界上贡献什么，而不是
强迫所有 bar 单调下降。如果 D1 的 logical plan 在 naïve physical execution 下不比 exact
快，这正是 D2 motivation，不应删除该结果。

### D1-A：Cost–fidelity frontier

**状态：复用 Q-SEM 已完成证据。**

| Dimension | Configuration |
|---|---|
| Data | KuaiRand/QB/QK × S/M/L × seeds 0–3 |
| Baselines | stale reuse、current projection、recent-token/closest selective baseline、progressive replay、compiled repair、exact |
| Metrics | measured GPU cost/full-exact、K/V recovery、score recovery、Top-k overlap、paired ranking delta |
| Output | 主文 aggregate Pareto；九个完整 cell 放 appendix |

主张是 shared affine repair 在多个 workload/capacity 上形成稳定的 cost/fidelity interior
point。严格 task-quality gate 只通过 6/9 cells 的事实必须保留；不能挑掉 negative endpoint，
也不能把 task quality 变成 cache admission oracle。

### D1-B：Source representation 与 repeated renewal

**状态：复用 R-KR frozen 结果。**

- source path：normalized capsule、direct-old-K/V、exact；
- source/operator：1/2/4 GPU，报告 program bytes、compile time、source bytes、preload 和
  break-even；
- lifecycle：682-record cohort 上 11 个连续 updates，下一步真正读取上一步输出；
- control：all-exact、bounded exact renewal、被拒绝的 threshold refresh；
- metrics：per-edge/cumulative GPU cost、exact fraction、migration depth、最低
  cache/score/top-100 fidelity、lineage。

主图可将 source-path bar 与 11-edge lifecycle curve 合并成两个 panel。它不宣称 organic
serving trace、在线最优 selector 或跨数据集 lifecycle generality。

### D2-A：Mechanism ablation

**状态：新必跑，R-KR/4 GPU 共 6 个 cells。**

所有 mixed variants 固定同一个 ActionPlan hash、owner map、embedding placement、
source/target endpoint 和 timer：

1. independently tuned all-exact；
2. naïve row-sharded fixed-action mixed；
3. owner-local retained repair，但仍写 contiguous target；
4. 再合并 `scheduled_exact` 与 `natural_exact` 的 physical exact pool，同时保留不同
   provenance；
5. 再加入 `(suffix, retained)` shape-aware lowering；
6. 再使用 segmented suffix-only target，形成 complete D2。

atomic publication 是正确性语义，不包装成一个性能优化 bar。必报：

- retained lookup 是否为零、old-K/V peer bytes；
- total lookup/remote-vector bytes；
- retained rewrite bytes；
- padding、collective count/exposed time；
- temporary HBM、segmented consumer time、complete wave time。

主图使用 cumulative mechanism bar，加 traffic/collective breakdown。R-KR 负责 shape
heterogeneity；大 embedding 与 shard-count 结论由下面 X2 scaling 提供。
这些 bar 只解释该累积顺序下的 marginal effect；机制存在交互，不能把每段差值写成彼此
独立、可任意相加的贡献。

### D2-B：1/2/4-GPU scaling

**状态：新必跑；与 D2-A 合计 17 个新 D2 timed cells。**

1. X2 weak scaling：每 rank 64 real records，exact/naïve mixed/complete D2，
   1/2/4 GPU，共九个 cells。
2. R-KR resident comparison：固定 682 records，2-GPU exact/complete D2 新增两个
   cells；4-GPU exact/complete D2 复用 D2-A 两行。R-KR 不做不可容纳的 1-GPU
   strict-COW resident point。

报告 max-rank makespan、records/tokens per second、strong/weak scaling efficiency、
off-rank vector bytes、collective count、rank imbalance 和 HBM。

X2 weak-scaling cohort 在 ActionPlan 冻结后按 action 与 extent strata 分配到 rank，确保
1/2/4 卡的 exact-token fraction 可比；不能让 owner hash 偶然改变 action mix。

X2 的 16.364 GiB embedding 本身仍能放入一张 A40，dense model 也能放入单卡。因此论文
不能声称“模型本身无法放进单卡”。可支持的结论是：在真实 row-sharded placement 和受限
K/V working-set HBM 下，D2 减少了 fixed-action workload 的物理通信、padding 与 rewrite。

### D3-A：Causal mechanism chain

**状态：新必跑，X2/2,048 records/2 GPU 共 8 个 cells。**

所有 mixed rows 使用同一 action/source/owner/target/timer：

1. all-exact S1 control；同规模 all-exact S0 由 M3 提供，E1 denominator 取两者更快者；
2. D1-only naïve lowering，在 action-oblivious S0/S1 中取更快者；
3. D1+D2 sequential groups；
4. D1+D2 whole-group two-slot；
5. input segmentation；
6. bidirectional input/output segmentation；
7. decoupled I/C/O granularity、route-major order；
8. selected ResidencyPlan。

报告 makespan、input-boundary wait、output-credit wait、CPU packing、H2D/D2H、publish、
GPU/collective、HBM/pinned footprint、planner prediction error。这个 causal chain 要分别
显示：

- generic whole-group overlap 带来多少；
- inner bidirectional pipeline 如何移动并消除 bottleneck；
- 最后的 cross-route coordination 还有多少增量。

如果最终 route-specific triples 仍都选择 `(8,8,8)`，就只能声称 planner 可表达并安全执行
不同 granularity，不能声称 asymmetric granularity 已带来收益。D3 的核心 bar 是完整
hierarchical pipeline 相对 strong S1，而不是单独夸大当前约 1% 的 order-only improvement。

### D3-B：Capacity、GPU、action mix、model 与 edge sensitivity

**状态：新必跑。与 D3-A 合计 33 个新 D3/E2E timed cells。**

#### Capacity

| GPUs | X2 records | Host COW footprint | Exact target payload | Methods |
|---:|---:|---:|---:|---|
| 2 | 1,024, 2,048 | 144, 288 GiB | 72, 144 GiB | tuned exact、tuned action-oblivious D1+D2、D3 |
| 4 | 2,048, 4,096, 5,120 | 288, 576, 720 GiB | 144, 288, 360 GiB | tuned exact、tuned action-oblivious D1+D2、D3 |

2-GPU/2,048 的三行复用 D3-A；其余是 12 个新 cells。5,120-record 点必须使用两 NUMA
node 的本地 ordinary-DRAM arena，不能落到 `/data`。
在 2-GPU/1,024 和 2,048 两点，capacity matrix 的 exact row 对应 formal S1，M3 另计
formal S0；最终 exact denominator 取两者更快值。其他点先独立调优 exact one-shot/S0/S1，
只冻结赢家进入 formal matrix。

表中的 COW footprint 对所有方法都保留：即使 all-exact 不读取 old K/V，committed old
epoch 在新 target 原子发布前仍然 authoritative，不能为了省 host DRAM 提前释放。
区别在于 all-exact 只读取 raw IDs/metadata 并写 target，而 mixed path 还读取所需 old-K/V
extents；这一本来就是两种算法的真实 source-state 差异。二者共享 source tier、target
endpoint、records、owner/topology 和完整 timer，但分别报告实际 source-read/H2D/D2H/
published bytes，不能为了“字节相同”给 exact 人为添加无用 old-K/V I/O。

#### Fixed-work GPU scaling

固定 X2/2,048 records，比较 1/2/4 GPU 的 tuned exact、tuned action-oblivious D1+D2、
D3。2/4-GPU 点从
capacity matrix 复用，只新增 1-GPU 三个 cells。

这张图是 strong scaling，同时 \(\rho_{\mathrm{KV}}\) 会从 14.29 降到 2.22，论文必须报告
这一变化。若需要 capacity-matched supporting plot，可选用：

| GPUs | X2 records | Transaction | Approx. \(\rho_{\mathrm{KV}}\) |
|---:|---:|---:|---:|
| 1 | 768 | 108 GiB | 5.36 |
| 2 | 2,048 | 288 GiB | 5.08 |
| 4 | 4,608 | 648 GiB | 5.00 |

capacity-matched plot 属于 supporting extension，不进入 66-cell paper core。

#### Action mix

X2/2,048 records/2 GPU 使用 label-free、predeclared retained-token budgets 生成 10/20/40%
exact plans，目标 record counts 为 205/410/819，并报告实际 record 与 token fractions。
20%/410 records 是主配置。all-exact 是共同 reference。
只新增 10%/40% 下的 tuned action-oblivious D1+D2 和 D3，共四个 cells。

#### Model sensitivity

X1/4,096 records/2 GPU 的 transaction footprint 是 256 GiB，与 X2 主点的 288 GiB 接近。
训练并冻结 X1 自己的 model edge、D1 program 和 ActionPlan，运行 tuned exact、
tuned action-oblivious D1+D2、D3 三个 cells。不能复用 X2 的 program 或 action plan。

#### Held-out update edge

- route profiles 来自与 benchmark users 不重叠的 512-user calibration pool；
- qualification users、final benchmark users 与 calibration users互斥；
- X2 增加一个 `theta1→theta2` edge；
- 在该 edge 上重新生成 edge-specific joint profiles 与 ResidencyPlan，再运行 tuned exact、
  tuned action-oblivious D1+D2、D3，共三个 cells。held-out edge 测的是机制在另一更新边
  的可重复性，不是把 `theta0→theta1` plan 直接跨 checkpoint 复用。

这组图回答 operating region 和 robustness，不支持 SSD、数据库、host-DRAM
oversubscription、online hotness 或 serving SLO。

### C1：Correctness、transaction 与 overhead

**状态：必做，但不额外计入 timed-cell 数。**

每个正式 cell 的 correctness pass 覆盖其完整 workload。总测试集还必须覆盖：

- direct-old-K/V 对 reference operator；
- replicated 对 row-sharded exact；
- segmented consumer 对 contiguous reference；
- S0/S1/D3 完整 payload 等价；
- 1/2/4 ranks、uneven rank、zero-delta 和 empty collective participation；
- 超过 \(2^{31}\) flattened offset 的 64-bit addressing；
- complete/exactly-once coverage、lineage 和 checksum；
- capacity/preflight mismatch；
- mid-H2D、mid-compute、partial D2H、missing/duplicate output、pre-commit abort；
- 所有失败下 old manifest 仍 authoritative，不能出现 mixed epoch。

在 720/792 GiB 点不能同时保留 old、reference target 和 candidate target。正确流程是：
先保留 committed old，运行 reference 并保存 per-extent digest/metadata，释放 reference
private target，按第 5.1 节的 timer-external reset 恢复同一 old manifest，再运行
candidate；任意时刻最多存在一个完整 private target。digest 比较、coverage 和抽样数值
oracle 仍必须覆盖完整 record universe，不能因内存不足改成只跑 partial workload。

overhead table 报告：

- source/target arena reservation 与 dirty first-touch time，以及 cold-allocation-inclusive
  job time；
- D1 fit/compile time 与 program bytes；
- D2 lowering/WavePlan time 与 bytes；
- D3 joint profile、planning time与 plan bytes；
- execution-only 与 plan-inclusive single-wave time；
- segmented-consumer cost；
- validation、commit、reclaim；
- HBM、pinned、ordinary-DRAM standing/transient bytes。

## 7. 精确运行预算

### 7.1 复用、不重跑

| Evidence | Existing units | Action |
|---|---:|---|
| Q-SEM motivation | 3 datasets × 3 tiers × 4 seeds = 36 model-version chains | 直接按现有 protocol 汇总 |
| D1 cost/fidelity | 9 cells 的 discovery + frozen-seed replication | 直接复用；不因系统矩阵重训 |
| R-KR source/operator/lifecycle | frozen single-configuration artifacts | 复用有效结果 |
| D2 W3、D3 M0/M1 development | 若干单次 development profiles | 只用于选机制和预算；不进入 paper table |

### 7.2 新 paper-core timed cells

| Family | New unique cells |
|---|---:|
| M2 logical-to-physical | 14 |
| M3 extra exact-S0 points | 2 |
| D2 resident ablation and 1/2/4 strong/weak scaling | 17 |
| D3 causal、E2E、capacity、GPU/action/model/edge sensitivity | 33 |
| **Total** | **66** |

每个 cell 为一次 correctness、一次 warmup、五次 measured：

- 66 correctness jobs；
- 66 warmup jobs；
- 330 measured jobs；
- **这 66 个 timed-cell bundles 共 462 complete executions**。

调优/profile runs 不算 paper cells。它们使用小的 disjoint profile cohort，且每个 baseline
有明确候选上限。C1 failure injection、额外 functional tests、训练、profile 和
materialization 都在 462 之外，因此 462 不是整个项目的总命令数。

### 7.3 时间和资源预估

现有 X2 development full job 约为几十秒量级。formal matrix 的合理规划是：

- X1/X2 新 edge 与 artifact preparation：数十分钟到数小时；
- profile/tuning：约 4–8 个独占 node-hours；
- 462 次 formal execution：约 6–12 个独占 node-hours；
- 576/720 GiB arena first-touch、materialization、checksum 与失败重跑：额外约
  4–8 个 node-hours；
- 整体预留 **2–3 个独占机器日**，分实验族 checkpoint，不作为一个长达数天的单命令。

单次 primary timed job 预期仍是秒到数分钟；最可能超过半小时的是最大 DRAM store 的首次
materialization/verification，而不是一次 measured wave。任何明显超出预算的 cell 先检查
page fault、swap、NUMA placement、allocator 或 store backend，不能直接把异常时间当作系统
现象。

## 8. 论文图表映射

主文尽量控制为六张复合图和两张表：

| Paper object | Content |
|---|---|
| Table 1 | datasets、quality/system models、embedding、dense parameters、K/V footprint、hardware |
| Figure 1 | M1 reuse–recompute、M2 logical-to-physical、M3 HBM boundary 三个 motivation panels |
| Figure 2 | E1 resident 与 out-of-core end-to-end 两个 panels |
| Figure 3 | D1 cost–fidelity frontier 与 11-edge bounded renewal |
| Figure 4 | D2 mechanism ablation 与 1/2/4 strong/weak scaling |
| Figure 5 | D3 causal mechanism chain 与 wait/bubble breakdown |
| Figure 6 | capacity、GPU、action mix、X1/X2/held-edge operating region |
| Table 2 | planning/compile/transaction overhead、memory footprint、correctness/failure coverage |

完整 3×3×4 quality cells、所有 raw timing samples、额外 metrics、可选 topology/XL stress 放
appendix，避免主文变成结果目录。

## 9. 结果不理想时如何回头

实验矩阵不是为了“证明已经决定好的结论”，而是为了暴露设计真正有效的范围。

- 如果 M2 bytes 随 exact fraction 下降但 time 几乎不变，先用 rank wait、padding、
  collective calls 和 GPU compute 找到兑换失败的位置；D2 不得仅凭 byte model 进入论文。
- 如果 D2 只在 R-KR 有效而 X2 scaling 无收益，检查 large embedding 下的 collective
  fragmentation、owner balance 和 exact pool；必要时回到 D2 lowering，而不是扩大数据
  掩盖问题。
- 如果 D3 只胜 sequential S0、不胜 independently tuned S1，它只是实现路径，不是第三个
  design。优先研究 DMA/affine/collective interference、phase-aware pacing 和
  collective-arrival-aware planning。
- 如果 D3 在 288 GiB 有效、在 576/720 GiB 消失，先检查 NUMA/locality 与 publication
  serialization；这正是 capacity plot 要揭示的 operating boundary。
- 如果 X1/X2 方向冲突，不把两个点平均；分别报告 compute-bound 与 movement-bound
  regime，并据此收窄系统 claim。
- 如果最大点因机器当前占用或 backend 失败，不能用 logical cap、稀疏文件或未触页内存
  替代。释放独占资源、实现正确 DRAM arena，或把该点诚实降为未完成 extension。

## 10. 核心完成条件与可选扩展

paper-core 完成意味着：

1. 66 个 timed cells 全部具备相同 revision 内的 correctness/warmup/5 repeats；
2. M2 明确给出 logical tokens、physical bytes 与 wall time；
3. D2 同时有 R-KR shape ablation 和 X2 large-embedding 1/2/4 scaling；
4. D3 同时有 strong S1、causal chain、288/576/720 GiB capacity 和 held-out edge；
5. full target 可被 segmented consumer 读取并进入下一 wave；
6. plan、publication、commit、reclaim 与 failure semantics 都在 overhead/correctness 表中
   闭合；
7. 所有 paper claim 都能指向一个同边界、同 revision、可复现的 result family。

核心稳定后才考虑：

- X2/5,632 records/4 GPU：396 GiB cache、792 GiB transaction，三方法共三个 cells；
- X3 32L/H2048：exact、D1+D2、D3 三个 cells；
- GPU0+GPU2 的跨 NUMA/SYS topology sensitivity；
- \(\rho_{\mathrm{KV}}\approx5\) 的 capacity-matched 1/2/4-GPU plot；
- 两个独立 4,096-record waves 的 long soak，只报告 across-two-transactions 累计
  1.125 TiB source-plus-private-target footprint；实际 DRAM read/write traffic 另报；
- synthetic embedding contention，仅作为 resource characterization，不叫 serving
  workload。

下一步不是直接启动 462 次 execution，而是依次生成：

1. formal QK cohort split 与 nested workload manifests；
2. X1/X2 model-edge、program 与 immutable ActionPlan artifacts；
3. NUMA-aware DRAM arena 和 38 GiB/rank preflight；
4. 四个新 protocol/config families：M2、D2、D3/E2E、correctness/transaction；
5. 先各跑一个两卡主点的 exact/naïve/full canary，确认 timer 与 bytes，再展开完整矩阵。
