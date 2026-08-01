# EvoKV 论文实验总蓝图

日期：2026-07-31

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
3. **X-QK：强制分片 embedding 与 out-of-core 系统证据。** 已物化的 X2 保留为两卡
   development/quality bridge；正式 paper-scale 配置 XP 使用同一 QK base-only 语义实体与
   HSTU core，固定 2,859,836×4,096 FP32 item table（43.638 GiB）和
   `E4096→H1536` owner-side projection。Table + dense/projection core 已约
   44.725 GiB，超过当前 Torch 可分配的单卡 44.424 GiB，尚未计 program、NCCL context
   或 workspace，因此完整 fixed state 物理上不能在一张 A40 复制。一次与性能无关的
   capacity/trainability qualification 只验证该 geometry；若失败，必须在任何 EvoKV
   timing 前显式修订 benchmark identity，不能从结果中换模型。主 out-of-core workload 覆盖
   144–720 GiB 的**单版本有效 K/V**，承担 Motivation 2、Motivation 3、D2 多卡、D3
   DRAM↔HBM 和端到端结果。

旧 fixed-length/full-private-target 蓝图的精确预算是 64–66 cells；该数字现已退役。
新的 HET/rolling/XP portfolio 预计在 **70–80 个 formal timed cells**，每个 cell 仍是一
次 correctness、一次 warmup 和五次 measured jobs，即约 490–560 次完整 executions。
精确去重数必须在 XP qualification、HET byte manifests、capacity-admitted rank points 和
strongest generic winners 冻结后重算，不能为了维持旧数字删除强 baseline。D1 已有的
36 条模型版本链不重训；旧 W3/D3 development 数字只用于选机制和预算，不进入论文结果表。

执行采用 **per-layer baseline first**，但不能机械地先跑完所有 baseline 再碰任何
proposed，因为 D3 的 mixed baseline 必须建立在已经冻结的 complete D2 stack 上。正确顺序
是先冻结 D2 foundation，再完成 D2 并冻结 stack，随后冻结 D3 foundation，最后才运行 D3
proposed。每一层都先验证数据、模型、分片、DRAM/HBM、timer、consumer endpoint 和独立
数值 oracle。调优 screen、Benchmark Qualification、D1 同 SLA comparator pass 和已有 D1
证据不计入 system timed cells，但必须保留全部配置与结果，不能只公开赢家。

主系统场景从一开始就承认 K/V extent 的自然异质性。`X-QK-HET` 是所有 headline
system experiments 的主 workload：它保留每条真实记录在更新边上的 old、retained、
evicted、append 和 target 有效长度。`X-QK-HOM` 对**同一批 HET records**使用 masked
max-context physical layout：不增加真实事件，不改变有效 history、D1 action 或质量语义，
只把不足 512 的 slots 物理 padding 成统一 shape。它用于 Motivation/ablation 的因果控制，
不能与 HET 二选一挑选较好结果。Q-SEM 本来就尊重真实 sequence length，R-KR 也已有丰富的
`(suffix, retained)` shape；因此 variable-length 不是 D3 临时引入的新场景，大容量只是让
tail、rank imbalance 和 pipeline interference 更明显。

正式 D3 采用 bounded rolling replacement，而不是完整 old epoch 与完整 private target
并存。每个 capacity group 完成
`stage → compute → writeback → validate → group commit → old-group reclaim` 后即可复用
旧组页面；每组始终有显式 version/lineage，未完成组仍指向 old，已提交组指向 new。论文
容量轴因此报告最高 720 GiB 的**单版本 cache**，host peak 是一份 live cache 加 bounded
in-flight shadow/staging。现有 QK M1 的 144-GiB old + 144-GiB private-target 开发结果保留
为历史机制证据，但不再定义 formal endpoint。

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
| Tenrec QK | 493,458,970 rows、5,022,750 users、3,753,436 raw items；只有 user 内 ordinal order | Q-SEM 使用 1,000/3,000/5,000 users；X-QK HET/HOM 从 1,880,275 个达到 96-event 边界的用户中选取，long-edge roles 从 25,770 个达到 576-event 边界的用户中选取 | 大 entity embedding、多卡 lookup、物理 out-of-core、容量与模型敏感性 | 将 ordinal window 写成“按天更新” |

QB/QK 是 Tenrec 同一 collection 的两个相关表，论文必须如实说明，不能把它们包装成两个
完全独立的数据来源。ZhihuRec 和 Taobao 的既有 negative boundary 不进入主矩阵。
R-KR 使用的是 4+12 preparation 的 945-user artifact；8+8 preparation 的 965 users
属于另一条冻结协议，不能混入这一 workload。

QK 没有跨用户全局时间戳，因此 X-QK-HET 只能称为“固定模型更新边上的
trace-grounded heterogeneous cache snapshot”：每条 record 的 items 与用户内顺序是真实的，
但不同 \(b_u\) 不代表这些用户在同一自然时间点同时到达该 ordinal。论文不能把它写成按天
共时 workload、到达过程或在线 trace；真实日历更新的语义证据由 KuaiRand 承担。这个边界
不影响它回答 embedding placement、valid-length execution 和 DRAM↔HBM 系统问题。

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

### 3.4 X-QK：系统模型与统一 workload

现有 X1/X2 与正式 XP 均使用 base-only entity address space：

- 2,859,835 个 base-active semantic rows，加一个 padding row；
- top-250k rows 可作为 prediction targets；
- 其余 2,609,835 rows 是 lossless context entities；
- stream-only item identity 映射到已有 context rows，不扩张 future-fitted vocabulary；
- embedding 按 row modulo 分片，dense core 在 rank 间复制。

| ID | Model | Dense parameters, excluding embedding | FP32 dense | Global FP32 embedding | FP16 K/V per record per version | Role |
|---|---:|---:|---:|---:|---:|---|
| X1 | 16L/H1024, 16×64, T=512 | 84,990,976 | 0.317 GiB | 10.909 GiB | 32 MiB | 第二个真实系统模型；model sensitivity |
| X2 | 24L/H1536, 24×64, T=512 | 285,571,584 | 1.064 GiB | 16.364 GiB | 72 MiB | 已完成的两卡 development 与大模型 fidelity bridge |
| XP | 24L/H1536 core、bias-free E4096→H1536 owner-side projection、T=512 | 291,863,040 | 1.087 GiB | 43.638 GiB | full-length calibration 为 72 MiB | D2/D3/E2E paper-scale primary |
| X3，可选 | 32L/H2048, 32×64, T=512 | 675,428,352 | 2.516 GiB | 由 XP 规则生成 | 128 MiB | 核心结果稳定后的 model stress |

X2 已有一个 development `theta0→theta1` edge。它使用 225,737 个 base eligible targets、
13,492 个 update targets，并在 13,426 个 held-out positive targets 上得到正常的正更新
信号。正式论文不直接复用它的 development timing；它只说明这一配置是可训练、可更新、
可执行的。X1 需要训练一个短 `theta0→theta1→theta2` 链，X2 需要补一个独立
`theta1→theta2` edge。XP geometry 已预声明为 2,859,835 个 base-only semantic rows
加一个 padding row，即 2,859,836-row E4096 FP32 physical table 和 H1536 core。
Qualification 只看 active-row coverage、单卡不可复制、
2/4-way 可执行和短 edge 正信号，不能看 EvoKV speedup。通过后，XP 的 exact baseline、
D1 program、ActionPlan 和全部 baselines 都使用同一个 geometry/hash。若它失败，先发布
新的 blueprint identity 和失败原因，再重新 qualification；不能在已有 timing 中挑一个
更有利的宽度。系统 scale 不要求每个 timing cell 重训多 seed，也不把一个 system seed
当作 D1 质量泛化证据。

E4096 是明确标记的 capacity-stress model scale，不冒充 QK 的默认工业配置。它是在同一
2,859,836-row 真实 item namespace 上预声明的标准 power-of-two 宽度，使
table+dense/projection 在不依赖软件 cap 的情况下超过单卡物理可分配量；bias-free owner
projection 将它还原到原有 H1536 trunk。X2/H1536 的同边界
结果承担 model sensitivity，防止论文把只有宽表 stress 才成立的结论写成普遍规律。

XP 的 embedding 构造遵守六条限制：

1. 物理 row 来自 base period 内真实出现的 item/entity/declared feature namespace，禁止用
   永不访问的 padding rows 把表填大；
2. 主 dtype 下全局 embedding 固定为 43.638 GiB；它与 dense/projection core 合计约
   44.725 GiB，在加入任何 runtime state 前已超过单卡 Torch allocatable bytes，直接记录
   `full-replication=capacity-not-admitted`；
3. 只有在 base-period checkpoint builder 中真正收到至少一次 optimizer update 的 row
   才计作 `active row`；仅分配、初始化或被 inference lookup 访问不算；
4. 对两个 formal model edges、全部 headline manifests 和合法 comparators，冻结
   semantic embedding-request union：至少包含 all-exact valid target tokens、每个冻结
   fixed-action plan 的 exact/append/fallback requests；HOM masked padding 不属于 semantic
   request。该 union 必须全部是 active rows，并把 row count/hash 写入 qualification
   manifest，且
   \(B_{\mathrm{active\ embedding}}+B_{\mathrm{dense/projection}}\) 自身必须超过冻结的
   single-card Torch allocatable bytes。这样强制分片结论不依赖大量未训练冷行；若任一
   条件失败，XP qualification 失败。按当前四舍五入数字，这要求约 99.3% 的 non-padding
   rows active；正式 gate 使用精确 bytes 而不是该百分比；
5. formal manifest 同时报告总/active rows、每行 update-count 分布、wave
   requested/unique/hot rows、active-byte coverage 和实际 remote-vector bytes；
6. 所有 exact/mixed 方法都允许在 embedding owner 上先执行相同
   `E4096→H1536` projection，再返回 H1536 vectors，禁止用跨卡传输原始 E4096 vectors
   人为放大 baseline；BF16/FP16 table/transport 与 hot-row replication 作为
   operating-region sensitivity，不能因 FP32 主配置而被省略。

XP checkpoint 使用 common-upstream、base-only row-coverage builder。它可以读取 formal
users 的 **base-period** histories，因为推荐系统的 \(\theta_0\) 本来就由过去数据训练；
它不能读取任何 update/final window、D1 action、route profile 或 timing。这样按时间隔离
而不是强行按用户隔离，避免长尾 row 只出现在某个 final user's base history 时形成不可满足
的 gate。Builder 在 4 ranks 上做
row-sharded sparse embedding update，并为 dense/projection 与 embedding 分开配置
optimizer；embedding 使用 row-wise state，必要时把 optimizer state 放在 ordinary DRAM，
禁止物化一份 full-table dense Adam gradient/state。具体 optimizer、offload、batch 和
checkpoint 路径在 qualification 开始前冻结，只由可训练性、显存与 active-row gate 决定，
不能查看 EvoKV timing。训练得到的 canonical global-row checkpoint 再验证性地 reshard
到 2/4-rank inference layouts；训练 world size 不是性能对比变量。

大 address space 不能靠未访问的冷行制造。当前 development 2,048-record characterizer
中，all-exact/mixed 分别触达 309,467/126,588 个 unique rows；formal table 必须同时报告
address-space rows、active rows/bytes、unique requested rows/coverage、request tokens
和 off-rank response bytes。16.364 GiB 是真实物化表容量，不自动等价于每个 wave 都
扫描整张表。

formal X-QK 用户角色在第一次运行前用一个 base-only stable-hash salt 冻结，之后不能按
结果更换：

| Role | Users | Use |
|---|---:|---|
| XP base-row checkpoint builder | base-period coverage corpus | common-upstream \(\theta_0\) training；可使用下游角色用户的 base histories，但禁止任何 update/final window；直到 active-byte 与 frozen request-union gates 同时通过 |
| `theta0→theta1` edge update training | 2,560 | 一次 32-exposure update epoch；记录 effective targets |
| `theta1→theta2` long-user subset | 2,048 | 从 raw length≥1,024 的 2,219-user pool 预选；第二个 32-exposure update，并用 `[576,608)` 检查正更新信号 |
| D1 program fit/calibration | 512 | exact K/V target collection、shared program fitting；records 不进入 final performance workload，准备成本进入 E1 update-inclusive |
| D3 route profile | 512 | baseline/profile tuning 与 service-rate estimation；records 不进入 final performance workload，edge-specific profile cost 进入 E1 update-inclusive |
| Held-out qualification | 512 | 冻结 plan 前的合法性、prediction 与 capacity qualification |
| Final system benchmark universe | 最多 65,536 | stable-hash HET universe；容量点按 valid K/V bytes 截断，不按结果换用户 |

D1 fit、D3 profile、qualification、final benchmark 与两组 **post-base** model-edge
training users 使用预声明 salt 分配为互斥角色。唯一例外是 common-upstream
\(\theta_0\) builder：它可使用这些用户严格位于 base period 的 histories，但不能看到其
update/final windows，也不能按 action 或性能选数据。QK 容量足够，没有必要接受未来标签
或 tuning 泄漏。25,770
只证明 576-event boundary；第二个 edge 的 `[576,608)` 推荐检查必须
单独审计上述 length≥1,024 long-user subset 在 entity mapping 后仍满足 608-event boundary。
system benchmark records 只要求其 cache history 覆盖到 target version。salt、实际 raw
user IDs、各 window 的 retained rows/targets 和交集审计都写入 formal manifest。

`X-QK-HET` 与 `X-QK-HOM` 共享模型、edge、entity table、用户选择 salt、**相同 record
IDs** 和 D1 action 语义。对每个 HET 用户令 \(b_u=\min(n_u,544)\)，要求
\(n_u\ge96\)；old cache 截止于
\(b_u-32\)，target 截止于 \(b_u\)，两者各保留最近最多 512 个真实 exposures。于是 old
length 为 64–512，target length 为 96–512；短历史是增长窗口，长历史自然退化为现有
`[0,512)→[32,544)` sliding path。manifest 必报 old/retained/evicted/append/target length
quantiles、512-token saturation、valid tokens 和物理 K/V bytes。HOM 不另选长用户，而
是把同一 HET record 的 old/target 左侧用 padding row 补成 512 slots，并严格 mask
padding；padded slots 不计入 valid tokens、semantic lookup 或质量指标，但其 allocated
K/V、padding compute 和搬运必须完整计量。这样 HET/HOM 只改变物理 shape regularity，
不会同时改变用户活跃度、item locality、dedup 上限或 owner assignment。

HOM 沿用 HET valid bytes 冻结的相同 records 与 nominal capacity point，但不能沿用 HET
group cuts。HET/HOM 分别按实际 allocated old input、target shadow、workspace 和 allocator
peak 做 micro-wave admission；HOM masked padding 必须完整计入，避免 matched control
越过同一 HBM/shadow budget。

### 3.5 XP-HET 单版本 K/V 容量档

容量由实际 target-version valid K/V bytes 定义，而不是由 record 数反推。formal builder
在排除 fit/profile/qualification users 后，按 base-only stable-hash order 取 nested HET
users，直到达到 byte threshold；允许的 overshoot 不超过一条真实 record，并同时报告
实际 records、old/target bytes 和 length quantiles。

| Target single-version valid K/V | 512-token full-length calibration | Intended use |
|---:|---:|---|
| 36 GiB | 512 records | resident/out-of-core crossover control |
| 72 GiB | 1,024 records | 双卡小容量 sensitivity |
| 144 GiB | 2,048 records | 双卡中间档 |
| 288 GiB | 4,096 records | 双卡 headline、fixed-work scaling |
| 576 GiB | 8,192 records | 四卡大点 |
| 720 GiB | 10,240 records | 四卡最大 paper-core 点 |
| 800 GiB，可选 | 约 11,378 records | 独占机器通过 qualification 后的最大 physical stress |

HET records 通常多于表中 full-length calibration 数，因为短历史使用更少有效 bytes。第一轮
paper-core 禁止复制用户；因此不存在跨副本 dedup 人为放大收益的问题。当前
machine-readable development cohort 只有 2,048 个固定长 records，必须保留为历史 M1
身份，不能被当成新的 HET formal cohort。

65,536 不是为了放大结果而任意设置的上限。QK audit 中有 1,880,275 个用户达到
96-event HET 最小边界；每个 target 至少有 96 个 valid tokens。即使 stable-hash universe
中的所有记录都恰好只有最小长度，720/800-GiB 也分别至多需要约 54,614/60,682 个互异
records，仍不需要复制用户。实际 builder 按 valid bytes 到达阈值即停止，并报告真实
record count；不能为了使用满 65,536 而继续加入记录。

若未来真的需要 trace multiplication，每个 replica 必须有独立 cache ID、owner 和物理
extent，完整支付 source/compute/target bytes，禁止跨 replica dedup，并标为 synthetic
capacity workload。它不能进入质量统计，也不能把 replica 数写成真实 dataset users。

## 4. 硬件与统一资源边界

### 4.1 GPU 与拓扑

- 4×NVIDIA A40；`nvidia-smi` 标称 46,068 MiB，即 44.988 GiB，当前 Torch/CUDA
  allocatable total 为 47,699,722,240 bytes，即约 44.424 GiB；
- Benchmark Qualification 必须先用 geometry-independent 的 1/2/4-rank launcher、
  NCCL context、allocator/workspace probe 和固定 safety margin 冻结共同
  \(B_{\mathrm{HBM,usable}}\)，再验证 XP，顺序不能倒置。38 GiB 只用于当前预算估计；
  预计 39–40 GiB/rank 的最终 cap 不能因 XP 或 EvoKV 的结果而上调。Candidate-specific
  full-runner preflight 只能拒绝该配置或触发一次 timing 前的全局重新 qualification，
  不能为某个方法单独改变 cap；
- GPU0/1 在 NUMA 0，彼此为 NV4；
- GPU2/3 在 NUMA 1，彼此为 NV4；
- 两个 pair 之间为 `SYS`；
- 两卡主实验固定 GPU0/1；四卡 rank 0/1 first-touch NUMA 0，rank 2/3
  first-touch NUMA 1；
- formal 四卡运行只在四卡均空闲、机器独占时启动；不能终止或挤占外部 GPU 进程来完成
  matrix。

X2 direct-old-K/V program 约 0.422 GiB。按当前 38-GiB development estimate，X2 的 HBM
账本为：

| GPUs | Embedding shard / rank | Dense + program + embedding fixed / rank | Dynamic budget under 38 GiB cap |
|---:|---:|---:|---:|
| 1 | 16.364 GiB | 17.850 GiB | 20.150 GiB |
| 2 | 8.182 GiB | 9.668 GiB | 28.332 GiB |
| 4 | 4.091 GiB | 5.577 GiB | 32.423 GiB |

该表仅描述已存在的 X2，不替代 XP 的正式 capacity ledger。XP qualification 通过后必须生成
自己的 1/2/4-way ledger；如果 fixed state 在 1 GPU 上已超过 usable budget，1-GPU XP
cell 诚实标为 `capacity-not-admitted`，并用 X2 运行同一代码路径的 local sanity。

当前 X2 development checkpoint 只有 world-size-2 的两个 modulo shards。正式 1/2/4-GPU
矩阵开始前，必须从同一 canonical global embedding 生成并验证 1/2/4-shard artifacts，
记录相同 global-table digest，并检查 reshard 前后的 lookup 输出一致；不能把现有两 shard
文件直接当成一卡或四卡 checkpoint。

1/2/4 rank 是后继 runner 从一开始就支持的参数，不等于每个模型在每个 rank 数都必须
capacity-admitted。R-KR 与 X2 用于覆盖 1-rank 路径；XP 负责 2/4-rank 强制分片主结果。
每个 cell 在运行时对自己的 fixed state、old-group input、target-group shadow、workspace
和 allocator peak 做实测 admission，不能只做静态 byte arithmetic。

表中的 fixed bytes 包含 EvoKV program；exact baseline 不被迫加载无用 program，而是报告
自己的 fixed bytes。统一报告 per-rank 物理压力：

$$
\rho_{\mathrm{KV}} =
\max_r
\frac{B_{\mathrm{live\ single\ version},r}}
{B_{\mathrm{HBM,usable}}-B_{\mathrm{fixed},r}}.
$$

formal artifact 使用 HET 的实际 valid bytes 和自然 owner map 计算 max-rank value；同时
报告每个 admitted group 的
\(B_{\mathrm{old\ input}}+B_{\mathrm{target\ shadow}}+B_{\mathrm{workspace}}\) 峰值。这样比
仅报告“分了多少组”更能解释 D3 operating region。exact 另报 raw-history source 与
target-working-set pressure。

### 4.2 Host DRAM

机器有两个约 504 GiB 的 NUMA node。formal run 必须：

- 在独占机器上先释放或复用旧 development arena，再检查 total `MemAvailable` 与
  per-node free memory；
- live cache arena 与 bounded group shadow/staging 按 rank 的 GPU NUMA first-touch；
- primary timer 前 reserve backing，并用 rank-specific non-payload sentinel 对 live
  cache arena 与 bounded shadow 的每个物理页做 dirty write-touch；只 read-touch、
  `MAP_POPULATE` 或 `mincore` 不能排除 Linux shared-zero-page/overcommit 假象；
- source old K/V 在 timer 前完整物化；每个 group 在其 input 已被完全消费前保持
  immutable，target payload 必须在 timer 内写入 shadow 或合法 replacement extent，
  通过 validation/commit 后才回收 old group；
- 记录 total/per-node RSS、`numa_maps`、major faults 和 swap-in/out；formal timer 内
  major-fault 与 swap 增量必须全部为零；
- 报告普通 DRAM、pinned DRAM、HBM allocated/reserved 的 standing 与 peak bytes。

前一轮 D3 development 曾在 `/dev/shm` 使用约 292 GiB old/target arena；它们不是 formal
matrix 必须与新 workload 并存的常驻资产，当前已经释放，准备快照中的
`MemAvailable` 约为 939 GiB。后续实验仍可复用 `/dev/shm`：每个实验族开始前先确认旧
arena 的 lineage 已冻结、所有 live mapping/file descriptor 已关闭，再释放或原地覆盖。
因此独占机器的规划边界是扣除 OS 与运行时余量后约 **900 GiB 可用 DRAM**，不能把运行
旧实验时的瞬时占用当成永久上限，也不能把旧 arena 与新事务重复计费；仅 `unlink`
一个仍被 mmap/open 的文件不代表物理页已经释放。

另一方面，单个 `/dev/shm` mount 的容量仍只有约 504 GiB。它可以继续承载 144/288 GiB
主点及其他可容纳实验，但 576/720/800 GiB 点不能只依赖这个 file backend。大点需要
NUMA-aware anonymous pageable-DRAM arena，或明确扩容且按 NUMA 放置的 tmpfs。不能用
`/data` 上的 NVMe 文件冒充 ordinary-DRAM result。materialization 在 primary timer 外，
但真实 physical pages、coverage、bytes 和 checksum 必须存在并被记录。tmpfs 与
anonymous arena 消耗的是同一套物理 DRAM，504 GiB mount capacity 与约 900 GiB
机器可用预算不能相加，二者必须进入同一个 total/per-node preflight。

### 4.3 最大物理边界

新的 rolling group boundary 允许在约 900 GiB 可用 DRAM 上物化一份大 cache，而不支付
完整 old+new 的 2× footprint：

- 720 GiB single-version cache 是 paper-core 最大点；
- 800 GiB single-version cache 只有在 OS/runtime 余量、两 NUMA node placement 和 bounded
  shadow 全部通过 qualification 后才进入可选 stress；
- “1 TB”若指 1 TiB/1,000+ GiB，当前机器无法作为普通 DRAM 物理结果；增加磁盘也不改变
  DRAM endpoint；
- generator、sparse file、未 dirty-touch 的 logical payload 或多个独立 wave 的累计 bytes
  都不能冒充一个 1-TiB live cache。

每个 group 的 commit 是原子 version/lineage pointer transition；整个 job 在最后一组完成时
结束，但中间允许存在显式标记的 old/new groups。禁止的是无版本标记的 mixed state，而
不是要求全局蓝绿式 epoch switch。

### 4.4 磁盘与产物生命周期

磁盘是 formal matrix 的实际资源边界，但容量快照不是冻结 protocol。清理旧 source
shards 与 DRAM arena 后，2026-07-31 的最新准备快照为：仓库所在 `/data` 分区总容量约
3.5 TiB、可用约 **503 GiB**；仓库约 64 GiB，其中 checkpoints 约 53 GiB；根分区可用
约 118 GiB。每个实验族开始前必须重新记录这些值，不能把本文快照当成未来运行时的
事实。大型临时文件不得通过 `/tmp`、默认 Torch 临时目录或 core dump 意外写入根分区。

即使每个 formal cell 只保留一份完整 consumer-ready target，多个 288–720 GiB cells 也会
快速进入多 TiB；最大点的一份 720 GiB target 已经超过当前全部空闲磁盘。因此
baseline-first 的“冻结结果”只指冻结 protocol、metadata、timing samples、digest 和小型
数值证据，绝不指冻结完整 old K/V 或 candidate target。rolling runner 在每次 measured
job 后先完成 digest/witness，再在 timer 外按相同 group 顺序从 checkpoint/raw history
重建 old cache 到同一 arena；不能为了重复实验长期保存第二份 old epoch。

长期模型资产采用单一 canonical 表示：

| Asset | Per version | Three versions |
|---|---:|---:|
| X1 dense + global embedding | 11.226 GiB | 33.678 GiB |
| X2 dense + global embedding | 17.428 GiB | 52.284 GiB |
| XP dense + global embedding | 约 44.725 GiB | 约 134.175 GiB |

当前 X2 `theta0/theta1` 已占约 35 GiB；补齐 X2 `theta2` 与 X1 三版本约新增
51.1 GiB。XP 的 checkpoint/optimizer/scratch 实测预算必须在 qualification 时补入
storage manifest；三版本 canonical inference state 约 134.175 GiB，因此当前磁盘未必阻止首个 baseline，但
很可能使持久化全部 1/2/4 layouts 不合理，并可能触发用户已允许的专用实验盘扩容。在当前
filesystem 下，1/2/4-GPU embedding layouts 不应全部长期复制。这里的 canonical 是具有
全局 row-order digest 的唯一持久化内容，允许由一套已验证的 sharded physical layout
承载，不要求额外复制一个 global tensor 文件。当前默认做法是从它为当前
`model edge × world size` 生成一个临时 layout，验证 global digest 与 lookup parity，
完成该实验族后回收 payload，只保留 shard manifest 与 digest；增加专用实验盘后可以
重新评估是否持久化这些可重建 layouts。

统一 retention policy 为：

- live K/V arena、group shadow、逐 extent oracle target、fit/profile K/V 和 staging buffers
  只驻 ordinary/pinned DRAM 或 HBM，禁止落入 NVMe、`/tmp` 或 swap；
- correctness reference 逐 extent 生成 digest 和固定 witness 后立即释放；correctness、
  warmup、五次 measured jobs 复用同一 DRAM arena，并在 timer 外重建相同 old source；
- 每个真实模型版本只长期保存 inference-only canonical checkpoint；optimizer state
  最多保留一个最新恢复点，并在 inference checkpoint 验证后回收；
- formal result bundle 只保存完整配置、source/code/environment hash、五个原始 timing
  samples、per-rank phase/resource counters、per-extent digest/Merkle root、coverage、
  lineage、失败位置和少量固定数值 witness；
- 禁止保存完整 output tensor、逐 token/K/V arrays 和无上限 debug log。当前已有单个
  JSON 接近 175 MB；若约 500 次 execution 沿用该格式，日志本身可能接近 85 GiB。formal
  schema 必须使用 compact counters、digest 和必要的压缩 per-record summary；
- checkpoint 写入、reshard scratch 和 result bundle 都采用有配额的 experiment-local
  目录；不允许把同一 checkpoint 复制进每个 cell 的结果目录。

在 XP qualification 尚未完成前，不再给出虚假的单一峰值数字。X1/X2 的新增量仍约
57–65 GiB；XP
三版本、一个 active derived layout、atomic checkpoint 临时副本和至多一个 optimizer
recovery state 将把新增峰值推到大约 **260–330 GiB**，具体值由 qualification 的真实
optimizer/scratch 实测更新。当前空间可以启动 builder/canary，但是否在 baseline family 之间反复
reshard、是否提前安装专用盘，由 storage preflight 决定。所有 builder/runner 必须在物化
前声明 `persistent_bytes` 与 `scratch_bytes`。准备阶段以物化长期资产之前的空闲量为基准：

$$
B_{\mathrm{available,initial}} -
B_{\mathrm{persistent,new}} -
B_{\mathrm{scratch,peak}}
\ge 100\ \mathrm{GiB}.
$$

长期资产已经写入后，runner 只检查
\(B_{\mathrm{current,free}}-B_{\mathrm{scratch,remaining}}\ge100\ \mathrm{GiB}\)，避免重复
扣除 persistent bytes。100 GiB 是当前 filesystem 的运维 safety floor，不是论文机制；
增加专用磁盘后可以在新 storage config 中重新设定，但必须保留明确余量与 atomic-save
空间。不足时先回收已验证的临时 layout/optimizer state，或暂停等待扩盘；不能通过把 K/V
spill 到 NVMe、减少 physical materialization 或删除仍未冻结 lineage 的唯一 checkpoint
来绕过 admission gate。

若后续增加一块约 1 TB 的专用实验盘，它不改变 paper-core 的语义，只改变哪些
可重建资产值得长期保留。以下标记用于扩盘后重新审查实验计划：

| Artifact | 当前 503 GiB free 下的策略 | 增加专用盘后的策略 | 标记 |
|---|---|---|---|
| X1/X2 六个 canonical inference checkpoints | 必须长期保留，约 85.962 GiB total | 原样保留并迁移到 content-addressed store | core，无需等扩盘 |
| XP 三个 canonical inference checkpoints | qualification 后保留，预计约 134.175 GiB | 优先迁移到 content-addressed store | core，扩盘触发点 |
| 当前 edge 的一个 derived shard layout | 临时生成并回收 | 频繁复用时允许持久化 | core，无需等扩盘 |
| X1/X2/XP 完整 1/2/4-GPU layouts | 不默认保留 | 可持久化并消除重复 reshard | `EXPAND-P1` |
| 多版本 optimizer/restart states | 只留一个最新恢复点；XP 大小由 row-wise/offload qualification 实测，不能沿用旧 22–35 GiB 估计 | 训练需反复回退时可按 edge 保留 | `EXPAND-P1` |
| 可选 X3 三版本及其 layouts | paper-core 前不生成 | 核心结果稳定后再预算 | `EXPAND-P2` |
| 完整 formal K/V outputs 或无上限 raw logs | 不保留 | 仍不保留 | 非扩盘项 |

`EXPAND-P1` 不是开始当前实验的 blocker；只有当重复 reshard/训练回退已经显著拖慢执行，
或需要冻结这些 checkpoint 作为跨机器复现实物时，才优先安装新盘并更新 storage manifest。
新增约 1 TB 原始容量通常能提供约 0.9 TiB 可用空间，足以容纳 P1 checkpoint families
与 scratch，但仍远小于保存整套 formal K/V outputs 所需的 6–30 TiB。

## 5. 统一计时、统计和公平性规则

### 5.1 正式 cell

每个新的 timed cell 固定：

1. 一次 complete-workload correctness pass；
2. 一次 untimed complete-workload warmup；
3. 五次 measured complete jobs。

五个 timing samples 用于描述 runtime variation，不是五个模型样本。报告 median、五个原始
样本或 min–max，并对同一 revision 的方法做 paired comparison。这里保留的是五条 compact
timing/resource records，不是五份 K/V output。质量统计仍以 training seed 为 replication
unit。

“Baseline first”指先完成并冻结 baseline implementation/config，不是把所有 baseline 的
五次测量按时间跑完后才测 proposed。同一 workload/revision 的 measured jobs 使用预声明
seed 的 randomized/interleaved blocks：两方法采用 AB/BA 交替，多方法采用轮转顺序；每个
method×repeat 都从相同 old checksum、已 dirty-touch arena 和相同资源上限开始。记录实际
运行顺序、GPU clocks/temperature/power、NUMA placement 和 reset time。若某一 block 出现
throttle、外部竞争、major fault 或 page-state 不对称，整 block 作废重跑，不能只删较慢
方法的样本。

每个 cell bundle 只物化一份 live cache arena 和 bounded group shadow/staging。
correctness、warmup、五次 measured job 以及 paired methods 都从同一个 old manifest 和
相同 source checksum 开始。timer 内每个 group 完成 writeback、validation、versioned
commit 和 old-group reclaim；最后一组提交后 job 才结束。为了下一次 job，benchmark
harness 在 timer 外从相同 checkpoint/raw-history source 按 group 重建 old cache 到同一
arena，并记录 reset/rematerialization 时间。即使在 720 GiB 点也不需要第二份完整 old 或
target cache。

Formal rolling endpoint 要求所有方法在 group commit 前保留该组 old extent，因此使用
共同 bounded replacement shadow；未验证 target 不得原位覆盖唯一 old copy。验证后执行
group-level pointer/allocator transition，再回收 old extent供下一组复用。不同方法可以
需要更少 shadow/workspace，但共享同一 host-capacity上限；禁止让某一方法保留完整 private
target，或让另一方法用不可恢复的原位覆盖换取优势。

### 5.2 Primary timers

resident 与 out-of-core 使用两个明确、不可混算的 timer。

**Resident timer。** HBM-resident old K/V、raw IDs/history、model、embedding shards 和
compiled program 已就绪。timer 包含当前 wave 尚未持久化而必须现场完成的 lowering、
compute/collective、target construction、consumer-ready setup、coverage/lineage validation、
group commit 和 reclaim。只有已经合法序列化、绑定当前 stack 且可复用的 WavePlan 才可把
lowering 移到 execution timer 外。

**Out-of-core timer。** ordinary-DRAM source 到 consumer-ready ordinary-DRAM target，包含：

- job 内的逻辑 extent allocation、ownership 和 ledger setup；
- source extent 读取到 bounded pinned staging；
- pageable→pinned packing、H2D；
- D2 compute 与 embedding collective；
- D2H、pinned→ordinary-DRAM group writeback；
- segmented/contiguous consumer-ready setup；
- per-group coverage/lineage validation、versioned commit、old-group reclaim，以及最后一组
  完成时的 job-final manifest。

为保证 formal timer 内没有 page faults，物理 backing arena 的 reservation 与逐页 dirty
write-touch 对所有方法对称地放在 timer 外；真实 target payload 的写入必须完整发生在
timer 内。

以下单独报告，不塞入 execution-only timer：

- 模型训练和 checkpoint materialization；
- source old-K/V 的第一次生成；
- D1 fit 与 program compilation；
- 已持久化 WavePlan 的一次性生成；
- D3 route profile 与 ResidencyPlan construction。

计时主次现在冻结如下：

- D2/D3 mechanism ablation、capacity 和 scaling 图以 **execution-only** 为主，标题和正文
  必须明确称 maintenance execution，不得称完整 end-to-end；
- E1 的 **end-to-end primary** 是 first-wave update-inclusive time：模型 checkpoint 已
  发布后，包含该 edge 必需的 D1 fit/target collection/program compile、D2 lowering/
  routing plan、D3 edge-specific profiling/ResidencyPlan construction 和完整 rolling
  execution。Streaming model training 是所有方法共同的上游过程，单独报告但不进入该
  timer；
- exact/generic baselines 同样计入自己每个 edge 必需的 routing/profile/plan construction；
  一次性的候选 screen/tuning 不计入任何方法，但完整 screen ledger 必须公开；
- 另报 execution-only、update-inclusive breakdown、跨多 wave 复用时的 amortized time 和
  break-even/reuse count。只有 update-inclusive row 获胜时，摘要才可使用“end-to-end
  speedup”；否则只能声明 execution speedup。

HBM resident 与 ordinary-DRAM endpoint 分成两个 panel，不能把二者的 raw time 串成一个
虚假 waterfall。

`cold-allocation-inclusive` 数字定义为 arena reservation、dirty first-touch 与第一次
execution 的和；source old-K/V 的首次生成仍作为独立 preparation cost 报告。重复实验的
timer-external manifest reset 也单列，不能藏进 warmup。这样同时保留生产 hot path、
冷启动成本和 benchmark 可重复性三个不同边界。

### 5.3 Baseline tuning

每个 baseline 在相同 \(B_{\mathrm{HBM,usable}}\) 下独立调优，不能让 EvoKV 使用更大的
group 或更多 buffer。除 HBM 外，还必须冻结 CPU affinity、CPU thread count、每 rank pinned-byte cap、
CUDA stream 上限、outer lookahead、outstanding drain credits 和 NUMA placement。
不同方法可以少用资源，但不能获得更高上限。调优只用独立 profile users：

- 先做 legality/correctness 与 capacity preflight；
- 一般候选空间按机制最多保留约 12 个有意义点，不做无边界 Cartesian product；核心
  all-exact denominator 是例外，使用下述不超过 48 个 legal joint combinations 的
  bounded grid；
- 每点一次 screen，top-3 做三次复测；
- 根据 median 选定，tie 时优先低 HBM、低 pinned bytes、低 padding；
- 选定后冻结到 qualification 和 final benchmark。

具体而言，D2/D3 的 exact denominator 不能只调 pipeline。在独立 profile users 上对
capacity-admitted placement/transport candidates 做 bounded joint grid：row sharding +
wave-scope dedup + owner-side projection 的 FP32 return、相同路径的 BF16/FP16 return、
hot-row replication + cold-row sharding，以及合法时的 full replication，分别与
routing/coalescing 和下面的 one-shot/S0/S1/fine-grained pipeline 联合。不得先按单独
placement 排名只保留前两名，因为低精度、dedup、replication 和 pipeline 可能交互。当前
合法组合上限冻结为 48；若实现后超过该上限，必须在看最终结果前发布保留交互项的 pruning
规则。joint grid 的 top-3 用相同三次复测选最终 winner。所有淘汰配置和
capacity-not-admitted 原因进入 appendix。

D2 的 exact denominator 也在 one-shot/two-stage execution candidates 中独立选最快者；
D2 的 fixed-action denominator 在 owner-local staged-contiguous 与
owner-local fused-contiguous 中独立选最快者。D3 的 exact denominator 在合法
one-shot、sequential S0、whole-group S1 和 bounded fine-grained exact pipeline 中独立
选最快者；mixed baseline 先在 S0、S1 和 generic fixed-FIFO bidirectionally segmented S2
中独立选择，再加入一个读取相同 stage profiles、但不知道 EvoKV action 语义的 generic
work-conserving/list scheduler。S2 只使用一套全局 segment 参数、固定 FIFO route order，
不生成 ResidencyPlan；profile-aware generic scheduler 把 group 当作 opaque jobs，使用
相同 service-time/capacity/credits 和 bounded-flow recurrence，但不能读取
compiled/exact 标签、route-specific parameter sharing、EvoKV stable-interleave objective
或改变 D1 actions。EvoKV 必须相对两者中更强的合法 winner 报告结果。

M3 为解释分组边界而保留 exact S0，并将上述 tuned generic exact winner 作为强对照；
每个 exact candidate 的 screen 结果进入 appendix。不能因为 development 中 S1 曾获胜，
就在所有容量上预设 whole-group double buffering 一定最快，也不能让 full D3 用
microbatch pipeline 去比较一个只做 whole-group overlap 的弱 exact/mixed baseline。

baseline 必须使用同一 source/target tier、GPU topology 和 timer。不同算法天然读取不同
source representation 时，分别报告其 logical/physical bytes；不能通过假装 source bytes
相同制造公平，也不能忽略多出来的状态。

所有参与 timed comparison 的 rows 都必须结束在同一个 consumer-ready contract：相同
target K/V/hidden dtype、layout、durability、coverage、group-version 和 reclaim 语义。
若一个 baseline 原生写 contiguous target，而正式 endpoint 是 segmented target，就必须
在 timer 内执行并计量 adapter；不能把 layout conversion 留给读者想象。exact、mixed 与
proposed 的 tuning 候选、赢家配置、淘汰原因和五次正式样本全部保存，appendix 至少报告
每个候选的 median、peak HBM/pinned bytes 和 legality。

### 5.4 Append accounting

所有 D2/M2 结果同时报告：

1. retained-prefix maintenance，不含 method-common append；
2. complete wave，包含 append。

compiled retained repair 可以是 zero lookup/zero embedding collective；compiled records
的新 token append 仍然必须查 embedding。任何“embedding-free”措辞都只能限定在
retained-prefix phase。

### 5.5 Per-layer baseline-first execution ledger

baseline 不是“先随便跑一个能工作的版本”，而是对应设计层的正式 foundation。下面是
accounting role；真正执行时按依赖分层冻结，而不是把所有 baseline 当成一个全局 barrier：

| Family | Baseline/control foundation | New formal cells |
|---|---|---:|
| M1/D1 semantic | 已冻结的 frozen-training control、stale reuse、current projection、结构化 partial-replay controls 与 exact；同 SLA structural comparison 另计 comparison pass | 0 |
| M2 | HET row-sharded partial-exact curve、capacity-admitted placement controls、HOM matched controls | 14 |
| M3 | 两个容量点的 exact S0；各点 tuned generic exact winner 复用 D3 capacity row | 2 |
| D2 resident | tuned exact、owner-local staged/fused fixed-action controls、X2 1-rank sanity、XP 2/4-rank 与 R-KR foundations | qualification 后重算 |
| D3/E1 out-of-core | tuned exact、tuned D1-only、S0/S1/S2/profile-aware generic winners，覆盖 capacity/GPU/action/model/held-edge | qualification 后重算 |
| C1 | 独立 semantic oracle、failure/group-lifecycle checks 和 hardware ceilings，不作为性能 cell | 0 |

旧 \(64+k\) cell accounting 来自 fixed-length X2 与 full-private-target boundary。新的
cell ledger 在 XP qualification、HET manifests 和 strongest generic scheduler 冻结后再去重；在这之前保留
family/row 设计，不伪造精确总数。现有数字只用于估算执行量，不能成为删掉强 baseline 或
真实 HET point 的理由。

执行顺序冻结为：

1. 整理 M1/D1 既有 controls，并完成 M2 与 D2 baseline foundation；
2. 运行 D2 rows 4–6、scaling 中的 complete D2，冻结同一 `stack_revision`；
3. 在该 stack 上运行 M3、D3/E1 exact、D1-only、S0/S1/S2/profile-aware generic
   foundations；
4. 最后运行 D3 input-only、route-specific triples、ResidencyPlan 与 sensitivity
   proposed rows。

每个 baseline cell 同样执行 correctness、warmup 和五次 measured jobs，不能在它所对应的
proposed row 跑完后才补一个较慢 comparator。每层 baseline 的完成 gate 是：

1. 所有 target 都通过独立 semantic oracle，而不只是 baseline 之间互相 hash 相等；
2. source、target、action/work manifest、model/embedding shard 和 endpoint hash 已冻结；
3. formal timer 内无 major fault、swap、隐式 rematerialization 或未计量 layout adapter；
4. 1/2/4-GPU shards、owner assignment、collective participation 和 exact counts 可重建；
5. 每个 independently tuned winner 确实胜过或不劣于自己的合法候选，完整 screen 可查；
6. paired methods 使用相同资源上限，运行波动能够由原始五次样本和 phase counters 解释。

任一 gate 失败时先修 foundation 并重跑该 baseline family。不能先跑本层 proposed，再通过
修改 baseline harness 让 speedup 出现。

### 5.6 Benchmark Qualification 入口

Benchmark Qualification 是正式 paper result 之前的独立准备模块，不阻止当前继续设计
workload、实现 baseline runner 或探索 D3。它当前只登记需要在正式矩阵前闭合的问题：

- XP 2,859,835 semantic rows + 1 padding row 的 2,859,836×4,096 physical FP32
  embedding、owner-side projection、短 edge trainability 与
  1/2/4-way shard parity；
- HET/HOM manifest 的长度分布、valid-byte accounting、自然 owner imbalance 与
  independent quality subset；
- 两组 model-edge training、D1 fit、D3 profile、qualification 和 final benchmark roles
  必须用户级互斥；
- timed fast path 中关闭诊断 hash/CPU round trip，并验证 instrumentation on/off 不改变
  方法排序；
- formal measured jobs 采用预声明的 randomized/interleaved method order，并记录
  clocks、temperature、power、NUMA/page state；baseline-first 只冻结实现，不固定为
  baseline 先测、proposed 后测；
- \(B_{\mathrm{HBM,usable}}\)、pinned cap、最大地址、1/2/4-rank launcher、NUMA placement、
  `/dev/shm`/anonymous arena、zero-swap/major-fault 和整机独占；
- exact、hot/full replication、dedup、低精度 transport、S2 与 profile-aware generic
  scheduler 的候选实现；
- group commit/reclaim、幂等恢复、segmented consumer 和下一 wave compatibility。

详细资格阈值、执行顺序和故障处置单独记录在
[11_benchmark_qualification.md](11_benchmark_qualification.md)。qualification 失败可以修改
后续 formal config 或 claim，但不能回写、重命名或升级现有 development artifacts。

## 6. 具体实验矩阵

### M1：Reuse–Recompute dilemma

**状态：复用已有证据，不新增 timed cells。**

| Item | Configuration |
|---|---|
| RQ | streaming update 有价值时，stale reuse 是否仍留下可恢复的 version-consistency gap？ |
| Data/model | Q-SEM 的 3 datasets × 3 tiers × 4 seeds，共 36 条模型版本链 |
| Controls | frozen-training control、current-model stale reuse、current-model exact recomputation |
| Metrics | BestRank/MeanRank、rank utility、NDCG/Hit、streaming value、reuse value、maintenance gap、staleness tax，以及 seed-0 resident full-prefix exact 的 absolute GPU time |
| Output | 一张 3×3 grouped heatmap；完整 per-seed 数值放 appendix |

三条 control 的语义必须写死。`frozen-training` 是 theta0 模型在同一完整 history/positives
上的 fresh full forward，用来测 streaming training 本身的价值；它不是 cache-maintenance
baseline。`stale reuse` 是 current checkpoint/scorer 加 old-version prefix K/V，并对同一
latest token 使用 current model。`exact` 是 current checkpoint 对同一完整 prefix 的
replay，再处理同一 latest token。

它支持“问题存在且与 workload/regime 有关”。它不支持“模型越大 gap 必然越大”、
“exact 是 ranking quality 上界”或“用 task labels 路由 cache”。
四个 training seeds 是质量 replication；已有 seed-0 resident exact timing 只用于说明
完整 prefix replay 的绝对计算代价，不能伪装成四个独立 cost samples。零维护 reuse 与
full inference 的 timer 不同，因此不把二者 raw time 相除成 end-to-end speedup。

另有一条已完成但不进入 formal M1 replication count 的 XP development foundation：
24L/H1536、43.638-GiB global FP32 embedding、两卡 row sharding、16,384 个训练用户与
4,096 个 qualification 用户。在一个 warm-up edge 后，三个普通版本边使用同一 999-negative
binding 和 FP16-storage/FP32-consumption endpoint，Exact-over-Reuse CE gap 为
0.01846/0.01068/0.01340。对应 D1 bridge 的 compiled/mixed gap recovery 为
63.9%/68.9%、55.3%/62.3%、70.0%/74.3%。这些数字固定 system benchmark 与 D1→D2
接口，但因为配置经过 development selection，不能升级为 formal seed replication。

### M2：Logical sparsity is not physical sparsity

**状态：新必跑，14 个 timed cells。**

当前设备可用性只允许 GPU0/GPU1，因此先实现并冻结所有两卡 cells；四卡 rows 保留在
paper matrix 中，但在用户明确恢复 GPU2/GPU3 前不得启动，也不能用两卡结果外推替代。

使用同一 XP-QK edge/address space 的 resident `X-QK-HET` 小切片。这个实验在介绍 D1 前
构造一个预声明的 partial-exact workload，不使用 D1 program 或 D1 输出挑选 exact
records：被选中的记录 exact replay retained prefix，未选中的记录保留原 K/V；两者都执行
相同的 32-token append。stable hash 在 old/retained-length strata 内选择记录，使目标
0/20/50/100% 指的是 **retained-token exact fraction**；同时报告实际 record fraction。
所有 M2 rows 使用 resident timer，不经过 ordinary-DRAM capacity grouping，避免把 D3
prefetch/writeback 混入 Motivation 2。

| Placement/control | GPUs | Exact fractions | Cells |
|---|---:|---:|---:|
| XP-HET row-sharded embedding | 2, 4 | 0%, 20%, 50%, 100% retained tokens | 8 |
| XP-HET strongest capacity-admitted placement control | 2, 4 | 20%, 100% retained tokens | 4 |
| XP-HOM row-sharded matched control | 2 | 20%, 100% retained tokens | 2 |
| Total |  |  | **14** |

strongest placement control 从 hot-row replication+cold-row sharding、wave-scope dedup 和
低精度 transport 的预声明 candidates 中独立选择；完整 replication 若超过 XP usable-HBM
则记录 `capacity-not-admitted`，并在 X2 qualification 上保留同逻辑 sanity。HOM 两行仅
回答 M2 结论是否由长度不均衡制造，不能替代 HET headline。

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

**状态：新增两个 exact S0 cells；相同容量的 tuned generic exact winners 计入 D3
capacity matrix。**

| Model/data | GPUs | Single-version valid K/V | Records | Methods |
|---|---:|---:|---:|---|
| XP-QK-HET | 2 | 144 GiB | manifest-derived | exact S0、tuned generic exact winner |
| XP-QK-HET | 2 | 288 GiB | manifest-derived | exact S0、tuned generic exact winner |

顺序 baseline 是完整的 capacity group：

```text
read/pack group → H2D → distributed exact → D2H → publish → next group
```

报告 ordinary-DRAM pack/publish、H2D、D2H、GPU、collective、input-boundary wait、
output-credit wait 和 peak HBM。GPU/collective-only phase sum只是诊断下界，不是同 endpoint
speedup denominator。

这个实验用于说明 out-of-core execution 是独立且真实的问题，以及最强通用 exact
pipeline 之后还剩多少 residual movement/interference；它不预先证明 D3 有效。generic
winner 从合法 one-shot、S0、whole-group S1 和 bounded fine-grained exact pipeline 中
独立调优，并与 E1/D3-B 共用。S0 无论输赢都保留为可解释的顺序分组 control；如果它
获胜，winner row 直接复用该 cell。所有 tuning rows 进入 appendix。其他 capacity/GPU
点也独立选择合法候选中的最快 exact，不能预设 double buffering 必胜。

### E1：端到端性能

端到端主图使用两个资源边界清楚的 panel。

#### Resident panel

1. XP-QK-HET 按每 rank 相近 valid K/V bytes 的 2/4-GPU weak scaling：
   fastest exact、strong owner-local fixed-action mixed、complete D2；与 D2-B 共用。
   strong mixed 在 staged/fused contiguous candidates 中独立选快者。
2. X2-QK-HET 的 1/2/4-GPU supporting path使用同一 runner，提供 1-rank local ceiling 和
   rank-count sanity；不能替代 XP 的强制分片 headline。
3. R-KR 682 records 的 resident comparison，capacity-admitted 的 1/2/4 GPU：
   exact 与 complete D2；最终 rank points 由共同 usable-HBM qualification 冻结。

#### Out-of-core panel

XP-HET/288-GiB single-version/2 GPU 的四行来自 D3-A：

1. independently tuned all-exact，在合法 one-shot/S0/S1/fine-grained exact 中取快者；
2. fixed D1 actions + owner-local contiguous execution，并给予其合法的 generic pipeline
   tuning；若它仍不能输出 canonical target，则 adapter 必须计时；
3. D1+D2，在 S0/S1/S2/profile-aware generic candidates 中取快者；
4. D1+D2+D3。

X2-HET/约 288-GiB single-version/2 GPU 的三行来自 D3-B model sensitivity：

1. independently tuned exact；
2. tuned generic D1+D2（S0/S1/S2 winner）；
3. full EvoKV。

报告 absolute makespan、records/s、valid tokens/s、speedup、peak HBM、pinned/DRAM bytes、
phase breakdown、validation/commit/reclaim。主张是每层在自己的资源边界上贡献什么，而不是
强迫所有 bar 单调下降。第二行已经包含 owner-local 语义，因此只能称为
`owner-local contiguous fixed-action`，不能用它宣称完整 owner-compute 增量。如果 D1 的
logical plan 在这一强物理执行下仍不比 exact 快，这正是 D2 motivation，不应删除该结果。

### D1-A：Cost–fidelity frontier

**状态：EvoKV primary replication 已冻结；同 SLA structural comparator 需要新的
comparison protocol。**

| Dimension | Configuration |
|---|---|
| Primary EvoKV/endpoints | KuaiRand/QB/QK × S/M/L × seeds 0–3 |
| Structural controls | 九个预选 discovery checkpoints；不能赋予四-seed replication 含义 |
| Endpoint references | stale reuse、current projection、exact |
| Structural discovery baselines | fixed deep suffix、plain progressive-prefix replay、recent-token rectangles、arbitrary contiguous intervals |
| Primary integrated EvoKV actions | compiled affine repair、exact fallback |
| Supporting D1-only action | residual-delta progressive repair；只进质量/算法消融，不进入 D2/D3 headline |
| Metrics | measured GPU cost/full-exact、K/V recovery、score recovery、Top-k overlap、paired ranking delta |
| Output | 主文同 SLA frontier 与 aggregate EvoKV result；九个 discovery cells 的完整 raw-action frontier 放 appendix |

主张是 shared affine repair 在多个 workload/capacity 上形成稳定的 cost/fidelity interior
point。严格 task-quality gate 只通过 6/9 cells 的事实必须保留；不能挑掉 negative endpoint，
也不能把 task quality 变成 cache admission oracle。

这里不能把不同 protocol 的“已选择点”直接画成一个 Pareto。plain prefix 与
recent/interval discovery 使用过 20% recovery selector，而 D1 primary 是 50%，并以
75/90% 为 secondary operating points，而且两族历史 artifact 的 split seeds/probe users
也不同。正式同 SLA comparison 因此不是一次 JSON 重汇总：它必须先冻结一个新的
comparison protocol，在九个 discovery model-version cells 的共同 probe users 上重新
evaluate/timing structural actions，再用 50% recovery target 为每个 family 独立选择最低
实测成本的合法 action；depth/rank/token rectangle/interval 不能看 final users 或 seeds
1–3。这个 pass 不重训模型，也不改变 36 条 quality replication chains，但其 action-level
runs 在 manifest 审计后单独预算，不计入 system timed-cell portfolio。若不做这个新 pass，
主文就不能声称已有同 SLA frontier，只能在 appendix 报 historical raw actions 的
descriptive frontier。

D1-A 仍然只是 resident-kernel frontier：current projection/compiled 需要 old
`Norm(x)`，residual tier 需要 BF16 hidden suffix，reuse 读取 old K/V，exact 读取 raw
history。每条线必须报告 auxiliary-state bytes、fit/compile cost 和 amortization；主文不能
把 0.121× resident kernel cost 直接称为 end-to-end 系统 speedup。physical source 闭环由
D1-B 单独回答。

### D1-B：Source representation 与 repeated renewal

**状态：复用 R-KR frozen 结果。**

- source panel A1：`/data` ext4 buffered-POSIX normalized capsule 对同 boundary exact，
  保留 frozen no-transform/reuse-copy floor 与负结果；
- source panel A2：decoded pinned-DRAM-resident normalized capsule economics 对 paired
  pinned-DRAM exact；这是 1/4-GPU backup evidence，不冒充 ordinary pageable DRAM；
- source panel B：hot-HBM direct-old-K/V 对 hot-HBM raw-history exact；现有冻结结果没有
  hot-HBM no-transform control，若新增只能标 supporting measurement；
- source/operator：A1/B 报告 1/2/4 GPU，A2 只报告已有 1/4 GPU；各自报告 program bytes、
  compile time、source bytes、preload 和 break-even；
- lifecycle primary：Stage 4.9 corrected growing-history same-device rollout，682-record
  cohort 上 11 个 updates，下一步真正读取上一步输出；
- Stage 4.9 primary controls：fresh paired all-exact 与简单的
  `token_debt_total10` cost endpoint；`staggered_renewal_h12` 是 proposed；
- lifecycle supporting：Stage 4.6 fixed-history accumulation isolation，其中
  all-migrate/all-exact endpoints、matched-random、periodic/depth-only 与 fixed-quota
  candidates 只按其 selection/diagnostic role 报告；threshold refresh 只保留为 negative
  diagnostic，不作为唯一 comparator；
- metrics：per-edge/cumulative GPU cost、exact fraction、migration depth、最低
  cache/score/top-100 fidelity、lineage。

主图可将同边界 source-path bars 与 11-edge lifecycle curve 合并成两个 panel。
Stage 4.6 与 Stage 4.9 的 history、append 和 denominator 不同，不能把 0.2134× 与
0.100017× 拼成一条曲线。Stage 4.9 中 current-model append 在 retained migration 后执行，
并从 migration numerator 与 paired exact denominator 对称排除。closest external
`selective_contiguous` baseline 只放在 R-KR structural supporting panel：报告其
`certificate_passed=false` 的 profiled action，以及真正可发布的 exact fallback；它不属于
Q-SEM 3×3×4 aggregate。生命周期仍不宣称 organic serving trace、在线最优 selector 或
跨数据集 generality。Stage 4.9 只有 11 个 updates，短于 H12 的 12-edge horizon，因此
没有观察一个完整 renewal cycle，也不能宣称长期 deadline 已被完整验证。

### D2-A：Mechanism ablation

**状态：新必跑，R-KR/4 GPU 共 6 个 cells。**

所有 mixed variants 固定同一个 ActionPlan hash、owner map、embedding placement、
source/target endpoint 和 timer：

1. independently tuned all-exact；
2. owner-local staged-finalization、contiguous-target fixed-action mixed；
3. owner-local fused-finalization、contiguous-target fixed-action mixed；
4. segmented target，但不使用 shape-aware order，且 exact pools 不合并；
5. shape-aware segmented target，但 exact pools 仍不合并；
6. 再合并 `scheduled_exact` 与 `natural_exact` 的 physical exact pool，形成 complete
   D2，同时保留不同 provenance。

Row 1 是不同 ActionPlan/source semantics 的 all-exact operating reference，不属于
fixed-action cumulative chain；只有 rows 2–6 可以逐步归因。各行 factor 由下表冻结，
避免 rows 2–6 的一个 bar 偷偷改变两个机制：

| Row | Retained placement | Finalization | Target | Shape order | Exact pools |
|---:|---|---|---|---|---|
| 1 | exact reference | tuned exact | canonical | n/a | n/a |
| 2 | owner-local | staged | contiguous + timed canonical adapter | off | separate |
| 3 | owner-local | fused | contiguous + timed canonical adapter | off | separate |
| 4 | owner-local | fused | segmented | off | separate |
| 5 | owner-local | fused | segmented | on | separate |
| 6 | owner-local | fused | segmented | on | merged physically, provenance preserved |

现有 W3 所谓 `naive` 本身已经是 owner-local staged execution，因此正式论文不构造一个
故意搬运逐记录 old K/V 的弱 placement-oblivious headline。owner-local 在这里作为所有
fixed-action mixed rows 的强架构不变量，不单独声称 speedup；若以后实现
placement-oblivious supporting control，必须完整支付 old-K/V P2P，并与主六行分开。

group-version commit/reclaim 是正确性与资源生命周期语义，不包装成一个性能优化 bar。必报：

- retained lookup 是否为零、old-K/V peer bytes；
- total lookup/remote-vector bytes；
- retained rewrite bytes；
- padding、collective count/exposed time；
- temporary HBM、segmented consumer time、complete wave time。

主图把 all-exact 画成独立 reference，再对 rows 2–6 使用 cumulative mechanism bars，
并加 traffic/collective breakdown。R-KR 负责 shape heterogeneity；大 embedding 与
shard-count 结论由下面 X2 scaling 提供。这些 bar 只解释 rows 2–6 既定累积顺序下的
marginal effect；机制存在交互，不能把每段差值写成彼此独立、可任意相加的贡献，也不能
把 row 1→2 的差值算给任何 D2 mechanism。

### D2-B：1/2/4-GPU scaling

**状态：新必跑；与 D2-A 合计 17 个新 D2 timed cells。**

1. XP-HET weak scaling：每 rank 使用相近 valid K/V bytes/tokens，exact/strong contiguous
   mixed/complete D2，2/4 GPU。这里的 `strong contiguous mixed` 指在 staged/fused
   candidates 中独立选出的 strong owner-local fixed-action winner，不固定为较慢的
   staged v1。
2. X2-HET supporting scaling：固定同一 manifest 运行 1/2/4 GPU，验证通用 rank path 和
   local ceiling。
3. R-KR resident comparison：固定 682 records，在 capacity-admitted 的 1/2/4 GPU 上运行
   exact/complete D2；最终 points 在 usable-HBM qualification 后冻结。

报告 max-rank makespan、records/tokens per second、strong/weak scaling efficiency、
off-rank vector bytes、collective count、rank imbalance 和 HBM。

HET owner 由冻结 stable hash 决定，不为每个 GPU 数人为消除 imbalance。weak scaling
manifest 只在 length/action strata 上控制每 rank 的期望 valid bytes，不重排最终 owner；
必须报告 max/mean rank bytes、exact tokens 和 tail imbalance。吞吐主指标是 valid
tokens/s 与 physical GiB/s，records/s 只作辅助。XP 的 formal ledger 必须证明完整 fixed
state/replication 在 1 GPU 上不满足统一 admission；X2 只承担“同代码可运行”的 supporting
结论，不能代替 XP 的容量必要性。

### D3-A：Causal mechanism chain

**状态：新必跑，XP-HET/288-GiB single-version/2 GPU 共 9 个 cells。**

headline controls 与 causal rows 分开解释：

1. independently tuned all-exact winner，候选包含 one-shot/S0/S1/fine-grained exact；
2. owner-local contiguous fixed-action D1-only diagnostic，给予其合法的 generic pipeline
   tuning，并在 timer 内到达 canonical target；
3. D1+D2 sequential groups；
4. D1+D2 whole-group two-slot；
5. input segmentation；
6. generic fixed-FIFO bidirectional S2：独立调优一个 global \((I,C,O)\) triple，两条
   route 共用，不使用 route profile；
7. profile-aware generic list scheduler：读取与 EvoKV 相同的 stage profiles/capacity/
   credits，但不知道 action 语义；
8. route-specific I/C/O granularity：compiled/exact 各有一个 triple，但保持
   route-major order；
9. selected ResidencyPlan：完全复用 row 8 的两个 triples，只增加 stable interleave。

Rows 3–9 是因果链，不是七个各自独立调优的 headline 方法。它们必须固定完全相同的
HET WorkManifest、record/action/length/source-byte/target-byte multiset、capacity cuts、
route-internal order、\(B_{\mathrm{HBM,usable}}\)、
pinned-byte cap、CPU/streams、outer slots、one-lookahead 和 one-drain credit，只逐项打开
S1、input segmentation、global bidirectional triple、generic profile scheduling、
route-specific triples 和 route interleave。group admission 使用有效 source/target/
workspace bytes，不使用 record count。E1 与 D3-B 另外使用 independently tuned winner，
不能把 group-size 差异算成某个机制的贡献。

报告 makespan、input-boundary wait、output-credit wait、CPU packing、H2D/D2H、publish、
GPU/collective、HBM/pinned footprint、planner prediction error。这个 causal chain 要分别
显示：

- generic whole-group overlap 带来多少；
- inner bidirectional pipeline 如何移动并消除 bottleneck；
- 最后的 cross-route coordination 还有多少增量。

如果最终 route-specific triples 仍都选择 `(8,8,8)`，就只能声称 planner 可表达并安全执行
不同 granularity，不能声称 asymmetric granularity 已带来收益。D3 的核心 bar 是完整
hierarchical pipeline 相对 strongest generic S2/profile-aware scheduler 和 fastest
same-boundary exact 的结果，
而不是只相对 whole-group S1，或单独夸大当前约 1% 的 order-only improvement。若 D3
不能稳定胜过 strongest generic winner，它需要继续设计，不能靠较弱 baseline 获得第三个
贡献。一个 same-record HOM 的 `generic winner vs D3` pair 只作 supporting regularity
control，不复制整条九行链。

### D3-B：Capacity、GPU、action mix、model 与 edge sensitivity

**状态：新必跑。精确 cell 数在 XP qualification、HET manifests 与 generic winner
qualification 后去重冻结。**

#### Capacity

| GPUs | XP-HET single-version valid K/V | Records | Methods |
|---:|---:|---:|---|
| 2 | 144, 288 GiB | manifest-derived | tuned exact、tuned generic D1+D2、D3 |
| 4 | 288, 576, 720 GiB | manifest-derived | tuned exact、tuned generic D1+D2、D3 |

2-GPU/288-GiB 的三行复用 D3-A。576/720-GiB 点必须使用两个 NUMA node 的本地
ordinary-DRAM arena，不能落到 `/data`。
在 2-GPU/144 和 288-GiB 两点，capacity matrix 的 exact row 是 independently tuned
generic exact winner，M3 另计 formal S0；其他点采用相同 bounded tuning rule，只冻结
赢家进入 formal matrix。`tuned generic D1+D2` 在 S0/S1/S2 中独立选快者，因此没有额外
增加 capacity cells；profile-aware generic scheduler 也进入同一 winner selection。

所有方法共享 rolling group endpoint 和 host-capacity上限。all-exact 只读取 raw
IDs/metadata 并写 replacement group，mixed path 还读取所需 old-K/V extents；这一本来就是
两种算法的真实 source-state 差异。两者分别报告实际 source-read/H2D/D2H/writeback bytes，
不能为了“字节相同”给 exact 人为添加无用 old-K/V I/O，也不能让 mixed 隐藏额外 source。

#### Fixed-work GPU scaling

固定同一个 XP-HET/288-GiB manifest，比较 capacity-admitted 的 2/4 GPU tuned exact、
tuned generic D1+D2、D3；这里 generic winner 从 S0/S1/S2/profile-aware scheduler 中选择。
XP 的 1-GPU 点预计为 `fixed-state-capacity-not-admitted`。同一 runner 另在 X2-HET 上执行
1/2/4-GPU fixed-work supporting scaling，证明 rank-general code path，但不跨模型计算
speedup。两图都报告实际 \(\rho_{\mathrm{KV}}\)、per-rank bytes 和 topology。

#### Action mix

XP-HET/288-GiB/2 GPU 使用 label-free、predeclared retained-token budgets 生成
10/20/40% exact plans，并报告实际 record 与 token fractions。20% retained-token budget
是主配置；不再预设 205/410/819 这类等长 record counts。all-exact 是共同 reference。
只新增 10%/40% 下的 tuned generic D1+D2 和 D3，共四个 cells。

#### Model sensitivity

X2-HET 使用约 288-GiB single-version valid K/V，与 XP 主点按实际 bytes 匹配。冻结 X2
自己的 model edge、D1 program 和 ActionPlan，运行 tuned exact、tuned generic D1+D2、
D3 三个 cells。不能复用 XP 的 program 或 action plan。X1 只在核心结果稳定后作为更小
model extension。

#### Held-out update edge

- route profiles 来自与 benchmark users 不重叠的 512-user calibration pool；
- qualification users、final benchmark users 与 calibration users互斥；
- XP 增加一个独立 `theta1→theta2` edge；若训练成本先由 X2 bridge 完成，则 XP formal
  结果仍需在 freeze 前补自己的 edge-specific qualification；
- 在该 edge 上重新生成 edge-specific joint profiles 与 ResidencyPlan，再运行 tuned exact、
  tuned generic D1+D2、D3，共三个 cells。held-out edge 测的是机制在另一更新边
  的可重复性，不是把 `theta0→theta1` plan 直接跨 checkpoint 复用。

这组图回答 operating region 和 robustness，不支持 SSD、数据库、host-DRAM
oversubscription、online hotness 或 serving SLO。

### C1：Correctness、group lifecycle 与 overhead

**状态：必做，但不额外计入 timed-cell 数。**

每个正式 cell 的 correctness pass 覆盖其完整 workload。总测试集还必须覆盖：

- independent current-model full recomputation 对 exact runner 的逐 extent oracle；
- independent resident/reference D1+D2 semantic operator 对 mixed runner 的逐 extent
  tolerance oracle；
- 在独立 oracle 通过后，再检查 S0/S1/S2/D3 同 runner 的 byte digest 一致；互相相等
  本身不能排除 shared bug；
- direct-old-K/V 对 reference D1 operator；
- replicated 对 row-sharded exact lookup；
- segmented consumer 对 contiguous reference；
- 1/2/4 ranks、uneven rank、zero-delta 和 empty collective participation；
- HET 的 min/intermediate/max length、短历史无 eviction、长历史 sliding、非整除尾段和
  unpadded valid-extent oracle；
- 超过 \(2^{31}\) flattened offset 的 64-bit addressing；
- complete/exactly-once coverage、lineage 和 checksum；
- capacity/preflight mismatch；
- mid-H2D、mid-compute、partial D2H、missing/duplicate output、pre-group-commit abort；
- 任一失败后，每个 group/record 必须可判定为 old 或 new，已提交 group 幂等可读，未提交
  group 仍指向 old；禁止无 version/lineage 的 partial group。

在 720/800-GiB 点不能同时保留 old、reference target 和 candidate target。正确流程是：
在同一 live arena 上运行 rolling reference，逐 extent 保存 digest/metadata；timer 外从
checkpoint/raw history 按组重建相同 old source，再运行 candidate。digest、coverage 和
抽样数值 oracle 仍覆盖完整 record universe，不能因内存不足改成 partial workload。独立
reference 可以逐 extent 生成 digest 后立即释放，因此不需要额外完整 target。

另外重测两个不作为 speedup denominator 的 ceiling：

- 同一 work 的 HBM-resident compute-only ceiling；
- copy-only/isolated-stage perfect-overlap replay ceiling。

它们只解释当前配置还剩多少 movement/overlap headroom，不能与 ordinary-DRAM
consumer-ready end-to-end time 直接相除后宣称系统 speedup。

overhead table 报告：

- source/target arena reservation 与 dirty first-touch time，以及 cold-allocation-inclusive
  job time；
- D1 fit/compile time 与 program bytes；
- D2 lowering/WavePlan time 与 bytes；
- D3 joint profile、planning time与 plan bytes；
- execution-only 与 plan-inclusive single-wave time；
- segmented-consumer cost；
- per-group validation/commit/reclaim 与 job-final manifest；
- HBM、pinned、ordinary-DRAM standing/transient bytes。

## 7. 运行预算

### 7.1 复用、不重跑

| Evidence | Existing units | Action |
|---|---:|---|
| Q-SEM motivation | 3 datasets × 3 tiers × 4 seeds = 36 model-version chains | 直接按现有 protocol 汇总 |
| D1 cost/fidelity | 9 cells 的 discovery + frozen-seed replication | primary replication 直接复用、不重训；同 SLA comparator 必须新建 comparison protocol 和共同 probe，另计 action-level runs |
| R-KR source/operator/lifecycle | frozen single-configuration artifacts | 按同 source/timer boundary 与 Stage 4.6/4.9 protocol 重组，不跨边界合并 |
| D2 W3、D3 M0/M1 development | 若干单次 development profiles | 只用于选机制和预算；不进入 paper table |

D1 同 SLA baseline 工作按 **9 个 discovery-cell comparison bundles** 管理，不产生新模型
training chains。每个 bundle 的结构 action 数取决于 3/6/9-layer action manifest；正式运行
前必须先枚举并 hash 全部 depth/suffix/rectangle/interval candidates，届时冻结准确的
action-level correctness/timing run count。这个数量尚未由当前 artifacts 闭合，所以不能
塞进下面的 system-cell 总数假装已经精确。

### 7.2 新 paper-core timed cells

| Family | Current accounting |
|---|---:|
| M2 logical-to-physical | 14 |
| M3 extra exact-S0 points | 2 |
| D2 resident ablation、XP 2/4-rank、X2/R-KR rank support | XP/rank admission 后去重 |
| D3 causal、E2E、capacity、GPU/action/model/edge sensitivity | generic winners 后去重 |
| **Current envelope** | **约 70–80 cells** |

每个 cell 为一次 correctness、一次 warmup、五次 measured：

- 70–80 correctness jobs；
- 70–80 warmup jobs；
- 350–400 measured jobs；
- **合计约 490–560 次 complete executions**。

调优/profile runs 不算 paper cells。它们使用小的 disjoint profile cohort，且每个 baseline
有明确候选上限。C1 failure injection、额外 functional tests、训练、profile 和
materialization 都在上述公式之外；D1 同 SLA comparison 也按 action-level manifest 另计，
因此该 envelope 不是整个项目的总命令数。XP/HET qualification 完成后，本文必须把
envelope 替换成 exact de-duplicated ledger，再开始 formal execution。

### 7.3 时间和资源预估

现有 X2 development full job 约为几十秒量级。XP-HET formal matrix 的合理规划是：

- X1/X2 新 edge 与 artifact preparation：数十分钟到数小时；XP row-coverage builder
  可能达到数小时至一个独占机器日，qualification 后用实测替换；
- canonical checkpoint、program、digest 与 compact result 的峰值新增暂按 260–330 GiB
  envelope，XP optimizer/scratch qualification 后更新；
- profile/tuning：约 4–8 个独占 node-hours；
- 约 490–560 次 formal system-cell execution：约 8–16 个独占 node-hours；
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
| Figure 4 | D2 mechanism ablation、XP 2/4-rank forced-sharding 与 X2/R-KR rank support |
| Figure 5 | D3 causal mechanism chain 与 wait/bubble breakdown |
| Figure 6 | HET capacity、GPU、action mix、X2/XP/held-edge operating region |
| Table 2 | planning/compile/group-lifecycle overhead、memory footprint、correctness/failure coverage |

完整 3×3×4 quality cells、所有 raw timing samples、额外 metrics、可选 topology/XL stress 放
appendix，避免主文变成结果目录。

## 9. 结果不理想时如何回头

实验矩阵不是为了“证明已经决定好的结论”，而是为了暴露设计真正有效的范围。

- 如果 M2 bytes 随 exact fraction 下降但 time 几乎不变，先用 rank wait、padding、
  collective calls 和 GPU compute 找到兑换失败的位置；D2 不得仅凭 byte model 进入论文。
- 如果 D2 只在 R-KR 有效而 XP scaling 无收益，检查 large embedding 下的 collective
  fragmentation、owner balance 和 exact pool；必要时回到 D2 lowering，而不是扩大数据
  掩盖问题。
- 如果 complete D2 只胜 staged control、不胜 independently tuned fused-contiguous
  fixed-action baseline，不能把 fused finalization 的收益算给 segmented/shape-aware
  lowering。
- 如果 D3 只胜 sequential S0 或 whole-group S1、不胜 independently tuned S2 与
  profile-aware generic scheduler 中的强者，它只是通用流水实现，不足以成为第三个
  design。优先研究
  DMA/affine/collective interference、phase-aware pacing 和
  collective-arrival-aware planning。
- 如果 D3 在 288 GiB 有效、在 576/720 GiB 消失，先检查 NUMA/locality 与 group writeback
  serialization；这正是 capacity plot 要揭示的 operating boundary。
- 如果 X2/XP 方向冲突，不把两个点平均；分别报告 compute-bound 与 movement-bound
  regime，并据此收窄系统 claim。
- 如果最大点因机器当前占用或 backend 失败，不能用 logical cap、稀疏文件或未触页内存
  替代。释放独占资源、实现正确 DRAM arena，或把该点诚实降为未完成 extension。

## 10. 核心完成条件与可选扩展

paper-core 完成意味着：

1. 每层 baseline/control 先于本层 proposed 冻结，按第 5.5 节先完成 D2 stack 再建立 D3
   foundation；qualification 后冻结的全部去重 cells 均具备相同 revision 内的
   correctness/warmup/5 repeats；
2. M2 明确给出 logical tokens、physical bytes 与 wall time；
3. D1 comparator 不跨 recovery target、source tier 或 Stage 4.6/4.9 protocol 合并；
4. D2 同时有可归因的六行 R-KR ablation、XP 强制分片 2/4-rank evidence 和
   X2/R-KR 的 1-rank support；
5. D3 同时有 tuned exact、generic S2、profile-aware generic scheduler、HET causal chain、
   288/576/720 GiB single-version capacity 和 held-out edge，并相对 strongest generic
   winner 报告结果；
6. rolling groups 可被 segmented consumer 读取并进入下一 wave；
7. plan、group writeback、commit、reclaim 与 failure semantics 都在 overhead/correctness 表中
   闭合；
8. exact 和 mixed 均通过独立 semantic oracle，而不只是互相 byte parity；
9. 所有 paper claim 都能指向一个同边界、同 revision、可复现的 result family。

核心稳定后才考虑：

- XP-HET/800-GiB single-version/4 GPU，三方法共三个可选 cells；
- X3 32L/H2048：exact、D1+D2、D3 三个 cells；
- GPU0+GPU2 的跨 NUMA/SYS topology sensitivity；
- \(\rho_{\mathrm{KV}}\approx5\) 的 capacity-matched 1/2/4-GPU plot；
- 两个独立 576-GiB rolling waves 的 long soak；实际 DRAM read/write traffic另报，不能
  把累计 bytes 写成一个 1-TiB live cache；
- synthetic embedding contention，仅作为 resource characterization，不叫 serving
  workload。

下一步不是直接启动整个 490–560-execution envelope，而是依次生成：

1. formal QK role split、HET primary 与 HOM control manifests；
2. XP qualification 与 X1/X2/XP model-edge、program、primary compiled/exact ActionPlan；
3. 带 storage-configured safety floor（当前为 100 GiB）、临时 reshard 回收和
   compact-result schema 的磁盘 preflight；
4. 记录但暂不执行独立 Benchmark Qualification：NUMA-aware DRAM arena、共同 usable-HBM
   budget、1/2/4 rank、timing purity 与最大地址；
5. 四个新 protocol/config families：M2、D2、D3/E2E、correctness/group lifecycle；
6. 补齐并验证 fine-grained exact、staged/fused owner-local contiguous、generic S2 和
   profile-aware generic baseline runners；
7. baseline 开始前再按第 5.6 节执行 qualification 与 independent oracle canary；
8. 按第 5.5 节先跑 M2/D2 foundations，再完成并冻结 D2 stack；
9. 在该 stack 上跑 M3/D3 foundations，最后才运行 D3 proposed/ablation cells。
