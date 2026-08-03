# EvoKV Design 3 方向：Hierarchical Out-of-Core Pipeline

最后更新：2026-08-03

状态：**route-aware ResidencyPlan、grouped development E0 和 contribution diagnostics
已完成；正式重复、协议和论文结果尚不存在**。
本文档描述当前最可能的 D3 形态，不是阻塞机制发现的接口合同，也不能作为 paper claim 的
证据。D1/D2 的已有证据仍按各自冻结协议解释，但新的 D3 development 可以从轻量 adapter
开始，并允许根据 DRAM/HBM 实测形成新的跨层 `stack_revision`。

2026-07-31 起，论文级 successor benchmark 采用以下统一边界：自然变长
`X-QK-HET` 是 D1→D2→D3 的 headline workload，同记录 masked-512 `X-QK-HOM` 只作
matched control；集成路径的主动作是 `compiled|exact`；XP 用 base-period 真实语义实体
主动构造单卡容量不可行的 embedding/model foundation；D3 维护一份 live cache，并以
`stage → compute → writeback → validate → group commit → old-group reclaim` 滚动替换
capacity groups。本文记录的 fixed-512、GPU0/GPU1、complete old + private target M1
结果全部保留为历史开发证据，不能替代 successor 的正式结果。

2026-08-03 已完成可复用模型输入的收敛：QK LR0.15 theta0--theta4 为 primary，QB
`u30_e3` theta0--theta3 为 secondary。旧 fixed-512 D3 专用模型副本已删除，但 compact
结果和机制代码保留。新的 D3 runner 必须从 machine registry 绑定 checkpoint，并为选定
edge/records 重新生成 K/V、ActionPlan、WavePlan 与 ResidencyPlan。

具体的两卡 foundation benchmark、H12 semantic canary、真实 QK 物理 out-of-core
workload、分阶段实现和回退条件见
[DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md)。

## 0. 方向裁决

EvoKV 当前按同一迁移任务分成三个层次：

1. **D1 决定算什么。** 集成主路径输出 `compiled|exact` `ActionPlan` 和自然变长
   extent；progressive residual replay 保留为 D1-only supporting extension。
2. **D2 决定多卡怎样执行。** 它给出 owner、operator、physical compatibility、
   collectives、segmented layout 和 group-valid output contract。
3. **D3 研究物理 extent 何时进入和离开 HBM。** 它把超显存工作集切成 capacity-safe
   micro-waves，调度 DRAM→GPU→DRAM 流水，并完成 group commit/reclaim。

这三层是当前叙事骨架，不是永久接口。探索分成两条轨道：

- **isolation track：** 在一次 revision 中保持 D1/D2 工作不变，只研究搬运和调度；
- **co-design track：** 允许容量与通信结果反向调整 D1 action granularity、D2 owner/pool/
  layout，再为整个新 stack 重跑 baselines。

不允许的是把旧 baseline 与修改后的 candidate 直接比较；允许的是在运行前重新生成一个
完整、可记录的新 revision。

新的 D3 核心问题是：

> 当一份 live、自然变长的 K/V cache 超过每 rank 可用 HBM 时，怎样在 1/2/4-rank
> 执行器上组织 DRAM↔HBM、D2 分布式计算和滚动替换，并在必要时联合调整上游执行组织，
> 得到超过最强通用 profile-aware scheduler 的实际收益？

按显存容量顺序分组只是基础 baseline，不是贡献。普通 prefetch 或 double buffering 也不是
贡献。D3 必须证明：在相同 HET manifest、source-byte multiset、rolling endpoint 和容量
预算下，它相对独立调优的 fixed-FIFO segmented pipeline 与 profile-aware
work-conserving generic scheduler 中的强者仍有可归因收益，而不只是把 whole-group
buffer 切成 microbatches。HOM 只回答收益是否依赖长度异质性，不能取代 HET headline。

当前不主动恢复 organic mixed-version program graph 或复杂 renewal controller。但若简单
out-of-core profile 表明 action/owner/granularity 必须协同，允许先做小规模跨层候选，再根据
最终机制重新划分论文中的 D2/D3 边界。

## 1. 系统边界

目标数据路径为：

```text
ordinary host DRAM live group
  → bounded pinned input staging
  → H2D
  → D2 owner/shard execution
  → D2H
  → bounded pinned output staging
  → ordinary host DRAM replacement group
  → validation → group commit → old-group reclaim
```

边界内包括：

- CPU source→pinned 和 pinned→target copies；
- H2D/D2H；
- D2/D3 mechanism 图使用 execution-only timer；已冻结且绑定同一 stack、可跨 wave
  复用的 `ResidencyPlan` construction 可单列，但当前 run 必需的 planning 不得免费；
- E1 的 end-to-end primary 从 model checkpoint publication 后开始，必须计入该 edge 的
  D1 fit/compile、D2 lowering/routing、D3 profile/plan construction 和 rolling execution；
- D2 lookup、collective、compiled/exact/append compute；
- replacement-group writeback、validation、group commit、old-group reclaim 和 staging
  reclaim。

边界外包括：

- SSD、database、object store、remote storage 到 host DRAM；
- serving trace、hotness、online admission、foreground SLO；
- host-DRAM oversubscription；
- model-parallel dense trunk；
- durable cross-host transaction。

Successor benchmark 不要求 ordinary host DRAM 同时保存 complete source 和 complete
private target。任一时刻只需一份 live cache、当前/相邻 group 的 bounded replacement
state 与 staging；每组验证后独立提交并回收旧 extent。失败恢复只需区分该组的
`old-valid`、`replacement-ready` 和 `committed` 状态，不宣称全局 epoch 原子切换。
历史 M0/M1 的 complete private target endpoint 保留用于回归和 byte-parity，不再是正式
容量或事务边界。

## 2. 当前工作快照

### 2.1 D1 `ActionPlan`

D1 在一个 `stack_revision` 内记录 source/target version、policy、provenance、counts 和
content hash。H12 保留为 semantic canary；历史物理 M1 在 QK 上固定 2,048 个
512-token records，其中 410 exact、1,638 compiled。每条 record 固定：

- record id、`compiled|exact` action 与 reason；
- old/retained/delta/latest/target-prefix/final extents；
- previous-cache presence、last-exact version 和 migration depth；
- history 与 extent-identity hashes。

该历史 M1 使用 24L/H1536、16.364-GiB global FP32 embedding 和一个短
`theta0→theta1` edge。完整 committed old K/V 为 144 GiB，complete private target 也为
144 GiB；两 rank 各持有 1,024 records 和 72 GiB old/target payload。17 个 route-pure
capacity groups 是这组 S0/S1/D3 development artifacts 共同的固定物理边界。

Compiled program 是由 D2 adapter 绑定的独立 D1 artifact，不是 action-record field。
Successor 集成 benchmark 不引入 `progressive` route；若 D1-only supporting experiment
使用 progressive residual replay，必须在独立 protocol 中版本化 auxiliary state，D3
不能自行推断或将它混入 headline timing。

Successor 还从同一 QK base-only entity universe 导出两份共享模型、edge、用户选择规则和
action 语义的 manifest：

- `X-QK-HET` 保留自然 old/retained/evicted/append/target 长度，并以 valid K/V bytes
  定义容量、分组和吞吐；
- `X-QK-HOM` 复用相同 HET records、valid histories、actions、items 和 owners，只将物理
  layout masked-pad 到 512 slots，作为长度/shape 异质性的 matched control。

XP 固定为 2,859,835 个 base-period semantic rows 加一个 padding row，即
2,859,836×4,096 physical FP32 table（43.638 GiB）、
owner-side E4096→H1536 projection 和 24L/H1536 core。硬件 HBM cap 先独立冻结，
qualification 再验证单卡不可复制、2/4-rank admission 和短 edge；不得通过 inert padding
rows、raw-E4096 cross-rank returns 或性能结果制造/选择配置。强制分片只计算收到真实
optimizer update 的 active rows；两个 formal edges 上 all-exact 与所有冻结 fixed-action
exact/append/fallback 的 semantic-request union 必须全部 active 并记录 hash，且 active
embedding 加 dense/projection bytes 自身超过单卡可分配量。

Isolation track 不重新选择 action。Co-design track 可以在运行前生成新的 D1 plan，但必须为
新 revision 重跑 baseline。Recommendation labels、per-user drift 和预测 task gain 仍不能
作为 D3 routing signal。

### 2.2 从最小 `WorkManifest` 到正式 constraints

最初两卡 benchmark 只需从当前 D2 runtime 导出：

- record/action/owner；
- source/target extents 与 byte counts；
- operator/pool key；
- source/target locators。

正式 isolation evaluation 必须补充可重建的 capacity-independent constraints：

- logical GPU owner 和 embedding routing；
- operator；
- `(S,R)` compiled shape-bin membership；
- `F`-keyed exact-pool membership；
- collective dependency/order constraints；
- segmented target layout；
- per-group source/target version、coverage、lineage、validation、commit/reclaim contract。

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
- route-specific input/compute/output segmentation；
- 保持各 route 内部顺序的跨 route stable interleave；
- one-lookahead/one-drain prefetch/execute/writeback launch order；
- pinned-buffer credits、backpressure 和 NUMA/PCIe staging placement；
- replacement-group validation、versioned group commit 和 old-group reclaim 顺序。

Isolation track 的 scheduler/baseline 共享同一 `WorkManifest`。Co-design track 可以生成新的
ActionPlan/owner/pool/layout，但必须记录新 revision 并重跑对应 baselines。

## 3. 公平的 source contract

Isolation-track mixed baselines 使用相同的 action-required source-byte multiset：

- compiled 只读 valid retained old K/V；
- exact 读 raw history IDs，不读无用 old K/V；
- append 只读 suffix IDs；
- target 按 D2 segmented layout 分配和写回。

Progressive residual replay 只出现在 D1-only supporting family；若启用，必须显式读取并
计量 BF16 hidden suffix，不能改变 successor integrated baselines 的 source contract。

选择性读取是 isolation track 的共同执行行为，不是 proposed scheduler 相对 strong
double-buffer baseline 的创新来源。Co-design track 若改变 action/source bytes，必须报告变化
并在新 revision 下重跑 baseline。

All-exact 必然有不同的 ActionPlan、physical constraint plan 和 raw-history source bytes。它只与
mixed 方法共享：records、target model、ordinary-host source tier、target
dtype/layout/durability、GPU topology、per-rank HBM budget、timer 和 manifest endpoint。必须
单独报告实际 source bytes。

## 4. 候选机制

### 4.1 Capacity groups are the outer boundary

先从 D2 compatibility membership 建立 route-pure logical pools，再按每 rank 的
valid input、execution transient、replacement output 和 fixed state 做 capacity-safe cuts。
若一个 pool 超过 HBM，D3 必须切开；它不能声称保留一个物理 global launch。HET group
按 bytes 而不是 record count admission，并保留自然 owner imbalance。历史 M1 使用
17 个 group-64 cuts；successor 不继承这个固定 group size。顺序执行是 S0，
whole-group lookahead-1 是 strong S1；二者共享完全相同的 D1/D2 work。

### 4.2 Historical M1 mechanism: rate-matched route-aware pipeline

历史 M1 Strong S1 的外层流水为：

```text
prefetch group i+1 || execute group i || drain group i-1
```

它已经把 fair S0 从 48.238 秒降到 32.703 秒，但 whole-group input 仍留下
6.20--6.79 秒 boundary wait。该 development candidate 在每个 capacity group 内再切
microbatches：

```text
pageable pack j+1 || H2D j
unchanged D2 compute/collective
D2H j+1           || ordinary-DRAM publish j
```

这两级流水构成固定顺序的 precursor。输入和输出各交替复用两个已有 pinned components；
CUDA events 在 CPU 读取或覆盖 pinned storage 前建立生命周期边界。外层仍只有一个
prefetched group、一个 outstanding drain 和一个 main-thread collective issuer，因此收益
不是来自第三个 resident group、无界队列或改变 D1/D2 work。

Input-only segmentation 是必要的因果 probe：它把 makespan 降到 31.096 秒，同时将瓶颈转移
为 3.71--4.87 秒 output-credit wait。历史 v1 双向 segmentation 再将 makespan 降到
28.885 秒，相对 S1 为 1.133x；full outputs 与 S1 逐字节一致。由于 runner identity
不同，28.885 秒不能作为当前 plan 的 order-only control。

历史 `ResidencyPlan` 在这个 precursor 上再加入两个互相依赖的决定：

1. compiled/exact 各自独立的 input-segment、compute-batch、output-segment；
2. compiled 与 exact 两条内部有序序列之间的 stable interleave。

Planner 只组合来自同一个 no-plan source 的 compiled/exact profile；每个 stage 先取最慢
rank，尾组按离散 segment 数缩放，再使用与运行时一致的 one-lookahead/one-drain
bounded-flow recurrence。小搜索空间穷举所有保持两条 route 内部顺序的 interleave，大空间
使用 Pareto-beam DP。预测全局最优点 3% 内再按 HBM、pinned memory 和 segment 数选择，
该 tie 区间锚定全局最小值，不依赖遍历顺序。

当前选择仍为两 route `(8,8,8)`，但 launch order 变为
`[13,0,1,2,3,4,5,6,7,8,9,10,11,14,12,15,16]`。在同一 exact stack/hash 下，route-major
control 为 28.514442098 秒，selected order 为 28.147194647 秒，即 1.013047x、wall time
降低 1.2879%；相对 S1/fair S0 为 1.16186x/1.71379x。预测 29.244944224 秒，比实测高
3.90%。两 rank 各 77,309,939,712-byte target 均与 S1 逐字节一致，coverage
complete/exactly-once。

Plan 内嵌 profile，并绑定 compiler/program/source code、Torch/CUDA、GPU UUID/PCI、store
tier、groups、checkpoints、HBM 和 pinned limit；两 rank 在 target creation 和 collective
前统一执行 capacity preflight。历史 mechanism execution timer 将 plan/profile
construction 单列；这不覆盖 successor E1，E1 的 update-inclusive primary 必须计入
edge-specific profile/plan。当前 triple 仍对称，不能声称 route-asymmetric granularity
收益已验证；input-16/output-4 仅在各自观察点没有提升。相邻 identity-only revision 的
29.7169→28.0497（5.61%）只作为波动诊断。

### 4.3 Per-rank concurrent-buffer admission

对 micro-wave `i`，每个 rank 都必须满足：

```text
fixed(model + embedding shard + program + context)
+ input(i+1)
+ execution transient(i)
+ replacement/output(i-1)
<= physical HBM - allocator/safety margin
```

容量以共同 qualification 得到的每卡可用 HBM 为硬约束，不能只比较 aggregate bytes；
38 GiB 仅可用于预估，不能被写成固定事实。主 sweep 使用

$$
\rho_{\mathrm{KV}}=\max_r
\frac{B_{\mathrm{live\ single\ version},r}}
{B_{\mathrm{HBM,usable},r}-B_{\mathrm{fixed},r}}.
$$

其中 \(B_{\mathrm{fixed},r}\) 包含该方法实际需要的 model、embedding shard、program 和
runtime fixed state；exact 不被迫加载无用 program。每个 admitted group 还必须实测
old-group input、target-group shadow、workspace 和 allocator peak，不能仅凭
\(\rho_{\mathrm{KV}}\) 推断可执行性。

历史 M1 的 old+private-target 共 288 GiB，证明旧 endpoint 超出两张 A40，但这不是
successor 的 \(\rho\) 定义。正式 D3 主点要求**单个 live HET cache version** 本身形成
out-of-core capacity boundary；DRAM peak 是该 live version 加 bounded group shadow/staging，
而不是两个完整版本。历史
group-64/microbatch-8 的 peak allocated HBM 为 29.27/29.09 GiB，reserved 为
39.42 GiB。microbatch-16 更慢且把 reserved 提高到 41.54 GiB，所以不选。
当前 exact-stack pair 中，route-major/selected 的 observed reserved 为
39.42/35.25 GiB；这是一条开发态伴随观察，尚不能独立宣称为普遍的调度 memory saving。

### 4.4 Fallback, not the active mechanism

若 independently tuned/repeated E0 或 sensitivity 推翻当前 development-positive
crossover，才恢复 resource-complementary action packing、rank-aware byte balance、
collective-arrival-aware prefetch 或 D1/D2/D3 co-design。任何改变 actions、owners、pools
或 layout 的版本都必须获得新 `stack_revision` 并重跑其 S0/S1。

## 5. Motivation 与实验

### 5.1 Development characterization

H12/W2 M0 已完成最小接口验证，但真实物理结论来自 QK M1。相同 2,048 records、固定
D1/D2 work、17 groups 和 ordinary-DRAM endpoints 下，当前链路是：

```text
fair sequential S0        48.238 s
strong whole-group S1     32.703 s
input-only causal probe   31.096 s
historical v1 segmented   28.885 s
v3 route-major control    28.514442098 s
v3 route-aware plan       28.147194647 s
```

这条链同时报告 CPU staging、H2D/D2H、GPU compute、collective、pipeline waits、peak HBM
和 total completion time。旧 normalized-capsule path 仍只能作为风险警示，不能与
direct-old-K/V M1 pooling。

### 5.2 D3 baselines

当前主 baseline/candidate：

1. fair sequential capacity groups；
2. strong action-oblivious whole-group pipeline；
3. generic fixed-FIFO bidirectionally segmented pipeline：两条 route 共用一个独立调优的
   global `(I,C,O)` triple，不读 route profile；
4. profile-aware work-conserving generic scheduler：把每个 group 当作 opaque job，可以
   读取与 EvoKV 相同的 stage profiles、capacity、credits 和 bounded-flow recurrence，
   但不能读取 compiled/exact 标签、route-specific parameter sharing 或 EvoKV 的
   stable-interleave objective；
5. route-aware ResidencyPlan；
6. independently tuned same-boundary all-exact E0，其候选同时包含 grouped 与 bounded
   fine-grained exact pipeline。

E0 现有开发态顺序/双 slot 时间为 44.639/33.549 秒。补充的 owner-local、naive-staged
D1-only 路径为 57.597 秒，当前 binary 的 sequential D1+D2 rerun 为 49.753 秒。它们说明
D1 的逻辑稀疏性在超显存场景下不会自动转化为 wall-time 收益；D2 能减少连续重写和
collective fragmentation，但最终相对 strong all-exact 的明确 crossover 仍依赖 D3。
这些均为单次 development diagnostics，不是 formal waterfall；D1-only 也不是
placement-oblivious owner-compute ablation。

Isolation-track variants 共享 source-byte multiset 和 `WorkManifest`。Co-design variants 记录
不同的 action/owner/source bytes，并在同一新 revision 下重跑 baselines。所有 speedup 只在
ordinary-DRAM→GPU→ordinary-DRAM 同 endpoint 内形成。

机制因果链中，route-specific control 为 compiled/exact 分别选择一个 `(I,C,O)` triple，
但保持 route-major order；selected-order row 必须复用这两个 triples，只增加 stable
interleave。headline ResidencyPlan 可以独立联合调优，但不能拿它与不同 cuts/resources 的
control 做机制归因。

### 5.3 Metrics

- absolute wall time、records/tokens per second；
- CPU staging/H2D/D2H bytes、bandwidth 和 exposed time；
- GPU/collective busy、overlap 和 bubbles；
- compatibility fragmentation、launch/collective counts；
- peak per-rank fixed/input/execute/output HBM；
- HET old/retained/append/target valid-token/byte quantiles、natural owner imbalance；
- replacement group writeback、validation、commit、abort、old-group reclaim；
- full-payload K/V、hidden、score、Top-k、padding、lineage、manifest correctness。

## 6. 论文条件，不是开发前置 gate

D3 最终进入论文结果时应满足：

1. full-payload correctness 与 bounded-memory admission；
2. ordinary-DRAM rolling same-boundary sequential、whole-group double-buffer、generic
   segmented、profile-aware generic、D3 和 all-exact；
3. 同一 rank-parameterized runner 支持 1/2/4 rank；XP 以 capacity-admitted 2/4-rank 为
   headline，X2/R-KR 提供 1-rank sanity；
4. 相对 independently tuned fixed-FIFO segmented 与 profile-aware generic 中更强者的
   可归因收益；
5. 至少一个相对 fastest same-boundary all-exact 有意义的 operating point；
6. HET headline、HOM matched control、容量/action mix/model/held-edge sensitivity；
7. group validate/commit/reclaim 与下一 wave segmented-consumer compatibility；
8. report positive region 和 exact-preferred crossover；
9. 不把 development timing 写成 formal result，也不扩张到 serving、SSD 或 remote tier。

若只胜过 sequential loop、whole-group double buffer 或 fixed-FIFO，但不胜过
profile-aware generic baseline，D3 只是 implementation path。若 unavoidable old-K/V input lower bound 已慢于
same-boundary exact，继续调 scheduler 没有意义，应停止并重新研究 source
representation，而不是无限调参。

## 7. 当前实现与正式化缺口

历史两卡 adapter、M1 store、独立 route-specific segmentation、stable interleave、
plan/stack hash 和双向流水已经存在。它们依赖的上游事实是：

- immutable H12 `ActionPlan` 及其 hash；
- deterministic owner/embedding rules；
- Stage-A single-rank wave adapter；
- W3 owner-local `(S,R,F,record_id)` ordering、fixed-size resident extents 和 merged-exact
  execution membership；
- segmented destination implementation；
- D2 private-target、coverage、lineage 和旧 commit/abort semantics；
- real-history source extents 和现有 D1 exact/compiled operators。

Successor formal isolation evaluation 仍需要：

- HET primary/HOM control manifest；同一 HET valid-byte cohort 冻结 nominal
  record/capacity point，但 HET/HOM 分别按实际 allocated input、replacement shadow、
  workspace 和 padding 做 capacity-safe micro-wave packing；
- capacity-forced XP foundation 和 1/2/4-rank-capable runner；
- 一个 normalized、capacity-independent 的 D3-facing constraint schema；
- owner/operator/program/membership/collective/layout/group-version 的完整序列化；
- 对当前 runtime 的 record-coverage/parity 检查；
- rolling replacement、group validation/commit/reclaim 和 segmented-consumer closure；
- independently tuned fixed-FIFO、profile-aware generic 与 all-exact；
- held-out calibration/evaluation boundary、正式重复和 action/capacity/model sensitivity。

此外，正式 evaluation 不能依赖：

- W3 timing 是正式 D2 结果；
- resident D2 schedule 能放入 HBM；
- segmented target 已被 serving 或下一轮直接消费；
- direct-old-K/V ordinary-DRAM development timing 等同于 formal evidence；
- Python plan/materialization、CPU copy、pinned staging、commit 或 reclaim 免费；
- old+complete-private-target 容量等价于单版本 out-of-core 容量；
- fixed-512 record count 可以代表 HET valid-byte capacity；
- destination-v4 normalized capsule 能代表本 D3 source contract。

这些正式工件没有阻塞 M0/M1 mechanism discovery。当前 artifact 记录
`scientific_result=false`、`formal_design3=false`、`stack_revision`、per-rank capacity ledger
和 timer components。当前 plan/profile 自身已有 stable content hash，但它不是正式 D2
constraint exporter。Benchmark qualification 的项目登记在
[../11_benchmark_qualification.md](../11_benchmark_qualification.md)：它不阻止 workload、
runner、baselines 或 D3 机制设计，只有在结果要晋升为正式论文证据前才冻结具体配置和
protocol。

## 8. 实施顺序

1. 已从 H12/W2 导出最小 `WorkManifest` 并完成历史 M0；
2. 已构造 fixed-512 QK M1 ordinary-DRAM source/target 和历史物理 oversubscription；
3. 已完成历史 fair S0 与 strong S1；
4. 已构造并验证历史 generic fixed-FIFO bidirectionally segmented S2；
5. 已实现 route-aware ResidencyPlan，并在同一 exact stack/hash 上完成 route-major 与
   selected-order 配对、full byte parity 和 capacity preflight；
6. 已完成 same-boundary grouped E0 和 owner-local D1-only contribution diagnostics；
7. 下一步生成 QK HET primary/HOM control manifests；先用 HET valid bytes 冻结同一
   nominal cohort，再分别按 HET/HOM 实际 allocated source+shadow+workspace bytes 构造
   capacity-safe rolling groups，HOM masked padding 完整计入 admission；
8. 同时完成 XP qualification、rank-parameterized 1/2/4 runner、rolling group lifecycle 和
   exact/S0/S1/S2/profile-aware generic baseline foundation；
9. 在 baseline foundation 上重新 profile、探索并比较 D3；若修改上游形成新的
   `stack_revision`，为该 revision 重跑全部对应 baselines；
10. 结果晋升前再执行登记的 qualification、正式重复和 sensitivity，并在
    `docs/eval_protocol.md` 冻结 protocol family。

## 9. 更远期反馈层

Organic mixed-version program graph、program composition、communication-aware semantic selection
和 cross-wave bounded renewal 保留为 D1+D2+D3 之后的 future controller，不属于当前 D3。

该控制器若恢复，仍必须满足：

- 不使用 recommendation labels 或 task-quality admission oracle；
- exact renewal/depth/deadline 不可被 composition 取消；
- abort/reject 不推进 lineage；
- 允许已提交新组与尚未提交旧组显式共存；不把未验证、无版本或 lineage 错误的
  replacement group 当作 current；
- 需要独立 organic workload contract 和新的 protocol family。

当前不实现该反馈层，也不把它计入 EvoKV 的第三个 design。
