# EvoKV Benchmark Qualification Registry

最后更新：2026-08-03

状态：**实验配置与 runner 的待验证清单；不是冻结 protocol、paper result 或当前设计的
阻塞 gate。**

本文只记录在正式 baseline/result promotion 前必须回答的可行性、公平性和可复现性问题。
当前可以继续设计 HET/HOM workloads、实现 1/2/4-rank runner、构造 baselines 和探索 D3；
具体阈值、命令、失败回退和执行顺序在开始 qualification 时再单独讨论并冻结。

## 1. Workload 与模型身份

- 每个 qualification round 先通过
  `scripts/verify_evokv_selected_checkpoints.py --full-payload`，并把
  `selected_checkpoint_registry_development_v0.json` 的 hash 写入 round manifest。若使用
  新重训 chain，先建立新的 registry revision，不能悄悄覆盖当前 QK/QB identity。
- 从同一 QK base-only entity universe 生成自然变长 `X-QK-HET` primary manifest 和
  same-record、masked-512-layout `X-QK-HOM` control manifest。
- HET 必须记录 old、retained、evicted、append、target length quantiles、512-token
  saturation、valid tokens、valid K/V bytes、自然 owner/rank imbalance 和 stable-hash
  selection identity。
- QK HET 是固定 model edge 上由真实 IDs 与用户内 ordinal order 构造的 heterogeneous
  cache snapshot，不是跨用户共时 trace；manifest 与论文措辞都不得声称按天到达或自然
  同时发生。
- HOM 必须复用 HET 的 record IDs、valid histories、actions、items 和 owners；padding row
  严格 masked，并分别报告 valid 与 allocated bytes。它只用于 matched causal control，
  不得在 HET/HOM 之间按最终 speedup 选择结果。Nominal record/capacity point 由同一 HET
  valid-byte cohort 冻结，但 HET/HOM 各自的 micro-wave admission 必须使用实际 allocated
  input+shadow+workspace bytes；HOM padding 必须完整计入。
- XP 固定为 2,859,835 个 semantic rows 加一个 padding row，即
  2,859,836×4,096 FP32 physical item table（43.638 GiB）、
  bias-free `E4096→H1536` owner-side projection 和 24L/H1536 core；table +
  dense/projection 已约 44.725 GiB，超过单卡 Torch allocatable bytes。Qualification 只验证
  base-only semantic row coverage、单卡不可复制、2/4-rank admission、短 edge 正信号和
  D1 合法性，不能使用 EvoKV performance。若失败，必须在任何 timing 前发布新的
  benchmark identity 和失败原因，不能在已有结果上换 geometry。
- XP row 必须是可解释且真实触达的 entity/feature namespace，禁止用 inert padding rows
  制造容量。只有收到真实 optimizer update 的 row 才算 active；仅初始化、分配或
  inference lookup 不算。对两个 formal edges、全部 headline manifests 和合法
  comparators 冻结 semantic request union：至少包含 all-exact valid targets、所有冻结
  fixed-action plans 的 exact/append/fallback requests；HOM masked padding 不计。该 union
  必须全部 active，并记录 count/hash，且 active embedding bytes 加 dense/projection
  bytes 必须独立超过冻结的 single-card allocatable bytes。manifest 同时报
  total/active/requested/unique/hot rows、active bytes 和 update-count 分布。当前约
  99.3% 只是由 rounded sizes 得到的预估；pass/fail 使用精确 byte ledger。
- 在任何 EvoKV timing 前冻结 XP checkpoint builder：4-rank row-sharded sparse
  embedding update、embedding/dense 分离 optimizer、row-wise embedding state，以及必要
  的 ordinary-DRAM optimizer offload。禁止 full-table dense Adam gradient/state。
  Builder 是 common-upstream \(\theta_0\) training，可以读取所有角色用户的 base-period
  histories，但禁止 update/final windows、D1 actions、profiles 和 timing；post-base
  update/fit/profile/qualification/final roles 仍用户级互斥。若 active/request-union gates
  不能满足则拒绝该 benchmark identity。
- primary integrated ActionPlan 只包含 `compiled|exact`。progressive residual repair 保留为
  D1-only supporting extension，不进入 D2/D3 headline。

## 2. Semantic bridge

- Q-SEM 继续承担跨数据集、跨模型 tier、四训练 seed 的 D1 质量泛化。
- XP 的两组 post-base model-edge update training、D1 fit、D3 profile、qualification 和
  final benchmark users 必须由预声明 salt 分配并用户级互斥；common-upstream
  \(\theta_0\) base histories 是唯一明确例外。
- X2 和 XP 各选择一个独立 qualification subset，覆盖 length bins、compiled/exact actions
  和至少一个 model edge，报告 K/V recovery、score recovery、Top-k overlap 与 paired
  ranking delta。
- headline ActionPlan 在 qualification 后冻结；其余 performance cells复用该 plan，不要求
  每个 timing cell 重复推荐质量评价。
- 第二 model edge 只用于验证 edge-specific program/plan 和机制稳定性，不能跨 checkpoint
  复用旧 plan。

## 3. Timed path 与强 baselines

- correctness path 可以执行 hash、逐 extent oracle、`torch.unique` 和详细 lineage；primary
  timed path 必须关闭不属于生产执行的 GPU→CPU round trip、逐请求 SHA 和诊断同步。
- 对同一 workload 做 instrumentation on/off paired run，确认诊断不会改变方法排序。
- 冻结 randomized/interleaved measured-job order。两方法 AB/BA 交替，多方法轮转；记录
  clocks、temperature、power、NUMA/page state，异常时整 block 重跑而不是单边删样本。
- exact candidates 至少考虑 preplanned routing、wave-scope coalescing/dedup、低精度
  transport、fused/compiled dense execution和合法的 production-style embedding exchange。
- Exact winner 必须联合覆盖 `placement/transport × routing/coalescing × pipeline`。
  对预声明的至多 48 个 legal combinations 跑完整 bounded joint screen，再复测整体
  top-3；不能先按 placement 单项排名剪掉可能通过 dedup、精度或 pipeline 交互获胜的组合。
  若组合数超过上限，必须在看结果前冻结保留交互项的 pruning 规则。
- placement candidates 至少记录 full replication 的 capacity、hot-row replication +
  cold-row sharding、row sharding 与 dedup。超过共同 HBM budget 的候选标
  `capacity-not-admitted`，不能假装执行。
- D3 的 strongest generic denominator 同时包含 sequential S0、whole-group S1、
  fixed-FIFO segmented S2 和一个把 group 视为 opaque job、读取相同 stage profiles、
  capacity、credits 与 bounded-flow recurrence，但不知道 compiled/exact 标签、
  route-specific parameter sharing 或 EvoKV stable-interleave objective 的
  work-conserving/list scheduler。
- 冻结两个报告边界：mechanism figures 使用 execution-only；E1 end-to-end 使用
  first-wave update-inclusive cost，包含该 edge 的 D1 fit/compile、D2 lowering/plan、
  D3 profile/plan 和 rolling execution。只有后者获胜才可声明 end-to-end speedup。

## 4. GPU、拓扑与 HBM

- 后继 runner 从一开始参数化 1/2/4 ranks；某个具体模型可因 fixed-state capacity
  不被某个 rank 数接纳，但代码路径不能写死两卡。
- 记录 GPU UUID、PCIe、NVLink/P2P、NCCL topology 和 2-GPU NVLink pair 对跨 NUMA/SYS
  pair 的基础带宽。
- 先用 geometry-independent 1/2/4-rank launcher、NCCL context、allocator/workspace probe
  和固定 safety margin 冻结共同 \(B_{\mathrm{HBM,usable}}\)，再验证 XP；顺序不能倒置。
  Candidate-specific full-runner preflight 记录 allocated、reserved、NVML peak、workspace
  和 fragmentation，但只能拒绝配置或触发 timing 前的全局重新 qualification，不能为
  某个方法单独改变 cap。
- 覆盖最大地址首/中/末 extents、非整除尾段、empty collective participation 和超过
  \(2^{31}\) flattened element offset。
- 正式运行要求整机 CPU、GPU、DRAM 和 I/O 无外部竞争；动态时钟、温度、功耗和 throttle
  counters 进入 environment record。

## 5. Ordinary DRAM、NUMA 与 store

- 当前两个 NUMA nodes 各约 504 GiB；rank 0/1 的 CPU/memory first-touch 靠近 GPU0/1，
  rank 2/3 靠近 GPU2/3，并用 `numa_maps`/per-node counters 验证。
- `/dev/shm` 当前 mount 上限约 504 GiB，不等于约 900 GiB 的总可用 DRAM。576/720/800
  GiB points 需要扩容 tmpfs 或 NUMA-aware anonymous/shared arena，不能回退到 NVMe 文件。
- arena reservation 后逐页 dirty-touch；primary timer 内 major fault 和 swap delta 必须为
  零。
- 800-GiB point 只有在一份 live cache、bounded group shadow、OS/runtime余量和两 NUMA
  nodes 均通过时才进入可选 stress。当前机器不支持 1-TiB ordinary-DRAM live-cache claim。

## 6. Rolling group lifecycle

- 每个 capacity group 执行 input consumption、compute、writeback、validation、versioned
  commit 和 old-group reclaim；host peak 是一份 live cache 加 bounded shadow/staging。
- 所有 formal methods 在 group commit 前保留 old extent，并写入共同 bounded replacement
  shadow；未验证 target 不得覆盖唯一 old copy。commit 后才允许 allocator 复用旧页。
- 失败时每个 group/record 必须明确指向 old 或 new。已提交组可幂等读取，未提交组仍由 old
  extent 提供；禁止无 version/lineage 的 partial group。
- correctness、warmup 和 repeats 使用同一 arena。每次 job 后先保存 digest/witness，再在
  timer 外从相同 checkpoint/raw history 重建 old cache；reset 时间单独记录。
- segmented consumer 必须读取有效 length metadata，并通过下一 wave compatibility canary。

## 7. Promotion boundary

Qualification 只在以下时刻成为硬要求：

1. 冻结新的 D2/D3 result protocol；
2. 启动正式 repeated baseline/candidate cells；
3. 将任何新 timing、capacity、correctness 或 speedup 放入论文表图。

它不阻止当前继续完善实验蓝图、编写 baseline、扩展 runner 或寻找 D3 机制。任何失败都应
先修改后继 config/claim并生成新的 identity；不得回写或升级已有
`scientific_result=false` development artifacts。
