# CohortKV Design 2 Stage A handoff

日期：2026-07-29

状态：Stage A 已冻结，Stage B 入口为 **GO**。本 handoff 记录可由 Stage B 依赖的边界，
不构成多卡正确性、端到端性能或论文结果。

后续状态：Stage B 的实现与 W1/W2 证据已完成，但独立 W4 gate 尚未冻结；当前事实见
`DESIGN2_STAGE_B_HANDOFF.md` 和 `DESIGN2_DEVELOPMENT_STATUS.md`。本文件仍只定义 Stage A
边界，不倒写后续 W3 performance discovery。

冻结摘要：
[`configs/cohortkv_d2/stage_a_summary.json`](../../configs/cohortkv_d2/stage_a_summary.json)。
所有 Stage A characterization 均标记为 `scientific_result=false`。

## 1. Immutable inputs and hashes

Stage B 必须使用同一份 Stage-4.9 H12 step-1 动作，不得重跑 scheduler 或根据 D2
characterization 改写 action：

- ActionPlan：
  [`action_plan_theta1_theta2_staggered_renewal_h12.json`](../../configs/cohortkv_d2/action_plan_theta1_theta2_staggered_renewal_h12.json)
- ActionPlan content SHA-256：
  `c4bc383d28f3558fdd11be8788799aaa6f66e80f778a4670f781eb9295f0027e`
- ActionPlan file SHA-256：
  `3572a858111b1e9d08e4102512af46ef6a6d2b1fbe7ee7b2828162d28d58518d`
- 上游 Stage-4.9 artifact SHA-256：
  `bd2e7aa8b6bb4a2203ec416c9a69ecbe9c3eea43c3e4454664a449ffb516104f`
- prepared data SHA-256：
  `e03f3e80dacf9deccd5783d26a184d8ced7b339275bf13fa3b90de42a4b028b8`
- action partition SHA-256：
  `f8a89298ed9c5fd064b920a22d8f72c35fd2738ca7140bf5e0c0a62be647666a`
- theta2 checkpoint SHA-256：
  `37eb8189d36127b735e6b6482f82dc31927d625168e0d67eca9890d5a01f3c18`
- theta1→theta2 direct-old-K/V program SHA-256：
  `c0a2ff2de64200f482c6eb5097cbad4f7db8d200de385be4238bdcfbf9cf5e7d`

动作集合恰好覆盖 record IDs `0..681`：548 条 compiled、46 条 scheduled exact、88 条
natural exact，共 682 条。`scheduled_exact` 和 `natural_exact` 在 runtime 都执行 exact，
但必须保留各自 provenance。`134/682 = 19.6%` 是包含 natural exact 的 runtime exact-route
record fraction，不是 policy-only exact、lookup、communication 或 time fraction。

## 2. Outputs and artifacts

Stage A 冻结了以下可重建工件：

| 工件 | 作用 | 证据边界 |
|---|---|---|
| [`stage_a_plan_adapter_summary.json`](../../configs/cohortkv_d2/stage_a_plan_adapter_summary.json) | ActionPlan/WavePlan schema、单 rank 投影与静态 phase ledger | schema 和 projection |
| [`stage_a_exact_frontend_validation.json`](../../configs/cohortkv_d2/stage_a_exact_frontend_validation.json) | HSTU split frontend、exact helper、真实路径 lookup instrumentation | 单 GPU 真实模型 |
| [`stage_a_stage5_adapter_validation.json`](../../configs/cohortkv_d2/stage_a_stage5_adapter_validation.json) | D2 请求对 Stage-5 transaction 的行为保持 | 完整 682-record plan、合成小 K/V payload |
| [`stage_a_request_characterization.json`](../../configs/cohortkv_d2/stage_a_request_characterization.json) | phase/batch/owner request multiset、fan-out 与 dedup ceiling | 静态请求上限 |
| [`stage_a_capacity_characterization.json`](../../configs/cohortkv_d2/stage_a_capacity_characterization.json) | strict-COW 的 1/2/4-rank 容量账本 | 静态 admission |
| [`stage_a_p2p_topology.json`](../../configs/cohortkv_d2/stage_a_p2p_topology.json) | 四卡 peer topology、真实 extent copy/overlap 和 sampled owner imbalance | GPU microbenchmark |
| [`stage_a_summary.json`](../../configs/cohortkv_d2/stage_a_summary.json) | hashes、gates、决策和未支持主张的冻结入口 | Stage A aggregate |

实现入口位于 `design2_plan.py`、`design2_runtime.py`、`design2_metrics.py`，HSTU embedded
frontend 位于 `models/hstu.py`，统一 exact/jagged helper 位于 `migration/recompute.py`。
Stage B 不应重新把这些逻辑复制回 experiment script。

冻结时的仓库验证为：

- `pytest -q`：349 passed；
- `ruff check src tests scripts`：passed；
- `python scripts/freeze_cohortkv_design2_stage_a.py --check`：passed；
- `git diff --check`：passed。

## 3. Passed gates

- **G0 ActionPlan：pass。** ActionPlan 可从冻结上游 artifact 确定性重建，counts、record
  coverage、provenance 和所有输入 hashes 闭合。
- **G1 mechanical refactor：pass。** 在真实 theta2 16L/H512 模型上，旧 frontend 与
  FP32 embedded frontend 的 hidden、K、V、scores 和 Top-100 bitwise 相等；state-dict keys
  不变，training mode 被恢复。append helper 的 hidden/K/V/length 也 bitwise 相等。
- **P0.2 Stage-5 adapter：pass。** D2-adapted requests 与独立 direct Stage-5 requests
  在 normal commit、semantic fallback、mid-job abort 和 pre-commit abort 上得到相同
  decisions、manifest、payload bytes 与 readback。该验证使用完整 682-record action plan，
  但 K/V payload 是合成小对象。
- **P0.3 phase lookup：pass。** 真实 compiled retained 代表路径的 embedding lookup 为零；
  exact、delta append 和 latest append 均有正 lookup。完整 plan request replay 为
  `0 / 50,099 / 82,612 / 213,669 / 682` tokens。
- **P0.4 topology/microbenchmark：pass。** 四张 A40 的 direct-peer topology 完整测得为两个
  NVLink islands `{0,1}` 与 `{2,3}`；两个岛内方向、并发 copy、真实 compiled
  copy/compute overlap 以及 16-record owner imbalance sample 均完成。跨岛 NCCL 未测。
- **P0.5 request characterization：pass。** retained-prefix lookup 从 all-exact 的
  `637,954` 降至 mixed 的 `50,099`，即 `12.73×`；包含方法无关 append 后，
  `934,917 → 347,062`，即 `2.694×`，也就是 mixed 仍保留 37.1% lookup tokens。这些是
  D1 logical-plan/token accounting，不是物理通信或时间结果。D2 后续只做 fixed-action
  physical lowering，不得改变 requested actions。

因此 Stage B entry 为 **GO**。G2 distributed exact、G3 distributed communication 尚未开始；
G7 capacity claim 未通过；paper performance claim 未评估。

## 4. Falsified hypotheses

Stage A 已经证伪五个可能误导后续实现的假设：

1. “compiled record 整体不访问 embedding”是错的；只有 retained-prefix transform
   embedding-free，随后 delta/latest append 仍需 lookup。
2. 单张 A40 能容纳 strict-COW H12 wave 是错的。旧 K/V 为 `28,383,969,280` B，新 K/V 为
   `30,635,360,256` B，单是 strict-COW K/V 即 `59,019,329,536` B。
3. 四张 A40 构成均匀 direct-peer fabric 是错的；跨 NVLink island 没有 direct peer。
4. FP16/BF16 item-vector transport 与当前 exact frontend 机械等价是错的。二者可继续作为
   Stage B correctness candidate，但 FP32 是唯一已证明的 mechanical baseline。
5. “dedup 在所有合理 scope 下都低于 10%，可永久删除”是错的。大 coalescing scope 的
   静态 remote-return reduction 上限在 W2/W4 分别达到约 `17.35%/11.14%`，而 batch-4
   仅约 `3.19%/3.15%`。

因此 P5 dedup 的裁决是 **defer**：不进入 Stage B baseline，但保留为拿到真实 collective
bytes 后才决定的候选。

## 5. Remaining risks

- Stage A 没有执行 torchrun multi-rank owner-compute、row-sharded exact/append 或 collective
  ordering；单 rank 的正确性不能外推为 distributed correctness。
- 两个 NVLink island 之间的 NCCL route、其真实 bytes 和 tail 尚未知，可能改变 owner
  placement 与通信原语选择。
- physical collective bytes、actual remote fraction 和 exposed communication time 均未知；
  静态 token/vector bytes 不能替代这些指标。
- strict-COW 的 W2/W4 admission 尚未用实际 HBM-resident source manifest 和 per-rank CUDA
  context 验证；Stage B 可能反向暴露 Stage A 的容量模型过松。
- Stage-5 adapter 已闭合 transaction 行为，但没有跑真实 52.3-MB/record payload，因此不能
  支持 transaction 性能或容量结论。
- P2P owner execution 只覆盖 16 个 retained-length quantile records，不是完整 wave。

若 Stage B 的 world-size-1 SPMD 不能复现本 handoff 的 exact/ledger，或 W2 实测容量与静态
admission 矛盾，应先回到 Stage A 修复 process boundary、dtype 或容量账本，而不是继续堆叠
distributed logic。

## 6. Provisional assumptions

- W2/W4 的 old/new K/V placement 使用确定性 owner maps，但没有实际 HBM source manifest。
- CUDA context/allocator 以每 rank `2 GiB` margin 表示，而非实测常驻开销。
- row-sharded embedding dense bytes 是由已测 tensor layout 推导的估计值。
- current model tensor 为 `724,328,448` B，模型本身能放入单卡；当前容量压力来自 cohort
  strict-COW，不能写成“模型大到单卡放不下”。
- program file 为 `33,592,613` B，实际 tensor bytes 为 `33,587,200` B；容量账本只能使用
  后者。
- FP32 item vectors 是当前机械等价 transport；FP16/BF16、cross-island route、dedup 和
  work stealing 都必须由 Stage B 重新裁决。

## 7. Known unsupported claims

Stage A 不支持下列说法：

- multi-rank owner-compute 已正确；
- row-sharded exact 已正确或已节省物理通信；
- mixed wave 已获得端到端 speedup；
- G7 已通过，或当前模型不能放进单 GPU；
- compiled maintenance 已实测不干扰前台 embedding service；
- topology、overlap 或 sampled imbalance microbenchmark 是完整 wave 性能；
- 任一 Stage A 数字可直接进入论文主结果。

Table 8 只能作为历史动机。D2 的 record-DP、all-exact 和 mixed baseline 必须在同一个新
SPMD harness 内配对重测。

## 8. Historical first three Stage B diagnostics

以下是冻结 handoff 当时规定的三个 Stage-B diagnostic；当前均已执行，实时状态以
`DESIGN2_DEVELOPMENT_STATUS.md` 为准：

1. **World-size-1 SPMD parity。** 用一进程一卡的新 harness 重放 frozen ActionPlan，要求
   exact outputs、action counts、phase lookup tokens、requested/final/fallback accounting
   与 Stage A 完全相同。若不相同，停止并回查 process/context/frontend 边界。
2. **Two-rank sharded-exact truth test。** 在一个 NVLink island 上实现最小 row-sharded
   exact/append，逐 batch 记录 local/remote IDs、request/return physical bytes、collective
   order 和 FP32 output parity。先证明 correctness 与账本，再测试低精度或 dedup。
3. **Four-rank cross-island route test。** 用 NCCL 覆盖两个 island 和至少一条跨岛 route，
   验证所有 ranks collective 次序一致、失败可传播，并测真实 bytes/tail。完成前不实现
   topology-aware placement、overlap 或 work stealing。

只有这三项都闭合，Stage B 才继续扩展到完整 1/2/4-rank primitive matrix。
