# CohortKV Design 2 Stage B handoff

日期：2026-07-29

状态：**实现与 W1/W2 证据、W3 开发诊断和 C0 开发态闭合均已完成，Stage B 尚未冻结。**
唯一未闭合的 formal hard gate 是 W4 normal/hard-failure：物理 GPU2 当前被另一个用户的
长期 workload 总计占用 `44,253 MiB`（约 `43.22 GiB`），其中 VLLM EngineCore 占
`43,744 MiB`（约 `42.72 GiB`），另有 LMCache 占 `434 MiB`，不能安全运行四张独立 A40。
W2 cross-island、W3 和 C0 都不能替代 W4。

当前采用双 Gate：

- `stage_c_development_entry=go`：C0 已完成；允许在 W1/W2/W3 上继续针对性开发诊断；
- `stage_c_evaluation_entry=blocked`：在
  `configs/cohortkv_d2/stage_b_summary.json` 合法生成并通过 `--check` 前，禁止正式
  Stage-C integrated evaluation、full-cohort/timing 和论文证据。

实时状态和等待命令见 `DESIGN2_DEVELOPMENT_STATUS.md`。

所有本阶段 artifact 均为 `scientific_result=false` 的开发证据，不是论文结果。

## 1. Immutable inputs

Stage B 沿用 Stage A 冻结的 H12 step-1 动作：

- ActionPlan content SHA-256：
  `c4bc383d28f3558fdd11be8788799aaa6f66e80f778a4670f781eb9295f0027e`
- ActionPlan file SHA-256：
  `3572a858111b1e9d08e4102512af46ef6a6d2b1fbe7ee7b2828162d28d58518d`
- 动作数：548 compiled、46 scheduled exact、88 natural exact，共 682；
- runtime exact-route record fraction：`134/682 = 19.6%`，包含 natural exact；完整 mixed
  lookup fraction 仍为 `347,062/934,917 = 37.1%`，两者不可互换；
- source/target：`theta1 -> theta2`；
- record owner：`strict_cow_lpt`；
- embedding owner：`item_id % world_size`；
- exact item-vector transport：FP32；
- private K/V fragment：FP16。

Stage B 开始前已重新执行
`python scripts/freeze_cohortkv_design2_stage_a.py --check`。Stage A 的 15 个实现文件、
ActionPlan、上游、prepared data、checkpoint 和 program hashes 均未改变。

## 2. Implemented outputs

核心实现：

- `design2_distributed.py`：always-initialized SPMD runtime、collective phase guard、
  cooperative failure vote 和 metadata collectives；
- `design2_embedding.py`：modulo row-sharded FP32 lookup、exact 和 padded append；
- `design2_owner.py`：old-K/V owner-local compiled retained transform；
- `design2_transaction.py`：private-fragment coverage/owner/hash/capacity validation，只产生
  ready/abort，不发布 epoch；
- `design2_dev_wave.py` 与 `design2_dev_epoch.py`：16-record integrated-wave closure 和
  development-only epoch namespace；不实现正式 Stage-C publication；
- `run_cohortkv_design2_distributed_tests.py`：真实 theta1/theta2 sample、adversarial
  routing、capacity、placement 和 transaction worker；
- `run_cohortkv_design2_dev_c0.py` 与 `launch_cohortkv_design2_dev_c0.py`：显式 W1/W2/W3
  normal、W3 pre-commit abort 和 atomic development-status writer；
- `launch_cohortkv_design2_stage_b.py`：显式 W1/W2/W4 torchrun 与 bounded hard-failure
  launcher；
- `freeze_cohortkv_design2_stage_b.py`：只有 W1/W2/W4 normal、W2/W4 hard failure 和全部
  provenance/gates 均闭合时才生成 frozen summary。

当前正式 artifacts：

| artifact | 当前结果 |
|---|---|
| `stage_b_w1_primitives.json` | complete；全部 checks 通过 |
| `stage_b_w1_repeat_primitives.json` | complete；private component/fragment-set hashes 稳定 |
| `stage_b_w2_primitives.json` | complete；全部 checks 通过 |
| `stage_b_w2_cross_island_primitives.json` | complete；物理 GPU1/GPU3 supplemental |
| `stage_b_w2_hard_failure.json` | complete；显式 NCCL/物理 UUID；rank 1 exit 23，约 5.90 s 内收敛 |
| `stage_b_w4_primitives.json` | **missing；hard blocker** |
| `stage_b_w4_hard_failure.json` | **missing；hard blocker** |
| `stage_b_summary.json` | **不得生成，直到两个 W4 artifact 完成** |

另有三份不进入 Stage-B freeze 的 W3 development artifacts：

| artifact | file SHA-256 | 当前结果 |
|---|---|---|
| `dev_w3_sample_inputs.json` | `8e6d44b72efe83ced91309b453899ed6367fd5423060d089eb7eaf04c560bb2d` | 9-record deterministic W3 sample |
| `dev_w3_primitives.json` | `125b1c0f4b21c1efb1a29a3c2623941125e00a1882b0d90c3a9f28a53a591c72` | W3 NCCL normal；全部 checks 通过 |
| `dev_w3_hard_failure.json` | `16ecd88fd3b59904dbac5efdaf5b56e4b63bad5699092c0eb6656e0a49115149` | rank 1 exit 23；约 6.72 s；无残留进程组 |

三份 W3 artifact 均为 `scientific_result=false`，且明确记录
`formal_stage_b_gate=false`、`stage_c_evaluation_authorized=false` 和
`substitute_for_w4=false`；normal/hard-failure result 另外明确标记
`formal_stage_b_summary_eligible=false`。

## 3. Passed gates

- **Stage-A reverse audit：pass。** W1 的代表性 migrate/scheduled/natural exact 数值
  parity 通过；完整 ActionPlan phase ledger 精确回到 retained
  `637,954 -> 50,099`、natural `82,612`、append `213,669 + 682`、full
  `934,917 -> 347,062`。这关闭的是 D1 logical plan/lookup ledger，不证明 physical
  execution benefit。
- **W1 SPMD：pass。** 新 process boundary、forbidden full embedding、owner-local compiled
  retained、delta/latest append、private ready/abort 均通过。
- **W2 sharded exact/append：pass。** natural、scheduled、one-owner empty rank、
  padding-only、repeated ID、valid item ID 0、delta/latest 和 one-owner append 均与完整表
  FP32 reference bitwise 一致。
- **ID/byte reconstruction：pass。** 每 phase/rank 冻结 requested/local/remote/unique ID
  counts 和 hashes；per-peer send/receive ID hashes 转置匹配；counts/ID/FP32-vector tensor
  payload、off-diagonal bytes 和 collective-call 数可独立重建。
- **Compiled retained invariant：pass。** `item_lookup_calls`、embedding collective
  count/bytes 和 old-K/V P2P bytes 均为零；append lookup 单独计入。
- **Private transaction：pass。** 实际 K/V component checksums、owner coverage、capacity、
  phase traces 和 fragment-set hash 闭合；ready 与 synthetic abort 均不发布 target epoch。
- **W2 projected full-cohort capacity：pass as a Stage-B projection。** 两 rank 的 projected
  required bytes 均约 32.63 GB，对 47.70 GB device capacity 保留约 15.07 GB margin。
  这不是 G7，也不是实际 resident 682-record strict-COW run。
- **Hard failure W2：pass。** 重跑 artifact 显式绑定 NCCL、物理 GPU0/GPU1 UUID；rank 1
  `os._exit(23)` 使 torchrun 非零退出，未到 subprocess timeout，进程组无残留。
- **W3 development diagnostic：pass。** 物理 GPU0/GPU1/GPU3 上的三 rank NCCL normal
  覆盖 asymmetric owner/embedding split、empty-rank participation、collective order 和
  reverse return；rank-exit propagation 也有独立 bounded artifact。这令
  `stage_c_development_entry=go`，但不改变 Stage-B formal gate。
- **C0 development closure：pass。** 固定 16-record fixture 的 W1/W2/W3 normal 和 W3
  pre-commit abort 均闭合 route/coverage/owner/final-token、development pointer、source
  fixture 和 reference-release checks；无 timeout 或残留进程。它不满足正式 Stage-B/Stage-C
  gate。

## 4. Falsified implementation assumptions

1. `previous_cache_present` 不能区分 natural exact；本 H12 中 natural records 也可标记为
   present，但 retained overlap 为零。公共 delta 必须按 phase/reason 计账，否则会把
   `82,612` natural prefix 重复计入。
2. 同方向 ring 可以在 W2 偶然完成 return，但 W4 会把输出送到 `r+2`。old-K/V steal 与
   output return 必须使用相反方向，并冻结为两个 collective ordinals。
3. 把 report/timing 或 tensor shape+sum 当作 fragment checksum 不可靠且不可复现。
   Transaction hash 必须绑定实际 dtype/shape/length/K/V bytes，并排除 timing。
4. “sample transaction ready”不等于“full cohort capacity admitted”。W1 sample primitive
   可运行，但 projected strict-COW full cohort 需要约 62.46 GB，因此仍应拒绝。
5. 只记录 remote token count 会形成循环 byte check。必须保留 request ID evidence 和
   per-peer hash，才能从 owner rule 重建 payload。

## 5. Remaining risks and provisional assumptions

- W4 尚未证明四个独立 GPU 上的 all-zero split、collective order、ring return、两个 NVLink
  island 之间的 route、per-rank capacity 和 hard-failure termination。
- W2 GPU1/GPU3 supplemental 已经过物理 cross-island NCCL，但 world size 仍是 2，不能说明
  四 rank phase composition。
- W3 已证明三 rank 非对称 composition，但缺少物理 GPU2，仍不能证明四个独立 CUDA context、
  W4 owner map、四 rank capacity 或正式 W4 termination。任何 C0 成功都不能消除这项风险。
- source old K/V 在 worker 中只为 correctness fixture 临时从 source checkpoint 重建；
  不是 runtime source-read cost，也不是完整 resident source manifest。
- W2/W4 full-cohort capacity 是根据完整 682-record owner map、实际 per-process
  model/shard/program/context 初值和 projected old/new K/V 推导；实际 682-record HBM COW
  allocation属于 Stage C/G7。
- FP32 是唯一机械等价的 embedding transport。低精度、dedup、overlap、topology-aware
  placement 和 work stealing 均未进入 baseline。
- 记录的是 collective tensor payload，不是 NCCL wire bytes。

## 6. Unsupported claims

Stage B 当前及最终冻结后都不支持：

- formal W4/Stage-C 完整 682-record mixed wave 已执行；
- 一个 formal target epoch 已原子发布；
- formal mixed wave 已获得论文级端到端 speedup；
- G7 physical capacity 已通过；
- compiled record 整体 embedding-free；
- 真实 serving latency/SLO 或 formal embedding-tier isolation；当前只有单 offered-rate
  synthetic development contention probe；
- dedup、overlap 或 topology-aware placement 有收益；
- 当前模型无法放入一张 A40；
- 任一 Stage-B timing 可进入论文。

## 7. Closing the pending W4 gate

GPU2 安全可用后只执行：

```bash
python scripts/launch_cohortkv_design2_stage_b.py \
  --world-sizes 4 \
  --cases normal hard_failure \
  --visible-devices 0 1 2 3

python scripts/freeze_cohortkv_design2_stage_b.py
python scripts/freeze_cohortkv_design2_stage_b.py --check
```

若 W4 normal 或 hard-failure 失败，先按症状回到 Stage B；若 W1 parity/ledger 反向失败，
再回到 Stage A。不得用 W2 cross-island、逻辑四 rank 共卡或 Gloo 替代四张独立 A40。

## 8. C0 development 与正式 Stage C 入口

等待 W4 时已完成 C0：

1. 使用 `stage_b_sample_inputs.json` 的固定 16-record fixture，把
   compiled/scheduled-exact/natural-exact、delta/latest append 和 private fragments 接到
   development epoch state machine；
2. W1/W2/W3 normal 的 phase order、coverage、lineage、ownership、development pointer 和
   reference-release order 全部通过；
3. W3 pre-commit abort 保持 source pointer，未暴露 development target，并释放 private-target
   references；
4. 四个 run 和 aggregate status 均为 `scientific_result=false`、`formal_stage_c=false`，
   无 launcher timeout 或残留 process group。

C0 aggregate 是
`configs/cohortkv_d2/development/c0_status.json`，file SHA-256 为
`b193ddacb12be623e3e03e1a9f1cbdbde34cab4f7a3be9ec62d9af0ef27f9507`；五份最终 artifact
hash 见 `DESIGN2_DEVELOPMENT_STATUS.md`。normal 只发布 development namespace pointer；
artifact 明确 `target_epoch_published=false`。C0 不含 timing、full-cohort、capacity 或正式
epoch-publication 证据，因此完成后仍只维持 `stage_c_development_entry=go`。

C0 之后又完成了独立 W3 full682 physical-lowering discovery 与 full-payload validation。
它们记录在 `DESIGN2_DEVELOPMENT_STATUS.md`，不进入 Stage-B freeze，也不改变这里的 W4 gate。
其作用是冻结候选机制：`(S,R)` shape-aware extents、segmented suffix-only finalization 和
lineage-preserving merged exact pool；requested actions 始终不变。

只有 Stage B summary 冻结并通过 `--check` 后，才执行正式 Stage-C diagnostics：

1. 在同一 SPMD harness 中跑 one-shot/two-stage all-exact、naive sharded fixed-action mixed
   和 D2 physical-sparse mixed，冻结共同 source-read-to-post-commit boundary；
2. 在新 D2 protocol 下闭合 strict-COW commit/readback、fallback 和 abort；
3. 执行 frozen H12 的完整 W2/W4 mixed wave，并同时报告 D1 retained-prefix boundary 和
   D2 post-append integrated boundary。任何论文 timing 前先冻结新的 Stage-C protocol。
