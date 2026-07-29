# CohortKV Design 2 development status

日期：2026-07-29

这是 D2 当前执行状态的短台账，不改变
`DESIGN2_FINAL_PLAN.md` 的设计定义，也不改变
`DESIGN2_FOUR_STAGE_EXECUTION.md` 的正式 Stage-B/Stage-C gate。

## 0. 规模不足时的冻结回退原则

当前 312,145-row 配置先用于完成 W3 forced-sharding、真实通信和 integrated-wave
验证。若后续 phase/collective/GPU 计量表明问题迟迟不显著，而且根因是模型或真实访问规模
不足，不得直接把 D2 判为不存在，也不得只靠 cold rows 制造主结果。此时暂停当前
performance 路线，从此前已接受且 target semantics 正确的数据扩大真实 item/user 覆盖，
按 D1 相同流程训练一个 base version 和一到两个短 streaming versions，重新生成 old K/V、
compiled program、certificate、exact endpoint 与 D2 ActionPlan。该配置首先只承担 D2
mechanism discovery；不要求一开始完成长期、多 seed 的完整 D1 replication。机制收敛后再决定
是否补成论文主配置。

cold-row expansion 只允许作为独立 admission control；性能和设计结论必须来自真实 accessed
IDs、真实版本训练和实际 routed communication。是否触发这一回退，由现有 W3 integrated、
communication 和 interference 结果的瓶颈归因决定。

当前裁决：**不触发回退。** full682 W3 已同时暴露 naive physical execution 的明确失败和
shape-aware segmented lowering 的正向信号；现有模型足以完成 D2 mechanism discovery。只有正式
protocol 后该信号消失且瓶颈归因重新指向真实规模不足，才恢复本节路线。

## 1. 当前双 Gate

| Gate | 状态 | 允许/禁止 |
|---|---|---|
| `stage_c_development_entry` | **GO** | C0 开发态闭合已完成；仍可在 W1/W2/W3 上做 `scientific_result=false` 的针对性开发诊断 |
| W3 mechanism discovery | **COMPLETE** | 已允许并完成 `scientific_result=false` 的 192-record 与 682-record fixed-action 设计发现；它不是正式 Stage-C、W4 替代品或 paper evidence |
| `stage_c_evaluation_entry` | **BLOCKED** | 不得把 W3 discovery 升格为正式 Stage-C integrated evaluation、正式 target epoch、论文 timing 或 paper claim |
| Stage-B formal freeze | **BLOCKED** | 正式 W4 normal/hard-failure 和 `stage_b_summary.json` 尚缺失 |

这里的 GO 只解除短期开发阻塞，不是降低正式证据门槛。W3 不能替代 W4，C0 不能替代
Stage C，且当前不得生成 `configs/cohortkv_d2/stage_b_summary.json`。

## 2. 已完成的 W3 开发诊断

物理 GPU0/GPU1/GPU3 上的三 rank NCCL diagnostic 已完成。三份文件的 **file SHA-256** 为：

| artifact | file SHA-256 | 边界 |
|---|---|---|
| `configs/cohortkv_d2/dev_w3_sample_inputs.json` | `8e6d44b72efe83ced91309b453899ed6367fd5423060d089eb7eaf04c560bb2d` | 9-record deterministic W3 development sample |
| `configs/cohortkv_d2/dev_w3_primitives.json` | `125b1c0f4b21c1efb1a29a3c2623941125e00a1882b0d90c3a9f28a53a591c72` | W3 NCCL normal primitive diagnostic |
| `configs/cohortkv_d2/dev_w3_hard_failure.json` | `16ecd88fd3b59904dbac5efdaf5b56e4b63bad5699092c0eb6656e0a49115149` | rank-1 `os._exit(23)` bounded-failure diagnostic |

W3 normal 使用 `CUDA_VISIBLE_DEVICES=0,1,3`、`world_size=3` 和 NCCL。它通过三 rank
collective-order、empty-rank participation、sharded exact/append bitwise、owner-local
compiled、placement return、private ready/abort、ID/byte reconstruction 和 capacity-boundary
checks。hard-failure 在约 6.72 s 内非零退出，未到 launcher timeout，清理后无存活进程组。

这些产物都明确记录：

- `scientific_result=false`；
- `formal_stage_b_gate=false`；
- `stage_c_evaluation_authorized=false`；
- `substitute_for_w4=false`；
- 这三份 Stage-B primitive artifacts 未执行完整 mixed wave，未发布 target epoch，也未观测
  NCCL wire bytes；后续 §5 的独立 Stage-C development family 不倒写这些 artifact 的边界。

normal 与 hard-failure result 还明确记录
`formal_stage_b_summary_eligible=false` 和 `stage_c_development_evidence=true`；sample-input
artifact 只定义输入，不冒充 result。

W3 的价值是尽早暴露 W2 对称性可能掩盖的三 rank、非均匀分片、跨岛 route 和反向 return
问题；它不覆盖四个独立 CUDA context、W4 phase composition、四 rank 容量或正式 W4
failure termination。

## 3. W2 hard-failure 证据修订

`configs/cohortkv_d2/stage_b_w2_hard_failure.json` 已在显式 NCCL 和物理 UUID 绑定后重跑：

- file SHA-256：
  `49f2cf52550df9f12f6fbb7c5a37945effc91f1ae6d086c128954919f34cc015`；
- backend：`nccl`；
- visible devices：GPU0/GPU1；
- UUID：
  `GPU-c71993c2-1a63-2462-2d0c-33c8bf79108c`、
  `GPU-cfa93a9f-ec9d-4e1d-7130-4733ac28a5c4`；
- rank 1 exit code 23 被观察到，torchrun 非零退出，约 5.90 s 内收敛；
- 未到 subprocess timeout，退出与清理后均无残留进程组。

这关闭的是 W2 failure artifact 的 backend/device 证据绑定，不关闭 W4，也不冻结 Stage B。

## 4. C0 当前状态与边界

状态：**development diagnostic complete。** C0 已完成一次干净的 W1/W2/W3 normal 重跑和
W3 pre-commit-abort 重跑。runner 使用
`configs/cohortkv_d2/stage_b_sample_inputs.json` 的 16-record fixture（file SHA-256
`9a53fe5e57d09142ff87399823be7382395eae88e7ec51db357f756112e57f10`），不是 W3 primitive
diagnostic 使用的 9-record sample。

最终 artifact 的 **file SHA-256** 为：

| artifact | file SHA-256 | 当前结果 |
|---|---|---|
| `configs/cohortkv_d2/development/dev_c0_w1_normal.json` | `3fa690a03a7fcdfa94c7a425ef0c02158a14cc81a29f8eed713ccb4c6c15a76c` | W1 normal complete |
| `configs/cohortkv_d2/development/dev_c0_w2_normal.json` | `d151c51f31661d342ba82c04e5eafe71267623a5de973bb689993a8d9e30e323` | W2 NCCL normal complete |
| `configs/cohortkv_d2/development/dev_c0_w3_normal.json` | `868d1594486e23b7a3bd6d1d7848a343a5a9f052f7245f77dc233ba2c1c5c9dd` | W3 NCCL normal complete |
| `configs/cohortkv_d2/development/dev_c0_w3_pre_commit_abort.json` | `eae44b2e5137819102ff90106494c15e8335451b586ddf89f7ee07fff92778e3` | W3 pre-commit abort complete |
| `configs/cohortkv_d2/development/c0_status.json` | `b193ddacb12be623e3e03e1a9f1cbdbde34cab4f7a3be9ec62d9af0ef27f9507` | clean-launch aggregate complete |

四个 run artifact 的 route、coverage、owner assignment、final-token、compiled-retained
embedding bypass、COW source-fixture 和 development epoch checks 全部通过。normal 在 W1/W2/W3
均发布 development namespace pointer；W3 abort 保持 source pointer，并释放 development
private-target references。所有 launcher 均未 timeout，清理后无残留 process group。

这项闭合必须保持以下边界：

- `scientific_result=false`、`formal_stage_c=false`；
- 固定 16-record fixture，不得外推到 682-record full cohort；
- `performance_result=false`、`timing_claim=false`；
- normal 只发布 development namespace pointer；artifact 明确
  `target_epoch_published=false`，因此没有正式 target-epoch publication；
- COW source 只是 correctness fixture，reference release 不证明真实 HBM allocator reclaim；
- 没有 full-cohort capacity/admission 证据；
- 当前 closure 是结构、路由与事务 closure，不是独立 one-shot target-exact 数值 reference；
- 16-record fixture 没有覆盖 `delta_tokens=0`，而完整 ActionPlan 的 compiled record 575
  正好命中该条件；正式 full-cohort 前必须先关闭共享的 unchanged-append 分支；
- 不替代 formal Stage-B W4，也不触发 Stage-B freeze。

C0 完成只关闭了 sample integrated-wave 与 development state-machine 的接口风险。
`stage_c_development_entry` 保持 GO，但 `stage_c_evaluation_entry` 和 formal Stage-B freeze
仍保持 BLOCKED。

canonical development 目录只允许完整的 W1/W2/W3 normal 与 W3 pre-commit-abort 矩阵更新
`c0_status.json`。任何局部重跑必须指定独立 `--output-dir`，其 aggregate 状态标为
`partial`，不能覆盖这里的完整矩阵台账。

## 5. W3 设计验证：当前实测与修正

以下均为物理 GPU0/GPU1/GPU3 上的 development evidence，不是 formal Stage-C 或 paper
result。

本节的统一解释是：

> D1 已冻结 `548 compiled + 134 runtime exact` 的 logical action sparsity；D2 不改变这些
> requested actions，而是把它们 lowering 成 physical sparsity：compiled records 按 `(S,R)`
> 组织，retained destination 只写一次，suffix-only append 形成 segmented state，两个 exact
> reasons 保留 lineage 但共用一个按 `F` 的 physical pool。

这里的 physical sparsity 指不实例化多余 padding、retained rewrite 和 semantic-only phases，
不是稀疏张量。`134/682 = 19.65%` 只是 exact-route record fraction；完整 mixed lookup 仍为
`347,062/934,917 = 37.12%`，实际 one-way off-diagonal vector volume 为
`454.62/1,222.86 MiB = 37.18%`。因此不能从“约 80% records compiled”推出“约 80%
compute/lookup/communication 已删除”。

### 5.1 Request/communication

完整 H12/682 mixed 请求为 347,062 tokens，其中 231,861 个请求远端；requester-scope
unique 后为 246,086 tokens，其中 164,325 个远端。all-exact 公平 unique 后仍有 334,738
个远端，约为 mixed 的 2.04 倍。

`configs/cohortkv_d2/development/dev_wave_embedding_capsule_w3_v2.json` 验证了
plan-compiled vector-only capsule：

- mixed 单向 off-diagonal payload 为 320.947 MiB；
- 每 rank steady execution 只有一次 vector collective；
- count/ID collective bytes 为零；
- 输出与无 dedup reference bitwise；
- median max-rank steady execution 为 9.409 ms；
- 但当前 Python 全局 plan compiler 的 mixed max-rank compile 为 409.8 ms，materialization
  为 24.0 ms；all-exact 分别为 713.9/49.6 ms。

v1/v2 中 dynamic modes 调用了带 GPU→CPU ID SHA 的诊断 lookup，而 capsule 不做 SHA，
因此 dynamic-vs-capsule 的 wall-time ratio **已作废**，不得写入 claim。请求数、unique、
bytes、collective count、bitwise 和 capsule 自身 steady time 仍有效。

公平的无 SHA 重测已写入
`configs/cohortkv_d2/development/dev_wave_embedding_capsule_w3_v3.json`。mixed 结果为：

- one-batch no-dedup：steady 13.960 ms，含 plan 单 wave 35.563 ms；
- requester-scope wave unique：steady 11.151 ms，含 plan 单 wave 33.052 ms；
- compiled vector-only capsule：steady 9.419 ms，但含 manifest gather、compile 和
  materialization 的单 wave 为 100.397 ms；
- capsule 相对 dynamic wave unique 的 steady 收益约 15.5%，但当前额外 prepare 需要相同
  frozen manifest 共执行约 40 次才 break even；这不是 organic cross-wave reuse claim。

NumPy 向量化已把单进程完整 H12 compiler 从 mixed/all-exact 的约 335.7/864.0 ms 降到
28.0/69.4 ms，但三进程实际 mixed full prepare 仍达 91.0 ms，主要剩余项是 object gather、
各 rank 重复 tuple compiler 和 device materialization。因此当前裁决是：

- 保留 whole-wave unique coalescing；它在第一次执行即有约 7.1% plan-inclusive 改善，并有
  明确的 29.1% mixed off-diagonal byte reduction；
- vector-only capsule 暂不进入 integrated 主机制，只保留为 steady/control-plane
  characterization；
- 只有 device/tensor plan compiler、与 retained transform overlap 或真实 identical-manifest
  reuse 能关闭 plan-inclusive gate 时才恢复它。

### 5.2 Integrated pilot

`results/system/cohortkv_design2_integrated_w3_development_v1/pilot192.json` 在 192 条分层记录、
每 rank 64 条、B16 strict-COW 下得到：

- owner-mixed：3.025 s；
- one-shot all-exact：2.079 s；
- two-stage all-exact：2.275 s。

这是一个真实负结果：naive owner-mixed 比最强 exact 慢约 45.5%。phase 归因显示
compiled-retained 仅约 56–57 ms/rank，主要代价来自 delta/latest append 的重复 prefix
展开、完整 updated-cache 生成和 collective phase fragmentation。

这正是 D2 的 Motivation-2 bridge：D1 已经减少 logical full-history replay，但 naive 多卡
executor 仍因 padding、full-prefix rewrite 和 fragmented phases 失去收益。它不是 D1 action
selection 失败，也不能通过改变 exact budget 来“修复”。

`results/system/cohortkv_design2_integrated_w3_development_v2/pilot192_fused_finalization.json`
保持相同 action、owner、lookup multiset 和 COW endpoint，只把 compiled 的 delta+latest
合为一次 append，并让 exact route one-shot 到 final：

- staged owner-mixed：3.022 s；
- fused-finalization owner-mixed：2.700 s；
- one-shot all-exact：2.083 s；
- collective 从 39 降为 18 次/rank；
- fused 比 staged 改善约 10.7%，但仍比 exact 慢约 29.6%。

v3 随后实现 owner-local segmented append-only destination：retained target extent 只写一次，
incremental HSTU 只返回新 suffix K/V，最终逻辑 cache 由
`{retained segment, suffix segment}` 组成。B16 pilot 从 2.700 s 降到 2.357 s，但仍比
one-shot exact 2.084 s 慢 13.1%。该负结果定位出真正瓶颈不是 D1 retained transform
（仅约 56 ms/rank），而是 compiled suffix batch 的形状与同步。

### 5.3 收敛机制：wave-compiled segmented migration

D1 ActionPlan 已在 wave 开始前给出每条记录的 retained 长度 `R`、suffix 长度 `S` 和 final
长度 `F`。旧 executor 按 `F` 排 compiled records，但增量 attention 的主要 padding 代价近似
`B × S_max × (R_max + S_max)`。因此 v4/v5 把 D2 收敛为同一个 physical-wave lowering
问题，而不是三个互不相关的优化：

1. compiled extents 按 `(S, R)` 组织，而不是按 `F`；
2. retained finalization 使用 segmented append-only manifest，不重写整个 retained prefix；
3. `scheduled_exact` 与 `natural_exact` 保留不同 lineage/reason，但降低到同一个 full-final
   exact execution pool。

这三项都不改变 frozen requested action、owner map、exact budget、lookup multiset 或质量。
它们分别消除错误形状 padding、full-record destination rewrite 和语义标签造成的物理 phase
fragmentation。当前最准确的 D2 名称是：

> **Wave-Compiled Segmented Migration over Row-Sharded Embeddings.**

B16 pilot 的同一条消融链为：

| 版本 | 机制 | mixed makespan | 结论 |
|---|---|---:|---|
| v1 | naive staged owner-mixed | 3.025 s | 比 exact 慢 45.5% |
| v2 | phase-fused contiguous finalization | 2.700 s | 比 v1 快 10.7% |
| v3 | segmented append-only | 2.357 s | 比 v2 快 12.7%，仍未胜 exact |
| v4 | `suffix_retained` shape order | 1.542 s | 比 v3 快 34.6%，首次稳定胜 exact |
| v5 | merged physical exact pool | 1.493 s | collective 18→15/rank；再快约 3.2% |

extent-size discovery 只用于冻结后续规则，不是正式调参证据：

| pilot extent | merged mixed | paired one-shot exact | exact / mixed |
|---:|---:|---:|---:|
| B4 | 1.088 s | 1.970 s | 1.81× |
| B8 | 1.235 s | 2.011 s | 1.63× |
| B16 | 1.493 s | 2.087 s | 1.40× |

完整 682-record W3 discovery 中，B8 是 mixed 与 exact 的已测最佳共同点：

- merged segmented mixed：3.633 s；
- unmerged segmented mixed：4.055 s；
- one-shot all-exact：6.716 s；
- mixed 相对 exact 快约 45.9%，即 exact/mixed 约 1.85×；
- mixed lookup tokens 为 347,062，exact 为 934,917；实际 one-way off-diagonal vector
  volume 分别约 454.62 MiB 与 1,222.86 MiB；
- 三次 mixed makespan 为 3.6320/3.6334/3.6336 s；
- B8 因 full-wave 上略优于 B4 且 collective 更少，冻结为下一轮 full-wave 默认。

主 artifact 及 file SHA-256 为：

| artifact | file SHA-256 |
|---|---|
| `results/system/cohortkv_design2_integrated_w3_development_v5/pilot192_shape_append_merged_exact_b4.json` | `79c48c50cbbd23203899a16d4c21e92368736986477ca569eb82fe9af9d61cfa` |
| `results/system/cohortkv_design2_integrated_w3_development_v5/full682_shape_append_merged_exact_b8.json` | `228929768479aa4d5e65e7849428766f4ba8f9b629de7183e3b024828c3e1029` |

这些 artifact 仍明确记录 `scientific_result=false`、`formal_stage_c=false`、
`paper_performance_claim=false`。source fixture materialization、wave/history preparation、
validation、manifest publication、global commit 和 reclaim 尚未进入 primary timer。因此它们
证明“当前 D2 机制值得进入正式 Stage C”，不产生可直接写入论文表格的最终 speedup。

append-only 的精确措辞是 **destination finalization 不重写 retained prefix**，不是 retained
K/V 完全零拷贝。suffix attention 目前仍把每个 extent 的 FP16 jagged retained K/V 临时整理成
FP32 padded layout。分段 cache 也必须由 serving 或下一轮维护直接消费；若 publication 后仍强制
拼成完整连续 cache，当前收益可能只是延迟支付。

### 5.4 全 payload correctness

`results/system/cohortkv_design2_integrated_full_payload_development_v1/full682.json`
（file SHA-256
`d17e2d16cf521a249d9d50482fd7ce821ef7fce6053c28767478b58257aa8508`）
在物理 W3、B8、`suffix_retained` 下逐 extent 重放 contiguous 与 segmented 路径，覆盖
682/682 records、所有 exact reasons、zero-delta record 和约 15.32 billion 个有效
K/V/last-hidden 元素：

- 所有 record/route/element coverage 与 allclose checks 通过；
- 最大 K/V 绝对误差为 0.00390625；
- 最大 last-hidden 绝对误差约 `1.91e-6`；
- 无 non-finite；
- 双侧 SHA 不同且不宣称 bitwise，因为 suffix-only attention 改变了浮点归约顺序。

该验证不计时，只关闭完整 payload correctness，不关闭 publication/consumer 边界。

### 5.5 Secondary embedding-tier contention characterization

`results/system/cohortkv_design2_resource_isolation_development_v1/h12.json`
（file SHA-256
`e8fed655f8545d54eaca21a0c2d9384c122c7b1a530eb2edb60b6dd65b040ef2`）
使用同一真实 row-sharded checkpoint、独立 NCCL process group/CUDA stream 和每 rank
250 requests/s、128 tokens/request 的固定速率 synthetic foreground lookup。1.5 s 固定分析
窗口内：

| maintenance | foreground p99 | 20 ms deadline misses | maintenance max-rank wall |
|---|---:|---:|---:|
| idle | 0.973 ms | 0 | 0 |
| rank-local dense control | 1.108 ms | 0 | 68.1 ms |
| mixed embedding | 26.106 ms | 24 | 69.7 ms |
| all-exact embedding | 88.316 ms | 87 | 144.9 ms |

all-exact/mixed 的 unique-token、one-way vector bytes 和 maintenance wall 比分别约
2.04×/2.04×/2.08×。该结果支持“mixed maintenance 明显降低共享 embedding-tier
干扰”，但不支持“零干扰”：mixed 的 p99 仍超过 20 ms deadline。dense control 还表明主要尾部
并非同等时长的一般 SM/HBM compute 所致。

这是 deterministic contention probe，不是真实 serving workload；当前只有一个 offered rate
和一次 development run。它不是 Motivation 2、D2 核心机制或 paper-strength gate，只能作为
可选资源归因。若保留正式主张，仍需 rate sweep、重复、同 binary 交错执行和真实 publication
boundary。

## 6. 等待 GPU2 后的正式待办

GPU2 安全空闲后执行：

```bash
python scripts/launch_cohortkv_design2_stage_b.py \
  --world-sizes 4 \
  --cases normal hard_failure \
  --visible-devices 0 1 2 3

python scripts/freeze_cohortkv_design2_stage_b.py
python scripts/freeze_cohortkv_design2_stage_b.py --check
```

随后重新执行 Stage-A freeze check、tests、lint 和 diff check。只有 W4 两份 artifact 通过、
`stage_b_summary.json` 被 freeze script 合法生成且 `--check` 通过后，才能把 Stage B 标记为
frozen。

不得用以下方式“补齐”正式 gate：

- 四个进程共享三张或更少 GPU；
- Gloo 替代 NCCL；
- W2 cross-island 或 W3 development artifact 改名为 W4；
- 手工生成或编辑 `stage_b_summary.json`。

## 7. W4 闭合后的下一入口

正式 Stage-C evaluation 开始前仍需：

1. 回查 Stage-B handoff 和 frozen summary；
2. 在 `docs/eval_protocol.md` 下冻结新的 D2 protocol、action/config identity、timer、
   baselines、communication/capacity ledger 和 artifact schema；
3. 先跑 correctness/failure closure，再形成任何论文 timing；
4. 保持 D1 retained-prefix 与 D2 post-append integrated 两个计量边界分开；
5. 在同一 binary/protocol 下冻结 strong all-exact、naive sharded fixed-action mixed、
   segmented/shape-aware/merged-exact ablation；
6. 关闭 segmented consumer/next-wave、plan-inclusive preparation 和
   publication/commit/reclaim 边界。synthetic contention sweep 是 optional，不阻塞主线。

因此当前唯一准确的状态是：

```text
Stage B formal freeze        = BLOCKED on independent W4
Stage C C0 diagnostic        = COMPLETE
Stage C development entry    = GO
Stage C formal evaluation    = BLOCKED
paper evidence               = BLOCKED
```
