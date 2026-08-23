# EvoKV 下一阶段：冻结方法的规模验证

更新日期：2026-08-22。

## 目标

Yambda-50M、4L/H128/context512 已完成端到端 development 闭环。下一阶段不再
寻找更大的 gap，也不重新设计方法，而是回答：

> 冻结的 workload、release、partial action、profiler 和 scheduler，在更深、更宽、
> 更长上下文的模型上是否仍能执行，并保留相同方向的 H/S/frontier 结构？

首个规模点固定为：

```yaml
dataset: Yambda-50M
workload: F Explicit Feedback
models: [M0-F, M1]
seeds: [17, 37, 71]
architecture:
  layers: 8
  hidden: 256
  context: 1024
base: reuse_frozen_feature_schema_and_fit_protocol
GPUs: [0, 1, 2, 3]
```

这是同源模型规模验证，不冒充 Yambda-5B 或生产规模。

## 不随规模改变的内容

- N/R/F 的任务语义及 F 主 workload；
- Frozen Base + CC residual 部署分数；
- Full/Recent、Current/Reuse 和 recursive lineage；
- R0/R1/R2 release recipe；
- seed 17/37/71；
- action 的依赖语义；
- 1% sparse probe、固定 Ridge 和 allocator；
- 5%/10%/25% exact-equivalent token-layer budgets；
- target-free assignment 先封存、quality 后 join；
- dislike-only 指标完整报告。

允许因架构自然变化而重新计算的只有动作成本、KV bytes、batch size 和物理 runtime；
不得据此改变动作选择算法。

## 分阶段执行

### S0 — EvoKV v1 full-stack seal

封存 P11.1–P11.4 的合同、六种 dependency-closed actions、1% deterministic probe、
state features、StandardScaler + Ridge(alpha=1.0)、5%/10%/25% budgets、成本核算、
allocator、grouped executor、三个 seed 和统计方法。同时记录：

```text
code commit
config / contract hashes
manifest hashes
Frozen Base hashes
checkpoint hashes
assignment / raw-result / adjudication hashes
```

该封存版本命名为 **EvoKV v1**。之后不得用 4L、8L 或 θ3 结果继续调 predictor、
action、feature、probe、budget 或分配逻辑。

### S1 — 静态资源与覆盖审计

只计算：

- 参数量和 embedding/encoder/head 分解；
- Full-1024 单 batch 显存、KV bytes/state；
- 8L action 的 token-layer、attention-pair、read/write bytes；
- Yambda 中 Full-1024、Recent-32 的历史覆盖率；
- 预计 θ0、R0/R1/R2 和六 cell 的 GPU-hours、checkpoint 空间。

单卡 FP32 AdamW 已由真实一步 canary 判定 OOM；冻结执行改为一模型占四卡的
FSDP FULL_SHARD。若四卡 canary 无法完成，不擅自减模型或 history。

### S2 — Correctness canary

固定 M0-F seed17，仅做数十个用户/请求：

- Full 与 incremental Append 等价；
- rolling cap-1024 的逐事件 append/evict 正确；
- Base logits 对 Recent/Full 完全相同；
- candidate query 不写回 persistent state；
- R0 cache identity；
- Exact action 与 Current Full 对齐；
- grouped executor 与逐 UID executor 数值等价。

Canary 只修实现错误，不查看 H/S 是否“足够大”。

当前结果：通过。24.07 亿参数模型的单卡 FP32 AdamW 在 optimizer step OOM；四卡
FSDP FULL_SHARD canary 通过，每卡 peak reserved 约 40.997GB（85.95% A40），余量约
6.70GB。Full/Append、rolling cap、R0 identity、Exact、query immutability 全部通过；
grouped-vs-serial 最大绝对差约 `4.41e-6`。训练器还完成了一个 logical batch、dev
selection、约 9.63GB full checkpoint 写入/普通 HSTU key/shape round-trip，临时
checkpoint 随后删除。

### S3 — θ0 长期 H scale gate

训练且只训练 8L θ0 的 M0-F/M1、三个 seed。一次性比较：

```text
Base Only
Base + Recent-32
Base + Full-1024
Base + frozen compact-summary companion
```

完整报告 target-free H、log loss、ROC-AUC、dislike PR-AUC、Brier 和
dislike-only log loss。若 M0-F/M1 都未保留长期 H，则停止规模版本链；不能改 history、
task weight 或 checkpoint 来寻找正结果。

执行采用预注册的分阶段顺序：先运行 M0-F seed17 θ0 并立即裁决 H；H 通过后先完成该
seed 的 R0/R1/R2 最小版本链。只有 pilot chain 保留 `H→S→legal partial` 后，才补
seed37/71 与 M1。Pilot 只能授权后续 replication，不能单独升级为三-seed正式结论。

### S4 — R0/R1/R2 staleness scale gate

只有 S3 至少一个模型条件保留 H，才训练对应的 R0/R1/R2：

- R0 必须保持数值地板，否则停止查 lineage；
- R1 两边和 R2 原样复现，不调 update 强度；
- 报告 H、S、S/H companion、tail 和 quality；
- 所有三个 seed 保留。

#### S3/S4 pilot 结果（2026-08-23）

M0-F seed17 已完成 `theta0 -> R1 edge1 -> R1 edge2` 以及从 theta0 分叉的 R2。
本轮按用户指令只训练/评测 R1 与 R2，未运行 8L R0；因此不能把 4L 的 R0 identity
直接写成规模复现结果。

| Edge | H: Full1024 vs Recent32 JS | S: Exact rolling vs Reuse rolling JS | S/H | Reuse harm: log-loss gain |
| --- | ---: | ---: | ---: | ---: |
| R1 edge1 | 0.000850343 | 0.000148300 | 0.1744 | +0.00152275 `[0.00056077, 0.00245416]` |
| R1 edge2 | 0.000122303 | 0.000034853 | 0.2850 | +0.00014571 `[-0.00036910, 0.00063620]` |
| R2 | 0.000977428 | 0.000150880 | 0.1544 | +0.00113881 `[0.00013436, 0.00213231]` |

每条边的 H 和 S 都通过冻结的用户 bootstrap / panel 门。该结果表明 8L/H256/context1024
上长期状态与真实 rolling-cache staleness 均保留；R1 edge1 和 R2 还复现了 aggregate
任务质量伤害。它仍是单 seed development scale pilot：不能替代 M1、seed37/71、8L R0、
冻结 partial/scheduler replay 或 blind qualification。

封存 artifacts：

- raw seal：`results/scale_8l_v1/hs_raw_seal_v1.json`；
- adjudication：`results/scale_8l_v1/pilot/s4_*_m0_f_seed17_adjudication.json`；
- result contract：`configs/contracts/scale_8l_hs_result_v1.yaml`。

### S5 — Frozen partial 与 scheduler replay

不做新 tomography 搜索，只回放冻结动作：

- Layer0-Recent128；
- Layer0-Middle；
- Layer0-Full；
- Hybrid-Tail128；
- Exact-All。

由于总层数变为 8，Hybrid-Tail 的实际 token-layer/attention work 会自然变化，必须重新
计费。随后使用原 1% Ridge scheduler，比较同成本 uniform、metadata、random Exact 和
offline oracle；assignment 仍先封存。

当前已准备一键、可恢复、失败即停的 S5/S6 pilot 队列。它依次补训练 8L R0、运行四个
release 的 16-state canary、全人群六动作 target-free profiler、raw seal、action
adjudication、1%/2%/fixed-count/capped-rate scheduler replay，以及 assignment 封存后的
rolling quality validation。启动命令：

```bash
PYTHONPATH=src:scripts python scripts/run_scale_8l_method_full.py --run
```

状态查询：

```bash
PYTHONPATH=src:scripts python scripts/run_scale_8l_method_full.py --status
```

队列使用 GPU 0/1/2/3；R0 的四卡 FSDP 训练结束后，四个 action cell 分卡并行。任何
Exact 等价、R0 identity、raw hash 或 lineage gate 失败都会停止，且不会访问 theta3。

#### S5/S6 pilot 结果（2026-08-23）

队列已全部完成。R0 cache-producing 参数变化为 0，所有 action 的最大 logit 差为
`3.58e-7`。全人群 target-free recovery：

| Edge | Recent128 | Middle | Layer0-Full | HybridTail128 |
| --- | ---: | ---: | ---: | ---: |
| R1 edge1 | 21.31% | 31.03% | 24.06% | 43.42% |
| R1 edge2 | 50.16% | 74.87% | 68.30% | 55.97% |
| R2 | 22.37% | 73.17% | 96.35% | 21.26% |

1% Ridge 在 8/9 个 budget cells 中优于最强同成本非学习基线；唯一例外是 R2 25%，
recovery 低 `1.19` 个百分点。5%/10%/25% policy 的 grouped transition runtime 为
`2.97–13.74s`，对应各 edge Exact-All 的 `44.98–46.64s`。

用户等权 rolling quality 只在强发布 R2 的高预算点形成清楚恢复：25% policy 相对 No-op
改善 log loss `0.001234 [0.000117, 0.002311]`，与 Exact 差
`-0.000095 [-0.000302, 0.000114]`。R1 的 policy fidelity 虽恢复，但质量 CI 未稳定优于
No-op。该结果支持“冻结方法跨规模有效但 quality opportunity 依赖 release semantics”，
不支持“每个 release、每个预算都提升质量”。

### S6 — Probe population sensitivity

主方法仍为 1%，但规模实验同步报告：

```text
rate_1pct
fixed_count_64
fixed_count_128
fixed_count_256
capped_rate_min(1pct, frozen_fixed_cap)
```

主配置仍为 1%。fixed count/cap 必须在执行前由 4L population 和资源规则冻结；全部
sensitivity 都报告，不能根据 8L 或 θ3 结果选择一个替代主配置。本步骤只回答 profiler
成本能否在百万/亿级人口下有界，不重新选择 predictor。

## Scale-point 通过标准

规模点不要求每个绝对数字复制 4L，也不要求 S 变大。通过意味着：

1. correctness 与 R0 identity 完整通过；
2. 至少一个事前冻结模型条件仍有长期 H；
3. cache-producing release 中至少一个条件有可重复 S；
4. 至少一个冻结 partial 在三 seed 聚合上正恢复；
5. frozen scheduler 在相同成本下优于最强确定性非学习基线；
6. quality companion 没有被隐藏，rare dislike caveat 原样报告。

若只满足 H/S 而 scheduler 不通过，应报告“现象跨规模、方法未跨规模”；若 H 本身消失，
应报告当前长期状态对象不随该模型规模保留，而不是在 scale window 上重新开发 workload。

## B0 — θ3 blind contract

8L scale point 完成并冻结后、训练或打开 θ3 结果之前，创建并封存：

- θ3 训练/更新窗口、R0/R1/R2 recipe 和 model-admission gate；
- H、R0 numeric identity、S 和 same-cost scheduler 判定；
- aggregate quality non-inferiority/improvement criterion；
- dislike PR-AUC、dislike-only log loss 与 calibration companions；
- seed-level/clustered-bootstrap 统计；
- token-layer work、KV/history I/O、batched runtime 报告方式；
- H 消失、S 变弱、partial 失效、scheduler 优势消失或 quality 不稳定时的停止规则。

## B1 — 一次性 θ3 qualification

θ3 使用完全未查看的时间边，一次性执行 frozen EvoKV v1。Raw artifacts 先封存，再统一
计算指标。揭盲后不得调整 action、feature、Ridge、probe、budget、阈值或判定。

通过时，结论升级为：

> reproduced on a previously unseen temporal release under a frozen policy.

未通过时必须保留可定位的泛化边界，例如 H 保留但 S 弱、partial 保留但 scheduler 优势
消失，或 fidelity 恢复但 quality companion 不稳定；不得在 θ3 上继续开发。

## 更大数据

Yambda-5B、VK-LSVD 和 RecFlow 属于 scale point 之后的扩展：

- Yambda-5B：同源大数据/大模型主 qualification；
- VK-LSVD：百万到千万 materialized-state 的 population/I/O/makespan；
- RecFlow：真实多阶段 candidate workload 外部验证。

它们都必须重新审计时间、行为、catalog、candidate 和 population，不能直接继承 50M 的
风险排序或节省比例。

## 当前授权边界

S0–S2 已通过。S3 及以后属于长实验；资源预算、canary 和自动队列已经准备完毕，当前
停在首次 `M0-F seed17` 四卡训练，仍需用户显式启动。

## 论文同步产物

在 scale/blind 完成前只维护紧凑 paper jig，并准备四个固定图表，不扩写长稿：

1. 发布、persistent KV、No-op/Partial/Exact 和 budget allocator 的系统图；
2. N/R/F × R0/R1/R2 的 workload/release phase diagram；
3. No-op、fixed partial、metadata、Ridge、Exact-All 的成本—fidelity—quality frontier；
4. development、8L scale、recursive lineage、θ3 blind 的 qualification 表。
