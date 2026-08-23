# Archived: P9.10–P9.11 Runtime and Frontier Result

## P9.10 全人群 migration-only runtime

三个预注册条件覆盖 edge1 M0-F R2、edge1 M1 R2 和 edge2 M1 R1-edge2。每个条件使用全部 8,229/8,488 个 cutover states、真实 prefix-length 分布和 exact-length batching，重复三次。

平均 batch 约 9.4，只有约 47% batches 达到 16，因而结果包含 ragged population 的执行碎片。

| Action | ms/state 范围 | 单 GPU 全人群 kernel rollout |
|---|---:|---:|
| Layer0-Recent128 | 0.065–0.067 | 0.54–0.57 s |
| Layer0-Middle | 0.063–0.065 | 0.52–0.55 s |
| Layer0-Full | 0.061–0.062 | 0.50–0.52 s |
| HybridTail128 | 0.270–0.276 | 2.22–2.35 s |
| Exact-All | 0.475–0.482 | 3.91–4.09 s |

M0/M1 同架构的 edge1 runtime 相差约 1%，说明结果主要由 executor 与 state shape 决定。Pinned-memory PCIe proxy 约为 25.9–26.7 GB/s；它不是持久化存储吞吐。KV storage 仍只报告逻辑 read/write bytes。

Layer0 宽度的 kernel 时间接近，是当前小模型投影、launch 和 full-cache clone overhead 主导的结果；不能仅凭 prototype runtime 忽略 token-layer、raw-history 和 KV bytes。

## P9.11 正式 development frontier

主预算轴为 exact-equivalent recomputed token-layers。所有 action、cell 和负恢复 seed 均保留。比较：

- version-level best frozen action；
- random Exact allocation；
- top-risk Exact offline oracle；
- state×action near-optimal offline oracle。

### 5% budget

统一 action 因最便宜的全人群 partial 约需 6.8%，通常只能 No-op；random Exact 约恢复 5%。state×action oracle 的三-seed恢复率为：

| Release / Model | Recovery seed points |
|---|---:|
| R1 edge1 / M0-F | 59.2% / 58.8% / 84.0% |
| R1 edge1 / M1 | 54.0% / 65.6% / 72.2% |
| R1 edge2 / M0-F | 51.9% / 90.2% / 59.2% |
| R1 edge2 / M1 | 63.0% / 72.8% / 64.3% |
| R2 / M0-F | 80.1% / 85.0% / 84.9% |
| R2 / M1 | 38.3% / 55.3% / 93.9% |

### 10% budget

state×action oracle 在非 R0 cells 中恢复约 59.4%–99.7%，多数条件明显超过 uniform action、top-risk Exact 和 random Exact。

### 解释

这证明了 state-level、多动作分配的系统机会：低预算下，给大量状态做便宜 layer-0 projection、给少数状态做 Hybrid/Exact，显著优于只挑少量用户 Exact 或对全版本应用一个动作。

但 near-optimal policy 使用 sealed CurrentExact cutover-probe loss，部署时等价于先知道答案，只能作为 oracle upper bound。它不授权直接训练或宣称 scheduler。下一门是：只用 release-time cheap features、release metadata 和显式计费的小样本 probe，能否逼近 oracle frontier；policy assignment 必须先封存，再连接 held-out labels。

## 当前裁决

```yaml
legal_partial_executor: established_in_development
full_population_fidelity_recovery: established
heldout_quality_recovery: established_for_M1_R2
measured_migration_runtime: established
state_level_allocation_opportunity: established_as_offline_oracle
deployable_profiler_or_scheduler: not_yet
paper_qualification: not_yet
```
