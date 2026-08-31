# Yambda-500M Large 模型规模讨论稿

日期：2026-08-29  
状态：**供专家讨论；不是训练合同，不授权 Large 数据物化、canary 或长训练**

## 1. 建议结论

当前原定 Large 为 `8L/H256/8 heads/context1024`，使用全部 79,681 名合格用户。它约有
**572.3M 参数**，相对 Medium 的 266.3M 已扩大 2.15 倍，因此它本身是一个成立且稳健的论文
Large point，并不存在“8 层就等于 Medium”这一技术问题。

如果希望在不明显破坏数据—容量匹配的前提下，让最终 Large 再大一个清晰台阶，本文建议专家优先讨论：

> **Large 主方案：10 layers、hidden 320、10 heads（head dim 32）、context 1024，约 717.3M 参数。**

它比 Medium 的参数量大 2.69 倍，与 Large 相对 Medium 约 2.64 倍的 foundation 数据增长基本一致；
同时把层数从 6 提高到 10，论文表述上也能形成明确的 S/M/L 梯度。`12L/H320` 不建议直接作为
默认主方案：它只有 718.3M 参数，比 10L 仅多 0.14%，却多 20% token-layer work、20% attention
work 和 20% persistent K/V。

建议冻结顺序为：`10L/H320` 是首选；若事前资源 canary 因 activation、吞吐或稳定性不过门，使用
只由物理门触发的 `8L/H320` 预注册 fallback；若 embedding/FSDP 本身仍不安全，再回到已有
`8L/H256`。不能读取 Large quality 后再从三者中挑最好者。

## 2. Small、Medium 与 Large 数据/模型对齐

| Scale | 固定用户 | foundation catalog | foundation 训练请求 | 完整历史 listens | 模型 | 参数量 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Small | 10,000 | 781,677 | 约 642,483 | — | 4L/H128/4 heads/C512 | 100.4M |
| Medium | 30,000 | 1,380,509 | 2,004,404 | 171.05M | 6L/H192/6 heads/C1024 | 266.3M |
| Large（已有规划） | 79,681 | 2,224,809 | 约 5.30M（待物化） | 450.61M | 8L/H256/8 heads/C1024 | 572.3M |
| Large（本文首选） | 79,681 | 2,224,809 | 同上 | 450.61M | 10L/H320/10 heads/C1024 | 717.3M |

Large 的 5.30M 是根据同口径 population audit 中 Large/Medium foundation executable-request
比例 4.923M/1.862M = 2.64，再作用到当前 Medium manifest 的 2.004M known request 得到的规划值；
它不是已经封存的 Large manifest 数字。正式合同前必须物化并报告真实去重、冲突排除和 target-OOV
后的计数。

Medium → Large 的主要倍率是：

- 用户：2.66 倍；
- 完整历史 listens：2.63 倍；
- foundation 请求：同口径审计为 2.64 倍；
- catalog：1.61 倍；
- 每个 catalog item 的完整历史 listens：约 1.63 倍（123.9 → 202.5）。

因此 `H192 → H320` 的 1.67 倍宽度增长与每个 item 可用历史支持增长非常接近；catalog×hidden
共同使总参数增长到 Medium 的 2.69 倍，也与请求量的 2.64 倍接近。按这个口径，10L/H320 是
“稍微积极但仍匹配数据”的容量点，不是用有限数据硬撑一个夸张模型。

## 3. 为什么不能只看层数

当前 HSTU 的参数几乎全部位于 item embedding。以 Large catalog 计算：

| 候选 | 总参数 | HSTU blocks 参数 | FP32 weights | BF16 K/V / user | 全人口满 context BF16 K/V 上限 | 相对 Medium 满 context 单请求计算¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8L/H256 | 572.3M | 2.62M | 2.13 GiB | 8.0 MiB | 622.5 GiB | linear 2.37× / attention 1.78× |
| 8L/H320 | 716.2M | 4.10M | 2.67 GiB | 10.0 MiB | 778.1 GiB | linear 3.70× / attention 2.22× |
| **10L/H320** | **717.3M** | **5.12M** | **2.67 GiB** | **12.5 MiB** | **972.7 GiB** | **linear 4.63× / attention 2.78×** |
| 12L/H320 | 718.3M | 6.15M | 2.68 GiB | 15.0 MiB | 1,167.2 GiB | linear 5.56× / attention 3.33× |

¹ 只比较结构项，context 均为 1024；linear 近似按 `layers × hidden²`，attention 近似按
`layers × hidden`。它们不是实测 GPU wall-clock。

这解释了两个容易混淆的问题：

1. 在 H320 下从 8 层加到 12 层，并不会让 headline parameter count 明显变大：新增四层只增加约
   2.05M 参数（约 0.29%）；但它会把 contextual compute 和每用户持久状态提高 50%。
2. 若目标是让参数规模与 Large 数据量匹配，应主要提高 hidden width；若目标是提高 contextual
   depth，则可以加层，但必须单独为额外 compute、优化难度和近 1 TiB 的 population state 负责。

因此 10L/H320 是折中点：宽度完成主要参数扩展，10 层提供明确的深度升级；12 层增加的主要是系统
压力，而不是统计容量。

## 4. Context 不建议继续增大

Large 人口更多，但单用户历史长度分布与 Medium 几乎相同：pre-foundation median 分别约为 2,334
和 2,352，至少有 1,024 条历史的比例分别约为 67.9% 和 68.2%。这说明扩大 population 并没有让
单用户序列明显变长。

所以 Large 保持 context1024 最干净：它隔离 population/model scale，复用 Medium 已验证的时间因果
和 rolling semantics。把 context 改成 2048 会同时改变研究对象、attention 成本和 persistent-state
I/O，而且只有约三分之二用户能填满 1024，本轮没有足够依据承担这项变化。

## 5. 训练与机器资源的数量级

Medium v0 在两张 A40、global batch 32 上实测 219.48 分钟，峰值 reserved 约 10.3 GiB/rank。
Large 使用四张 A40、保持 global batch 32 时，依据 2.64 倍请求量和上述结构计算项做一阶外推：

- 8L/H256 foundation：约 10–16 wall-clock 小时；
- 10L/H320 foundation：约 16–31 小时；
- 12L/H320 foundation：约 19–38 小时。

这只是资源 envelope，不是承诺；CPU history loading、FSDP all-gather、实际长度和通信可能改变结果。
按 Medium 的 D14 实测继续外推，10L/H320 的 shared foundation 加四个完整 D14 update，训练本身大致
是 **1–2 天量级**；完整 Full/Reuse 评测还会额外占用数天，并可能比训练更慢。

历史 Yambda-50M 8L pilot 只能说明本机四卡 FSDP 路径曾经可执行；它使用旧 workload、旧 catalog
语义且已按清理记录退出当前结果链，不能拿来承诺本次显存、质量或 wall-clock。新 Large 必须重新做
真实架构的 backward/optimizer/checkpoint focused canary。

## 6. 对论文规模的判断

即使采用保守的 8L/H256，本轮也已经具备明确的大规模属性：79.7k persistent users、2.22M catalog、
572M parameters 和约 0.62 TiB 的全人口满 context BF16 K/V state 上限。论文应同时报告 population、catalog、总参数、
非 embedding 参数、context 和 persistent-state bytes，而不是只用“8 层/10 层”命名规模。

采用 10L/H320 后，最终点变为约 717M 参数和约 0.95 TiB population K/V 上限，更容易体现本文真正研究的
系统压力：一份大规模、跨 release 持久存在的用户状态如何在受限 compute/I/O 下收敛。这个规模比
Medium 足够明显，又没有进入 12L/H320 那种“参数几乎不增、成本继续上升”的低收益区间。

## 7. 建议交给专家裁决的问题

建议专家在以下两个选项中裁决，不继续展开无边界 architecture search：

1. **推荐：10L/H320/C1024，约 717M 参数。** 数据—参数倍率最匹配，深度/宽度都与 Medium 拉开；
   接受约 0.95 TiB population K/V 上限和更长的评测时间。
2. **保守：8L/H256/C1024，约 572M 参数。** 已有统一 scale contract 的规划值，资源风险最低，仍然
   是可信 Large；代价是模型参数增长略慢于数据增长，最终规模对比不如首选鲜明。

`12L/H320` 只在专家明确把“更深的 contextual reader / 更重的 state-system stress”视为独立实验目标
时采用；不能仅因为 12 层听起来更像 Large 就选择它。

专家裁决后，执行者再创建新的 prospective Large contract。现有
`yambda500m_unified_scales_v1.yaml` 是已冻结的数据/population 规划，不能原地改写；新合同应保留
79,681-user population、item-mapping hash、`[0,217)` foundation、`[217,300)` stream 和 seed17，
只显式覆盖最终 architecture，并在任何长训练前完成 label-free manifest、资源估计、四卡 focused
canary 和用户 launch。

## 8. 依据

- `configs/contracts/yambda500m_unified_scales_v1.yaml`：S/M/L population、原定模型与 context；
- `results/data_audit/yambda500m_scale_v1/population_audit.json`：人口、catalog、历史和同口径请求审计；
- `data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/manifest.json`：Medium 当前真实请求计数；
- `results/yambda500m_medium_seed17/full_reuse_matrix_v1/medium_scale_experiment_summary.md`：Medium
  训练/评测结果和实测资源；
- `results/checkpoint_cleanup_2026-08-24.md`：旧 8L pilot 与当前 Yambda-500M Large 的证据边界。
