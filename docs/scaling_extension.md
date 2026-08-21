# EvoKV 规模化扩展路线

更新日期：2026-08-18。

本文定义 37D 路线从 Yambda-50M development platform 向更大模型、更大数据和更大 persistent-state population 扩展时的验证边界。它是当前路线和论文设计的规模化补充，不冻结新的实验 contract，也不提前授权长时间或大规模实验。

## 1. 规模定位

Yambda-50M 当前承担的是**机制开发平台**，不是最终“大规模生成式推荐系统”结论的唯一依据。当前模型只有几百 MB，有效历史长度也约为 512；即使原始用户历史可以达到几千或几万条，也不能把自然历史长度直接写成模型已经验证过的 sequence scale。

50M 阶段要回答的是：

- release-time snapshot 如何冻结；
- Previous Full、Current Full、Reuse 和合法 partial path 如何定义 lineage；
- scheduler 如何只使用发布时可得的信息；
- exact-equivalent work 如何统一表达迁移预算；
- No-op、局部迁移和 Exact 如何在有限预算下分配给整个 materialized-state population。

因此，50M 的正确产物是稳定、可审计、可迁移的协议和机制，而不是把一个小 checkpoint 人工撑大后作为系统规模证据。

## 2. 什么可以跨规模复用

从 50M 迁移到更大 workload 时，优先迁移 EvoKV 的系统抽象，而不是迁移小模型上的具体数值规律。

| 可迁移的抽象 | 规模迁移时仍需重新验证的事实 |
| --- | --- |
| 发布时冻结 materialized state 的规则 | 哪些用户或状态风险最高 |
| 合法 Full/Reuse/partial lineage | Reuse 相对 Exact 的具体 regret、tail 和 downstream gap |
| future-information exclusion | 哪些 metadata 最能预测风险 |
| exact-equivalent work 与预算 frontier | 给定预算可节省的比例，例如 20% 或 40% |
| No-op、局部迁移、Exact 的动作接口 | 风险是否随模型宽度、层数、候选集合或历史长度变化 |
| state-version debt 与 rollout accounting | I/O、worker-hours、makespan 和吞吐的绝对规模 |

换句话说，协议、lineage、预算和控制器接口可以作为方法骨架复用；用户级风险排序、节省比例、候选集合效应以及模型 release regression 都必须在新规模上重新测量，不能从 50M 外推。

## 3. 分阶段扩展路线

### 阶段 A：Yambda-50M——机制稳定与 development evidence

继续使用当前同源 workload，完成现有路线尚未闭合的工作：

1. 完成 multi-panel cutover label 和 external-validity gate；
2. 生成 append `0/1/2/4/8/16` 的 dilution curve；
3. 在 development edges 上冻结 metadata-only risk ranker、budget points、executor 和 accounting；
4. 在 blind edge 上做 controller qualification；
5. 再加入 partial path、3d robustness、independent seed 和 recursive lineage；
6. 记录协议中与规模无关的接口、审计字段和失败条件。

本阶段不承担“大模型、大数据”主 qualification，也不因为原始历史很长就宣称已验证 1K--4K sequence scale。

### 阶段 B：Yambda-5B——同源大模型与大数据主 qualification

方法冻结后，将同一套 protocol、manifest、lineage、预算定义和评价指标迁移到 Yambda-5B。它与 Yambda-50M 属于同源体系，时间语义、行为定义和 release protocol 有机会直接复用，但仍须先完成新数据审计和新的 workload contract；“同源”不是跳过重验证的理由。

这一阶段的目标是让模型和数据自然进入真正更大的范围，而不是通过无语义的 padding 凑参数。可探索的 foundation 配置包括：

- 约 8 层；
- hidden dimension 256 或 512；
- 历史长度 1K--4K；
- 随着 item vocabulary、embedding table、模型宽度和层数增长，模型进入数 GB 到几十 GB 的范围。

具体配置必须由显存、训练吞吐、有效历史覆盖和 workload contract 共同决定，不能在当前文档中预先把某一个配置当成冻结结果。Yambda-5B 阶段至少要重新确认：

- Full 与 incremental Append 的 correctness 和 precision tolerance；
- 版本链、release cutover、snapshot population 和 OOV/catalog policy；
- no-op/exact oracle frontier 与 50M 的方向是否一致；
- 发布前 feature 的风险排序是否跨边、跨 seed、跨规模仍然有效；
- 模型大小、有效历史长度、materialized-state 数量和迁移工作量的实际测量。

论文中的“大模型、大数据”主 qualification 应由这一阶段承担。50M 上观察到的规律可以作为 hypothesis 或 development evidence，但不能直接充当 5B 的结果。

### 阶段 C：VK-LSVD——population 与 system-scale 验证

在更大的用户和交互规模上，VK-LSVD 更适合承担 population/system-scale 验证：重点不只是单个模型 checkpoint 的大小，还包括百万到千万级 materialized states 下的：

- persistent KV footprint 和分层存储占用；
- KV read/write 与 history input I/O；
- migration work、worker-hours 和 pipeline makespan；
- scheduler 吞吐、完成时间和连续 release 下的 state-version debt；
- 不同状态数量、prefix length 分布和资源预算下的 rollout completion。

这一阶段可以验证系统压力和跨 population 场景的稳健性，但不能把新的用户规模自动写成模型质量或兼容性结论。数据是否具有可用时间语义、行为定义和 candidate/evaluation 条件，仍需先经过独立 audit。

### 阶段 D：RecFlow——真实 candidate workload 补充

RecFlow 更适合补充真实的多阶段 candidate workload。当前已知的研究风险是 compatibility risk 可能依赖 candidate set；因此，真实候选日志可以用来检查：

- 同一 persistent state 在不同 candidate set 下的 semantic risk 是否改变；
- canonical probe、固定候选集与真实候选阶段之间的相关性；
- multi-stage candidate filtering/reranking 对 migration frontier 的影响；
- candidate workload 变化是否使原有 risk ranker 或预算策略失效。

RecFlow 是 candidate-protocol 和跨场景验证的补充，不替代 Yambda-5B 的同源大模型 qualification，也不应在没有数据审计的情况下被写成生产请求 SLO 证据。

## 4. 规模化时必须同时增长的维度

最终的“大规模”应是多维度共同成立，而不是一个几十 GB checkpoint 的单点展示：

| 维度 | 50M 的角色 | 后续主验证 |
| --- | --- | --- |
| 数据交互量 | 机制开发与时间协议 | Yambda-5B；再到 VK-LSVD 的系统压力 |
| 用户数量 | 小 population 的完整协议闭环 | Yambda-5B 的大数据 qualification；VK-LSVD 的百万--千万状态 |
| item vocabulary / embedding | 当前 foundation 的真实配置 | Yambda-5B 的自然参数增长 |
| 模型宽度、层数 | correctness、lineage、scheduler 开发 | 5B foundation 的数 GB--几十 GB 范围 |
| 有效历史长度 | 当前约 512 的机制验证 | 5B 上重新验证 1K--4K；不得由原始历史长度替代 |
| persistent KV 总量 | 小规模 state-level frontier | VK-LSVD 上的 footprint、I/O 和 rollout |
| candidate workload | 固定、可审计的 profiler/quality manifest | RecFlow 的真实多阶段 candidate 补充 |

只有这些维度按阶段同步扩大，论文中的“大规模”才同时具有模型、数据、状态和系统含义。

## 5. 规模迁移的准入条件

规模迁移不是把脚本和 checkpoint 直接复制到新数据集。每个阶段至少需要以下证据链：

1. **数据审计**：时间单位、行为定义、排序、重复、catalog/OOV、用户/item 覆盖和 candidate 条件已明确；
2. **模型审计**：Full/Append correctness、有效历史覆盖、显存/存储占用和训练/推理成本已测量；
3. **协议复现**：snapshot、release gap、manifest、lineage 和 future-information exclusion 与当前 contract 对齐；
4. **机制重测**：oracle frontier、risk ranker、executor 和 state-version debt 在新规模上重新测量；
5. **系统会计**：报告 state 数量、KV footprint、read/write、worker、makespan 和 completion，而不是只报告一个 checkpoint size；
6. **证据分层**：development、scale qualification、system-scale validation 和 candidate-workload validation 分开标注，不跨阶段拼接成单一结果。

如果某一阶段无法满足这些条件，它可以作为数据或模型的 exploratory bring-up，但不能承担对应的论文规模结论。

## 6. 论文中的规模化表述

推荐采用以下分层表述：

- Yambda-50M：验证科学问题、合法 cache lineage、风险定义、预算 oracle、scheduler 和 migration mechanism 的 development platform；
- Yambda-5B：验证同一方法能否在同源大数据和真正更大 foundation 上成立，是“大模型、大数据”主 qualification；
- VK-LSVD：验证百万到千万 materialized states 下的 population 和 system-scale 行为；
- RecFlow：验证真实多阶段 candidate workload 对 compatibility risk 和 scheduler 的影响。

不能使用的表述包括：仅凭 50M 原始长历史声称已验证长序列；仅通过扩大 hidden dimension 或 embedding table 声称大型 workload；把 50M 的 risk pattern 或节省比例直接外推到 5B；把离线 candidate proxy 写成真实线上 P99、QPS 或业务 SLO。

## 7. 资源与执行纪律

当前阶段仍遵守最小数据和最小计算原则。Yambda-5B、VK-LSVD 和 RecFlow 在方法和 contract 冻结前只是后续扩展候选，不应提前下载、训练或启动长实验。进入下一阶段后，也应先做 canary、schema/time audit、Full/Append correctness 和小规模 migration smoke，再决定是否扩大 population、模型和 worker 数量；默认不保留大 checkpoint、日志、生成数据或 processed corpus。
