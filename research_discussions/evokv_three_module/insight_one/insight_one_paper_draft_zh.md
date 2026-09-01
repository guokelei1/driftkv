# EvoKV Design I / Insight 1 专家讨论稿

日期：2026-09-01。

> 本文包含供专家讨论使用的工作背景和一段面向论文正文的中文候选稿。正文中的 `【待填】` 是尚未完成或尚未冻结的实验数字，不能在测量前替换成模拟结果。本文不授权任何新训练或长实验，所有 Current Exact splice 仍只用于结构诊断。

## 讨论背景：EvoKV 整体工作与 Insight 1 的位置

现代生成式推荐模型会持续训练并发布新版本，但线上用户状态不会随模型发布自然更新。在 HSTU/Transformer 推荐系统中，一个用户的大量历史交互已经被 Parent 模型编码为 persistent K/V，并长期保存在状态存储中。当 Current 模型通过上游评测并确定发布后，系统可以直接让它读取 Parent K/V，从而几乎不付出迁移成本，却会因状态与新模型不兼容而损失一部分本应获得的推荐收益；也可以让 Current 模型重放每个用户的完整历史，生成 Current Exact K/V，但这会产生巨大的全人口计算、raw-history 读取、状态写回和迁移时间。EvoKV 研究的正是在模型已经确定发布以后，如何以远低于 Exact-All 的成本，使海量长期用户状态演化到能够被 Current 模型有效消费的形式。它不讨论模型何时发布，也不是一般意义上的 KV cache 压缩。

整篇论文计划沿三个递进模块展开。One-Release Migration 首先回答一次 Parent→Current 切换中究竟应该迁移什么，并寻找低成本但接近 Current Exact 的状态表示；Continuous Multi-Release Migration 随后研究这种近似状态如何在连续版本链中反复演化，同时限制累积误差并在必要时触发更强 repair 或 Exact Rebase；GPU Migration Runtime 最后把前两部分确定的迁移算子组织成面向海量 ragged 用户状态的高吞吐执行。三部分分别解决迁移对象、长期正确性和物理执行问题，共同组成从“单次可行”到“连续可信”再到“真实可运行”的完整设计。

当前已封存的 Small 结果已经确认了问题和方案的基本可行性。在约一万用户、四层 HSTU、512 context 和五条连续 release edges 上，Current Full 的发布收益会被直接复用 Parent K/V 稳定侵蚀；在四条常规正收益边上，Reuse 吞噬了 25.5%–47.9% 的 AUC 发布收益。进一步在 3,000 个用户、每用户 64 个无标签 candidate probes 上观察到，同一用户的主要历史读取作用以及 Exact–Reuse readout difference 在候选之间高度共享。基于这一结构形成的 lightweight Per-user Reader Offset（PRO）只使用约 9.1% 的 Exact 理论 FLOPs，并为每个用户额外保存 512 个标量，即约 2 KiB sidecar；在五条冻结 release edges、217,584 个 rolling 请求上，它相对 Reuse 的 AUC 5/5 正向，log-loss 3/5 正向，五边平均 AUC 和 log-loss 均有改善。这些结果证明 reader-side 迁移路线可行，但不同版本边的恢复幅度仍不一致，现有 PRO 还不是已经闭合的最终设计。30,000-user、六层 HSTU、1024 context 的 Medium 训练及 Full/Reuse matrix 也已完成并再次确认核心兼容性问题，但 candidate-shared reader structure 与 PRO 尚未在 Medium 上验证；Large 则用于进一步检验模型规模、用户人口和 persistent-state footprint 的扩展性。

One-Release 最终需要给出一种具有推荐系统辨识度的迁移机制，而不能退化成普通的 Transformer KV repair。一个有说服力的结果应当在约 10%–20% Exact 计算预算内恢复大部分 Reuse–Current Exact 差距，并在不同 release edges 和更大规模上保持稳定，同时准确计入所需的状态、history I/O 和额外持久化开销。现有 PRO 已经证明这条路线能够在低成本下稳定改善 AUC，但质量恢复和最坏版本边仍有提升空间；因此，Design I 仍需要用一条清楚的 Insight 链解释，为什么常见 KV 重算粒度不够，以及 recommendation reader 提供了什么更合适的迁移对象。

本节讨论的 Insight 1 位于这条 Insight 链的起点。它不是要直接证明 PRO 的最终机制，也不是在 Evaluation 中比较若干完整系统，而是先检验一个更基础的设计选择：LLM KV 复用中常见的关键层、重要 token 和连续窗口局部重算，能否在 persistent Transformer recommendation state 上提供有利的重算量—恢复率关系。实验对三类方法分别构造有限且事先冻结的候选配置，并在每个预算下报告 best-observed result。如果其中任何一类在很小重算范围内已经恢复大部分 Current Exact 差距，One-Release 就应继续发展相应的局部重算路线；如果三类曲线都只能随着覆盖范围扩大而渐进恢复，它们便不足以成为主迁移抽象，下一项 Insight 才有理由转向 Current HSTU reader 的聚合边界，寻找紧凑、候选共享的用户级 compatibility correction。

因此，这份材料希望专家重点判断三件事：前文对现有 LLM 局部重算范式的归纳是否准确，一组按 release edge 展开的 best-observed 成本—恢复曲线是否足以支撑结论，以及由这一负面观察转向 recommendation-specific reader correction 的逻辑是否自然。下面给出拟进入论文的 Insight 1 正文。

## 论文正文草稿

### 4.1 少量 KV 重算能否恢复大部分跨版本差距？

当 Current 模型完成发布后，直接复用 Parent K/V 几乎没有迁移开销，却会损失一部分 Current 模型本应获得的推荐收益；Exact-All 使用 Current 模型重放完整用户历史，能够消除状态版本失配，但需要承担全人口历史读取、模型计算和状态写回。因此，我们希望在迁移开销与状态恢复质量之间取得更好的权衡，以明显低于 Exact-All 的成本尽可能保留 Current 模型的推荐收益。

在语言模型 KV cache 研究中，一种常见的降本思路是利用 cache 中的局部性，只保留或重算对输出更重要的部分。这类工作主要采用三种局部性假设。层局部性认为少数敏感层可能承担主要作用，[DroidSpeak](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_liu-yuhan_prepub.pdf) 据此识别并重算关键层组；稀疏位置局部性认为完整序列中只有少量 token 持续影响后续输出，[H2O](https://arxiv.org/abs/2306.14048)、[Scissorhands](https://arxiv.org/abs/2305.17118)、[SnapKV](https://arxiv.org/abs/2404.14469)、[TOVA](https://arxiv.org/abs/2401.06104)、[CacheBlend](https://arxiv.org/abs/2405.16444) 和 [Cache-Craft](https://arxiv.org/abs/2502.15734) 分别利用 attention importance 或 cache deviation 缩小保留或重算范围；连续窗口局部性则认为一段连续上下文可能包含主要有效信息，例如 [StreamingLLM](https://arxiv.org/abs/2309.17453) 使用最近窗口和少量起始位置。尽管任务不同，这些方法都试图利用 KV cache 中局部区域的重要性，以减少处理完整状态的开销。

与文本生成不同，Transformer recommender 的 KV cache token 对应用户历史中的交互事件，而新的候选 item 在评分时需要读取这些长期持久化的历史状态。因此，我们不能直接假设语言模型中的三类局部性同样适用于推荐状态迁移。为检验这一点，我们使用冻结的 Medium workload，包括 30,000 个用户、六层 HSTU、1024 context 和五条连续 Parent→Current release edges，并为 layer、token 和 window 三类方法分别设置有限的层组合、稀疏 token masks 和连续窗口候选。每个候选都从 Parent K/V 出发，仅将选中部分替换为 Current Exact K/V；图 X(a)–(e) 分别报告五条 release edge 上各 family 的 best-observed recovery，图 X(f) 给出 edge-equal aggregate。所有子图使用相同坐标：横轴为更新的 K/V 数量占 Exact-All 的比例，纵轴将 Reuse–Current Exact functional gap 的恢复归一化到 0–1；横轴只表示理论 KV 重算规模，不代表真实执行时间。

如图 X(a)–(e) 所示，三类方法在每条 release edge 上都能随着重算范围扩大而逐步缩小 Reuse–Current Exact 差距，但低重算比例下没有出现明显的恢复跃升。各 edge 的具体恢复幅度有所不同，layer family 没有表现出由少数关键层带来的突跃，token 和 window family 的恢复也主要随覆盖位置增加而渐进上升；这一共同趋势在【待填：重复边数】条 edge 上成立。图 X(f) 进一步显示，在相同的【待填：代表性预算 B】下，三类 family 的 edge-equal best-observed recovery 分别为【待填：L_B】、【待填：T_B】和【待填：W_B】；要达到 80% recovery，则分别需要更新约【待填：L80】、【待填：T80】和【待填：W80】的 K/V。局部更新能够修复部分跨版本误差，但较高恢复率普遍需要较大的理论重算比例。

**Insight 1：语言模型 KV cache 中常见的层局部性、稀疏位置局部性和连续窗口局部性，在持久化 Transformer 推荐状态上没有形成有利的重算量—恢复率权衡；即使采用冻结候选空间中的 best-observed 配置，要恢复大部分 Reuse–Current Exact 差距仍需重算相当比例的 K/V。**

这一结果促使下一项 Insight 转向 Current recommendation reader，考察分散的状态差异能否在读取过程中形成更紧凑的用户级结构。

## 图注草稿

**图 X：三类 KV 局部性假设在连续模型更新中的 best-observed 理论重算量—恢复率曲线。** 子图 (a)–(e) 依次对应 (v_0\!\to v_1)、(v_1\!\to v_2)、(v_2\!\to v_3)、(v_3\!\to v_4) 和 (v_4\!\to v_5)，子图 (f) 为五条 edge 的 edge-equal aggregate。所有子图使用相同坐标和颜色：横轴为需要更新的 K/V 数量占 Exact-All 全量 K/V 的比例，纵轴为归一化 functional gap recovery，其中 0 对应 Reuse，1 对应 Current Exact；layer、token 和 window 曲线在每个预算下报告各自冻结候选中 recovery 最高的配置。Exact-KV replacement 只用于构造理想化局部重算结果，真实依赖闭合计算、state/history I/O 和端到端运行时间在后续系统评测中单独报告。
