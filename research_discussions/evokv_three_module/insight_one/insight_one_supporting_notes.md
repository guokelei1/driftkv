# EvoKV Design I / Insight 1：文献依据与实验口径

日期：2026-09-01。

本文是 [`insight_one_paper_draft_zh.md`](./insight_one_paper_draft_zh.md) 的当前支持材料，只保留正文所需的文献依据、测量口径和结论边界。它不是实验结果，也不授权新的训练或长实验。所有 Current Exact K/V splice 都只能用于 Small/Medium 上的结构诊断，不能进入可部署 action、release decision 或 scale frontier。

## 当前研究问题

Insight 1 检验的是：LLM KV 复用中常见的局部重算思路，在持久化 Transformer 推荐状态上能否取得有利的重算量—恢复率权衡。当前正文只考察三类最直接的局部结构：选择关键层、选择离散的重要 token，以及选择连续的历史窗口。我们不试图证明所有可能的局部性都不存在，也不把有限候选搜索包装成数学意义上的最优上界。

论文希望观察的是完整成本—恢复曲线，而不是预先规定某个重算比例必须达到某个恢复率。三类方法随着重算范围扩大获得正向恢复是合理现象；真正有区分度的问题是，高 recovery 是否在很小的重算范围内快速出现，还是只有覆盖较大比例状态后才能逐步接近 Current Exact。

## 文献依据

关键层路线最接近的代表工作是 [DroidSpeak](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_liu-yuhan_prepub.pdf)。它面向同架构 fine-tuned LLM 的跨模型 KV sharing，通过 profiling 识别少量 sensitive layers，并只重算 critical layer group。它为 EvoKV 提供的直接问题是：一条 Parent→Current release edge 中，是否也存在少量层能够承担大部分恢复。PyramidKV、MiniCache 和 Layer-Condensed KV Cache 分别讨论逐层预算、相邻层冗余和架构重训，它们可以解释相邻设计空间，但不能被一个关键层实验一并“反驳”。

离散 token 路线包含几种不同但相关的选择依据。[H2O](https://arxiv.org/abs/2306.14048)、[Scissorhands](https://arxiv.org/abs/2305.17118)、[SnapKV](https://arxiv.org/abs/2404.14469) 和 [TOVA](https://arxiv.org/abs/2401.06104) 分别利用累计 attention、重要性持续性、观察窗口或当前 query attention 选择对后续计算更重要的位置；[CacheBlend](https://arxiv.org/abs/2405.16444) 和 [Cache-Craft](https://arxiv.org/abs/2502.15734) 则在非 prefix cache reuse 中选择更需要重新 contextualize 的 token 或 token-layer entries。这些工作的选择信号不同，当前主图不声称完整复现其中任何一个系统，只把它们共同启发的“少量离散位置可能承担大部分恢复”作为待测结构。

连续窗口路线的代表是 [StreamingLLM](https://arxiv.org/abs/2309.17453)。它真正使用的是 recent window 与少量 attention sinks，而不是证明 recent-only 永远足够。因此，EvoKV 的窗口候选不能只挑一个有利的 tail 长度，而应在事先冻结的窗口位置和长度集合中进行公平比较。窗口获得部分正向恢复不等于形成了低成本迁移边界；只有短窗口能够恢复大部分差距时，这条路线才表现出强时间局部性。

[Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)、[Mixture-of-Translators](https://arxiv.org/abs/2607.28979) 和 Activated-LoRA KV reuse 属于不同边界：它们整体翻译表示或通过模型协同设计制造兼容性，并不依赖少量层、token 或窗口。因此，当前 Insight 1 的局部重算结果既不检验也不否定这类 global translation；涉及 matched target K/V 或 native target trajectory 的拟合也不属于 EvoKV 当前允许的 action。

## 单图实验设计

主图采用一组共享坐标的 small multiples，而不是把所有 release edges 叠在同一坐标中。当前测量对象为冻结的 Medium workload：30,000 个用户、六层 HSTU、hidden 192、六个 heads、context 1024，以及五条 D14 Parent→Current release edges。子图 (a)–(e) 分别对应五条相邻 edge，子图 (f) 报告 edge-equal aggregate；每个子图都只包含 layer、token 和 window 三条曲线。三类方法共用相同的用户历史、Current reader 和冻结 candidate semantics。对于一个候选局部范围，从 Parent K/V 出发，只将被选中的部分替换为对应的 Current Exact K/V，其余状态保持不变。

每一类方法都必须在读取结果前冻结一组规模可控的候选配置。层候选可以包含单层、连续层组和有限的非连续层组合；token 候选由若干事先规定的选择规则生成 masks；窗口候选来自冻结的位置与长度网格。在每一个理论重算预算下，图中报告各类候选中 recovery 最高的配置，称为 best-observed result。正式记录必须给出每类候选的数量、生成规则与预算网格；在这些内容冻结以前，不预写 8、16 或 32 等候选数量。

Best-observed 只减少结论对单个弱 selector 的依赖。除非确实遍历了一个定义明确的有限空间，否则不得使用 exhaustive、oracle、upper bound 或“全局最优”等表述。尤其是离散 token 组合通常无法穷举，Current-aware 的强选择过程也只能称为诊断性候选生成器。尚未测试的 selector 仍可能移动曲线，因此正文结论必须限定到现有 LLM-inspired 局部重算范式及本实验覆盖的候选空间。

## 测量与报告口径

对 release edge $e$，记 Parent persistent state 为 $C^P$，Current Exact state 为 $C^C$，选择范围 $S$ 后的 hybrid state 为 $C^S$，固定 Current reader 的输出为 $F_C(C;Q)$。统一的 label-free functional recovery 为

\[
R_e(S)=1-\frac{D\!\left(F_C(C^S;Q),F_C(C^C;Q)\right)}
{D\!\left(F_C(C^P;Q),F_C(C^C;Q)\right)}.
\]

Reuse 对应 0，Current Exact 对应 1。每条 release edge 必须单独报告，再做 edge-equal aggregate，不能由请求更多的 edge 主导结果，也不能只挑有利 edge。Future labels 不得用于选择候选、预算或 winner；AUC 和 log-loss 留给后续 executable design evaluation。

图的横轴是理论上需要重新生成的 K/V 数据比例，纵轴是上述 recovery。横轴不是 GPU 执行时间，也不是依赖闭合后的真实 FLOPs；它没有计入发现候选位置、hidden-state propagation、不规则执行、state I/O 和写回成本。该口径有意对局部重算更有利，真实计算与运行时间由后续 Runtime/Evaluation 单独测量。

正文结果应同时提供两种可读比较。第一种是在同一个代表性理论重算比例 $B$ 下，报告 layer、token 和 window 的 best-observed recovery；第二种以统一的 80% recovery 为参照，报告三类方法分别需要重新生成多少 K/V。随后说明曲线是早期快速跃升还是随覆盖范围渐进上升，并报告该趋势在多少条 frozen release edges 上重复。实际数字未测量和封存前只能保留占位符，不能用模拟值冒充结果。

## 当前结论与边界

若测量结果符合当前预期，正文 Insight 使用以下一句：

> **现有 LLM KV 复用中常见的局部重算策略——选择关键层、重要 token 或连续窗口——在持久化 Transformer 推荐状态上未能形成有利的重算量—恢复率权衡；要恢复大部分 Reuse–Current Exact 差距，仍需重算较大比例的 K/V。**

这句话不等于“推荐 K/V 没有重要位置”，也不等于对所有未来 selector 的不可能性证明。如果任一 best-observed 曲线在很小重算范围内已经恢复大部分差距，当前结论必须修改，并继续检验该结构能否在不读取 Current Exact history 的情况下稳定识别，以及能否 dependency-closed 地执行。只有当局部候选的成本—恢复曲线确实呈渐进关系时，正文才可以使用上述 Insight。

已有 candidate-broadcast evidence 不进入这张主图，也不用于新增 per-candidate token Route。它只为下一项 Insight 提供线索：如果 entry-level 局部重算没有形成有利权衡，应进一步考察跨版本差异经过 Current HSTU reader 聚合后是否形成紧凑的用户级 correction。Reader-stage concentration、AV boundary 和 sidecar 公式属于下一节，不在 Insight 1 中提前展开。

## 当前图形状态

Insight 1 当前没有可作为论文证据的成图。旧机制四面板模拟图及其归档版本已经删除。正式图采用 2×3 布局：五个 edge-specific panels 加一个 edge-equal aggregate panel，并在所有 panel 中固定相同横纵轴、预算点、颜色和图例。Layer、token 和 window 分别使用三种固定颜色；图中不预画人为规定的“高恢复区域”，也不填入未经测量的模拟结果。
