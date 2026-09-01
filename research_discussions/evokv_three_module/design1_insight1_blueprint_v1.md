# EvoKV Design I — Insight 1 写作与图表蓝图

日期：2026-08-31。

> **状态说明（2026-09-01）：本文件的四轴结构已被新版论证取代。** 后续 primary-paper 审计又表明，简单的 layer、important-token、recent-window 三分法仍然过粗：token support 必须区分 serving salience、migration error 与 candidate-conditioned utility，recent family 也必须包含 anchors。当前实验设计参考是 [`insight_one/related_work_motivation_taxonomy_v1.md`](./insight_one/related_work_motivation_taxonomy_v1.md)，它从 17 篇核心/边界工作推导出四个 premise probes；可直接回填实验结果的中文论文正文见 [`insight_one/insight_one_paper_draft_zh.md`](./insight_one/insight_one_paper_draft_zh.md)。英文稿与模拟图见 [`insight_one/insight_one_draft.md`](./insight_one/insight_one_draft.md)，目前只保留为写作和版式草稿。本文件仅保留为讨论演化记录，不应继续指导主图或正文。

本文设计 EvoKV 论文中 One-Release Migration 的第一个 Insight，包括它在 Design 章节中的角色、核心命题、主图结构、每个面板的数据需求、正文段落、开头与结尾过渡，以及现有证据和新增实验之间的边界。本文是研究讨论稿，不是新的实验合同、结果裁决或实验授权；尚未测量的图形只表示 prospective hypothesis，不得预写成论文结论。

## 一、Insight 1 应该说什么

“传统 KV 方法不 work”本身不够成为一个论文 Insight。它既容易把相关工作写成 strawman，也会被现有正结果反驳：Tail-128、parameter-only joint mapping、K-only/V-only exact splice 都能恢复一部分 Current–Reuse gap。真正值得写成 Insight 的结构命题应当是：

> **Insight 1: Cross-version mismatch has no stable low-budget locality in KV space.**

对应的中文含义是：模型发布造成的 recommendation-state mismatch 在存储它的 K/V 坐标中是分布式的；它不能稳定地归因到某一侧 K/V、少数层、某个时间区域或一小组可廉价识别的重要 token。沿这些局部轴进行的乐观修补要么留下很大的功能 residual，要么需要接近稠密重算的计算范围。

这句话比“现有方法失败”更强，也更准确。它把各种 naive adaptation 统一成一个待检验的假设：这些方法都假定 mismatch 在 KV tensor 的某个轴上具有 locality。EvoKV 要否定的不是某篇 LLM 论文，而是这种 locality 是否能够在持续更新的推荐 Transformer 中形成稳定、低预算的迁移边界。

它和后续 Insight 2 应形成一个明确对照：

> **Insight 1: distributed in KV space.**  
> **Insight 2: concentrated in reader-function space.**

如果这两个命题都得到证据，Design I 的推导会很自然：既然在 state tensor 中找不到稳定的局部修补位置，就不再迁移“哪些 KV”，而去寻找这些分布式差异经过 Current reader 后形成的紧凑功能像。

## 二、Design 章节怎样进入 Insight 1

Design 章节不应一开始就介绍 PRO。前面先用一个很短的总体入口，把已发布模型、Parent state 和两种极端方案接起来，然后提出“迁移单位是什么”这一设计问题。推荐使用下面的英文逻辑作为章节开头：

> Once a Current model has passed release admission, EvoKV faces a state-only decision: directly reuse a Parent-produced state, or reconstruct the entire history under the Current model. A natural middle ground is to repair the stored KV tensors in place. Existing KV techniques make such repair tractable by exploiting locality across tokens, layers, or K/V coordinates. We first ask whether release mismatch in a recommendation Transformer exhibits any of these localities.

这一段只承担三个任务：第一，重申此时模型已经确定发布，不再讨论 admission；第二，把 Reuse 和 Current Exact 作为两个端点；第三，把传统 KV 方法归纳成 token、layer 和 coordinate 三种 locality hypothesis。不要在这里展开 related work，也不要提前出现 sidecar、reader offset 或 PRO。

随后用一个统一、无标签的功能恢复指标连接四个面板。对 release edge `e` 和 intervention `m`，建议定义：

\[
\mathrm{Recovery}_e(m)
=1-\frac{D(F_t(C^m),F_t(C^{\mathrm{Exact}}))}
{D(F_t(C^{\mathrm{Reuse}}),F_t(C^{\mathrm{Exact}}))}.
\]

其中 `F_t` 是 Current 模型的 reader/output function，`D` 使用冻结的 mean absolute probability gap、normalized score RMS 或同等的 label-free functional distance。图中将 Reuse 固定为 0，Current Exact 固定为 1；负值表示干预比 Reuse 更差。每条 release edge 单独计算 recovery，再做 edge-equal aggregate，不能让请求数较大的 edge 主导结论。Insight 图不使用 AUC/log-loss label 来选择方法，正式推荐质量留在 Design qualification 和 Evaluation。

## 三、主图应该是一张四面板组合图

建议只使用一个统一编号的 `figure*`，内部包含 `(a)–(d)` 四个小面板，而不是四张彼此分散的图。四个面板都回答同一个问题、共享同一 recovery 纵轴和五条 release edge，因此一张组合图的逻辑最强，caption 也可以一次说明哪些路径是 optimistic exact-splice diagnostics、哪些是 executable baselines。

版式优先采用两行两列，而不是一行四列。一行四列会使五边散点、方法名和误差范围过小；两行两列大约占双栏页面三分之一高度，仍可在其下放入三到四段正文，使整个 Insight 控制在约一页以内。阅读顺序为左上 K/V、右上 layer、左下 token、右下 quality–cost summary。

整张图的暂定标题可以是：

> **Common KV-space repair axes fail to expose a robust low-cost migration boundary.**

图注必须明确：每个彩色点或细线表示一条 held-out release edge，黑色粗点表示 edge-equal mean；0 和 1 分别对应 Reuse 与 Current Exact；exact splice 只用于定位 mismatch，不能作为可执行迁移动作。

### Panel (a)：K/V coordinate locality

第一个面板检验 mismatch 是否主要集中在 K 或 V 一侧。横轴放置 `Current K + Parent V`、`Parent K + Current V` 和 `Current K + Current V`，纵轴为 functional gap recovery。主图只需要五条 edge 在两个单侧 exact splice 上的 10 个有效数据点，Current Exact 作为 recovery=1 的参考线，不需要再画大量请求级点。

这一面板已经有完整初步数据。Current-K-only 的五边 recovery 为 37.2%–50.9%，edge-equal mean 为 45.9%；Current-V-only 为 32.8%–55.7%，mean 为 47.9%。两侧都能恢复一部分，但没有一侧稳定主导，也没有一侧接近 Exact。正文结论只能写“mismatch is coupled across K and V”，不能写成 K/V 完全不可分，也不能把 exact splice 称为一种部署方法。

为了让面板视觉更直接，可以将每条 edge 用同色线连接 K-only 与 V-only，显示 winner 会随 edge 改变；黑色菱形表示五边均值。这个面板不需要新的训练，现有 sealed diagnostic 已足够生成初稿。

### Panel (b)：layer locality

第二个面板检验少数层是否构成稳定的修补边界。现有 cumulative lower-layer exact splice 已显示：只替换 lower-1 的平均 recovery 为 19.2%，范围为 −9.3%–44.3%；lower-2 平均为 57.1%，范围为 44.2%–80.7%；必须替换 lower-3/4 layers 才达到 83.4%–96.0%，平均 90.3%。这已经证明 layer 0 不是充分热点，并表明 mismatch 会经过 early/middle dependency 传播。

但现有数据还不能写“没有重要层”或“所有层同等重要”，因为尚未穷举单层和等数量 layer subset。正式 Panel (b) 应增加一个很小的 prospective exact-splice oracle：对 Small 的四层枚举所有 layer subsets，对每个 subset size `k=0..4` 报告该 edge 上恢复最好的 subset。图中横轴是被替换为 Current Exact 的层数，纵轴是 oracle-best recovery，五条 edge 各一条线。Small 每条 edge 只有 16 个 subset，计算规模很小；Medium 的六层版本可在机制冻结后做相同验证。

这个 oracle envelope 是对 layer selection 极其有利的诊断上界。如果一层或两层已经在所有 edge 上获得高恢复，那么“无 layer locality”的命题被否定，layer repair 应成为候选设计；如果即使 oracle 也需要三层或更多，才能有力地写“the mismatch is not cheaply layer-localized”。这一 falsification rule 应在实验前固定。

### Panel (c)：token locality and importance selection

第三个面板检验 mismatch 是否集中在最近 token 或一小组稳定的重要 token。当前 Tail-128 只说明 dependency-closed recent replay 在五条 edge 都有正恢复，但恢复仅为 19.4%–25.8%，平均 23.1%；它还不能证明 recent region 比 old、middle 或 random region 更敏感。现有 History Utility probe 的 utility–harm Spearman 只有 −0.007 到 0.152，均值 0.068，说明基于用户或粗历史 utility 做选择接近随机，但这也不能替代 token-level selector 实验。

正式 Panel (c) 建议在同一批 label-free request/candidate probes 上使用三个固定 token budget，例如 32、64、128 positions，对比五种 selector：recent、uniform/random、Parent-reader heavy hitter、跨固定 candidate panel 聚合的 candidate-shared relevance，以及只用于诊断上界的 Current-delta oracle。每种 selector 选出的 positions 都用 Current Exact K/V splice 进行乐观干预，从而先回答“这些位置是否承载 mismatch”，而不是把任意位置重算的 dependency cost混入结构判断。recent selector 另保留 dependency-closed replay companion，供后续成本图使用。

主图预计包含 `5 selectors × 3 budgets × 5 edges = 75` 个 edge-level aggregate，不画请求级散点。用 selector 的 edge-equal mean curve 和跨边 min–max ribbon 即可。Current-delta oracle 必须使用虚线并明确标注 `diagnostic only`，不得拟合 mapper、成为 action 或参与 release-time decision。

这一面板有两种都很有价值的可能结果。如果连 Current-delta oracle 在 10%–20% position budget 下也很弱，说明 mismatch 本身在 token 轴上是稠密的；如果 oracle 很强但所有 label-free selector 都弱，说明稀疏 support 可能存在，却无法在不支付 Current Exact 或 per-candidate 成本时可靠识别。只有当某个可执行、candidate-amortized selector 在所有 held-out edge 上稳定接近 Exact，Insight 1 的 token-locality 命题才应被推翻。不要在数据产生前预写“没有重要 token”。

旧、middle、recent、random 的等宽 128-position exact-splice 对照建议同时执行，但可以放到 appendix 或 Panel (c) 的小 inset；它主要负责避免把 Tail 的可执行因果边界误写成“Tail 天生最重要”。

### Panel (d)：quality–cost frontier of KV-space repairs

第四个面板把前三个结构诊断收束为系统结论。横轴使用同一套 conservative causal FLOPs，相对 Exact-All 归一化；纵轴仍为五边 functional gap recovery。每个方法画一个 edge-equal mean 点，并用竖线给出五边范围。背景中可以淡色标出论文真正希望进入的区域，例如 `<20% Exact compute` 且恢复大部分 functional gap；具体阈值在 prospective contract 前保持为视觉参考，不写成已经冻结的 gate。

现有数据已经能提供若干初步点：mass-aware 64-carrier repair 的 causal attention work 为 Exact 的 20.3%，恢复 9.5%–25.0%，均值约 19.4%；parameter-only joint translation 的旧物化实现成本为 Exact 的 32.2%，恢复 21.6%–64.1%，均值 43.0%；dense Tail-128 的 causal attention work 为 43.7%，恢复 19.4%–25.8%，均值 23.1%；translation 与 Tail-128 组合恢复 38.7%–72.6%，均值 57.0%。Exact 位于 `(100%,100%)`，Reuse 位于 `(0%,0%)`。

translation+Tail 的完整公平成本以及新增 layer/token selectors 的 dependency-closed cost 必须用统一 cost model 重新封存后才能进入正式图，不能把若干百分比简单相加后当作论文数字。Panel (d) 的作用不是证明每个方法数值为负，而是显示在 KV-space 中获得更高恢复需要同时扩大 layer、token 或全状态变换范围，当前不存在一个稳定落入低成本高恢复区域的局部动作。

## 四、四段正文的建议写法

第一段负责提出并统一 locality hypotheses。它可以写成：传统 KV compression/reuse 工作分别利用重要 token、layer-wise allocation、selective recomputation 或 K/V mapping；这些机制在原任务中有效，但它们共同假定误差在存储状态的某个轴上可被局部化。为了避免任务和指标混淆，EvoKV 不直接把 LLM generation accuracy 拿来比较，而是在同一 Current reader、同一用户历史和同一真实 candidate 下，对四类乐观 intervention 使用统一的 normalized functional recovery。

第二段解释 Panel (a) 和 Panel (b)。重点不是罗列所有数字，而是先给结论再给最少证据：K-only 与 V-only 都只恢复约一半且 winner 跨 edge 改变；layer-0-only 很弱甚至可能恶化，恢复大部分 gap 需要三层中的大部分网络。最后一句写成“the mismatch is coupled in tensor type and propagated across depth”，而不是“all layers are equally stale”。

第三段解释 Panel (c) 和 Panel (d)。为了让内部图稿先具有完整的视觉形态，当前用一组明确标为 simulated 的数值替代空白：在 6.25%/12.5%/25% token budget 下，Random 暂画为 4/7/12%，Recent 为 9/16/23%，Heavy hitter 为 11/20/31%，candidate-shared relevance 为 15/27/39%，diagnostic oracle 为 28/46/64%。这些数值只定义曲线、标注密度和预期的辨析能力，既不是预测，也不能进入论文结论。Panel (d) 继续使用已有的 preliminary measured point：例如 Joint K/V map 需要 32.2% Exact-equivalent compute，平均恢复 43.0%。这一段应承认 joint mapping 的正结果，因为它是通向下一 Insight 的线索：global version translation 能修复一部分 mismatch，但它与 contextual replay 仍有稳定互补 residual，因此仅做通用 K/V mapping 不完整。

第四段用一句结构结论和一句问题转换结束。建议的英文原文是：

> These results do not show that release mismatch is unstructured. Rather, they show that its structure is not stably localized in the coordinates in which the state is stored. The consistent gain of a joint version map—and its complementarity with Current replay—suggests a shared functional structure beyond local KV repair. We therefore stop asking *where the stale KV entries are* and ask *how the Current reader misinterprets the stale user evidence*.

这一结尾应直接接到 Insight 2 的第一句：

> If a high-dimensional state error has no stable local support, it may still have a low-dimensional functional image. We next trace the Exact–Reuse correction through the HSTU reader and across the candidate bank.

这样，Insight 2 不是突然出现的另一个实验，而是 Insight 1 唯一自然的下一步。

## 五、可以直接写入的核心句与图注草稿

Insight 1 最核心的一句话建议保留为：

> **KV-space locality is the wrong migration abstraction: optimistic repairs along K/V, layer, and token axes either leave a large functional residual or require near-dense work.**

主图图注可以暂定为：

> **Figure X: Cross-version recommendation-state mismatch has no stable low-budget locality in KV space.** We measure label-free functional gap recovery on five Parent→Current release edges, with Reuse normalized to 0 and Current Exact to 1. Each colored mark denotes one edge and black marks show edge-equal means. (a) Exact K-only or V-only splices each recover only part of the gap, with no stable winner. (b) An optimistic layer-subset oracle requires a large fraction of the network to approach Exact. (c) Matched-budget position and importance selectors test whether a compact, candidate-amortized token subset can carry the mismatch. (d) Executable KV-space adaptations remain outside the desired low-cost/high-recovery region. Exact splices and Current-delta selectors are diagnostic interventions, not migration actions.

在 Panel (b)、Panel (c) 的新结果完成前，caption 中必须明确写出 `simulated`，相应句子保持为实验问题，不能把当前示意曲线提前写成肯定结果。当前内部图为 Panel (b) 暂设的 layer-subset oracle envelope 是 0/28/61/90/100%，为 Panel (c) 暂设的五组 selector 曲线如上；正式 probe 完成后应整体替换，而不是只替换不利于论点的点。

## 六、现有证据、待补实验和统一人口

Panel (a) 的 K/V exact splice、Panel (b) 的 cumulative lower-layer preliminary、Panel (d) 的 Tail、translation、mass-aware carrier 和组合 recovery 已经存在，可以立即生成内部草图。Panel (b) 仍缺 exhaustive layer-subset oracle；Panel (c) 缺等宽位置对照、matched-budget token selectors 和 diagnostic oracle；Panel (d) 缺新增方法的统一 dependency-closed FLOP accounting，以及 translation+Tail 的正式完整成本点。

为了避免四个面板由不同人群和不同 candidate semantics 拼接，正式 Insight 图最好在一个新的 prospective、UID-disjoint、label-free Small probe 上统一重放。建议每条 edge 固定约 256 个 eligible 用户的首个 append-free request，并使用相同的真实 exposed candidate set或同一冻结 candidate panel；五条 edge 合计约 1,280 个 user-edge。所有 intervention 共用 Parent Reuse、Current Exact、request、candidate 和 output metric。现有 sealed 数据只用于确定实验问题和画内部概念图，不再用旧 AUC/log-loss结果选择 selector、阈值或图中赢家。

Small 用于完成结构发现和反例；一旦四个 panel 的 hypothesis、selector 和预算冻结，Medium 只需重复关键 panel 或在 appendix 报告同方向，而不重新挑方法。Large 不应为了补 Insight 1 的图而启动新的长训练。

## 七、与相关工作的边界

Insight 1 可以引用几类 KV 工作来说明 locality hypotheses 的来源，但不能声称它们本来就在解决跨模型推荐用户状态迁移。H2O 和 SnapKV以少量重要 token 为基础压缩同模型 KV；PyramidKV 利用 layer-wise information funneling 分配不同层的 cache budget；CacheBlend 在 RAG 场景选择性重算部分 token 来融合预计算 cache；cross-model KV mapping 工作则显示模型家族间可能存在可学习的线性映射。EvoKV 对这些思想做的是 recommendation-release adaptation，并测试它们的核心 locality assumption 是否仍成立。

其中，任何依赖 matched Current target K/V 训练 mapper 的方法都不满足当前 EvoKV 的 no-target-KV-fitting contract，只能作为 related-work foil，不能成为 release-time action。正文中应把“different task and contract”说清楚，避免把同模型 cache compression、RAG context fusion 和跨版本 persistent-state migration混为一谈。

## 八、最终建议

这一段最合适的论文结构不是“展示四个 naive baseline 都很差”，而是“依次检验四种 KV locality，并发现没有一种形成稳定的低预算接口”。一张两行两列的组合图、三到四段正文和一个从 `where in KV` 转向 `effect at reader` 的结尾，足以构成完整的 Insight 1。它占用约一页，但只引入一个概念：**distributed KV-space mismatch**。

只要新增 layer/token probes 支持这一结构，Insight 1 就能非常自然地交给 Insight 2；如果某个 optimistic layer/token oracle 反而在低预算下稳定接近 Exact，也应当接受 falsification，重新让该局部性成为 Design I 的候选机制。这样的写法既避免预设结论，也使每一张小图都承担明确的推理作用。
