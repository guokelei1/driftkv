# EvoKV 三段式论文框架收敛稿

日期：2026-08-31。

本文是在综合 `expert1.md`、`expert2.md`、`expert3.md` 三份专家讨论稿、当前论文设计与已封存实验观察的基础上，对 EvoKV 三项核心设计所做的一次二次收敛。它同时吸收专家意见和新的独立判断，目的是确定论文应该讲成什么样、三个模块各自真正需要解决什么问题，以及怎样避免把论文扩张成三个彼此分离的研究方向。本文是研究讨论稿，不是新的实验合同、结果裁决、训练授权或已经验证的论文结论；其中涉及的质量门槛、连续误差形式和 Runtime 完成标准均是建议性的研究判断。

## 总体判断

目前设想的三段结构是成立的，而且具备一篇完整论文所需要的递进关系。最合适的组织方式不是把 One-Release、Continuous 和 GPU Runtime 写成三个同等独立的小课题，而是把它们写成同一个核心发现依次跨过三道障碍：One-Release 找到低成本迁移的正确语义对象，Continuous 让这种对象能够跨版本反复演化而不失控，GPU Runtime 则把已经确定的迁移算子真正变成高效的物理执行。三部分在正文中都可以占一面到一面半左右，但它们的学术角色不必完全等重。第一部分是语义和机制核心，第二部分补上长期闭包与可靠性，第三部分负责兑现系统收益。这样的结构是一项设计从“单次可行”走到“长期可信”，再走到“真实可运行”，而不是三篇小论文的拼接。

一个完整的系统设计点并不要求复杂理论或大量公式。更重要的结构是：先明确一个真实障碍，再用可复现、可被反例推翻的实验观察定位其原因，然后让机制尽可能直接地从观察中推出，最后用清楚的接口和结果说明问题被解决到什么程度。EvoKV 的科研简洁性应当来自三个模块始终围绕同一种状态语义，而不是通过增加大量 operator、controller rule 或系统组件来制造贡献数量。整篇论文可以共享一个核心不变量：系统维护的不是在 tensor distance 上接近 Current KV 的副本，而是旧用户证据在 Current reader 下的功能作用，以及这份功能作用随版本演化时可接受的误差边界。

需要特别修正一个容易让 Continuous 显得过于简单的表述：One-Release 并不是每次都把旧 cache 完整加工成一份新的 Current cache。当前 lightweight PRO 不物化完整 Current KV，它产生的是由较老版本主体 KV、面向 Current reader 的紧凑 sidecar，以及发布后由 Current 模型生成的新 append 共同组成的复合状态。到了下一次发布，输入已经不是同质的 Current Exact cache，而是一条带有 producer、sidecar 和 append 谱系的近似状态。假如每次都写回完整 Current KV，连续迁移确实容易重复执行，但第一部分的计算与写回优势也很可能随之消失。因此，怎样让这种紧凑复合状态对下一次迁移保持闭包，正是第二项设计真正成立的原因。

## One-Release：从分布式 KV 失配到紧凑 reader correction

第一组 Insight 实验可以从传统 KV repair 的粒度是否正确开始，但结论不应被预设成“层之间没有差异”或“完全不存在重要 token”。当前证据显示，tail replay、parameter-only mapping 和部分层重算都能获得一定恢复；它们的问题不是完全无效，而是在低预算下只能捕获分布式失配的一部分，且不同成分往往互补，没有形成一个稳定、完整的稀疏修补边界。更准确的负面观察是：Parent→Current mismatch 在 K/V 表示空间中跨 token、K/V 和多层传播，逐 token、逐层、单独 K/V、tail 或通用 mapping 都难以在很低成本下稳定接近 Current Exact。这样既保留强基线的真实能力，也能说明为什么继续在高维 KV 空间中寻找局部修补位置不是最合适的主线。

真正能够推出主设计的正面 Insight，可以概括为：跨版本差异在 state space 中是高维且分散的，但在 recommendation reader 的功能空间中会显著集中。现有实验已经观察到，Exact−Reuse 的读取差异主要由 candidate-shared 方向主导，最早稳定出现在 HSTU 的 `activated(qK)·V` 聚合边界，并能在一段真实 rolling 请求中保持方向；与此同时，HSTU 的非归一化聚合要求 compact state 显式保留 evidence mass。这意味着推荐系统的特殊结构并不是泛泛的“不同用户需要不同修正”，而是同一份长期用户证据会被大量 candidate 和多个请求重复消费，版本失配因而可以被分解成一个 release-shared 的 reader 变换和一个紧凑的 user-specific functional residual。

可以用一个抽象关系表达这一点：

\[
R_t(q,C_u^{\mathrm{Exact}})
\approx
R_t(q,C_u^{\mathrm{Parent}})+G_t(q)z_{u,t},
\]

其中，`R_t` 表示 Current reader 对用户状态的功能读取，`G_t` 表示 Current 请求侧对紧凑适配状态的消费方式，`z_{u,t}` 是按用户生成但可跨 candidate、并在 bounded rolling horizon 内复用的版本适配状态。lightweight PRO 的计算来源也应被准确表述：它之所以便宜，不是因为 Exact 不能在 candidate 之间共享，而是因为它不让完整长历史重新经过 Current 模型的所有层，也不物化和写回完整 Current KV；它把版本共享的 reader-pushed map 与少量 Current contextual carriers 结合，只生成一份小型 per-user reader offset。

因此，Design I 的正文可以非常紧凑。先用少量等预算图说明 token/layer/tail/KV mapping 有用但不完整；再用 reader-stage、candidate-shared、真实 exposed candidate 和跨请求 persistence 实验说明功能差异在 reader 边界集中；随后给出一到两个公式解释 PRO 怎样估计和消费 `z_{u,t}`；最后报告质量、理论计算、状态读取和写回成本。`CAST / PATCH / GROUP / SCALE` 可以继续作为强基线和机制证据，但不宜重新成为四个并列的论文 headline。

当前约 9.1% Exact FLOPs、约 2 KiB sidecar、AUC 5/5 正向已经足以证明路线成立，但五边平均 Reuse harm recovery 约为 34.7%，log-loss 只有 3/5 正向，说明它仍是一个有力原型，而不是最终闭合的第一项贡献。建议性的强度判断是：约 10% 成本若能稳定恢复一半以上的 Reuse harm，或者不超过 20% 成本能够在新的 release edge 和规模上恢复约 70% 甚至更多，同时不存在不可预测的灾难边，就会形成很有说服力的主设计。下一步最值得解释的不是继续增加相似 carrier，而是剩余误差究竟主要属于 score bias、temperature 等校准分量，还是仍然存在未被当前 sidecar 表达的 request/candidate-conditioned 功能方向。这个 residual decomposition 将决定现有 PRO 只缺一个小校准项，还是需要增加一种真正不同的低维方向。

One-Release 向下一模块交付的也不应只有一个效果不错的 sidecar。它至少需要交付一种明确的 canonical migrated-state 类型、一跳 transition 的输入输出语义、sidecar 在真实 append 和 old-state coverage 变化下的 transport 或 expiry 规则，以及一个能够表示剩余功能误差的量。这里不必立即建立复杂 predictor，但必须让 Continuous 知道上一轮产生的究竟是什么状态、哪些部分可以继续组合、哪些误差尚未消除。

## Continuous：canonical composition 加 bounded control

Continuous 采用“算法层变化加边界控制”的结构是正确的，但算法层变化不能只是调整 carrier 数、repair width 或重复调用 One-Release。它真正应当解决的是状态闭包：每轮迁移之后都重新输出同一种 Current-compatible canonical form，只保留一个面向当前版本的适配对象，而不是不断叠加 `v0→v1`、`v1→v2`、`v2→v3` 的 correction chain。最值得验证的连续 Insight 是，release-shared 的版本变换能否相对最近的 Exact anchor 直接组合或重新表达，而多跳真正积累的主要是规模较小的 user-context residual。如果这一结构成立，第二项设计就可以复用第一项设计的核心分解，但形成一个新的递推算法：共享版本部分被组合或折叠，已有 sidecar 被运输到新的 reader 空间，只有无法安全运输的 residual 才需要新的 Current evidence、加强 repair 或 Exact Rebase。

连续迁移的风险不是每一版机械地多出同样大小的误差，而是下一版迁移器开始消费上一版产生的近似状态，旧误差可能被新的 reader 放大，并在未来 query distribution 下才暴露。一个足够简单的分析关系可以写成：

\[
d_{t+1}\leq \alpha_t d_t+\epsilon_t,
\]

其中 `d_t` 是当前 lineage 相对最近 Exact anchor 的功能债务，`ε_t` 是本次近似迁移新引入的 residual，`α_t` 则表示已有误差被下一版本 reader 放大或削弱的程度。论文不应在没有证据时强行宣称所有版本上 `α_t<1`。当前 controlled dilution 中已经存在明确反例，因此“新行为自然把旧状态误差几何稀释”不能成为算法保证的物理前提。更可信的目标是让 `d_t` 可观测、可保守估计并始终被控制在一个预定义 envelope 内，而不是证明无限版本下必然渐近收敛。

控制层只需围绕这个债务建立一个简洁闭环。sampled Current Exact 是即时传感器，用无标签 reader/score fidelity 校准债务并发现分布外 release；真实行为标签是较慢的配置级安全审计，不能进入同一请求或单个用户的 future-label 调度；Rebase 是债务越界后的 reset actuator，而不是默认每隔固定版本执行的主体算法。再增加一个最大近似深度 `H`，系统就可以保持一个清楚的三段行为：债务安全时执行便宜的 canonical composition，风险升高时使用更强 repair，证书失效或达到硬上限时 Exact Rebase。控制器不需要扩张成大量规则，只要能保证近似不在 lineage 中悄然无限积累。

Continuous 真正解决的实验标准不是把若干 direct one-hop 结果串在一起，也不是只画十个版本后 AUC 看起来没有下降。实验必须执行真实 recursive lineage，使上一轮的近似输出成为下一轮的实际输入，并在每一版保留 Current Exact reference。它需要比较无反馈连续迁移、固定周期 Exact Rebase、简单 age/depth 控制，以及 canonical composition 加 debt/shadow feedback。在相同质量 envelope 下，如果后者能以较低 shadow rate 和较少 Rebase 保持最坏或高分位 Exact gap 有界，并在总成本上显著优于最佳固定周期策略，这一模块才算真正成立。如果必须频繁 Exact 或使用很高 shadow rate 才能避免越界，那么 Continuous 仍然只是隐式的周期重算。

## GPU Runtime：固定版本算子上的流式 ragged 执行

第三项设计应当主动收住边界，不把论文重新扩张成一个独立的 state-store 或 I/O 系统问题。I/O 可以很重，也必须被实测，但它在这里应当是 roofline、流水线约束和需要被隐藏的成本，而不是新的研究对象。Runtime 的核心问题应是：在前两项已经固定了迁移公式和 per-user plan 后，为什么通用 GPU 执行不能兑现低 FLOPs，以及如何针对这种迁移 workload 组织真正高吞吐的执行。

这一 workload 最有特色的物理结构是：一次 Parent→Current release 会让海量用户执行完全相同的版本变换，算子和版本参数高度同质，而用户状态长度、carrier 数和 lineage plan 高度 ragged。通用 HSTU/PyTorch 执行容易把它拆成大量很小的 kernel，并产生 padding、kernel launch、低 occupancy 和中间状态物化。一个聚焦的 Runtime 设计可以把整条 release edge 编译成常驻 GPU 的 migration program，按 transition signature 与长度分桶或做 segmented execution，把 reader transform、`qK`、activation、weighted-`V` reduction、mass handling 和 sidecar layout 融合起来，不生成完整 translated prefix 或 Current KV。跨用户共享的是同一条 Parent→Current transition，而不是把 candidate sharing误写成 Exact 不具备的计算优势。

prefetch、double buffering 和异步 writeback 的角色是让 state/history 搬运与 GPU 计算重叠，避免 GPU 因输入不连续而空转；它们是执行机制的支撑，不必扩张成新的存储体系。版本一致的 commit 也需要作为 persistent-state runtime 的正确性接口，防止后台迁移用旧 snapshot 覆盖 cutover 后的新 append，但它可以保持为一个简洁的执行约束，而不是另一项主贡献。整个 Runtime 的一句话原则可以是：固定一条 release transition，让 ragged 用户状态只流过融合程序一次，只生成最终 sidecar，不做隐藏的全量 Current-state 物化。

Runtime 是否足以成为第三个设计点，需要相对真正有竞争力的基线来判断。它不能只击败逐用户 Python，而应比较已经做好 batching、长度 bucketing、异步 prefetch 的通用 PyTorch 或编译实现。成功结果应同时表明：专用 migration kernel 相对优化后的 unfused PRO 有显著吞吐提升；算法的低理论计算能够转化为真实 GPU-hours 和全人口 makespan 的数倍改善；执行中没有隐藏的完整 KV 写回或中间物化；在必要的 serving coexistence 测试中干扰处于预定义范围。若 profiling 最终显示 source-state scan 成为主瓶颈，论文可以把它作为 roofline 事实并通过重叠来接近下界，而不必再扩大范围去解决一个完整的分布式 I/O 系统。

## 对三份专家意见的取舍

三份专家稿共同收敛出的正确主线是：One-Release 迁移的是 Current reader 对旧用户证据的功能作用，而不是 K/V tensor 本身；Continuous 必须把递归近似显式记账，不能依赖自然 append 或固定周期重算掩盖误差；Runtime 必须证明逻辑低成本没有被 ragged execution、中间物化和隐藏全量工作抵消。这些共同判断应当保留。

第一份专家稿提出的 reader contract、anchor/debt/audit/rebase 和 read-once execution 很有启发性，但如果把 transaction、service curve、状态 I/O 和 serving interference 全部扩成独立问题，第三部分会超过当前论文所需范围。第二份专家稿使用“表征—动力学—物理”概括三层问题，这个抽象值得吸收；但它进一步提出的稳定兴趣簇、item embedding 漂移主导和行为反馈带来几何收缩，目前没有充分证据，后两点分别受到 contextual Transformer co-adaptation 结果和 controlled dilution 反例的直接限制，不应进入论文前提。第三份专家稿最接近当前应采用的克制结构：用 functional adaptation state 统一 One-Release，用 canonical form 和 debt 统一 Continuous，再把 Runtime 限定为没有隐藏全量工作的 GPU execution。

在这些意见之上，可以进一步形成一个更紧的共同结构。One-Release 发现并利用“release-shared transform 加 user-specific residual”的分解；Continuous 尝试沿版本链组合 release-shared 部分，并只为无法安全传递的 residual 记债；Runtime 则把 release-shared transition 固定在 GPU 上，让大量 ragged user state 流过同一个融合程序。也就是说，三个模块不是各自引入一个新概念，而是在语义、时间和物理三个层面反复利用同一项结构发现。这种概念复用正是整篇论文最可能形成简洁感和整体感的地方。

## 最终收敛

整篇论文可以最终概括为：EvoKV 首先发现跨版本用户状态差异虽然分散在 KV 空间中，却在 HSTU 的 candidate-shared reader 边界集中为紧凑的用户适配状态；随后把这种状态规范化为可跨 release 组合、债务有界的长期表示；最后把同一版本转换编译成面向海量 ragged 用户的融合 GPU migration program。对应的三句话原则是：One-Release 在 reader 功能空间迁移用户证据，而不是重建完整 KV；Continuous 每一跳都回到同一种 canonical state，并让未消除的误差显式有界；Runtime 固定版本变换，只让必要的用户状态流动一次，并直接产出最终 sidecar。

这套三段结构已经是一份完整而且相对优美的科研框架。后续判断每一块是否足够成为核心设计，可以使用两个最重要的强度检查：如果 Continuous 最后只有一个 Rebase 状态机而没有 canonical composition，它还不够成为独立算法贡献；如果 Runtime 最后只有通用 bucketing 和 prefetch，而没有利用固定版本边、ragged state 和无完整 KV 物化形成专用数据流，它更适合作为实现优化。反过来，只要这两个缺口得到实验证实，三项设计就能够以相近篇幅组成一条统一而有递进关系的论文主线。
