你的修正是对的。上一版把**系统论文的证明负担抬得太高了**，不必要地引入了“HSTU 是否优于 SASRec”“是否稳定利用严格顺序”“每个自然日更新是否都带来正 gap”等算法论文问题。

最新版的破局方向应当收敛为：

> **不再证明 HSTU 是更好的推荐模型，而是把 HSTU 视为一种已经成立的、具有持久化用户状态的生成式推荐架构，研究模型版本更新后，海量旧 KV 状态应当何时保持不动、何时近似演进、何时选择性重算、何时完整重算。**

HSTU 原始工作已经在公开数据和工业部署中验证了其针对高基数、非平稳流式推荐场景的有效性；我们的论文没有必要重新承担这一算法有效性证明。([arXiv][1])

---

# 一、最新版核心问题：不是“如何恢复一个必然为正的 gap”

原来的问题隐含了一个过强假设：

[
Q(\theta_t,\mathrm{Full}) >
Q(\theta_t,\mathrm{Reuse})
]

而且希望它在每一个相邻版本上成立。

现在应改成：

> 一个已发布的新模型 (\theta_t)，继续读取由旧模型 (\theta_{t-1}) 产生的持久化 KV，会形成一个跨版本混合执行状态。不同版本更新、不同用户、不同历史长度、不同层和不同请求时刻上的失配风险并不相同。系统需要在质量风险和迁移成本之间做出状态演进决策。

设用户 (u) 的旧状态为：

[
C_u^{t-1}=F(\theta_{t-1},H_u)
]

当前模型精确状态为：

[
C_u^t=F(\theta_t,H_u)
]

EvoKV 生成或保留的状态为：

[
\hat C_u^t
]

目标不再是要求所有边都满足正 NDCG gap，而是：

[
\min_{{a_u}}\sum_u \mathrm{Cost}(a_u)
]

满足：

[
D!\left(
G(\theta_t,\hat C_u^t),
G(\theta_t,C_u^t)
\right)\leq \epsilon
]

其中 (a_u) 可以是：

* 不迁移；
* 快速恢复；
* 选择性重算；
* 精确重算。

同时在应用指标上满足：

[
Q(\mathrm{EvoKV})
\geq
Q(\mathrm{Full})-\eta
]

这里的 Full 是**当前模型的正式执行语义参考**，但不是 ranking 理论上界。你已有实验已经证明，当前模型更新如果发生负迁移，Reuse 在某些未来样本上可能偶然优于 Full；因此不能把所有 `Recompute < Reuse` 都当作 evaluator 错误。

这个问题更强的地方在于，它天然包含三类真实情况：

| 情况   | 现象               | 正确系统行为   |
| ---- | ---------------- | -------- |
| 状态兼容 | Full 与 Reuse 很接近 | 不迁移      |
| 局部失配 | 只有部分用户、层或请求受影响   | 选择性演进    |
| 严重失配 | Reuse 明显偏离当前模型   | 精确或大范围重算 |

因此，**零 gap 和负 gap 不再全是失败数据**。零 gap 可以证明全量迁移是浪费；正 gap 可以证明不加判断地复用存在风险；不同用户和版本之间的差异则证明需要控制器。

---

# 二、把模型发布和状态迁移彻底分开

上一版把“当前模型是否值得发布”也放进 EvoKV，实际上又扩大了论文范围。

更合理的边界是：

## 模型发布决策：外部输入，不是本文贡献

模型团队已经决定发布 (\theta_t)。它可能通过了线上实验、shadow evaluation、内部模型门禁或人工审批。

EvoKV 不负责判断：

> (\theta_t) 是否是一个比 (\theta_{t-1}) 更好的推荐模型？

EvoKV 只负责判断：

> 在 (\theta_t) 已经发布的前提下，旧状态 (C^{t-1}) 是否仍然足够兼容？如果不兼容，应该花多少成本恢复？

离线公开数据上没有真正的生产发布系统，所以可以采用一个冻结的 **qualified transition protocol** 来模拟已发布版本：

1. 使用统一训练方法产生版本；
2. 在发布前数据或固定 validation 上达到预设质量门槛；
3. 不根据后续 Recompute–Reuse 符号决定是否保留该边；
4. 所有候选更新及其状态兼容结果完整报告。

这只是构造合理的系统 workload，不需要把模型 admission 做成新的研究贡献。类似的跨模型 KV 工作也会先构造有意义的 sender–receiver 模型对，再研究状态复用，而不是证明任意两个 checkpoint 都值得迁移。DroidSpeak 明确要求 receiver 在对应任务上优于 sender，再分析跨模型 KV 复用。([arXiv][2])

---

# 三、HSTU 只需要通过“最小合理性检查”

你的判断完全正确：**不需要再做 HSTU 对 SASRec、BERT4Rec、GRU4Rec 等一大堆算法比较。**

论文不是在回答：

> HSTU 是否是这个数据集上最准确的推荐算法？

而是在回答：

> 一个持久化用户 KV 的状态化推荐系统，在模型版本更新时如何低成本维持状态一致性？

因此，HSTU 基模只需要满足三个最低条件。

## 1. 推荐质量没有坏到失去意义

主表中只需要：

* Random；
* Popularity 或简单 retrieval prior；
* No-history；
* Full HSTU。

目的不是赢 SOTA，而是说明：

* 模型明显优于随机；
* 在合理候选场景下，历史分支有非零贡献；
* 当前模型可以作为一个正常工作负载。

你已有的 v81 已经证明，在控制候选热度后，Full 相对 no-prefix 的用户 bootstrap 区间明显为正。这个证据足以说明 HSTU 并非完全没有使用历史状态。

## 2. KV 确实位于有效执行路径上

只要证明：

[
G(\theta_t,C_u^t)
\neq
G(\theta_t,\varnothing)
]

或者 Full 与 no-history 在 logit、AUC、NDCG、margin 中存在可测差异即可。

没有必要进一步要求：

[
\text{Natural Order} >
\text{Shuffle}
]

除非论文明确声称系统维护的是“严格时间顺序建模状态”。

更稳妥的论文用词应该是：

> **history-conditioned persistent state**

而不是：

> **精确编码自然时序关系的 sequential state**

这样 v81 中“模型稳定读取历史 item 集合，但没有稳定证明自然顺序更优”的结果就不是阻塞项。当前材料也已经给出了这种修改方向。

## 3. KV 的持久化确实带来计算收益

这一点不需要重新证明 HSTU 算法优越性，只需通过系统 profiling 说明长历史 Full Recompute 的成本显著高于增量 append。

近期工作已经把这一系统场景当成现实问题：MTServe 研究生成式推荐中每用户长期持久化 KV 的多级缓存，CollectiveKV 研究跨用户 KV 压缩，TransX 则在生产推荐中把行为编码移到 nearline，并将用户表示缓存到磁盘。([arXiv][3])

---

# 四、EvoKV 的最新版系统设计

原来的三条执行路径可以保留，但论文重心需要从：

> “设计一个始终有效的 KV 迁移算子”

转变为：

> “设计一个能够判断是否迁移、迁移哪些状态、选择哪条路径、如何在全局预算下执行的状态演进系统。”

这会显著降低整个论文对单一 migration 算子成功率的依赖。

## 第一层：Version Compatibility Profiler

新版本发布时，在极少量 canary 用户状态上运行：

* Full；
* Reuse；
* 若干层级或路径级 partial recovery。

不使用未来 target 标签，主要收集：

* 当前模型输出分布的 KL/JS divergence；
* target-independent 的 Top-K overlap；
* candidate rank displacement；
* score margin 变化；
* 各层局部替换后的敏感度；
* 不同历史长度、状态年龄和用户活跃度下的差异。

这里可以使用少量 exact current KV 做 profiling，因为它只用于：

* 估计风险；
* 选择固定执行路径；
* 校准阈值。

它不用于监督一个任意自由度的旧 KV→新 KV mapper，因此不会违反你现有协议中“不使用少量用户 exact target KV 拟合自由变换器”的底线。

Profiler 输出版本级兼容性画像，例如：

```text
θt-1 → θt

0–3 层：高度兼容
4–5 层：部分用户敏感
6–7 层：普遍敏感

短历史用户：低风险
长历史、高活跃用户：高风险
首个发布后请求：高风险
当前模型 append 增多后：风险下降
```

## 第二层：多路径状态演进

建议保留四个动作，而不是强迫所有用户进入同一迁移流程。

| 风险级别 | 动作              | 说明                 |
| ---- | --------------- | ------------------ |
| 很低   | No-op Reuse     | 旧状态直接继续使用          |
| 较低   | 快速恢复路径          | 使用你现有的结构化 Design 1 |
| 中等   | 选择性重算           | 仅重算敏感层、敏感分片或指定状态区段 |
| 高或未知 | Exact Recompute | 保证当前模型正式语义         |

关键改变是：

> Design 1 不再承担“所有版本、所有用户都要恢复”的任务，只承担它最适合的中低风险区域。

如果快速恢复在某类更新上效果一般，系统可以转向选择性重算或 exact；如果某个版本本来就兼容，则直接 no-op。这样论文不会被某一个 approximate operator 的失败拖死。

## 第三层：Budgeted State Evolution Controller

控制器综合三类信息：

### 模型侧

* 哪些层发生更新；
* 参数变化幅度；
* attention/readout 路径是否变化；
* canary 上各层敏感度。

### 用户状态侧

* prefix 长度；
* KV 大小；
* 状态版本年龄；
* 用户近期请求概率；
* 当前模型已经 append 的新行为比例；
* 用户所属风险分组。

### 系统侧

* KV 位于 HBM、DRAM、SSD 还是 HDD；
* 读取旧 KV 的 IO 成本；
* partial/full recompute 的 GPU 成本；
* 发布后的迁移截止时间；
* 当前训练和推理负载。

全局优先级可以简单定义为：

[
Priority(u)=
\frac{
P_{\mathrm{request}}(u)\cdot
\widehat{\Delta D_u}
}{
\widehat{\mathrm{MigrationCost}_u}
}
]

其中：

* (P_{\mathrm{request}}(u)) 是用户近期被访问的概率；
* (\widehat{\Delta D_u}) 是迁移后预计恢复的语义偏差；
* (\widehat{\mathrm{MigrationCost}_u}) 是计算和 IO 总成本。

不必一开始就用复杂的 learned policy。**基于 profiling 的规则、分桶和经验阈值完全符合系统论文范式**。后续再比较简单规则、回归预测和 oracle 即可。

---

# 五、你现有的“首请求效应”应升格为核心系统观察

现有实验发现：

* 发布后第一个请求最容易暴露旧 prefix；
* 随着更多行为由当前模型 append，旧状态的作用逐渐被稀释。

这不是一个评测麻烦，而是很有价值的系统机制。

可以定义当前模型 append 比例：

[
\rho_u=
\frac{n_{\mathrm{current\ append}}}
{n_{\mathrm{old\ prefix}}+n_{\mathrm{current\ append}}}
]

通常 (\rho_u) 越大，旧版本状态所占比例越低。

因此：

* 刚发布、长旧 prefix、高活跃用户：优先迁移；
* 已有大量当前模型 append 的用户：迁移优先级下降；
* 长期不访问用户：根本不需要发布时主动迁移，可以首次访问时按需处理。

这会产生一个非常有“系统味道”的设计：

> EvoKV 不追求发布瞬间把全体用户状态全部更新，而是让状态以风险感知、访问驱动、预算约束的方式逐渐收敛到新版本。

这比“所有用户 KV 做一次批量转换”更符合亿级用户系统。

---

# 六、实验不再追求十全十美，而追求一个最小闭环

新版实验只需要回答四个 Research Question。

## RQ1：这个系统问题是否真实存在？

不要求所有更新都产生正 NDCG gap。

只需要预先定义几类更新：

* routine fine-tuning；
* 中等表示路径更新；
* major block/model refresh。

然后完整报告它们的结果。

理想的 characterization 不是“8/8 都正”，而是出现三类状态：

1. 有些版本高度兼容，Full≈Reuse；
2. 有些版本明显失配，Reuse 偏离 Full；
3. 同一版本中不同用户和请求的风险高度不均匀。

这三类结果共同说明：

> 全量重算和全量复用都不是合理的统一策略。

“只需要在某些情况下证明问题”是成立的，但这些“情况”必须由**更新语义和 workload 条件预先定义**，不能在看完 NDCG 后只挑正边。

如果最后发现：

* routine fine-tuning 基本不需要迁移；
* major model refresh 明显需要迁移；

那论文就应当诚实声称：

> EvoKV 面向会改变 cache-producing representation 的模型发布，而不是所有微小日常更新。

这依然是一个合理、清晰的系统问题。

## RQ2：EvoKV 能否判断哪些状态需要演进？

评估：

* 高风险状态召回率；
* 不必要迁移比例；
* 状态风险预测 AUC；
* fidelity SLO violation rate；
* oracle 与 controller 的差距。

这里的关键不是推荐 NDCG，而是 controller 是否把有限资源花在正确状态上。

## RQ3：EvoKV 能否用更少成本逼近 Full？

主要画一张 Pareto 曲线：

[
\text{Fidelity / Quality}
\quad\text{vs.}\quad
\text{Compute + IO Cost}
]

系统 baselines 只需要：

* Reuse All；
* Full Recompute All；
* Random Refresh；
* Hot-user-only；
* 固定最后 (k) 层重算；
* version-level uniform policy；
* EvoKV。

不需要 SASRec、BERT4Rec 等算法 baseline 出现在这里。

## RQ4：系统能否扩展到海量状态？

报告：

* 发布后 refresh burst；
* time-to-freshness；
* GPU hours；
* IO bytes；
* P50/P95/P99 在线延迟；
* 不同 HBM/DRAM/SSD/HDD 配置；
* 1M、10M、100M 用户外推或 trace-driven replay；
* 与在线推理共存时的吞吐影响。

---

# 七、推荐的数据集组合需要调整：Yambda 值得升为主数据集

广泛比较之后，我认为之前的 RecFlow + VK-LSVD 方向没错，但还缺了一个与你的时间更新问题更匹配的数据集：**Yambda-5B**。

## 第一主数据集：Yambda-50M 或 Yambda-500M

这是目前最适合验证“连续版本—长期用户状态—日级更新”的公开数据之一。

它提供：

* 约 11 个月的行为；
* 5 秒精度时间戳；
* timestamp-sorted 的用户序列格式；
* 50M、500M、5B 三种规模；
* listen、like、dislike 等多类行为；
* 区分自然行为与推荐驱动行为的 `is_organic`；
* 官方 Global Temporal Split。([arXiv][4])

尤其重要的是，它的官方 benchmark 本身采用：

* 300 天训练；
* 30 分钟发布 gap；
* 1 天测试；

并明确把它解释为近似日级离线模型和用户状态更新。我们不需要照搬“测试期冻结用户状态”的设置，但可以直接利用原始时间戳重新构造合法的当前模型 append lineage。([arXiv][4])

Yambda 的音乐场景还有一个潜在优势：用户具有较长历史、重复消费和相对稳定的兴趣结构，比 KuaiRand 的高基数稀疏短视频 next-item 更容易形成可预测的连续状态。

建议先用 Yambda-50M 完成机制开发，再扩展到 500M 做系统和质量验证，不需要一上来处理 5B。

## 第二主数据集：RecFlow

RecFlow 最有价值的不是规模，而是它提供了工业推荐漏斗中的真实阶段样本：

[
\text{retrieval}
\rightarrow
\text{pre-ranking}
\rightarrow
\text{coarse ranking}
\rightarrow
\text{ranking}
\rightarrow
\text{reranking}
]

它包含约 42K 用户、937 万请求和 19.24 亿阶段样本，覆盖 37 天。

这能够解决你目前最棘手的候选集争议：

* 不再用 uniform negative 把 popularity-only 变得异常强；
* 不再事后构造容易或困难候选；
* 直接冻结某一真实 serving stage 的候选；
* 在同一候选合同下比较 Full、Reuse 和 EvoKV。

因此：

* Yambda 负责**连续时间和长期状态**；
* RecFlow 负责**真实候选与 serving semantics**。

## 系统规模数据：VK-LSVD 或现有 QK/QB trace

VK-LSVD 有 10M 用户、近 20M 视频、超过 40B 曝光，持续六个月，并按周提供全局时间有序的 Parquet 文件；它还提供可直接使用的用户子集和内容 embedding。([arXiv][5])

但没有必要要求它同时承担所有质量实验。它更适合：

* 迁移工作量统计；
* 用户活跃度分布；
* 状态长度分布；
* 周级版本演进；
* 10M 级 trace-driven scheduling；
* 系统规模外推。

已有 QK/QB 如果更方便做大规模 KV payload 和吞吐，也可以继续作为系统 workload。

## KuaiRand 的新定位

KuaiRand 不需要丢掉，但应降级为：

* 历史结果连续性；
* 随机曝光 robustness；
* evaluator 和 lineage 回归；
* major-update stress case。

它不再承担“唯一证明自然日更新必然导致 cache stale”的任务。

---

# 八、现有 v69、v80、v81、v82 可以组成一套很好的动机

你的历史结果其实已经近似形成了三种系统状态。

## v69：严重失配 workload

v69 中完整重置并重训上四层，七条开发边均出现正点估计，但其结构化更新明显强于普通 fine-tuning，绝对 NDCG 也只有约 0.10。

新定位：

> major representation update 下的 high-risk transition。

它不是用来证明“普通日更通常如此”，而是证明：

> 当更新真正改变 cache-producing path 时，跨版本旧 KV 可以造成明显质量风险。

## v80：兼容或近似 no-op workload

v80 有较高 sampled NDCG，但主要由 popularity prior 提供，HSTU 边际贡献小，第二条边 Recompute–Reuse 接近零甚至略负。

新定位：

> 某些版本和候选场景中，状态迁移没有足够收益，Full refresh 是浪费。

它是 no-op path 的动机，不再作为失败链。

## v81/v82：条件性与异质性 workload

v81 证明 HSTU 使用历史状态，但 gap 区间跨零；v82 则显示候选难度会显著改变 gap 的可见性。

新定位：

> state staleness 不能只由参数变化量决定，还与服务候选、用户历史及请求阶段共同相关，因此需要 runtime profiling 和状态级决策。

于是过去的叙事从：

> “我们不断尝试但无法得到稳定正链”

变成：

> “我们发现状态兼容性具有强烈的版本、用户、请求和候选依赖性，统一迁移策略并不存在。”

后者明显更像一个系统发现。

---

# 九、创新性上必须避开的新风险

现在不能只把 EvoKV 写成：

> 选择一些层重算，剩下的旧 KV 继续复用。

因为近期跨模型 LLM KV 工作已经覆盖了：

* 基于层敏感度的选择性重算；
* 低秩状态转换；
* transition layer patch；
* 硬件感知的重算与 IO 比例搜索。([arXiv][2])

推荐领域近期也已经出现：

* MTServe：持久化推荐 KV 的 HBM–DRAM 层级管理；
* CollectiveKV：跨用户 KV 压缩；
* Memory Layer：训练—服务 item embedding freshness；
* TransX：nearline 用户行为编码与持久化缓存。([arXiv][3])

所以 EvoKV 的真正差异化不能只是某个局部 KV mapper，而应该是：

1. **连续模型 lineage，而不是一次 sender→receiver 转移；**
2. **每用户长期持久化状态，而不是一次 prompt；**
3. **模型更新后旧 prefix 与当前模型 append 共存；**
4. **状态风险会随请求和 append 逐渐稀释；**
5. **系统主动决定 no-op、近似恢复、选择性重算或 exact；**
6. **在亿级用户和多层存储上做全局预算调度；**
7. **控制多版本累计误差和状态版本年龄；**
8. **以当前模型 fidelity SLO 为正确性约束。**

这才是最值得押注的创新点。

---

# 十、最新版最小论文闭环

最终不需要十全十美，只要形成下面这个闭环：

### Observation 1

持久化推荐 KV 与模型版本绑定；模型更新后，旧状态与当前模型形成混合语义。

### Observation 2

状态兼容性不是固定的：

* 有些更新不需要迁移；
* 有些更新只影响部分状态；
* 有些更新必须重算。

### Design

EvoKV 使用少量 canary profiling，结合版本、用户、请求和硬件信息，在四条路径间决策，并进行预算约束的全局状态演进。

### Result

与 Full Recompute 相比：

* 显著降低总计算、IO 和发布后 burst；
* 在预设 fidelity/质量界限内逼近当前模型；
* 对兼容版本避免无意义迁移；
* 对高风险版本恢复部分或大部分 stale-state 损失。

### 明确不声称

* HSTU 优于 SASRec；
* 每一次日级更新都导致 KV 失效；
* Full 的 NDCG 永远优于 Reuse；
* HSTU 在所有数据上都稳定利用严格自然顺序；
* EvoKV 对任意版本更新都必须使用同一种迁移算法。

一句话概括最新版论文：

> **EvoKV is a version-aware state evolution system that determines when and how persistent user KV states should migrate across model updates, preserving current-model semantics under a bounded recomputation and I/O budget.**

真正的破局点，就是把成功条件从：

> “调出一条所有边都为正的 Recompute–Reuse 链”

改成：

> **“证明状态兼容性本身具有异质性，并设计一个比全量复用和全量重算都更合理的决策与执行系统。”**

下一轮实验应停止继续扩展 KuaiRand 八版本链，直接做 **Yambda-50M 的三版本小链 + RecFlow 固定阶段候选**。模型侧只做 Random、Popularity、No-history、Full 四项 sanity check；随后集中验证 routine、moderate、major 三类更新下的 no-op、selective、exact 三种系统区域。

[1]: https://arxiv.org/html/2402.17152v3 "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
[2]: https://arxiv.org/html/2411.02820v4 "DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving"
[3]: https://arxiv.org/html/2604.22881v1 "MTServe: Efficient Serving for Generative Recommendation Models with Hierarchical Caches"
[4]: https://arxiv.org/html/2505.22238v2 "Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval"
[5]: https://arxiv.org/html/2602.04567v2 "VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation"
