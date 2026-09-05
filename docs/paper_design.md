# EvoKV 论文总体设计

更新日期：2026-09-05

本文记录论文的问题、证据链、方法结构与比较范围。完整 Design 正文见
[Insight 2 / Design 1 论文稿](insight2_design1_expert_brief.md)，技术约定和评价协议见
[experimental_design.md](experimental_design.md)。Motivation 与 Insights 已有对应观察，
Sketch-to-Sketch 方法处于设计阶段。

## 1. 研究问题

持久化推荐系统的模型参数与用户历史状态具有不同的更新周期。Current 模型通过完整重算路径
证明了质量提升，发布时却需要继续读取 Parent 生成的用户 K/V。已有 Motivation 显示，这种
状态与读取参数的不兼容会损失部分模型更新收益。

EvoKV 研究如何在受限计算和 I/O 预算内完成发布时的状态迁移，使用户尽快受益于已经准入的新模型。
一次 Parent → Current 更新用于说明基础流程，连续发布进一步处理混合来源状态和目标切换。
全人口 Exact-All 需要重新处理历史并写回完整 K/V；本设计保留普通缓存，通过短摘要转换改善
Current 对历史的读取。

模型准入与缓存迁移分开处理。Current 首先通过 Parent/Current Full-only admission；
被拒绝的候选保持 Parent 服务和缓存谱系。已经准入但兼容性损失较小的更新，可以采用预先规定的
No-op 策略。

## 2. 从 Motivation 到 Design

### 2.1 Insight 1：局部重算难以覆盖分散的版本差异

在已观察的版本边上，少量 token、单层或 K/V 单侧替换难以稳定恢复主要差距。较高恢复率需要
较大状态覆盖，而可执行重算还需承担上游依赖。该观察引导 Design 1 保留逐事件历史，把迁移工作
集中到较小的附加状态。

### 2.2 Insight 2：历史聚合提供了紧凑的修正位置

旧 K/V 通过 Current query 的键匹配和值聚合影响预测。Medium discovery 在聚合边界观察到较
紧凑的响应修正：shared/low-rank oracle 分别恢复 95.34%/99.46% 的预指定功能差距。
固定 offset 的持续使用结果进一步说明，服务中需要能随 query 与存活历史变化的修正状态。
这些诊断为设计选择提供依据；具体结果保持原始证据范围。

### 2.3 方法连接

EvoKV 用 Migration Sketch 将两项观察连接起来。普通 K/V 继续提供完整历史的基础响应；
提前维护的摘要让发布任务能够快速读取每个用户的状态；共享 Translator 学习摘要的版本变化；
当前 query 再将变化转化为历史聚合处的修正。

论文的叙事因此沿着同一个问题推进：模型收益受到旧状态影响，局部重算难以低成本覆盖差异，
reader 聚合提供修正位置，Sketch-to-Sketch 则定义如何生成、使用和维护这份修正状态。

## 3. Design 1 的组件与流水线

| 组件 | 输入与输出 | 在流水线中的作用 |
| --- | --- | --- |
| Migration Sketch | 已生成的逐事件 K/V → 分段摘要 | 提前准备发布时可快速处理的用户状态 |
| Sketch Translator | 带 producer 的 Source Sketch → Current Sketch | 校准目标版本的共享转换，并用于全人口发布 |
| Paired Reader | 当前 query、普通 K/V、新旧摘要 → 修正后的历史聚合 | 将摘要变化转化为本次请求的响应修正 |
| State Evolution | 新事件、淘汰或再次发布 → 同步的缓存和目标视图 | 管理历史变化、混合来源、目标切换和版本退出 |

### 3.1 Overview：从摘要写入到版本修正

普通 K/V 提供基础历史，Source Sketch 保存翻译输入，Current Sketch 保存面向服务目标的读取
参照。writer 随事件生成摘要，Translator 在校准后完成发布迁移，Paired Reader 在每层注入
摘要响应差，状态管理使这套路径跨事件与模型更新反复执行。

以下四节按状态、转换、读取和持续管理组织。成本分别归入各组件，并在评价时累计。

### 3.2 Migration Sketch：版本化的历史摘要

Source Sketch 随 K/V 生成维护。事件进入固定 segment 和 slot，每个 slot 保存 K/V 累加量、
数量及时间位置统计，翻译和读取时形成平均 K/V。事件归属保持固定，追加时累加、淘汰时扣除。

同一 release family 使用相同的摘要格式及事件划分，Parent/Current 的对应 slots 描述相同历史。
参考配置每段最多 64 个事件、每层两个 slots；边界对齐的 1,024-event 窗口有每层 32 个 slots，
在线半满段与发布封段的容量单独预留。

提前维护摘要将聚合分摊到正常写入期。已有缓存首次接入时，从旧 K/V 构造 Source Sketch，
初始化工作计入系统成本。

### 3.3 Sketch Translation：共享校准与发布迁移

Current 完成正常训练和 Full-only admission 后，冻结两版 backbones，在独立的校准用户集上
训练共享 Translator。两版模型处理相同可见历史，形成对应摘要与响应教师。

参考 Translator 接收同用户待迁移 Source 的跨段、跨层信息，输出同结构的 K/V 残差。
事件数量、mask、时间和位置由状态维护路径保留。跨段上下文帮助网络利用前序历史的摘要，
网络容量在开发阶段确定。

训练结合摘要重建和响应差两个目标：前者对齐 Current 摘要载荷，后者对齐线上使用的修正量。
响应监督使用同一个 Current query 读取两版完整状态和两份摘要；query 来自逐层修正的部署路径。

正式发布对全量用户执行摘要翻译并保存结果。Source revision、模型版本和格式信息随结果提交，
用于连接发布、并发更新和请求读取。Current 已准入但尚无有效摘要的用户按 Reuse 路径服务，
迁移覆盖与完成时间一同计量。

### 3.4 Paired Reader：查询驱动的历史响应修正

每层先用当前 query 读取普通缓存，再读取新旧两份摘要，将摘要响应差加入基础历史聚合。
普通 K/V 保留完整历史响应，Source Sketch 提供旧版本参照，Current Sketch 提供翻译后的参照。
paired subtraction 将附加路径用于版本变化，避免重复加入整份历史摘要的响应。

HSTU reader 按 slot 事件数量加权，使用原模型的匹配激活、偏置和缩放规则。修正在聚合后
归一化之前合并，然后执行原有门控、输出投影与残差，继续形成下一层 query。
候选评分的暂态状态只用于本次请求。

### 3.5 State Evolution：历史更新与连续模型发布

新事件在 paired read 提供的历史上下文上完成 Current 前向，并写入 Current K/V 与 Source
Sketch。发布时封存 Parent 的未满 segment，让新事件进入 Current segment。

淘汰先扣除 Source 中的事件贡献，再删除对应普通 K/V。参考 Translator 使用跨段输入，因此
Source 内容变化后刷新该用户全部剩余旧段的翻译结果，并以统一修订提交。

Parent segments 逐步退出时，直接 paired read 的开销随之减少。持续服务评价同时关注新 K/V
在修正历史上生成后的质量变化，使用相同的追加、保留、淘汰与位置约定进行比较。
producer 标记承担版本路由功能。

连续发布时，某用户可能同时持有多个旧版本生成的 K/V。每个新目标版本的共享 Translator
以 producer 标记为条件，读取这些段的原始 Source 并生成面向新目标的视图，替换上一轮
translated sketch。校准按预先确定的发布与事件顺序保留实际生成的混合来源输入，使用最新
目标模型提供教师；不以独立相邻边的 Full 摘要拼接代替连续服务轨迹。

请求绑定一个目标模型及匹配的摘要修订，迟到的旧目标结果不再提交。新目标视图就绪前使用
该目标的 Reuse，整体切换后待在途请求结束再回收旧视图。仍有存活缓存引用的来源纳入版本
支持，未覆盖用户计入 Reuse 与整体质量，来源无引用后退出支持集合。

## 4. 论文写作与主张

Design 正文固定为五节：总体架构、版本化摘要、校准与发布迁移、查询修正、持续状态管理。
每节有独立职责，发布并入版本转换，追加、淘汰和连续发布并入状态管理，成本按组件说明。
只保留“摘要翻译”和“paired response correction”两条核心公式；
摘要统计、训练项和成本累计主要用文字解释。
固定尺度、代表位置、更新顺序等技术细节在实验设计中维护。

论文提出一个学习式近似迁移方案，其有效性通过质量、持续服务和系统成本评价。
各模块的名称说明它在流水线中的职责，论文以操作定义和实验支撑这些职责；
不要求为每个模块建立与完整重算逐元素等价的证明。

方法贡献由整条流水线承担：producer-time 摘要维护、release-time 共享翻译、
query-conditioned paired read，以及支持追加和淘汰的状态更新。Translator 是其中的共享映射组件。
比较时为直接 mapper、fixed offset、Current-Sketch-only、通用压缩、PRO 和 low-rank controls
匹配监督、容量与资源，检验这些组件及其连接带来的实际收益。

## 5. 成本与评价问题

一次性发布计算包含教师生成、Translator 训练、必要的摘要初始化和全人口翻译，以相同人口、
历史及执行口径的 Exact-All 为参照，设计目标为其 0%–20%。持续服务另计 writer、paired read 和
摘要刷新，并在同一服务时段内比较总开销。I/O、存储、完成时间、延迟和吞吐分别报告。

32/1024 的摘要配置对应约 6.25% 的额外历史 QK/AV 算术工作，存储按实际累加精度、翻译载荷、
统计字段与边界容量计算。Translator 与实现未确定前，旧“0.60% 翻译”和“约 7% 存储”不作为
当前方法成本结论。

评价围绕以下问题展开：

- 共享 Translator 是否能让未见用户受益于 Current 模型？
- 摘要、翻译和 paired read 各自对最终质量贡献多少？
- 追加、淘汰和模型连续发布后的持续服务表现如何？
- 一次性发布和持续请求的成本在哪些运行条件下具有优势？

具体协议保留真实摘要参考、Translator 泛化、逐层执行、持续更新和实测成本的比较。
主要质量目标为恢复至少 80% 的 Current Exact–Reuse 差距，90% 为进一步目标；
评价同时报告原始任务指标，并按预定规则处理差距很小或反号的情况。
全部冻结 edges 和 seeds 均报告，训练 seed 是统计重复单位。

## 6. 范围与证据管理

当前完整实例是 HSTU candidate-conditioned explicit-feedback ranking，设计覆盖一次已准入
Parent → Current 更新及连续发布的状态管理。现有实验证据和执行合同仍限于原单边范围；
多来源校准、发布重叠与完整连续轨迹需独立协议。其他 attention 结构需适配 reader，
next-item/retrieval、RecFlow 和 theta3 留在独立设计与合同范围内。

已有 Motivation、Insight 1 locality 和 Insight 2 response-contraction 结果保持原样，
oracle、PRO 与 generic controls 的数字不充当 Sketch-to-Sketch 方法结果。
实现状态和后续执行授权由 [experimental_design.md](experimental_design.md)及相应合同管理。
