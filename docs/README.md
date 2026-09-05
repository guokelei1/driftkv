# EvoKV 文档入口

更新日期：2026-09-05

当前论文主线研究 **模型发布时的 persistent state migration**，以一次 Parent → Current 更新
说明基础流程，并在设计中覆盖连续发布的状态管理。Motivation、
Medium Insight 1 与 scoped Insight 2 已形成；当前 Design 1 是
**Sketch-to-Sketch State Migration**：

- ordinary Parent K/V 保持不变；
- 固定、支持加减更新的 writer 随 K/V 生成 producer-native Source Sketch；
- Current checkpoint 定版后冻结 backbones，独立校准面向该目标版本的共享 Translator；
- release 时只把 Source Sketch 翻译为 Current Sketch；
- request 时由同一 Current query paired-read 新旧 sketches，并将 response difference 注入 HSTU
  聚合后归一化之前。

参考 Translator 使用同用户待迁移 Source 的跨段、跨层信息；连续发布时额外按段识别 producer，
将混合来源的原始 Source 直接翻译到新目标。Source 淘汰更新后刷新剩余旧段的翻译结果。
确定性摘要提供写入期预计算与发布期快速访问，不宣称新增普通 K/V 所缺失的信息。

历史 AV/PRO、common-mode canonical writer、per-version interpreter \(D_v\) 和 clean writer \(G_v\)
均不再定义当前 Design 1。它们只作为历史证据、旧候选或 baseline 保留。
连续发布的状态接口与版本管理纳入当前设计；更广泛的 Runtime 控制、RecFlow 和 theta3 仍在范围外。
现有观察和执行合同保持单边范围，新增连续发布设计不扩大运行授权。

## 当前权威入口

1. [Insight 2 / Design 1 论文稿](insight2_design1_expert_brief.md)：当前 Design 1 的唯一规范设计稿；
   第 1–2 节连接 Motivation 与 Insights，第 3 节包含总体架构、版本化摘要、校准与发布迁移、
   查询修正、持续状态管理五个小节，仅保留两条核心公式。
2. [论文总体设计](paper_design.md)：稳定的问题定义、Insight 1 → Insight 2 →
   Sketch-to-Sketch Design 1、架构边界和论文主张。
3. [论文具体实验设计](experimental_design.md)：现有 Medium V0–V5 资产、sealed discovery 证据、
   真实摘要参考、Translator calibration、closed-loop evaluation 和成本合同。
4. [核心 Motivation 与 Observation](motivation_observations.md)：已经封存的 motivation、观察和结果。
   结果数字保持原证据范围，不因 Design 1 更新而改写。
5. [Insight 2 / Design 1 探索计划](../research_discussions/evokv_three_module/insight_two/insight_two_exploration_plan.md)
   与[追加式探索日志](../research_discussions/evokv_three_module/insight_two/exploration_log.md)：保留发现
   过程、候选、反例与历史裁决，不再作为现行 Design 规范。
6. [当前 KV-only 接口裁决](../research_discussions/evokv_three_module/insight_two/current_kv_only_interface_adjudication.md)：
   汇总 KV-only generator-closure 缺口、强 generic control 和已退休机制。
7. [Medium Insight 1 locality 结果](../results/yambda500m_medium_seed17/insight1_locality_v1/analysis/report.md)：
   五条 D14 edge、34 个预指定 Exact-KV splice 的无标签诊断。
8. [Medium 全轮实验总结](../results/yambda500m_medium_seed17/full_reuse_matrix_v1/medium_scale_experiment_summary.md)：
   六个 D14 V0–V5 checkpoints、Full/Reuse 和数据范围的事实入口。

若文档冲突：

- 观察事实以 sealed result、[motivation_observations.md](motivation_observations.md) 和对应合同为准；
- 当前 Design 1 机制以 [insight2_design1_expert_brief.md](insight2_design1_expert_brief.md) 为准；
- prospective 执行必须同时满足 [experimental_design.md](experimental_design.md)、新合同和仓库授权规则。

## 当前状态

已经具备 ordinary K/V、Full/Reuse、reader instrumentation、diagnostic response-difference injection、
Insight 1 负结果和 Insight 2 oracle contraction evidence。尚未实现固定 Sketch writer、edge Translator、
production paired reader、完整 append/eviction lifecycle、release executor 和方法 qualification。

因此：

- oracle 95.34%/99.46%、PRO 或 generic-rank 结果不能表述为 Sketch-to-Sketch 方法结果；
- 旧 0.60% translation 与约 7% storage 不作为当前方法估计；32/1024 配置的 paired read
  约 6.25% 只表示 QK/AV 算术增量，FP32 Source 的存储需按实际精度重新计量；
- Current-produced appended K/V 不能未经 closed-loop 实验称为 clean/exact；
- 现有 V0–V5 backbone 可以冻结复用，但旧 cache 没有预存 Source Sketch，第一轮需要 backfill。
- 真实 Current Sketch 是差分压缩的诊断参考，不是所有 learned response 方法的严格上界；
- 本设计尚未证明持续追加收敛，Parent 淘汰不能作为误差消失的证明。

## 历史与支撑文档

- [Insight-Driven State Refinement Develop Map](insight_develop_map.md)保留旧 reader correction、
  typed coordinate、evidence mass、PRO 和负结果，不决定当前 Design。
- [One-Release State Refinement 与 Typed Plan IR](typed_state_refinement_algebra.md)保留旧
  CAST/PATCH/GROUP/SCALE 语义和 strong baselines，不决定当前 Design。
- [Medium 训练推进方案](medium_scale_training_plan.md)保留已完成训练记录；当前默认不重新训练
  V0–V5 backbone。
- [Large 模型规模讨论稿](large_scale_model_sizing_discussion.md)与
  [Large 训练记录](large_scale_training_and_qualification_plan.md)是历史 scale 材料，不授权本轮训练。

## 维护规则

- 只同步更新本 README、paper design、experimental design 和唯一指导稿；不为同一机制创建重复真源。
- 总体设计写稳定概念；实验设计写如何否证，不把预期或 oracle 写成方法结果。
- Design 正文说明组件职责、输入输出、设计动机和执行连接；有效性由实验评价，技术约定放在
  experimental design，避免把正文写成逐模块等价证明或反例清单。
- frozen contracts、hashes、raw seals、adjudications、negative results 和 invalidations 不覆盖、不改写。
- diagnostic Exact-KV/stage splice 永远不是 executable action；KV coverage 不能冒充 GPU FLOPs。
- Design 1 包含连续发布的混合来源、目标切换和版本退出规则；翻译始终从原始 Source 出发，
  不串联上一轮 translated sketch。多版本校准及连续轨迹评价需独立 prospective contract，
  不将已有五条相邻边结果合并冒充连续迁移结果。
- HSTU 的 additive paired-read 公式不能原样推广到 softmax attention；后者需要 numerator/normalizer
  sufficient-statistics adapter。
- Translator calibration 涉及 Current-derived teacher。运行前必须建立 prospective contract，明确
  disjoint calibration population 的共享 edge-level supervision，并继续禁止 evaluation-user/per-user
  target fitting。
- 代码与结果状态必须明确区分 specified、implemented、canary-validated 和 paper-qualified。
