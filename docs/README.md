# EvoKV 文档入口

当前文档保留 README、五份研究文档、Medium 执行/结果文档、Large sizing 裁决输入和 Large
正式执行记录，各自职责不同：

1. [论文总体设计](paper_design.md)：概念层问题定义与论文结构：Background & Motivation、简洁的
   System Overview、candidate-shared reader compatibility correction 等三条 recommendation-specific Insight、
   One-Release Refinement、Debt-Bounded
   Continuous State Evolution、GPU Transformation Runtime。内容应稳定，避免写入某次实验细节。
2. [论文具体实验设计](experimental_design.md)：当前架构、数据、版本训练、对照路径，以及 Design I、
   bounded-debt + Exact-shadow Continuous、Runtime、规模/外部验证的分阶段研究协议。
3. [核心 Motivation 与 Observation](motivation_observations.md)：目前已经观察到的 HSTU-native
   motivation、D14/E14 数字、固定 one-release 路径的完整 rolling AUC、版本年龄结果、3,000-user
   recommendation-state structure、signed causal/真实 exposed candidate 复核、reader stage、
   跨请求 persistence、history-basis 负结果、AV-sidecar 4/5 score canary、无 translated-prefix
   物化的轻量 PRO 正确性/成本、五边全人口 rolling quality，以及最新 progressive PRO 无标签
   error decomposition/C32-C48-C64 fidelity frontier、companion 和结论边界。本次 v0–v5 探索与专家意见后的推进另有一份可直接用于
   复核与专家讨论的[单篇总结](../results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/expert_discussion_summary.md)。
4. [Insight-Driven State Refinement Develop Map](insight_develop_map.md)：论文第 3 章的研究支撑文档，
   将 reader-correction/typed-coordinate/evidence-mass Insight、Design 0、已有证据、反证边界和补强路线放在同一设计链中；它不是
   独立的论文 Insight 章节，也不声称已经推出 Continuous 或 Runtime。
5. [One-Release State Refinement 与 Typed Plan IR](typed_state_refinement_algebra.md)：Design I 底层的
   `CAST / PATCH / GROUP / SCALE` typed semantics、聚焦机制实验、成本边界，以及向 Continuous
   交付的 lineage metadata；不将四个 operator 包装成四条并列论文 Insight。
6. [Medium Full-only 训练推进方案](medium_scale_training_plan.md)：冻结 day217、30k/6L Medium 的
   D7/D14 训练矩阵、GPU2/3 双卡执行、Full-only→Reuse admission 顺序、资源估算与 launch gate；
   当前 checkpoint、Full/Reuse、D7 forced diagnostic 与 D14 v5 扩展均已完成，文档保留实际执行记录。
7. [Medium 全轮实验总结](../results/yambda500m_medium_seed17/full_reuse_matrix_v1/medium_scale_experiment_summary.md)：
   已完成 seed17 的 checkpoint、D7/D14 Full/Reuse、D7 forced diagnostic、D14 v5、统一同-cohort 百分比、
   运行成本、异常边、结论边界和专家讨论问题。
8. [Large 模型规模讨论稿](large_scale_model_sizing_discussion.md)：对齐 Small/Medium/Large 的人口、
   catalog、请求、参数、persistent-state 和 A40 资源，比较 8L/H256、10L/H320 与 12L/H320；它是
   专家裁决时使用的历史输入，训练授权不由该文档提供。
9. [Large 训练与验证执行记录](large_scale_training_and_qualification_plan.md)：记录专家裁决后冻结的
   10L/H320、D7/D14 训练与 Full 协议、真实 manifest、四卡 canary、runtime 选择，以及 2026-08-31
   最终在首个正式 Reuse/PRO cell 前取消并于 Full-only 后结束的范围与证据边界。

README 是仓库入口；上面五份研究文档、Medium 执行/结果文档与 Large 两份文档是当前文档层次。
代码、合同、脚本和测试的职责以仓库目录及其 README 为准，不再维护另一套路线文档。

文档维护规则：

- 总体设计只写相对稳定的概念和论文边界；
- 具体实验设计写“如何做”和“希望观察什么”，不把预期写成结果；
- Insight-Driven State Refinement Develop Map 服务论文 Design I：每条 Insight 直接收束到对应
  mechanism，不再维护独立 Design Principles 层，也不把候选机制写成已验证；
- Typed State Refinement 文档只记录 One-Release instruction semantics 和 handoff contract，
  不把诊断 residual splice 直接加入 scale action set，也不代替 Continuous 设计；
- 核心结果文档只写已经封存或可复核的 observation，不把未来设计写成已验证；
- 新结果先更新核心 observation，再按需同步实验设计；不要重新创建阶段性路线文档；
- Continuous 和 Runtime 在研究成熟前统一维护在总体设计与具体实验设计中，不新建分散的路线文档；
- 旧 archive、legacy、开发阶段报告和重复路线说明已删除。
