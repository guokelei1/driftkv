# EvoKV 文档入口

当前文档保留 README 与五份研究文档，各自职责不同：

1. [论文总体设计](paper_design.md)：概念层问题定义与论文结构：Background & Motivation、简洁的
   System Overview、内嵌三条 Design Insights 的 One-Release Refinement、Debt-Bounded
   Continuous State Evolution、GPU Transformation Runtime。内容应稳定，避免写入某次实验细节。
2. [论文具体实验设计](experimental_design.md)：当前架构、数据、版本训练、对照路径，以及 Design I、
   bounded-debt + Exact-shadow Continuous、Runtime、规模/外部验证的分阶段研究协议。
3. [核心 Motivation 与 Observation](motivation_observations.md)：目前已经观察到的 HSTU-native motivation、D14/E14 数字、固定 one-release 路径的完整 rolling AUC、版本年龄结果、companion 和结论边界。
4. [Insight-Driven State Refinement Develop Map](insight_develop_map.md)：论文第 3 章的研究支撑文档，
   将三条 Design Insights、对应机制、已有证据、反证边界和补强路线放在同一设计链中；它不是
   独立的论文 Insight 章节，也不声称已经推出 Continuous 或 Runtime。
5. [One-Release State Refinement 与 Typed Plan IR](typed_state_refinement_algebra.md)：Design I 底层的
   `CAST / PATCH / GROUP / SCALE` typed semantics、聚焦机制实验、成本边界，以及向 Continuous
   交付的 lineage metadata；不将四个 operator 包装成四条并列论文 Insight。

README 是仓库入口；上面五份文档是研究内容的唯一文档层次。代码、合同、脚本和测试的职责以仓库目录及其 README 为准，实验设计文档只引用它们，不再维护另一套路线文档。

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
