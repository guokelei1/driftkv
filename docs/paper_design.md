# EvoKV 设计入口

论文叙述的唯一来源是 [paper/main.tex](../../paper/main.tex)。
当前方法为 **Cross-Version Cache Adaptation**；当前 Design 已完成设计，尚未实现和验证。
仓库中的摘要、Translator、native 响应修正和历史连续执行器可供复用，不证明当前方法有效。

## 设计与实现职责

| 职责 | 实现时要回答的问题 |
| --- | --- |
| 写入摘要 | 如何从真实持久缓存维护紧凑状态，并正确处理追加和淘汰？ |
| 发布翻译 | 如何在独立校准数据上学习共享版本转换，明确教师访问和成本？ |
| 读取修正 | 如何在当前查询路径中应用兼容性修正，保持原模型的读取语义？ |
| 状态维护 | 如何处理多 producer、重复发布、旧状态与后代继承误差？ |

具体算式、章节结构和论文机制以正文为准；本文件不维护第二份 Design 草稿或旧章节编号。
四个接口需一起构成首个可运行原型，研究目标是以较小的 Exact-All 成本恢复实质质量缺口。

## 工作入口

- [适配开发计划](design/plan.md)：冻结六层范围、接口和下一步。
- [实验设计](experimental_design.md)：资产入口、因果数据、对照、评价及成本定义。
- [决策记录](design/iterations.md)：历史方案变化、失败原因及对应证据。
- [对比方案](../src/hstu_kvcache/baselines/README.md)：按层重算、尾部重算、K/V 翻译。
- [模型训练](unified_training_2026_09/README.md)：Medium、Large、Max 逐版状态。
- [Motivation 与 Insight](motivation_observations.md)：测量及其解释范围。

## 历史证据边界

旧 native C、均值/PCA 条件修正和 paired-summary 原型属于不同探索阶段，
不把其结构、超参数、旧“Design 1冻结”或同步执行状态视为当前设计承诺。
[质量报告](../results/design/analysis/native_base_quality4091_01_report/report.md)、
[FLOPs 报告](../results/design/analysis/native_flops_01/report.md)和
[机制证据](design/expert_route_2026-09-07.md)仍保留；
单边质量、教师干预、理论人口摊销均不建立完整连续迁移或实际加速结论。

2026-09-22 移除本页重复的旧实现叙述与论文章节映射。修改前原文及旧技术草稿可从
[文档快照](../results/README.md#2026-09-22-文档精简)查阅；结果和冻结协议未改。
