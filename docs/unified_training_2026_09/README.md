# 本轮 Yambda 统一训练

**2026-09-23：Max已完成V1–V5的逐版开发训练与评测。**
V5从选定的V4 epoch2训练两轮，两个端点均已在E14 [287,301)评价；端点选择与准入记录见
[Max运行与结果](../../results/unified_training_2026_09/max/README.md)。早期暂停及启动决定保留为历史记录。

本轮从2026-09-15开始规划，逐个完成Yambda的Medium、Large、Max训练与评价。当前[Medium V0–V5模型链](../../results/unified_training_2026_09/medium/README.md)已固定；V2–V5的四条新训练边均通过准入和相对AUC >1%目标，Large工作链也已整理。各规模仅评估既定E14窗口。

## 入口

[三规模模型版本清单](model_versions.md)汇总各版选用权重、训练天数和相对 AUC 提升。

Large 已形成完整 [V0–V5工作链](../../results/unified_training_2026_09/large/README.md)：V0–V3历史复用，V4采用本轮2 epochs，V5按用户要求暂定1 epoch。V5原准入结果与备选端点评测仍在链文档中，备选权重已按清单清理。

- [训练计划](plan.md)：三档规模、数据与成本估算、研究原则和待定设置；本轮计划的维护入口。
- [讨论与执行记录](records.md)：按日期追加决定、数据准备进展、探针和运行结论。
- [配置目录](../../configs/unified_training_2026_09/README.md)：本轮开发配置与后续运行配置。
- [结果目录](../../results/unified_training_2026_09/README.md)：本轮数据统计、探针、训练与评价结果。

旧模型、合同及结果作为历史参考，保留原位置；新结果使用本轮目录，明确绑定配置、数据和模型版本。
