# 本轮 Yambda 统一训练

**2026-09-19：Max V1已按新指令恢复，连续2 epochs，保留并评测两个端点；其余后续训练仍待指令。** 2026-09-18暂停点继续作为历史记录。 Medium/Large版本链及Max V0均已保留；Max 10%用户E14抽样AUC为0.653509。后续逐版训练、评测的安排和时间估算见[计划](plan.md)，等待用户恢复本方向。

本轮从2026-09-15开始规划，逐个完成Yambda的Medium、Large、Max训练与评价。当前[Medium V0–V5模型链](../../results/unified_training_2026_09/medium/README.md)已固定；V2–V5的四条新训练边均通过准入和相对AUC >1%目标，Large工作链也已整理；Max V0已完成并封存。仅评估E14。

## 入口

Large 已形成完整 [V0–V5工作链](../../results/unified_training_2026_09/large/README.md)：V0–V3历史复用，V4采用本轮2 epochs，V5按用户要求暂定1 epoch。V5原准入限制及不完整E14范围在链文档明确记录，2 epochs备选保留。

- [训练计划](plan.md)：三档规模、数据与成本估算、研究原则和待定设置；本轮计划的维护入口。
- [讨论与执行记录](records.md)：按日期追加决定、数据准备进展、探针和运行结论。
- [配置目录](../../configs/unified_training_2026_09/README.md)：本轮开发配置与后续运行配置。
- [结果目录](../../results/unified_training_2026_09/README.md)：本轮数据统计、探针、训练与评价结果。

旧模型、合同及结果作为历史参考，保留原位置；新结果使用本轮目录，明确绑定配置、数据和模型版本。
