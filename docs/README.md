# 文档地图

论文设计以 [论文正文](../../paper/main.tex) 为唯一叙述来源。当前 Design 已完成设计，尚未实现和验证；仓库保留模型训练、历史适配探索和 Motivation/Insight 的文档与代码供后续复用。

| 要解决的问题 | 入口 |
| --- | --- |
| 当前论文方法 | [论文正文](../../paper/main.tex) |
| 设计入口与实现边界 | [paper_design.md](paper_design.md) |
| 适配原型的计划与迭代记录 | [design/plan.md](design/plan.md)；[design/iterations.md](design/iterations.md) |
| 三个对比方案的设计、CPU 原型及 Insight 1 接口 | [按层重算、尾部重算、直接 K/V 翻译](../src/hstu_kvcache/baselines/README.md) |
| 数据因果、准入、评价与成本定义 | [experimental_design.md](experimental_design.md) |
| 本轮 Yambda 三档统一训练 | [计划与记录](unified_training_2026_09/README.md) |
| RecFlow 当前协议与决策 | [开发计划](recflow/plan.md)；[完整结果](../results/recflow/README.md) |
| 已有 Motivation 与 Insight 证据 | [motivation_observations.md](motivation_observations.md) |
| Medium 训练与评估链 | [本轮模型链](../results/unified_training_2026_09/medium/README.md) |
| Large Full-only 与工作链 | [版本选择及准入限制](../results/unified_training_2026_09/large/README.md) |
| 结果、删除范围与恢复限制 | [结果索引](../results/README.md) |
| 图表生成器与来源 | [图表索引](../figures/README.md) |
| 可执行入口 | [脚本索引](../scripts/README.md) |

开发和实验遵循 [AGENTS.md](../AGENTS.md) 的研究原则：优先最小可验证实验，保留因果、评价和证据边界。

一处维护一类信息：论文写方法，计划写当前设置和下一问题，迭代记录写独有决策，
结果目录写完整指标与失败，脚本README写用法。导航不复制结果表或已完成任务的启动指令。
