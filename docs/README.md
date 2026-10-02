# 文档地图

本页按问题提供入口。当前 18 个 Yambda 模型、15 条 Full/Reuse 边、四种局部重算、
Q-v5/H-v4 及 RecFlow 六模型链均已完成；论文 Design 尚待完整实现与验证。
当前Design 1目标为读时校正恢复≥80%、新增计算≤15%；方法选择与实施顺序见[开发计划](design/plan.md)。

| 要查的问题 | 主要文档 | 代码或结果 |
| --- | --- | --- |
| 数据如何处理、模型如何逐版训练 | [训练链路](unified_training_2026_09/README.md) | [配置](../configs/unified_training_2026_09/README.md)、[脚本](../scripts/README.md) |
| 18 个模型分别是什么、用了哪些日期 | [固定版本清单](unified_training_2026_09/model_versions.md) | [训练结果](../results/unified_training_2026_09/README.md) |
| Full/Reuse 差多少 | [评价定义](experimental_design.md) | [15 条边结果](../results/unified_reuse_2026_09/README.md) |
| 四种重算与 Q/H 做了什么 | [Motivation 索引](motivation_observations.md)、[Q/H 设置](design/read_correction_probe_2026_09.md) | [重算结果](../results/selective_recompute_2026_09/README.md)、[Q/H 结果](../results/read_correction_2026_09/motivation_final/README.md) |
| RecFlow 的数据、模型与限制 | [RecFlow 协议](recflow/plan.md) | [脚本](../scripts/recflow/README.md)、[结果](../results/recflow/README.md) |
| 论文设计、固定开发集合与已实现部分 | [设计入口](paper_design.md)、[开发计划与集合](design/plan.md) | [六模型开发配置](../configs/design/medium_v0_v5_development.json)、[Design 1 执行入口](../scripts/design_one/README.md)、[历史接口](../scripts/design/README.md) |
| 历史方案为什么改变 | [设计决策](design/iterations.md)、[训练记录](unified_training_2026_09/records.md) | [历史证据索引](../results/README.md) |
| 图在哪里、如何生成 | [图表索引](../figures/README.md) | 各图对应的生成器与点表 |
| 数据、指标和成本如何定义 | [实验设计](experimental_design.md) | 对应实验的冻结配置与原始评分 |
| 修改后应该检查什么 | [测试索引](../tests/README.md) | 与改动相关的局部检查 |

论文方法的叙述来源是 [paper/main.tex](../../paper/main.tex)。
历史原型、旧附录诊断和当前三规模 Motivation 各有自己的结果，不能互换。

## 文档维护

- README 提供当前状态、职责和有效入口；模型清单维护版本选择，结果目录维护指标与失败。
- plan 保留实际设置与未完成的问题；records/iterations 只保留影响解释或复现的决定。
- 更新原文并修正引用，重复过程说明合并到已有文档，不新增 cleanup 或整理报告。
- 已封存的报告、invalidation 和源码快照保留原始结论与哈希；导航注明它们的历史用途。
  `data/raw/` 随数据下载的说明作为上游来源快照，本仓库的数据操作以脚本和配置索引为准。
- 过时命令、已删图片和不存在的权重不作为可用入口。已有恢复限制见[结果索引](../results/README.md)。

研究与执行规则统一见 [AGENTS.md](../AGENTS.md)。
