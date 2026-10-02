# Yambda 三规模训练结果

Medium、Large、Max 各 V0–V5，共 **18 个选定模型**，训练和 Full 评价已完成。
权重统一保存在各规模的 `seed17/checkpoints/v0..v5/checkpoint_100.pt`。
各版数据窗口、epoch、batch 与相对 AUC 提升统一见
[模型版本清单](../../docs/unified_training_2026_09/model_versions.md)。

| 规模 | 最终选择 | 结果入口 |
| --- | --- | --- |
| Medium：6L/H192，30,000 用户 | V0/V1 历史复用；V2 两轮，V3–V5 各一轮 | [模型与 Full](medium/README.md) |
| Large：10L/H320，79,681 用户 | V0–V3 历史复用；V4 两轮，V5 一轮 | [模型、Full 与原准入](large/README.md) |
| Max：16L/H320，200,000 用户 | V0–V2 一轮；V3–V5 两轮 | [数据、模型与 Full](max/README.md) |

Large V5@1 的原四项准入未全部通过；用户选定该端点用于当前实验。
其余端点选择和历史比较保持原记录，不把模型清单等同于自动服务推广。

## 训练代码与证据

- [训练设置](../../docs/unified_training_2026_09/plan.md)、
  [关键训练记录](../../docs/unified_training_2026_09/records.md)、
  [执行配置](../../configs/unified_training_2026_09/README.md)。
- 原运行目录保存合同、配置、训练结果、Full 原始评分、封存和 adjudication；
  选定权重的旧入口通过相对符号链接兼容。
- 清单外权重和优化器恢复 payload 已删除，保留的分支评测不表示其权重仍存在。
  Max V3 的训练来源 V2@2 也已删除，精确重放该历史训练需先重建此端点。
- 论文仍引用的历史 Medium D14 V2–V5 实体权重另行保留；
  它们与当前 Medium 同名版本不同，按目录和哈希区分。
- 日志归档和历史资产的可用范围见[结果总索引](../README.md)。

## 后续评价

选定模型的 [15 条相邻 Full/Reuse 边](../unified_reuse_2026_09/README.md)已经完成，
是当前 Motivation 的模型输入；其余实验入口见[文档地图](../../docs/README.md)。
