# 本轮统一训练结果

2026-09-23：Medium/Large工作链已保留；Max已完成V1–V5的逐版开发训练与评测，
V5两个端点均已评价。各规模的checkpoint选择与准入结果见规模索引。

计划入口：[Yambda 三档训练计划](../../docs/unified_training_2026_09/plan.md)。Medium 最终 V0–V5 已固定。

| 规模 | 当前状态 | 入口 |
| --- | --- | --- |
| Medium：6L/H192，30,000 用户 | V0–V5已固定；四条新训练边均通过E14准入和相对AUC >1%目标 | [模型链及结果](medium/README.md) |
| Large：10L/H320，79,681 用户 | V0–V5工作链已整理；V4为2 epochs，V5暂定1 epoch | [版本链、选择状态及结果](large/README.md) |
| Max：16L/H320，200,000 用户 | V0–V5已有端点与评测；用户选定通过准入的V5@2 | [版本选择、运行及结果](max/README.md) |

2026-09-23按[三规模版本清单](../../docs/unified_training_2026_09/model_versions.md)整理权重：
各规模仅保留选定的六个实体 checkpoint，Max 统一置于 `max/seed17/checkpoints/v0..v5/`；
清单外的8个实体权重和5份优化器恢复目录已删除。运行日志与状态文件已校验归档至
`nogit/archives/yambda_unified_training_20260923_logs.tar.gz`。原合同、原始分数、封存、
adjudication、汇总和失败结果仍保留；历史记录中“备选权重保留”的描述以本次清理状态为准。
另据论文第三章所用图表核对，旧 Medium D14 Motivation 六版路径保留；其余34份
与第三章无直接关系的旧 Medium/Large 实体权重已删除，6份同哈希旧副本改为选定权重的符号链接。
旧目录301份运行日志/状态文件归档至`nogit/archives/yambda_historical_training_20260923_logs.tar.gz`。
