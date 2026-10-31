# Yambda 三规模训练与固定模型

Medium、Large、Max 各 V0–V5，共 **18 个选定模型**，训练与 Full 评测已完成。
15 条相邻边的 Reuse 也已完成。Medium/Large 使用 Yambda-500M，Max 使用 Yambda-5B；
三个规模均采用 seed17、context1024 和既定 E14 评价窗口。

| 查找内容 | 入口 |
| --- | --- |
| 各版权重、训练天数、epoch、batch、相对 AUC 提升 | [模型版本清单](model_versions.md) |
| 三规模最终数据与训练配方 | [训练设置](plan.md) |
| 端点选择、训练分支、数值检查及恢复记录 | [关键记录](records.md) |
| 可复用的最终执行配置 | [配置目录](../../configs/unified_training_2026_09/README.md) |
| Full 指标、原始评分、封存和逐版结果 | [训练结果](../../results/unified_training_2026_09/README.md) |
| 15 条边的 Parent Full / Current Full / Reuse | [Reuse 结果](../../results/unified_reuse_2026_09/README.md) |

Large V5 使用用户选定的 epoch1；其原四项准入未全部通过，原判定保留。
Max V5 使用 epoch2。端点清单是当前模型使用入口；保存的旧合同和分支记录描述原实验，
不代表仍有待执行任务，也不代表清单外权重仍可用。
