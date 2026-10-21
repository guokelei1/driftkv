# 配置入口

2026-10-19本轮实际执行配置位于[system_study_2026_10/](system_study_2026_10/)，
当前正式实验配置与结果集中在[experiments/](../experiments/README.md)，目标和恢复状态见其中的README与STATUS；本目录保留已有配置。
下文“当前”配置说明属于旧研究阶段，不覆盖本轮入口。

本目录保存实际执行配方与冻结合同。当前三规模训练、15边Full/Reuse、
四重算、Q/H及RecFlow A–F链均已完成；配置中的原始阶段名和状态属于当时的执行记录。
版本选择以模型清单为准，现存配置不代表等待启动的任务。

| 用途 | 入口 |
| --- | --- |
| Yambda 数据处理、三规模训练与逐版 Full-only | [unified_training_2026_09/README.md](unified_training_2026_09/README.md) |
| 当前18个模型端点 | [固定版本清单](../docs/unified_training_2026_09/model_versions.md) |
| 15条相邻 Reuse 边 | [adjacent_e14.json](unified_reuse_2026_09/adjacent_e14.json) |
| 四种局部重算 | [selective_recompute_2026_09/plan.json](selective_recompute_2026_09/plan.json) |
| 当前 Q-v5 / H-v4 读取修正 | [结果、运行计划与保留来源](../results/read_correction_2026_09/motivation_final/README.md) |
| 当前三部分Design的开发输入 | [Medium V0–V5 配置](design/medium_v0_v5_development.json)、[历史UID 名单](design/medium_v0_v5_users.json)；当前冻结面板、质量/成本口径见[计划](../docs/design/plan.md) |
| RecFlow 六层 A–F 六模型链 | [window_6l_expanded_u4096_seed17.json](recflow/window_6l_expanded_u4096_seed17.json) |
| 冻结合同与历史依赖 | [contracts/README.md](contracts/README.md) |

当前实验及论文证据引用的合同保持原样；修改新实验设置时使用对应流程的配置入口。
旧 Small、D7 和 Large endpoint sweep 不是活动配置入口。
10k用户／7144教师结果因论文附录引用继续保留，不取代当前三规模 Motivation 设置。

当前实现为local-KV32＋joint16和继承flag限定的按需code维护；固定质量面板为
4096用户/51,731请求，完整成本使用独立30k人口事件账。256拟合/16验证用户与评价用户分离。
10k用户、12,699人开发池及theta3属于历史探索，不替代当前输入或作为待执行队列。
当前候选仍是开发证据，尚无独立确认；没有新训练或人口实验排队。
新长训练及正式人口实验遵守仓库授权边界，RecFlow角色与数据访问按其独立协议执行。
