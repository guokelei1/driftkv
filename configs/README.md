# 配置入口

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
| Design 1/2/3 的固定开发输入 | [Medium V0–V5 配置](design/medium_v0_v5_development.json)、[UID 名单](design/medium_v0_v5_users.json)；最新质量/成本目标与开放方法选择见[计划](../docs/design/plan.md) |
| RecFlow 六层 A–F 六模型链 | [window_6l_expanded_u4096_seed17.json](recflow/window_6l_expanded_u4096_seed17.json) |
| 冻结合同与历史依赖 | [contracts/README.md](contracts/README.md) |

当前实验及论文证据引用的合同保持原样；修改新实验设置时使用对应流程的配置入口。
旧 Small、D7 和 Large endpoint sweep 不是活动配置入口。
10k用户／7144教师结果因论文附录引用继续保留，不取代当前三规模 Motivation 设置。

当前 Cross-Version Cache Adaptation 尚待完整实现和验证；已固定六模型链、Design 1 的10,000名
有请求用户及连续阶段12,699人开发池，见[设计计划](../docs/design/plan.md)。
小规模探索记录必要设置；长训练与正式评价需要前瞻配置、资源估计、focused canary及用户启动。
用户已撤销笼统 target-KV fitting 禁令；旧限制只解释各自的历史实验。
theta3及RecFlow用户角色和数据访问边界按各自协议执行。
