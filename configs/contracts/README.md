# 合同索引

本目录保存当前模型、数据与评测所依赖的冻结合同，以及仍需解释保留证据的历史协议。
活动配置从[配置索引](../README.md)进入，不能仅按合同文件名判断当前版本。
合同中的 `prospective`、资源预算和旧阶段状态保留封存原文；实际完成状态由对应结果索引维护。

| 范围 | 当前入口 |
| --- | --- |
| Yambda 人口、时间窗口与共享数据 | `yambda500m_scale_population_v1.yaml`、`yambda500m_streaming_windows_v1.yaml`、`yambda500m_unified_scales_v1.yaml` |
| Medium / Large / Max 的训练与 Full-only | [统一训练配置](../unified_training_2026_09/README.md)及其 `frozen_parent` 合同 |
| 18个选定权重和训练窗口 | [模型清单](../../docs/unified_training_2026_09/model_versions.md) |
| 15条 Reuse 边、四种局部重算、当前 Q/H | [配置索引](../README.md)中的各独立计划与封存记录 |
| RecFlow 六层 A–F 六模型链 | [当前配置](../recflow/window_6l_expanded_u4096_seed17.json)；[协议](../../docs/recflow/plan.md) |

旧 Medium / Large 训练合同仍解释当前选定前缀及共享数据的来源；其中 D7 或旧方法字段
不代表活动实验。Small PRO合同和理论运算记录仅因原 Large 合同的哈希依赖保留。
旧 Large canonical与endpoint sweep已被本轮选定链替代，不再提供启动入口。
论文附录引用的10k用户／7144教师历史结果继续按原配置解释。

当前 Design 尚未实现和验证。旧六层原型不等于当前方法；历史合同中的拟合限制只约束原实验。
长训练和正式评价仍需对应资源检查与用户启动，旧合同存在不构成启动授权。
具体保留与恢复边界见[结果索引](../../results/README.md)。
