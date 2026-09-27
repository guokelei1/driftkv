# 适配探索与对比脚本

本目录包含已完成的AUC/功能诊断、公共原语与历史native适配流水线。
当前Design尚未实现和验证；接口计划见[适配计划](../../docs/design/plan.md)，
方案变化见[决策记录](../../docs/design/iterations.md)。结果和运行预算在各实验目录维护。

## 统一 AUC 诊断入口

| 用途 | 输入准备与执行 | 设置、结果 |
| --- | --- | --- |
| 6L四边、原四预算 | `prepare_unified_auc_inputs.py`；`run_unified_auc_parallel.py`；核心 `run_unified_auc.py` | [原始结果](../../results/insight/unified_auc_10k_20260920/README.md) |
| 6L九档教师预算 | `prepare_expanded_auc_inputs.py`；`run_expanded_auc.py` | [扩展结果](../../results/insight/unified_auc_10k_teachers7144_20260920/README.md) |
| 10L五边 | `prepare_large_auc_inputs.py`；`run_large_auc_parallel.py`；`run_large_auc.py` | [配置](../../configs/insight/large_unified_auc_10k_20260920.json)、[结果](../../results/insight/large_unified_auc_10k_20260920/README.md) |
| 16L两边 | `prepare_max_auc_inputs.py`；`run_max_auc.py` | [配置](../../configs/insight/max_unified_auc_10k_20260921.json)、[结果](../../results/insight/max_unified_auc_10k_20260921/README.md) |

`launch_large_auc.sh`、`launch_max_auc.sh`保留主诊断和退出状态记录，
旧自动出图步骤已移除。以上运行已完成，路径不是待执行队列；不要覆盖既有结果。
保留图表、ledger与显示口径见[图表索引](../../figures/README.md)。

公共依赖包括 `competitor_data.py`、`competitor_models.py`、
`unified_auc_baselines.py`、`expanded_read_calibration.py`、
`large_auc_primitives.py`、`max_auc_models.py`；它们仍会导入旧probe中的公共函数。
`unified_auc_cost.py`、`large_auc_cost.py`、`max_auc_cost.py`提供理论成本。
`check_large_expanded_auc.py`核对校准一致性，`probe_large_auc_ridge.py`是独立pilot；
不要因文件名历史化而删除调用依赖。

## 六层共享读取修正：已完成的概率保真度诊断

`run_query_read_probe.py`为执行入口，`query_read_probe.py`实现
共享 `b+Aq` 与 `b+Aq+Tr`，`analyze_query_read_probe.py`读取元数据分析费用。
[配置](../../configs/insight/query_read_6l_2560_20260920.json)和
[完整结果](../../results/insight/query_read_6l_2560_20260920/README.md)维护UID划分、指标及成本。

必须传入 `--config` 和新 `--output`；可用 `--canary`、`--pilot`、
`--calibration-from`、`--device`、`--batch-size`、`--threads`。
扩大评价从pilot参数原样复用，校验来源hash；分batch释放缓存。
该诊断是静态概率保真度，不是推荐AUC、摘要必要性或连续状态结论。

前序[摘要诊断](../../results/insight/shared_read_6l_20260920/README.md)、
[item/response诊断](../../results/insight/user_information_6l_20260920/README.md)
及source_snapshot保留。相关runner中的公共函数仍被AUC流程引用；
历史表格生成器已清理，不能因改图重跑实验。

## 独立 Motivation 2 功能指标接口

`run_insight1_competitors.py`是单边入口，`competitor_probe.py`统一执行
LR/TR/KT并评分。方法公式与限制见[baselines](../../src/hstu_kvcache/baselines/README.md)。
`--scale medium|large`选择本轮模型链，`--edge-index 0..4`选择边；
`--describe`仅查看元数据，不加载权重或用户历史：

```bash
PYTHONPATH=src:scripts python scripts/design/run_insight1_competitors.py --scale medium --edge-index 1 --describe
```

实际评价需提供新 `--output` 和 `--uids` JSON，键为 `evaluation`、`fit`、
可选 `selection`；组内唯一、组间互斥，来自所选规模人口。KT拟合在CPU，
`--device`控制模型、缓存和评分。`selection`省略时源层排名标为in-sample。
`--layer-intervals`使用从0开始、两端包含的层号，`--tail-lengths`、`--map-ks`、
`--ridge`、`--batch-size`按该次协议显式选择，具体可用参数以CLI为准。

64候选bank来自全部评价UID的发布前历史，使用各自词表的recent/old-only/novel规则；
`--max-users`、offset和batch不能改变bank。Full-only准入按所选边核验：
Large V5@1的原失败不因开发诊断变成已准入，也不自动换V5@2。

输出原始分数、指标及配置/来源hash。概率缺口恢复用均值之比，
不是逐用户比值均值；K/V更新比例不是FLOPs。这个接口不同于上面的真实反馈AUC实验，
且不建立完整连续迁移。数据、公式和入口检查见[测试索引](../../tests/README.md)。

## 历史适配原型与清理范围

`data.py`、`run.py`、`diagnose_query_holdout.py`、
`diagnose_summary_objective.py`、`diagnose_decoder_closure.py`保留场景和共享执行原语。
`diagnose_native_input.py`、`diagnose_native_coverage.py`、`fit_native_ablation.py`、
`native_service.py`保留旧native C/消融路径；`evaluate_native_base.py`、
`run_native_base.py`及相关report/audit读取历史结果。
旧query-affine拟合已退休，runner只保留通过 `--evaluation-from`读取既有权重的评价路径。

`docs/design/expert_route_2026-09-07.md`仍由历史runner纳入hash，不能只按文档长度删除。
2026-09-22删除的四个无调用诊断入口及源码快照见
[清理记录](../../results/README.md#2026-09-22-入口清理)；旧结果、合同和哈希未改。
