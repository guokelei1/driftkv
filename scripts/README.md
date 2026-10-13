# 当前脚本入口

只将当前方法、Motivation/模型链和必要公共依赖作为活动代码。
已完成失败分支的专用入口退出；历史结论见[结果索引](../results/README.md)。

## 当前Design

| 用途 | 入口 |
| --- | --- |
| local-KV32＋joint16发布前校准 | [calibrate_lightweight_local_kv.py](design_system/calibrate_lightweight_local_kv.py) |
| 既有依赖flag限定row code的连续评测 | [evaluate_lightweight_support_mask.py](design_system/evaluate_lightweight_support_mask.py) |
| 保存分数的配对分析 | [analyze_lightweight.py](design_system/analyze_lightweight.py) |
| 当前按需状态与维护 | [lightweight_lazy.py](design_system/lightweight_lazy.py) |
| 完整计算量与一次重建口径 | [lightweight_budget.py](design_system/lightweight_budget.py)、[once_rebuild_accounting.py](design_system/once_rebuild_accounting.py) |
| 固定真实人口的事件形状和按需成本 | [profile_lightweight_lazy.py](design_system/profile_lightweight_lazy.py)、[共享事件计数](design_system/profile_costs.py) |
| 冻结面板和共享教师采集 | [design_two/prepare.py](design_two/prepare.py)、[capture.py](design_two/capture.py)、[common.py](design_two/common.py) |

主评价输入为`results/design_system_2026_10/phase2/representative4096/panel/`，
匹配参照分数为同级`development4096/scores.parquet`；原始校准权重位于
`results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration/`。
最新门控候选复用这些权重，不重复拟合。方法和角色定义见[计划](../docs/design/plan.md)。

`design_system/`其余校准/评测文件提供上述入口实际调用的共享步骤。
旧Value-only、decoder-only和full-item校准文件只保留归一化、教师行采集、联合拟合等
当前使用的函数，旧CLI已移除；文件名不表示仍有这些实验队列。
`design_one/`只保留历史装载、数据核对与配对指标工具。
`design_two/`保留持续轨迹、捕获和事件计数依赖；它们的目录名不表示旧任务强度方案仍被选中。
共享legacy代码保留是因为当前导入链仍需要它，不作为旧实验重跑队列。

## 模型、数据与Motivation

| 用途 | 入口 |
| --- | --- |
| 原始数据下载 | [download_scale_datasets.py](download_scale_datasets.py) |
| Yambda人口、窗口、共享数据 | [prepare_yambda500m_scale_populations.py](prepare_yambda500m_scale_populations.py)、[plan_yambda500m_streaming_windows.py](plan_yambda500m_streaming_windows.py)、[build_yambda500m_unified_scales.py](build_yambda500m_unified_scales.py) |
| 请求清单 | [build_yambda500m_hstu_native_matrix_manifest.py](build_yambda500m_hstu_native_matrix_manifest.py) |
| 基础训练及固定版本执行 | [train_yambda500m_foundation_fsdp.py](train_yambda500m_foundation_fsdp.py)、[unified_training/](unified_training/) |
| 已完成Full与相邻Reuse结果、公共读写工具 | [unified_reuse_2026_09/](unified_reuse_2026_09/README.md) |
| 四种局部重算 | [selective_recompute_2026_09/](selective_recompute_2026_09/) |
| Q-v5/H-v4数值实现 | [read_correction_v5/](read_correction_v5/)、[read_correction_v4/](read_correction_v4/) |
| RecFlow数据、六模型链 | [recflow/](recflow/) |
| 论文图生成 | [图表索引](../figures/README.md) |

完成实验的临时launcher、GPU探针及85份执行源码复制已清理。重新测量基线需要先准备新实验输入；
保留方法实现不表示历史已删除的校准权重或历史包仍在。不要自动续跑已完成或停止的实验。
