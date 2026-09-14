# 脚本入口

此目录保留数据处理、六层/十层训练评估、Motivation、Insight 和历史六层适配探索的流程。当前 Design 已完成设计，尚未实现和验证；历史脚本不代表当前设计已可执行。正式运行仍需对应合同、canary 与用户授权。

| 用途 | 入口 |
| --- | --- |
| 六层 Medium 训练与 Full/Reuse 矩阵 | `run_yambda500m_medium_full_reuse_matrix.py` |
| Medium V5 扩展 | `run_yambda500m_medium_d14_v5_extension.py` |
| Motivation 完整旧 producer 对比 | `run_yambda500m_medium_d14_direct_long_age_reuse.py` |
| 十层 Large 基础训练和 Full-only 队列 | `run_yambda500m_large_qualification.py` |
| Large V4/V5 训练轨迹 | `run_yambda500m_large_v3_v4_epoch_sweep.py`；`run_yambda500m_large_v4e2_to_v5_epoch_sweep.py` |
| Insight 1 局部替换诊断 | [insight_one_locality/README.md](insight_one_locality/README.md) |
| Insight 2 响应修正和持续性诊断 | [insight_two/README.md](insight_two/README.md) |
| 六层适配原型、校准与连续评价 | [design/README.md](design/README.md) |
| 论文图片 | [../figures/README.md](../figures/README.md) |

数据入口为 `download_scale_datasets`、`prepare_yambda500m_scale_populations`、`plan_yambda500m_streaming_windows` 和 `build_yambda500m_unified_scales`。manifest、训练器与 evaluator 必须显式接收合同、输出路径和数据 manifest。

旧 Design 2/3 的实验入口已删除。历史删除范围与恢复限制见 [结果索引](../results/README.md)。
