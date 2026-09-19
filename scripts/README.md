# 脚本入口

此目录保留数据处理、六层/十层训练评估、Motivation、Insight 和历史六层适配探索的流程。当前 Design 已完成设计，尚未实现和验证；历史脚本不代表当前设计已可执行。正式运行仍需对应合同、canary 与用户授权。

| 用途 | 入口 |
| --- | --- |
| RecFlow 独立数据准备与开发探针 | [recflow/README.md](recflow/README.md) |
| 本轮六层 V1 → V2 两 epoch 训练与 Full-only | `unified_training/run_medium_v2_2epoch.py` |
| 六层 Medium 训练与 Full/Reuse 矩阵 | `run_yambda500m_medium_full_reuse_matrix.py` |
| Medium V5 扩展 | `run_yambda500m_medium_d14_v5_extension.py` |
| Motivation 完整旧 producer 对比 | `run_yambda500m_medium_d14_direct_long_age_reuse.py` |
| 十层 Large 基础训练和 Full-only 队列 | `run_yambda500m_large_qualification.py` |
| Large V4/V5 训练轨迹 | `run_yambda500m_large_v3_v4_epoch_sweep.py`；`run_yambda500m_large_v4e2_to_v5_epoch_sweep.py` |
| Insight 1 局部替换诊断 | [insight_one_locality/README.md](insight_one_locality/README.md) |
| Motivation 2（原 Insight 1）：当前 6L／10L 的三个对比方案 | [design/run_insight1_competitors.py](design/run_insight1_competitors.py)，[接口与用法](design/README.md) |
| Insight 2 响应修正和持续性诊断 | [insight_two/README.md](insight_two/README.md) |
| 六层适配原型、校准与连续评价 | [design/README.md](design/README.md) |
| 论文图片 | [../figures/README.md](../figures/README.md) |

数据入口为 `download_scale_datasets`、`prepare_yambda500m_scale_populations`、`plan_yambda500m_streaming_windows` 和 `build_yambda500m_unified_scales`。manifest、训练器与 evaluator 必须显式接收合同、输出路径和数据 manifest。

旧 Design 2/3 的实验入口已删除。历史删除范围与恢复限制见 [结果索引](../results/README.md)。

## Max 统一训练准备入口

- `unified_training/prepare_max_data.py`：5B/20万用户处理，复用历史选择规范。
- `unified_training/audit_max_data.py`：人口、映射、原始行数、请求因果性及真实历史对照。
- `unified_training/probe_max_resources.py`：四卡短资源探针。
- `unified_training/run_max_v0.py`：准备好的V0单epoch入口；`--print-command`只检查命令，正式执行要求已记录用户明确启动。
- `unified_training/run_max_v0.py --resume`：使用最新完整恢复点继续同一四卡任务；自动保留原运行日志。
- `unified_training/check_max_recovery.py`：四卡合成数据的连续训练/跨进程恢复对照与保存成本检查。
- `unified_training/evaluate_max_v0_sample.py prepare|canary|evaluate`：固定20,000/200,000用户的V0 E14抽样Full评价；复用现有打分和AUC实现，先封存原始分数再关联标签。单模型无需构造虚假的Parent/Current比较。

设置、预算和当前准备状态见[Max记录](../results/unified_training_2026_09/max/README.md)。

- `unified_training/prepare_max_v1.py`：固定V0接续训练与三模型Full评测的资源canary；生成运行所需通过记录和预算。
- `unified_training/launch_max_v1_epochs12.sh`：调用现有版本链runner，在后台连续训练Max V1两轮，保留两个端点并完成同窗口Full评测。
