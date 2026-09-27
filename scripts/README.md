# 脚本入口

此目录保留数据处理、六层/十层/十六层训练评估、Motivation、Insight 和历史六层适配探索的流程。当前 Design 已完成设计，尚未实现和验证；历史脚本不代表当前设计已可执行。正式运行仍需对应合同、canary 与用户授权。

| 用途 | 入口 |
| --- | --- |
| RecFlow 独立数据准备与开发探针 | [recflow/README.md](recflow/README.md) |
| HSTU 算子数值／性能、旧权重与两卡训练检查 | [benchmark_hstu_backends.py](benchmark_hstu_backends.py)；[check_hstu_backend_fsdp.py](check_hstu_backend_fsdp.py)，[测量范围](../results/backend_acceleration/2026_09_19/README.md) |
| 本轮六层 V1 → V2 两 epoch 训练与 Full-only | `unified_training/run_medium_v2_2epoch.py` |
| 六层 Medium 训练与 Full/Reuse 矩阵 | `run_yambda500m_medium_full_reuse_matrix.py` |
| 当前三规模选定链的15条相邻 Reuse 边 | [独立入口与说明](unified_reuse_2026_09/README.md) |
| Medium V5 扩展 | `run_yambda500m_medium_d14_v5_extension.py` |
| Motivation 完整旧 producer 对比 | `run_yambda500m_medium_d14_direct_long_age_reuse.py` |
| 十层 Large 基础训练和 Full-only 队列 | `run_yambda500m_large_qualification.py` |
| Large V4/V5 训练轨迹 | `run_yambda500m_large_v3_v4_epoch_sweep.py`；`run_yambda500m_large_v4e2_to_v5_epoch_sweep.py` |
| Insight 1 局部替换诊断 | [insight_one_locality/README.md](insight_one_locality/README.md) |
| Motivation 2（原 Insight 1）：当前 6L／10L 的三个对比方案 | [design/run_insight1_competitors.py](design/run_insight1_competitors.py)，[接口与用法](design/README.md) |
| 原四预算记录：6L、10000用户、四边统一真实反馈AUC | [四GPU入口](design/run_unified_auc_parallel.py)，[48线程输入准备](design/prepare_unified_auc_inputs.py)，[结果与协议](../results/insight/unified_auc_10k_20260920/README.md) |
| 当前完成：共享读取九档教师预算至7144，评价10k不变 | [扩展入口](design/run_expanded_auc.py)，[固定bank输入准备](design/prepare_expanded_auc_inputs.py)，[完整结果](../results/insight/unified_auc_10k_teachers7144_20260920/README.md) |
| 历史 6L Insight：256 校准／2560 评价的概率保真度 | [design/run_query_read_probe.py](design/run_query_read_probe.py)，[配置与方法](design/README.md#六层共享读取修正已完成的概率保真度诊断)，[结果与成本](../results/insight/query_read_6l_2560_20260920/README.md) |
| 历史 Insight 2 响应修正和持续性诊断 | [insight_two/README.md](insight_two/README.md) |
| 六层适配原型、校准与连续评价 | [design/README.md](design/README.md) |
| 论文图片 | [../figures/README.md](../figures/README.md) |

数据入口为 `download_scale_datasets`、`prepare_yambda500m_scale_populations`、`plan_yambda500m_streaming_windows` 和 `build_yambda500m_unified_scales`。manifest、训练器与 evaluator 必须显式接收合同、输出路径和数据 manifest。

旧 Design 2/3 的实验入口已删除。历史删除范围与恢复限制见 [结果索引](../results/README.md)。

## Max 统一训练入口

- `unified_training/prepare_max_data.py`：5B/20万用户处理，复用历史选择规范。
- `unified_training/audit_max_data.py`：人口、映射、原始行数、请求因果性及真实历史对照。
- `unified_training/probe_max_resources.py`：四卡短资源探针。
- `unified_training/run_max_v0.py`：准备好的V0单epoch入口；`--print-command`只检查命令，正式执行要求已记录用户明确启动。
- `unified_training/run_max_v0.py --resume`：使用最新完整恢复点继续同一四卡任务；自动保留原运行日志。
- `unified_training/check_max_recovery.py`：四卡合成数据的连续训练/跨进程恢复对照与保存成本检查。
- `unified_training/evaluate_max_v0_sample.py prepare|canary|evaluate`：固定20,000/200,000用户的V0 E14抽样Full评价；复用现有打分和AUC实现，先封存原始分数再关联标签。单模型无需构造虚假的Parent/Current比较。

设置、预算和逐版运行状态见[Max记录](../results/unified_training_2026_09/max/README.md)。

- `unified_training/prepare_max_v1.py`：Max接续训练与Full评测的资源canary；`--execution-config`选择版本，生成运行所需通过记录和预算。
- `unified_training/launch_max_v1_epochs12.sh`：调用现有版本链runner，在后台连续训练Max V1两轮，保留两个端点并完成同窗口Full评测。
- `unified_training/check_max_backend.py`：所选Max权重的16层FP32/BF16前向与反向Torch/Triton对照；`--expected-epochs`核对父端点，合成输入，不读取质量标签。
- `unified_training/prepare_max_profile_fixture.py`、`profile_max_training.py`、`summarize_max_training_profile.py`、`analyze_max_profile_attribution.py`：真实Max V2训练batch的CPU准备、四卡完整step对照及只读trace分析；[测量范围与结果](../results/backend_acceleration/max_profile_2026_09_19/README.md)。已完成的训练间隙调度与2026-09-21专用恢复脚本已退休，操作证据保留。
- `unified_training/launch_max_v2_1epoch.sh`：从用户选定的V1 epoch1接续V2一轮，四卡global80，使用auto后端，随后进行完整E14父子评测。
- `unified_training/launch_max_v2_epoch2.sh`：从保留V2 epoch1权重再增加一轮；明确记录AdamW/RNG重置和累计epoch2，保留最终optimizer恢复点，随后评价原V1基线并报告对epoch1的描述性变化。
- `unified_training/launch_max_v3_epochs12.sh`：从选定V2 epoch2初始化V3，四卡global80连续两轮、保存两个端点及恢复点，再统一完整E14评测。
- `unified_training/launch_max_v3_from_v2e1_epochs12.sh`：独立替代分支，从V2 epoch1初始化V3，连续两轮并统一评测；原V2 epoch2初始化分支完整保留。
- `unified_training/launch_max_v4_epochs12.sh`：从选定V3 epoch2初始化V4，四卡连续两轮，保留两个端点及恢复点，训练后统一完整E14评测。
- `unified_training/launch_max_v5_epochs12.sh`：从选定V4 epoch2初始化V5，四卡连续两轮，保存两个端点和恢复点；统一评价E14 [287,301)。

## 四种局部重算 baseline（2026-09-26）

[selective_recompute_2026_09/](selective_recompute_2026_09/README.md) 独立准备层、尾部、
偏差和 query 引导重算的15条边预算曲线；每边固定3k诊断用户，四卡共用因果回放、
按方法保存结果。提供准备、短 canary、断点恢复、完整成本和12张图的生成程序。
`launch.sh` 默认只预览；正式 tmux 运行等待用户明确 OK。
