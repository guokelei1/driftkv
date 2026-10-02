# 脚本入口

本目录维护数据处理、训练、评价与结果汇总入口；模型和数值原语放在 `src/`。
三规模18模型、15条Full/Reuse边、四种局部重算、当前Q/H及RecFlow六模型链已完成。
查已有结果先进入[结果索引](../results/README.md)，查模型选择先进入
[固定清单](../docs/unified_training_2026_09/model_versions.md)。保留脚本不代表需要重跑。

## 当前工作流

| 用途 | 入口 |
| --- | --- |
| 原始数据下载 | [download_scale_datasets.py](download_scale_datasets.py) |
| Yambda-500M 用户、窗口与共享数据 | [prepare_yambda500m_scale_populations.py](prepare_yambda500m_scale_populations.py)、[plan_yambda500m_streaming_windows.py](plan_yambda500m_streaming_windows.py)、[build_yambda500m_unified_scales.py](build_yambda500m_unified_scales.py) |
| HSTU 请求清单 | [build_yambda500m_hstu_native_matrix_manifest.py](build_yambda500m_hstu_native_matrix_manifest.py) |
| Max 5B / 20万用户数据处理与核验 | [prepare_max_data.py](unified_training/prepare_max_data.py)、[audit_max_data.py](unified_training/audit_max_data.py) |
| 三规模训练器 | [train_yambda500m_foundation_fsdp.py](train_yambda500m_foundation_fsdp.py) |
| Max V0；三规模后续版本及 Full-only | [run_max_v0.py](unified_training/run_max_v0.py)；[run_medium_v2_2epoch.py](unified_training/run_medium_v2_2epoch.py)，由 `--execution-config` 选择规模与版本 |
| 训练设置与18个选定端点 | [执行配置](../configs/unified_training_2026_09/README.md)；[版本清单](../docs/unified_training_2026_09/model_versions.md) |
| Full 原始评分与指标汇总 | [evaluate_yambda500m_release_candidates_raw.py](evaluate_yambda500m_release_candidates_raw.py)、[adjudicate_yambda500m_release_candidates.py](adjudicate_yambda500m_release_candidates.py) |
| 15条相邻 Reuse 边 | [unified_reuse_2026_09/README.md](unified_reuse_2026_09/README.md) |
| 四种局部重算：层、尾部、偏差、query 引导 | [selective_recompute_2026_09/README.md](selective_recompute_2026_09/README.md) |
| 非线性 Q-v5 / H-v4，当前90点 | [结果与现存执行入口](../results/read_correction_2026_09/motivation_final/README.md)；[共享依赖说明](read_correction_2026_09/README.md) |
| RecFlow 六层 A–F 六模型链 | [独立脚本](recflow/README.md)；[当前配置](../configs/recflow/window_6l_expanded_u4096_seed17.json)；[已完成结果](../results/recflow/README.md) |
| 当前 Design 1：固定一万人参照、读时校正与开发实验 | [design_one/README.md](design_one/README.md) |
| Design 的历史可复用接口与边界 | [design/README.md](design/README.md) |
| 图片与逐点数据导出 | [图表索引](../figures/README.md) |

`evaluate_yambda500m_foundation_raw.py` 等公共读取实现由现有评测复用。
Medium V0/V1、Large V0–V3 的原训练来源仍记录在历史启动器与冻结合同中；
当前模型一律按选定清单解析，旧 D7、Large canonical/sweep 和 Small 队列不再作为活动入口。

## 共享实现与历史命名

保留 `read_correction_v5/`、`read_correction_motivation/` 与v4校准/评测，
以及v1/v2共享拟合、读取和成本代码。旧v1/v2/v3、Max诊断及v4独立队列入口已退休。
封存哈希原样保留；旧队列不能在当前源码树按旧哈希直接续跑，已存结果可正常绘图。

## 后端与历史证据

[benchmark_hstu_backends.py](benchmark_hstu_backends.py) 和
[check_hstu_backend_fsdp.py](check_hstu_backend_fsdp.py)检查 HSTU 算子与训练后端；
结果见[后端记录](../results/backend_acceleration/2026_09_19/README.md)。
Max资源、恢复及性能检查位于 `unified_training/`，其范围由
[Max训练记录](../results/unified_training_2026_09/max/README.md)说明。

论文附录引用的[10k用户原四预算](../results/insight/unified_auc_10k_20260920/README.md)与
[7144教师扩展](../results/insight/unified_auc_10k_teachers7144_20260920/README.md)
保留为历史证据，不列为当前执行流程。清理与恢复边界见[结果索引](../results/README.md)。

长训练和正式评价按对应配置、资源检查与启动授权执行；一般的用法或导入检查使用
`--help`，不启动已完成的任务。图表生成器集中在 `figures/src/`，只读取已有结果。
