# 定向验证入口

按改动选择最小检查，关注公式、时间/缓存因果、数据划分和指标聚合。
文档与简单目录调整检查文本和链接；不默认运行全量 pytest，也不按覆盖率增加测试。

| 改动范围 | 现有检查 |
| --- | --- |
| Attention、增量状态与后端 | `test_attention_backends.py`、`test_append_only_kv.py`、`test_state_transition.py` |
| Yambda 数据、训练与 Full 评价 | `test_yambda_data.py`、`test_yambda_incremental_time_contract.py`、`test_yambda500m_streaming_windows.py`、`test_foundation_*.py` |
| 原始数据下载选择 | `test_download_scale_datasets.py` |
| 旧 Medium D14 producer 对照 | `test_medium_full_reuse_pipeline.py`、`test_medium_d14_direct_long_age_reuse.py` |
| 当前四种局部重算 | [selective_recompute/](selective_recompute/)：layer、tail、deviation、query 各自独立，另含 cohort、cost、evaluator、aggregation |
| 读取修正公共算式与拟合 | [read_correction/](read_correction/)：query/history、校准、成本与评价 |
| 当前 H-v4 非线性与 token read | [read_correction_v4/](read_correction_v4/) |
| 当前 Q-v5 非线性特征与优化 | [read_correction_v5/](read_correction_v5/) |
| Design 1 读时校正与因果回放 | [design_one/](design_one/)：摘要算式、响应/KV视图、紧凑读取及训练梯度等价、组合读取与冻结来源、流式ridge、拟合隔离、真实滚动场景/出生上下文及native写入不变 |
| 保留的历史摘要/状态原语 | `test_adaptation.py`、`test_summary_objective.py`、`test_functional_summary.py` |
| 旧 KV 映射与读取参考 | `test_baseline_kv_translate.py`、`test_reader_compatibility_correction.py`、`test_candidate_shared_causal.py` |
| RecFlow 数据、模型、指标与评价 | `test_recflow_data.py`、`test_recflow_model.py`、`test_recflow_metrics.py`、`test_recflow_evaluation.py` |

例如修改状态追加/淘汰时：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_state_transition.py
```

CUDA/Triton 用例需要相应设备；无设备时跳过相关 GPU 项。其他检查主要使用小型输入，
不自动扫描完整数据集或执行正式评测。真实模型的短 canary 和性能测量由相应
[实验入口](../scripts/README.md)单独记录。

旧 competitor、Insight 局部替换/oracle 与 Large canonical 的专属测试已退出。
保留历史原语的测试通过，不等于当前完整论文 Design 已实现或验证。
只在共享改动影响广泛、局部检查不足或用户明确要求时运行完整测试。
