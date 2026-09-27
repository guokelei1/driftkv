# 必要测试入口

测试服务论文结论和快速迭代。只运行当前改动涉及的用例，不默认运行全量 pytest，
不以覆盖率、测试数量或所有边界组合为目标。

优先保留关键公式与参考实现的一致性、时间与缓存因果、数据划分、指标聚合，
以及确实发生过且会改变研究结论的错误回归。显示格式、重复抄写配置常量、
已退出路线和仅重复实现的测试不作为长期维护目标。遇到明确冗余时删减，
无需为普通开发先做一轮完整测试审计。

按改动选择入口：

- test_attention_backends：CUDA Triton 与 PyTorch 的算式／梯度、旧缓存和 mask 对照；
  无 CUDA/Triton 时跳过 GPU 项。六层和已有权重的短测量见 `scripts/benchmark_hstu_backends.py`。
- test_foundation_*：训练、manifest、指标和缓存谱系。
- test_yambda_data、test_yambda_incremental_time_contract、test_yambda500m_streaming_windows：
  用户选择、OOV、反馈时间边界、增量时间与窗口；使用小型输入，不扫描本机完整数据。
- test_medium_d14_direct_long_age_reuse、test_large_d14_canonical_chain：当前 Medium 动机链与 Large canonical 链。
- test_append_only_kv、test_state_transition：追加、淘汰与状态依赖。
- test_baseline_*：三个对比方案的 CPU 公式、入口 hidden、真实尾部重放及跨层映射；不运行数据集评价。
- test_insight1_competitor*：三方案统一评分、概率缺口聚合、校准 UID 隔离及独立入口；使用 CPU 合成数据和临时 I/O，不读取真实评价用户。
- test_competitor_*：本轮 Medium／Large 模型清单解析、规模词表与候选构造、显式 UID 隔离及 10 层方法端点；只用元数据或合成输入。
- test_adaptation：固定归属的摘要增减、原算子的count权重和局部刷新等价性；真实六层梯度/追加canary在Design runner内。
- test_summary_objective、test_functional_summary：当前摘要/响应诊断的公式和增量淘汰参考。
- test_insight_one_locality：论文局部替换诊断。
- test_insight_two_functional_boundary：候选划分、聚合指标、rank-0/low-rank、持续性统计。
- test_reader_compatibility_correction 及其余 reader 测试：当前 Insight 诊断所依赖的公共接口。
- test_download_scale_datasets：当前 Yambda 数据选择边界。
- test_recflow_data、test_recflow_model、test_recflow_metrics、test_recflow_evaluation：
  独立 RecFlow 开发轨道的请求因果边界、滑窗时间输入、生成概率、检索指标，
  以及完整面板/采样子面板各自的聚合与随机基线；使用小型合成数据。

例如，修改 Insight 2 的聚合指标，只执行对应指标测试：

```bash
PYTHONPATH=src pytest -q tests/test_insight_two_functional_boundary.py -k score_metrics
```

局部代码修改可用一个相关测试或 tiny reference 对照确认；低影响修改不要求新增测试。
文档、绘图、路径整理按需做文本、链接或出图检查。实验的小样本检查可以同时承担 canary，
无需再套一层等价验证。仅在共享改动影响广泛、无法用局部检查覆盖实际风险，
或用户明确要求时运行完整测试。检查通过后继续研究任务，不惯性扩大测试范围。

四种局部重算的定向检查在 `tests/selective_recompute/`：层、尾部、偏差、query
各有独立目录，共享 evaluator／cohort／cost／aggregation 检查对应因果时间流、
用户筛选、实际算子计算量和同请求聚合。运行 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/selective_recompute`。真实模型短检查使用
`scripts/selective_recompute_2026_09/run.py --probe`，不等同正式评测启动。
