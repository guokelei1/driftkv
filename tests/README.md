# 必要测试入口

测试服务论文结论和快速迭代。只运行当前改动涉及的用例，不默认运行全量 pytest，
不以覆盖率、测试数量或所有边界组合为目标。

优先保留关键公式与参考实现的一致性、时间与缓存因果、数据划分、指标聚合，
以及确实发生过且会改变研究结论的错误回归。显示格式、重复抄写配置常量、
已退出路线和仅重复实现的测试不作为长期维护目标。遇到明确冗余时删减，
无需为普通开发先做一轮完整测试审计。

按改动选择入口：

- test_foundation_*：训练、manifest、指标和缓存谱系。
- test_yambda_data、test_yambda_incremental_time_contract、test_yambda500m_streaming_windows：
  用户选择、OOV、反馈时间边界、增量时间与窗口；使用小型输入，不扫描本机完整数据。
- test_medium_*、test_large_*：六层和十层现行队列、模型选择记录与运行边界。
- test_append_only_kv、test_state_transition：追加、淘汰与状态依赖。
- test_adaptation：固定归属的摘要增减、原算子的count权重和局部刷新等价性；真实六层梯度/追加canary在Design runner内。
- test_design_quality：配对UID bootstrap的计数实现与显式重复用户一致，覆盖概率并列及单类别重采样；使用`PYTHONPATH=src:scripts`。
- test_design_ridge：完整UID删除的岭回归残差，对照显式删除后重拟合；使用`PYTHONPATH=src:scripts`。
- test_design_kernel：非线性source kernel的自由截距方程、未拟合输入预测和发布后时间分解。
- test_design_query_view：query坐标折叠、count加权、标量/批量发布和逐层消退的张量参考。
- test_insight_one_locality：论文局部替换诊断。
- test_insight_two_functional_boundary：候选划分、聚合指标、rank-0/low-rank、持续性统计。
- test_reader_compatibility_correction、test_pro_lazy_reader 及其余 reader 测试：
  当前 evaluator 所依赖的公共接口和保留对照。
- test_release_cost_benchmark、test_download_scale_datasets：成本单位、随机模型和数据选择。

例如，修改 Insight 2 的聚合指标，只执行对应指标测试：

```bash
PYTHONPATH=src pytest -q tests/test_insight_two_functional_boundary.py -k score_metrics
```

局部代码修改可用一个相关测试或 tiny reference 对照确认；低影响修改不要求新增测试。
文档、绘图、路径整理按需做文本、链接或出图检查。实验的小样本检查可以同时承担 canary，
无需再套一层等价验证。仅在共享改动影响广泛、无法用局部检查覆盖实际风险，
或用户明确要求时运行完整测试。检查通过后继续研究任务，不惯性扩大测试范围。
