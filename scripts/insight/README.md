# 共享 reader 与历史对照函数

这里不再提供独立实验入口。六个模块因当前流程的实际依赖保留：

| 模块 | 保留原因 |
| --- | --- |
| candidate_shared_causal.py | Insight 2 使用原生历史聚合和 block 更新函数 |
| reader_compatibility_correction.py | 当前响应修正、持续性诊断及 Full/Reuse evaluator 的公共接口 |
| parameter_maps.py | Large 原资源 canary 的参数映射，以及 lazy reader 测试参考 |
| pro_lazy_reader.py | Large 原资源 canary 经同一 evaluator 使用的 PRO 路径 |
| pro_lazy_cost.py | Large runner 的既有成本计算与对应测试 |
| broadcast_residual.py | 保留 reader 的物化参考及单元测试 |

旧 Small probe、PRO 实验队列和 adjudication 启动器已经退出活动目录。
当前论文的方法定义见 [paper_design.md](../../docs/paper_design.md)；
这些旧名称不代表新方法已经实现，也不授权运行旧对照实验。

旧 one_release_refinement 构造器和 evidence-measure 分支已删除。
Small 结果目录只剩 Large 冻结合同直接校验的 theoretical_compute.json；
不要再把它看成完整的 Small 实验。
