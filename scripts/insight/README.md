# Insight 共享 reader 原语

本目录不提供独立实验入口，只保留当前 Insight 2 使用的两个公共模块：

| 模块 | 保留原因 |
| --- | --- |
| `candidate_shared_causal.py` | 原生历史聚合和 block 更新函数 |
| `reader_compatibility_correction.py` | 响应修正与持续性诊断的公共接口 |

Small/PRO 的参数映射、lazy reader、成本模型及其测试已经退休。当前论文的方法定义见
[paper_design.md](../../docs/paper_design.md)。
