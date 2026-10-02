# Design 可复用历史接口

当前论文 Design 尚未实现和验证。本目录保留native读取服务和共享拟合的最小依赖，
作为后续实现起点；当前Motivation实验在
[四重算](../selective_recompute_2026_09/README.md)与
[Q/H](../../results/read_correction_2026_09/motivation_final/README.md)的独立目录。

## 保留的10文件闭包

| 文件 | 保留职责 |
| --- | --- |
| `native_service.py` | 历史 `NativeRelease` 读取与视图接口 |
| `query_read_probe.py`、`shared_read_probe.py` | 共享query/response拟合及摘要条件化数值函数 |
| `data.py`、`run.py` | 历史场景、模型绑定与共享执行函数 |
| `diagnose_query_holdout.py` | 成对历史、候选与读取参考 |
| `diagnose_summary_objective.py` | 源投影、共享拟合及层安装 |
| `diagnose_decoder_closure.py`、`diagnose_native_input.py` | native接口仍调用的输入和拟合函数 |
| `../insight_two/common.py` | 固定历史数据接口与公共统计 |

模块名中的 `run` 或 `diagnose` 沿用历史命名；保留理由是上述函数依赖，
不把它们列为待启动实验。旧query-affine训练已退休，既有接口也不实现当前论文的
完整影响sketch、支持策略或运行时。

模型侧另保留 [adaptation/](../../src/hstu_kvcache/adaptation/) 的7个模块：
`__init__.py`、`reader.py`、`summary.py`、`functional_summary.py`、
`translator.py`、`state.py`、`inference.py`。
其中原生读取被当前Q/H使用，状态与摘要接口保留追加/淘汰等数值参考。

## 退休分支与保留证据

旧6L/10L/16L的10k AUC队列、preview、旧competitor功能探针、Insight局部替换/oracle/
持续性队列及不再使用的native报告入口已退出活动源码树。对应结果、封存哈希、
失败记录和必要源码证据仍保留；原队列不能按当前源码树直接续跑。

- 论文附录仍描述的[原10k AUC](../../results/insight/unified_auc_10k_20260920/README.md)
  与[7144校准用户扩展](../../results/insight/unified_auc_10k_teachers7144_20260920/README.md)。
- 历史[六层native质量](../../results/design/analysis/native_base_quality4091_01_report/report.md)
  与[FLOPs](../../results/design/analysis/native_flops_01/report.md)。
- 论文3.1旧Medium质量与成本表，以及其余历史结果见[结果索引](../../results/README.md)。

[适配计划](../../docs/design/plan.md)维护下一步；
[决策记录](../../docs/design/iterations.md)保留当时的实验选择；
`docs/design/expert_route_2026-09-07.md` 提供历史机制报告导航；原运行所用文档以封存源码快照为准。
