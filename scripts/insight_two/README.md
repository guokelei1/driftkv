# Insight 2：论文保留的两个诊断

本目录仅保留历史聚合响应修正和 14 天持续性诊断。
它们使用六层 Medium 冻结模型，不训练推荐模型；oracle 数据不能冒充 EvoKV 方法结果。

## 1. 响应修正位置与秩

- run_low_rank_canary.py：同一执行器支持 canary 和 discovery。
- adjudicate_low_rank.py：canary 裁决。
- adjudicate_discovery.py：论文使用的用户等权、版本等权裁决。
- low_rank_correction.py、common.py：公共实现和固定数据接口。
- estimate_discovery_runtime.py：既有 discovery 资源估计。

论文表格来源：
results/yambda500m_medium_seed17/insight2_functional_boundary_v1/
discovery_functional_boundary/analysis_v2/frontier.csv。

只使用 analysis_v2；旧 analysis 的 INVALIDATED.md 和原结果留作审计，不能重新引用旧口径。
论文 shared offset / rank-1 的 95.34% / 99.46% 是诊断数字，不是学习式版本转换结果。

## 2. 持续性

- run_temporal_persistence_diagnostic.py：真实追加和淘汰轨迹。
- temporal_persistence.py：固定时间桶、覆盖率、漂移与状态校验。
- adjudicate_temporal_persistence.py：裁决和汇总。
- estimate_temporal_persistence_runtime.py：既有资源估计。

论文使用 diagnostic_temporal_persistence_v1/discovery/analysis/report.md。
重新观察修正的 93.39%、冻结修正失效、按旧历史占比缩放的 33.85% 均保留原口径。

## 运行边界

脚本及其输入哈希保持不变，默认输出仍拒绝覆盖。正式命令需要新的明确运行授权，
不能为了改图重跑 discovery。只展示结果时直接读取封存表和报告。
当前论文使用 LaTeX 表格和文字，没有独立的 Insight 2 绘图脚本；
表格排版及数据入口统一见 [图表索引](../../figures/README.md)。

旧 rank-0 独立入口、时间系数、coreset、attention-cone、KV-only replay 等探索脚本已删除。
论文使用的原始结果、合同和 invalidation 保留；其他探索 raw 已删除，
旧结论与失败记录只在压缩包中，见 [结果索引](../../results/README.md)。

已知历史问题：functional-boundary 合同所记录的 research_plan SHA-256
与清理前已被追加更新的文档不匹配。原合同和清理前文档均保留，未放宽哈希检查。
这不改写封存 raw 或 analysis_v2；若未来获准重跑，须先解决该输入快照问题。
