# Insight 2：保留的两个历史诊断

本目录仅保留历史聚合响应修正和 14 天持续性诊断。
它们使用六层 Medium 冻结模型，不训练推荐模型；oracle 数据不能冒充 EvoKV 方法结果。
当前模型链上的共享读取诊断位于
[scripts/design/run_query_read_probe.py](../design/run_query_read_probe.py)，
采用 [256 校准／2560 评价配置](../../configs/insight/query_read_6l_2560_20260920.json)，
比较共享 `b+Aq` 与增加实际用户响应的 `b+Aq+Tr`；数值原语位于
[query_read_probe.py](../design/query_read_probe.py)。
固定设置与全部已执行修正见 [design 入口](../design/README.md#六层共享读取修正已完成的概率保真度诊断)，
结果见[本次记录](../../results/insight/query_read_6l_2560_20260920/README.md)。
该历史共享诊断比较两个共享分支，逐用户教师近99%或旧动态93.39%不作为方法效果。
新诊断单独保存，不覆盖本目录历史脚本、合同或结果。
此前 [128 用户 item／response 试验](../../results/insight/user_information_6l_20260920/README.md)
和[摘要试验](../../results/insight/shared_read_6l_20260920/source_snapshot/README.md)作为历史记录保留。

## 1. 响应修正位置与秩

- run_low_rank_canary.py：同一执行器支持 canary 和 discovery。
- adjudicate_low_rank.py：canary 裁决。
- adjudicate_discovery.py：历史有效的用户等权、版本等权裁决。
- low_rank_correction.py、common.py：公共实现和固定数据接口。
- estimate_discovery_runtime.py：既有 discovery 资源估计。

历史响应修正表来源：
results/yambda500m_medium_seed17/insight2_functional_boundary_v1/
discovery_functional_boundary/analysis_v2/frontier.csv。

只使用 analysis_v2；旧 analysis 的 INVALIDATED.md 和原结果留作审计，不能重新引用旧口径。
历史表中 shared offset / rank-1 的 95.34% / 99.46% 是逐用户教师诊断数字，
其中 shared 指同一用户跨候选共享，不是跨用户预测，也不是学习式版本转换结果。

## 2. 持续性

- run_temporal_persistence_diagnostic.py：真实追加和淘汰轨迹。
- temporal_persistence.py：固定时间桶、覆盖率、漂移与状态校验。
- adjudicate_temporal_persistence.py：裁决和汇总。
- estimate_temporal_persistence_runtime.py：既有资源估计。

历史记录为 diagnostic_temporal_persistence_v1/discovery/analysis/report.md，不进入本次 Insight 主文。
重新观察修正的 93.39%、冻结修正失效、按旧历史占比缩放的 33.85% 均保留原口径。

## 运行边界

脚本及其输入哈希保持不变，默认输出仍拒绝覆盖。正式命令需要新的明确运行授权，
不能为了改图重跑 discovery。只展示结果时直接读取封存表和报告。
历史响应与持续性证据仍保留。共享响应表的旧生成器 `shared_read_insight.py`
已清理，输入 `results/insight/query_read_6l_2560_20260920/diagnostic/` 仍保留；
当前图表及数据入口见 [图表索引](../../figures/README.md)。

旧 rank-0 独立入口、时间系数、coreset、attention-cone、KV-only replay 等探索脚本已删除。
论文使用的原始结果、合同和 invalidation 保留；其他探索 raw 已删除，
历史清理包目前在本地缺失，其恢复限制见 [结果索引](../../results/README.md)。

已知历史问题：functional-boundary 合同所记录的 research_plan SHA-256
与清理前已被追加更新的文档不匹配。原合同保留，但对应计划原文现已不在工作区，未放宽哈希检查。
这不改写封存 raw 或 analysis_v2；若未来获准重跑，须先解决该输入快照问题。
