# Insight 2 历史数据接口

2026-09-30 清理后，本目录仅保留 `common.py`，供
[Design 的共享接口](../design/README.md)读取冻结数据和公共统计。
旧响应修正、逐用户 oracle 及持续性诊断启动器已退休。
当前读取修正实验见[Q-v5 / H-v4](../../results/read_correction_2026_09/motivation_final/README.md)。

## 保留的历史证据

- `results/yambda500m_medium_seed17/insight2_functional_boundary_v1/`：
  有效口径为 `discovery_functional_boundary/analysis_v2/`；旧 `analysis`
  及其 invalidation 原样保留。shared offset / rank-1 的 95.34% / 99.46%
  是逐用户教师诊断，其中 shared 指同一用户跨候选共享，不是跨用户预测。
- 原持续性诊断 `diagnostic_temporal_persistence_v1/discovery/analysis/report.md`
  保留重新观察修正、冻结修正失效及旧历史比例缩放的原结论。
- [历史共享读取记录](../../results/insight/query_read_6l_2560_20260920/README.md)、
  [128 用户信息探索](../../results/insight/user_information_6l_20260920/README.md)
  和[摘要试验源码记录](../../results/insight/shared_read_6l_20260920/source_snapshot/README.md)
  保留；这些结果不代表当前完整 Design。

合同、原始数据、哈希和裁决未改写，旧队列不能在当前源码树直接续跑。
functional-boundary 合同记录的 research_plan 哈希与清理前追加修改的文档不匹配，
原计划快照当前缺失；该历史问题未通过放宽校验掩盖。
此前清理包的缺失及恢复限制见[结果索引](../../results/README.md)。
