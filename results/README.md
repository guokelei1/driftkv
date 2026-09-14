# 模型与论文证据索引

当前目录保留论文原始证据、六层/十层模型及必要训练评估过程。
废弃四层实验和无关探索原始数据已删除，不再把所有历史结果并列为活动入口。

Git 保存本目录的报告、紧凑 summary/adjudication、seals、训练元数据和分析 CSV；
原始 parquet/npz、权重、日志、进度文件、诊断图片和 history/ 恢复包留在本地。
忽略或取消追踪不会删除这些本地证据。analysis 与被修正的 analysis_v2 一起保留，
有效口径以原 report/INVALIDATED.md 为准。规则见 [../.gitignore](../.gitignore)。

| 用途 | 路径 |
| --- | --- |
| 六层 Medium 模型、Full/Reuse 与 Motivation | yambda500m_medium_seed17/full_reuse_matrix_v1/ |
| 十层 Large 当前 V0–V5 指针及范围 | [canonical chain](yambda500m_large_seed17/canonical_D14_v0_v5_v1/README.md) |
| 论文重算成本表 | [report](release_cost_random_weight_v1/report.md) |
| Insight 1 五条更新的完整诊断 | yambda500m_medium_seed17/insight1_locality_v1/ |
| Insight 2 响应修正表 | yambda500m_medium_seed17/insight2_functional_boundary_v1/discovery_functional_boundary/analysis_v2/ |
| Insight 2 持续性结果 | yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1/discovery/analysis/ |
| 数据审计 | data_audit/yambda500m_scale_v1/ |

Medium 的 V0–V5 每个版本均训练一 epoch。Large canonical 序列的 V4/V5 为两 epoch；
Large 的 post-hoc working-lineage 说明和原始失败记录保留，不能当作独立 qualification。
为保留六层/十层模型的完整训练评估依据，这两条 full/reuse 或 qualification 链未拆散，
包括 D7、未采用的 endpoint、epoch 对照和必要 canary。它们不是当前论文的方法效果。

Insight 1 的 formal_raw、analysis、正式 canary 和资源估计保留；
论文图显示前三条更新及其平均值，但底层五条更新的证据没有删减。
重复 batch benchmark 和中断 formal 的原始输出已删除，其摘要在清理包中。

Insight 2 只保留两个论文诊断及其 rank-0/low-rank canary 和资源估计。
旧 analysis 的 INVALIDATED.md、原视图及有效原始数据仍在；论文只引用 analysis_v2。
其他 estimator、coreset、paired-functional 和 activation 探索不再留在活动结果树。

Small 目录仅保留一个 theoretical_compute.json，因为 Large 冻结合同直接校验它的哈希。
这不是完整 Small 实验，也不是当前 EvoKV 的成本结论。Small 模型和其余原始结果已删除。
成本表的五个当前测量与对应随机权重保留，未采用的两组配置移除。

## 历史归档与恢复边界

2026-09-05 的两轮清理先移除旧入口，再依据后续授权删除废弃 Small 权重与无关原始结果。
保留 19 个 Medium、25 个 Large 模型 payload 和 5 个成本随机模型；模型路径、seals、
训练与 admission 记录保持原状。当前论文证据包括 Motivation 全部 15 个旧 producer 对比、
Insight 1 全部五边及 34 个配置、Insight 2 全部 ranks/stages/edges 和持续性桶，
以及原有 canary、资源估计、失败结果和 invalidation。

删除范围包括旧 Small 实验、Insight 1 重复 benchmark 与中断 formal、Insight 2 的
estimator/coreset/paired-functional/activation 等废弃探索、两组未采用成本配置。
共删除 3,184 个结果文件（38,875,237,060 bytes），其中 95 个旧四层权重和 2 个成本随机权重。
删除以完整废弃分支为单位，没有从保留实验中挑除负结果。

- [旧源码与文档归档](history/cleanup_2026-09-05.tar.gz)：210 个成员，保存第一轮退出的
  代码、设计讨论及修改前文本。SHA-256：
  `7161e3bdcb69fc8e83b7d773ae7c6ce02cdf2cab8e786883556efd9fc9b95da6`。
  对应 Git 基线为 `d39a3382cc0953880088bc2aa6307ad0f2a565a4`。
- [废弃结果摘要与清单](history/cleanup_results_2026-09-05.tar.gz)：1,757 个成员，保存旧结论、
  失败说明、invalidation 文本、修改前源码/文档和逐文件删除清单 `plan.json`。SHA-256：
  `ef782d6b63be72e5dfe82c49868e7a485d012f5168df2fa233c319d24f582093`。

第一包是入口清理时的快照，其中“模型与结果未动”不描述第二轮之后的状态。
第二包不包含已删除的权重及 parquet/npz 等原始数据；没有外部备份就无法从包中恢复这些材料。
清单记录原路径、大小和非权重数据哈希，权重仅记录元数据。可以在仓库根目录查看：

```bash
tar -tzf results/history/cleanup_2026-09-05.tar.gz
tar -xOf results/history/cleanup_results_2026-09-05.tar.gz plan.json
```

历史源码和摘要仅在独立目录按需提取，不把整包覆盖解压回仓库，也不根据旧合同恢复训练队列。
Small 仅存的 theoretical_compute.json 与旧 PRO 合同仍是 Large 合同的哈希依赖；
其他历史合同中的 Small 指针可能已不可用。旧 PRO 公共函数只服务保留的 canary 与测试。
更早的清理范围仍见 [2026-08-24 记录](checkpoint_cleanup_2026-08-24.md)。

已知协议问题：Insight 2 functional-boundary 的 research_plan 在清理前已被追加修改，
合同哈希与保留原文不匹配，具体哈希见 [冻结协议依赖](../research_discussions/README.md)。
原合同、原文、raw 与 analysis_v2 均保留，未跳过校验；未来重跑前须解决该输入快照问题。
