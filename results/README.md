# 模型与论文证据索引

当前目录保留论文原始证据、当前六层/十层/十六层模型及第三章 Motivation 所用旧 Medium D14 模型。
废弃四层实验和无关探索原始数据已删除，不再把所有历史结果并列为活动入口。

Git 保存本目录的报告、紧凑 summary/adjudication、seals、训练元数据和分析 CSV；
原始 parquet/npz、选定权重和诊断图片留在本地；过程日志按下述清理记录归档。
忽略或取消追踪不会删除这些本地证据。analysis 与被修正的 analysis_v2 一起保留，
有效口径以原 report/INVALIDATED.md 为准。规则见 [../.gitignore](../.gitignore)。

| 用途 | 路径 |
| --- | --- |
| HSTU Triton 后端升级：六层数值、性能和已有权重兼容 | [2026-09-19 开发验证](backend_acceleration/2026_09_19/README.md) |
| 16层Max完整训练步、四卡通信及编译预热成本 | [2026-09-19 性能剖析](backend_acceleration/max_profile_2026_09_19/README.md) |
| 本轮 Yambda 三档统一训练与逐版评测 | [本轮结果索引](unified_training_2026_09/README.md) |
| 三规模15条相邻 Reuse 边的已完成评测 | [独立目录与结果](unified_reuse_2026_09/README.md) |
| 四种选择性重算 baseline：15条边、每边3000用户，60条曲线已完成 | [运行记录、用户清单与资源检查](selective_recompute_2026_09/README.md) |
| 六层 Medium 模型、Full/Reuse 与 Motivation | yambda500m_medium_seed17/full_reuse_matrix_v1/ |
| 十层 Large 当前 V0–V5 模型 | [本轮选定链](unified_training_2026_09/large/README.md) |
| 论文重算成本表 | [report](release_cost_random_weight_v1/report.md) |
| 当前完成：统一 AUC、10000 用户、四次更新、九档教师预算至7144 | [扩展结果与完整预算](insight/unified_auc_10k_teachers7144_20260920/README.md)，[原四预算记录](insight/unified_auc_10k_20260920/README.md) |
| Insight 1 五条更新的完整诊断 | yambda500m_medium_seed17/insight1_locality_v1/ |
| 历史六层 Insight：256 校准／2560 评价的概率保真度 | [2026-09-20 结果、协议与成本](insight/query_read_6l_2560_20260920/README.md) |
| 历史 Insight 2 逐用户响应修正 | yambda500m_medium_seed17/insight2_functional_boundary_v1/discovery_functional_boundary/analysis_v2/ |
| 历史 Insight 2 持续性结果 | yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1/discovery/analysis/ |
| 数据审计 | data_audit/yambda500m_scale_v1/ |

保留的旧 full/reuse Medium 链 V0–V5 每个版本均训练一 epoch，不代表本轮 consolidated 链的训练设置。
Large canonical 序列的 V4/V5 为两 epoch；
Large 的 post-hoc working-lineage 说明和原始失败记录保留，不能当作独立 qualification。
2026-09-23按论文第三章筛选旧模型：旧 Medium D14 的六版 Motivation 路径保留，
其中 V0/V1 链接至哈希相同的本轮选定权重；旧 Large D14 的 V0–V3 也链接至本轮选定权重。
D7、训练探针及未进入第三章图表的旧 Large 备选权重已删除，原始结果、seals、adjudication
和失败记录仍保留。历史 [Large canonical chain](yambda500m_large_seed17/canonical_D14_v0_v5_v1/README.md)
只记录当时的开发选择，当前模型以[三规模清单](../docs/unified_training_2026_09/model_versions.md)为准。

Insight 1 的 formal_raw、analysis、正式 canary 和资源估计保留；
旧论文图显示前三条更新及其平均值，但底层五条更新的证据没有删减。
重复 batch benchmark 和中断 formal 的原始输出已删除，其摘要在清理包中。

历史 Insight 2 保留两个诊断及其 rank-0/low-rank canary 和资源估计。
旧 analysis 的 INVALIDATED.md、原视图及有效原始数据仍在；有效历史口径为 analysis_v2。
其他 estimator、coreset、paired-functional 和 activation 探索不再留在活动结果树。

当前完成结果是[九预算统一10000用户AUC诊断](insight/unified_auc_10k_teachers7144_20260920/README.md)：
四条相邻边共用1024-event发布快照、真实E14反馈和pooled AUC，五panel各保留四版本曲线。
212773条反馈、四边各37路径；原27路径未改，新增10路径，Reuse复算与原值差0。
扩展四GPU执行969.436秒、每卡峰值3.19GiB，CPU教师双缓存125.58GiB；
[原运行](insight/unified_auc_10k_20260920/README.md)498.896秒和全部证据继续保留。
256校准下共享offset／增加用户响应的等权四边恢复均值为26.90%／52.59%，后者四边均优于
前者；1024／7144教师下两臂均值为27.09%／45.14%、27.27%／45.61%，V1→V2全部新增
预算两臂仍负。全部预算与对比结果保留，旧Motivation和Insight证据转为历史入口。
[被替代的5872用户AUC pilot](insight/auc_read_6l_20260920/README.md)保留两次启动失败，
没有完成的AUC结果。

历史概率保真度 Insight 使用本轮六层链 V1→V2、V2→V3：256 校准、128 pilot、2560评价UID互斥。
pilot的校准参数固定后原样复用，正文显示共享 `b+Aq` 与 `b+Aq+Tr` 的U/P对照。
后者 U=8.05%／17.90%，P=65.54%／72.85%，2164／2154人优于Reuse；
前者 U=-818.36%／-200.01%，所有负尾、pilot及canary失败质量均保留。
[旧128人item／response记录](insight/user_information_6l_20260920/README.md)及其
[表格快照](insight/user_information_6l_20260920/source_snapshot/README.md)完整保留，
其中已执行的逐用户教师和两原生端点不删除。历史近99%教师与动态诊断不进入本次主文论证。
[前一摘要试验](insight/shared_read_6l_20260920/diagnostic/summary.json)与
[源码／旧表快照](insight/shared_read_6l_20260920/source_snapshot/README.md)也完整保留，
不作为当前问题的择优候选。这些历史概率结果不证明摘要必要性、推荐 AUC、连续适配或完整 Design。

Small 目录仅保留一个 theoretical_compute.json，因为 Large 冻结合同直接校验它的哈希。
这不是完整 Small 实验，也不是当前 EvoKV 的成本结论。Small 模型和其余原始结果已删除。
成本表的五个当前测量与对应随机权重保留，未采用的两组配置移除。

## 历史归档与恢复边界

2026-09-05 的两轮清理先移除旧入口，再依据后续授权删除废弃 Small 权重与无关原始结果。
当时保留 19 个 Medium、25 个 Large 模型 payload 和 5 个成本随机模型；模型路径、seals、
训练与 admission 记录保持原状。当前论文证据包括 Motivation 全部 15 个旧 producer 对比、
Insight 1 全部五边及 34 个配置、Insight 2 全部 ranks/stages/edges 和持续性桶，
以及原有 canary、资源估计、失败结果和 invalidation。

删除范围包括旧 Small 实验、Insight 1 重复 benchmark 与中断 formal、Insight 2 的
estimator/coreset/paired-functional/activation 等废弃探索、两组未采用成本配置。
共删除 3,184 个结果文件（38,875,237,060 bytes），其中 95 个旧四层权重和 2 个成本随机权重。
删除以完整废弃分支为单位，没有从保留实验中挑除负结果。

- 旧源码与文档归档 `history/cleanup_2026-09-05.tar.gz`：历史记录为210个成员，保存第一轮退出的
  代码、设计讨论及修改前文本。2026-09-22检查时本地文件不存在，不能作为可用恢复包。历史SHA-256：
  `7161e3bdcb69fc8e83b7d773ae7c6ce02cdf2cab8e786883556efd9fc9b95da6`。
  对应 Git 基线为 `d39a3382cc0953880088bc2aa6307ad0f2a565a4`。
- 废弃结果摘要与清单 `history/cleanup_results_2026-09-05.tar.gz`：历史记录为1,757个成员，保存旧结论、
  失败说明、invalidation 文本、修改前源码/文档和逐文件删除清单 `plan.json`。
  2026-09-22检查时本地文件不存在。历史SHA-256：
  `ef782d6b63be72e5dfe82c49868e7a485d012f5168df2fa233c319d24f582093`。

第一包是入口清理时的快照，其中“模型与结果未动”不描述第二轮之后的状态。
第二包不包含已删除的权重及 parquet/npz 等原始数据；没有外部备份就无法从包中恢复这些材料。
历史清单记录原路径、大小和非权重数据哈希，权重仅记录元数据。
上述包目前均缺失，恢复须先找到外部副本；本轮没有重建或验证其内容。

历史源码和摘要仅在独立目录按需提取，不把整包覆盖解压回仓库，也不根据旧合同恢复训练队列。
Small 仅存的 theoretical_compute.json 与旧 PRO 合同仍是 Large 合同的哈希依赖；
其他历史合同中的 Small 指针可能已不可用。旧 PRO 公共函数只服务保留的 canary 与测试。
更早的 `checkpoint_cleanup_2026-08-24.md` 在当前工作区也不存在；此处保留缺失状态，不补写旧记录。

已知协议问题：Insight 2 functional-boundary 的 research_plan 在清理前已被追加修改，
历史记录称合同哈希与当时原文不匹配。当前 `research_discussions/README.md` 及对应
research_plan 文档已不在工作区，不能继续声称原文可用。原合同、raw 与 analysis_v2
保留，未跳过校验；未来重跑前须解决该输入快照缺失及哈希问题。

## 2026-09-22 入口清理

删除已完成的专用恢复脚本 `resume_max_v2_epoch2_20260921.sh`、训练间隙调度脚本
`run_max_profile_after_training.py`，以及旧 Design 的 `finalize_base_method.py`、
`diagnose_constant_affine.py`、`report_mechanism.py`、`audit_native_response_scale.py`。
它们没有其他 Python 模块调用；仍被当前诊断复用的公共实现继续保留。

六个文件删除前的实际源码保存在本地
[源码快照](history/cleanup_sources_2026-09-22.tar.gz)，SHA-256：
`a0453c5f328f535f644f8ed77ab1b4eb68f8224328b53f4d659610c351377466`。
这是本轮工作区快照，不保证与更早封存的各次执行源码相同，也不包含结果或权重。
原配置、哈希和结果未改写；大模型训练入口、数据、恢复点与实验原始证据均未清理。
Large/Max AUC launcher 移除了已不存在的自动出图程序，保留诊断和退出状态记录。

## 2026-09-22 文档精简

精简AGENTS、设计入口、实验设计、RecFlow计划、适配决策记录和两份脚本README；
重复的Medium/Large短导航合并到实验定义及模型链索引。独有的协议变化、失败原因、
授权和未开放数据边界继续保留；完整结果由各实验目录维护，原指标、合同和seals未改。

十份修改前文档保存在本地[全文快照](history/docs_before_cleanup_2026-09-22.tar.gz)，
SHA-256：`9426c6f9f0be0790f5f7045f4e7ac36ff1bcb73cd7161d063e5237a0f95d69d8`。
包含旧paired-summary技术草稿、历史预算和被删除的两个短导航；这是本轮实际工作区
快照，不是旧运行源码hash的替代证明。未删除任何实验结果、模型或数据。
