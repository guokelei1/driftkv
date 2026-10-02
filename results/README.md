# 模型与实验结果

## 当前结果入口

| 主线 | 结果与来源 |
| --- | --- |
| Yambda 数据与三规模 18 个选定模型 | [统一训练](unified_training_2026_09/README.md)、[版本清单](../docs/unified_training_2026_09/model_versions.md)、[数据审计](data_audit/yambda500m_scale_v1/) |
| 15 条相邻边 Full/Recompute 与 Reuse | [完整对照](unified_reuse_2026_09/README.md) |
| 四种局部重算，每种 15 条边 | [用户面板、预算曲线与成本](selective_recompute_2026_09/README.md) |
| 非线性 Q-v5 / H-v4，90 个当前实测点 | [当前结果](read_correction_2026_09/motivation_final/README.md)、[图表](../figures/README.md) |
| Design 1：固定 Medium 一万人开发 | [当前目标与实施顺序](../docs/design/plan.md)、[全部迭代](../docs/design/iterations.md)、[当前紧凑读修正](design_one_2026_10/round20_compact_training/evaluation_10k/summary.json)、[首批参照](design_one_2026_10/rank8_10k/summary.json)、[匹配Q/H](design_one_2026_10/benchmark_10k/summary.json) |
| RecFlow 4096 用户 A–F 六模型链 | [数据处理、模型与随机对照](recflow/README.md) |
| 论文重算成本表 | [测量报告](release_cost_random_weight_v1/report.md) |
| HSTU 后端数值与性能 | [六层验证](backend_acceleration/2026_09_19/README.md)、[Max 剖析](backend_acceleration/max_profile_2026_09_19/README.md) |

以上训练与 Motivation 评测已完成。Design 1 第二十轮达到质量、计算与存储共同目标：
紧凑Item64＋Response256恢复84.98%、新增FLOPs10.46%、额外持久状态16.67%，AUC0.670193。
校准占8.66%、服务新增计算占1.80%；当前无待启动任务。第十八轮完整K/V参照为
83.65%/13.46%/100%额外状态，旧rank8参照为59.45%/0.753%。
完整 Design 的持续适配尚未验证。历史原型结果见
[设计记录](../docs/design/iterations.md)，保留接口见[实现边界](../docs/design/plan.md)。
当前实验的全部预算、负结果、失败和 invalidation 随原始记录保留。

### Design 1 本地保留范围

`design_one_2026_10/`的首批五轮保留17组一万人正式逐请求分数、全部summary、校准元数据、
launch与shard来源/控制记录，以及bootstrap、裁决和精确执行源快照。
固定请求面板和四份有界历史包继续供下一轮使用，全部正负结果均可从最终分数重算。

首批可直接加载的校准权重保留四份：`calibration_rank8_c256`、`calibration_rank0_c256`、
`calibration_pure_mean8_no_response_c256`、`calibration_pure_rank0_no_response_c256`。
其余16份已结束搜索/小样本/无效校准的权重已删除，元数据和原哈希仍在；
重做这些旧配置需要重新校准，历史configuration不是当前待运行队列。

第六轮新产物单独位于`round6_nonlinear/`：两份正式校准、两组完整一万人分数、
资源canary与执行源归档均保留。第七轮`round7_logit_refinement/`保留联合logit校准、
三个完整评价点及其canary/来源。新增两轮共5个完整点，无配置达到80%恢复目标。
第八轮`round8_rolling_context/`保留普通与context KV32的两份滚动校准、
两组完整评分和95份源码归档；恢复分别42.94%/42.04%，费用10.435%/10.452%，
未改善。第九轮完整维度映射另存`round9_affine_kv/`，最新执行状态见迭代记录。
第九轮完整仿射40.73%/9.396%，第十轮`round10_capacity_queries/`宽64的两个点为
60.11%/10.076%与59.13%/13.135%。第十一轮`round11_item_features/`保留新增Current
物品embedding后的完整校准与一万人评分，恢复77.63%/费用11.836%。
第十二轮`round12_item_refinement/`的响应/联合精炼分别77.87%/13.809%、78.95%/13.987%；
第十三轮`round13_item_duration/`联合训练延长至12轮后76.90%/14.677%，未改善。
第十四轮`round14_mixed_refinement/`真实混合状态精炼为48.97%/14.776%，未改善。
第十五轮`round15_producer_scale/`的学习缩放60.06%/14.7796%、固定第0层对照69.49%/11.8381%，
均未改善。第十六轮`round16_recent_queries/`仅调整校准query分布，结果78.03%/11.836%。
第十七轮`round17_relative_logits/`相对logit目标为78.38%/13.987%，未改善。
第十八轮冻结Item视图后补充响应残差，恢复83.65%/费用13.465%，结果已固定在
`round18_item_response/`：校准权重、完整逐请求分数、分项成本、控制和100份执行源均保留。
第十一轮校准是其冻结来源依赖，继续保留；此结果仅覆盖固定Medium V4→V5开发点。
第十九轮`round19_compact_hidden/`保留紧凑等价执行、六个完整点和四个较短Item训练
未达80%的结果，含原始分数、校准、费用、存储与执行源。第二十轮
`round20_compact_training/`是当前选定入口：fresh64轮紧凑Item校准、100轮Response校准、
完整一万人分数、控制、127份执行源与全部费用保留。其`item_calibration/`与
`calibration/`是当前必要权重，`evaluation_10k/final_check.json`核对固定对照、费用和源码。

2026-10-01按用户要求删除1218份冗余产物（208.71 MiB）：已核对一致的unit与shard
分数副本、已汇入shard记录的unit明细、重复inputs、完成日志、被全量结果替代的canary
分数以及上述旧权重。完成目录作为只读结果使用；已删分片不再提供原地断点续跑。
旧记录里的这些载荷路径与哈希描述原运行，不表示载荷仍在磁盘。
小样本数值控制、费用与时间汇总仍保留；canary逐行分数和单位级进度不再保留。

已结束的reference诊断脚本保存在`source_snapshots/reference_probe_v1.py`，
无调用的旧state包装器精确源在`source_before_producer_mass.tar.gz`中；
源码快照用于恢复历史来源，不作为活动执行入口。

## 本地资产与必要历史依赖

- **权重**：18 个 Yambda 选定模型和 RecFlow A@epoch3、B–F@epoch1 六端点。
  论文旧 Medium D14 producer 对照还保留四份额外实体权重；旧 V0/V1 与
  Large V0–V3 的别名指向对应选定权重。
- **数据**：Yambda-500M/5B 和 RecFlow 的原始数据、处理存储、映射、请求面板与时间窗。
  旧命名的中间清单只要仍被当前链路使用，就继续保留。
- **Max 对照**：所选 V2@epoch1 的同窗口 Parent Full 位于
  `v3_from_v2e1_epochs12_4gpu_b80_cpu14`。对应 V2→V3 的更新收益为
  1.252 个 AUC 百分点，Reuse 损失占 50.3%，详见当前对照表。
- **论文旧证据**：3.1 的 Medium D14 对照与成本表、附录引用的
  [10k 用户实验](insight/unified_auc_10k_20260920/README.md)及
  [7144 教师扩展](insight/unified_auc_10k_teachers7144_20260920/README.md)继续保留。
  这些旧实验不作为当前 15 边的方法入口。
- **历史合同**：Small 的 theoretical-compute 记录仍是 Large 原合同的哈希依赖。
  Insight 2 的有效历史口径为 `analysis_v2`，原 analysis 与 invalidation 同样保留。

## 保留范围与恢复限制

旧 RecFlow 分支及中间权重、旧 Large sweep/canonical、Yambda D7、KuaiRand
和 Small 请求清单已退出。上层历史 summary 中出现旧路径，不表示该 raw 或权重仍在；
现存内容以以上索引为准。精确退出路径保存在已有的
[机器清单](history/repository_cleanup_2026_09_30.json)，不另设整理文档。

所选 Max V3@epoch2 的实际训练父端点为 V2@epoch2，该旧父权重已删除，
配置和结果仍在；重放那次训练需先恢复或重新生成父权重。

RecFlow 原运行的 22 份执行源可由 19 份当前匹配文件与 3 份
[精确源码副本](recflow/source_snapshots/expanded_u4096_seed17_launch_2026_09_18/README.md)恢复。
旧 Design 活动源码在 Git 基线中保留，具体来源及已缺失的历史执行哈希见
[来源记录](history/retired_design_sources_2026_09_30.json)。
历史代码不作为当前已验证的执行队列。

Insight 2 functional-boundary 的原 research-plan 文本已缺失，历史合同记录的
哈希也与当时后改文本不符；原合同、raw、analysis_v2 保留，不能跳过该限制重跑。

### 历史源码与文档快照

| 本地记录 | 可用性与范围 |
| --- | --- |
| [2026-09-22 源码快照](history/cleanup_sources_2026-09-22.tar.gz) | 存在，6 个旧入口的当时工作区源码；不是更早运行哈希的替代证明 |
| [2026-09-22 文档快照](history/docs_before_cleanup_2026-09-22.tar.gz) | 存在，10 份旧文档、原技术草稿和预算；只作历史参考 |
| `history/cleanup_2026-09-05.tar.gz` | 本地缺失；历史 SHA-256 为 `7161e3bdcb69fc8e83b7d773ae7c6ce02cdf2cab8e786883556efd9fc9b95da6` |
| `history/cleanup_results_2026-09-05.tar.gz` | 本地缺失；历史 SHA-256 为 `ef782d6b63be72e5dfe82c49868e7a485d012f5168df2fa233c319d24f582093` |

9 月 22 日两个现存包的 SHA-256 分别为
`a0453c5f328f535f644f8ed77ab1b4eb68f8224328b53f4d659610c351377466`、
`9426c6f9f0be0790f5f7045f4e7ac36ff1bcb73cd7161d063e5237a0f95d69d8`。
旧 9 月 5 日源码包对应 Git 基线 `d39a3382cc0953880088bc2aa6307ad0f2a565a4`。
上述包均不包含已删模型或逐请求 raw；不能据此声称这些载荷可恢复。
更早的 checkpoint-cleanup 说明也已缺失，不补写其内容。

Git 保存源码、配置和紧凑证据；权重、原始评分、大型数据与恢复包按
[.gitignore](../.gitignore)留在本地。只克隆仓库不足以重评或生成依赖 raw 的图。
