# 结果索引与保留范围

## 当前Design证据

2026-10-15[Query correction／码接续探索](query_relay_2026_10/README.md)已完成：
冻结Current item门控A通过主行为任务的开发及用户迁移确认，原反馈任务仍未达联合目标。
质量、完整计算、全局内存与有限发布边界见[报告](query_relay_2026_10/report.md)。
相应工程优化与复验独立保存在[Implementation索引](implementation_2026_10/README.md)，不覆盖原质量结论。

2026-10-14新增[Design3状态来源分析](design3_state_2026_10/README.md)：只读历史形状统计与CPU合成结构检查，未训练或启动GPU实验；不构成新机制质量或完整成本验证。

2026-10-14新增[共享紧凑码探索索引](code_structure_2026_10/README.md)。
CPU 表达空间检查与成本预筛不能代替真实连续回放；当前选定方法的历史结论保持原样。

| 证据 | 入口 |
| --- | --- |
| 最新继承flag范围候选：84.8066%／18.4375% | [配置](design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_support_lazy/configuration.json)、[完整结果](design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_support_lazy/development4096/summary.json)、[配对区间](design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_support_lazy/development4096/paired_original.json) |
| 未门控匹配控制：76.1364%／18.4295% | [结果](design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/development4096/summary.json)、[原校准费用](design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration_summary.json) |
| 同局部权重local-only：63.9851%／12.8870% | [结果](design_system_2026_10/framework_review_2026_10_05/local_only/development4096/summary.json)、[joint配对](design_system_2026_10/framework_review_2026_10_05/local_only/paired_joint.json) |
| 用户组成敏感性 | [cohort_stability.json](design_system_2026_10/framework_review_2026_10_05/cohort_stability.json) |
| V5教师输出与候选相对误差 | [机制分析](design_system_2026_10/framework_review_2026_10_05/joint_composition_v5/analysis.json)、[请求对分解](design_system_2026_10/framework_review_2026_10_05/score_pair_structure.json) |
| eager/lazy事件账 | [lifecycle_cost.json](design_system_2026_10/framework_review_2026_10_05/lifecycle_cost.json) |
| 固定30k一次重建账 | [原生事件与成本](design_system_2026_10/phase2/population_cost/summary_complete.json)、[按需编码账](design_system_2026_10/lightweight_read/lazy_codes/population_cost/summary.json) |
| 第4.4节部分刷新：保留96.04%全量码收益、刷新编码节省9.41% | [最终报告](partial_refresh_2026_10/report.md)、[本轮索引](partial_refresh_2026_10/README.md)、[3360质量](partial_refresh_2026_10/benchmark3360/quality/summary.json)、[固定30k完整成本](partial_refresh_2026_10/population_cost_raw/summary.json) |
| 第二轮等价编码复用：维护节省35.10%／49.94%，收益保留≈100% | [新报告](partial_refresh_2026_10/round2/report.md)、[NoCache质量](partial_refresh_2026_10/round2/nocache3360/quality/summary.json)、[Static4k质量](partial_refresh_2026_10/round2/static4096_3360/quality/summary.json)、[维护及完整成本](partial_refresh_2026_10/round2/factored_cost/summary.json) |

质量样本固定4096用户/51,731请求，成本人口固定30k。最新候选恢复区间为
[51.635%,102.364%]；相对原控制的平均AUC增益区间跨零，V5回退。
这仍是开发证据，不是独立确认。额外要求重构校准source-KV时成本为21.9382%。

新增部分刷新证据使用同一冻结Medium版本链及原权重：strict策略在另一组全部3360位
合格benchmark用户/69,361请求上保留96.04%（UID95%区间93.35%–97.47%）。整体95%
保留margin尚未确立；短历史和V5损失更明显，纯新1024用户上全量码本身也低于Reuse。
完整新增成本18.34294%，相对原始全量码仅节省0.51314%；原生输入重构敏感性可超过20%。
此处的旧保留文件数量是2026-10-07清理时统计，不包含后续新增部分刷新资产。

第二轮按用户澄清以码刷新／维护计算为验收分母：NoCache35.10%／Static4k49.94%节省，
3360用户69361请求的全量码收益区间约[99.99994,100.00009]%，下界>90。
完整新增计算只省8.87%／12.61%，完整/Rebuild分别16.80301%／16.11196%；Static4k含原校准
source重构19.61258%仍在20%内。主要来自编码函数内复用，两个点均全量码；长度/优先刷新
失败分支和原新用户回归保留，不能称新盲确认或全面迁移成功。

## 必要运行资产

- Yambda三规模V0–V5共18个选定基础模型，见[模型清单](../docs/unified_training_2026_09/model_versions.md)。
- RecFlow六层A–F共6个选定模型，以及两条数据轨道的原始/当前处理数据。
- `design_two_2026_10/benchmark_panel/`：当前校准角色、历史与请求。
- `design_system_2026_10/phase2/representative4096/panel/`：当前质量面板；同级merged分数保留作为匹配参照。
- `lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration/`：当前联合及局部起点权重、元数据。
- 当前门控、未门控和local-only的合并分数；当前方法实现见[脚本索引](../scripts/README.md)。
- Motivation的15条Full/Reuse最终汇总、局部重算总表及90个Q/H点的匹配合并分数，支持现有图生成器的哈希检查。
- Max早期20,000用户评价名单：保留唯一的历史成员记录，不保留该次原始评分。

这些资产的文件身份、大小和修改时间在精简前后核对一致。

## Motivation与模型链

| 内容 | 入口 |
| --- | --- |
| 18模型训练结果 | [Medium](unified_training_2026_09/medium/README.md)、[Large](unified_training_2026_09/large/README.md)、[Max](unified_training_2026_09/max/README.md) |
| 15条相邻Full/Reuse边 | [汇总](unified_reuse_2026_09/README.md) |
| 四种局部重算 | [完整结果](selective_recompute_2026_09/README.md)、[summary](selective_recompute_2026_09/summary.json) |
| 当前Q-v5/H-v4的90点 | [结果](read_correction_2026_09/motivation_final/README.md)、[图与点表](../figures/out/read_correction_2026_09/motivation_final/README.md) |
| RecFlow六模型链 | [结果](recflow/README.md) |

## 已结束探索的关键证据

失败和不利结果继续保留原路径，不将其改写为成功。

| 探索 | 保留结果 |
| --- | --- |
| 平衡整体/相对教师误差：72.3082%／18.5421%，不采用 | [summary](design_system_2026_10/lightweight_read/fullitem8_score_modes_joint16_lazy/development4096/summary.json)、[配对差异](design_system_2026_10/lightweight_read/fullitem8_score_modes_joint16_lazy/development4096/paired_original.json) |
| Key辅助监督：65.6239%／19.2861%，不采用 | [summary](design_system_2026_10/lightweight_read/fullitem8_key_aux32_joint16_lazy/development4096/summary.json) |
| rank12与匹配rank8容量控制，均未改善 | [evaluation_summary](design_system_2026_10/lightweight_read/value_capacity_pair/evaluation_summary.json) |
| 旧Item64/Response与rank/query/decoder等探索 | [历史迭代](../docs/design/iterations.md)、`design_one_2026_10/`各轮最终summary及关键区间；设置和费用见汇总与历史记录 |
| 旧formation、任务强度与交互协调 | `design_two_2026_10/`最终summary、analysis及用户角色绑定；[历史说明](../docs/design/exploration_history.md) |
| 512用户独立确认的不利结果 | `design_two_2026_10/confirmation/`的summary、analysis及角色选择记录 |
| Fresh口径的纠正与完整成本失败 | [once_rebuild_reassessment](design_system_2026_10/phase2/once_rebuild_reassessment.json) |
| 更早Insight、随机权重成本及原型 | `insight/`、`design/`、`release_cost_random_weight_v1/`等目录的分支最终汇总；重复逐边、逐worker报告已清理 |

## 2026-10-07评分分片与运行状态清理

用户进一步授权删除已整理、不再使用的原始评分/请求文件及旧运行状态。
`results/`的Parquet从**5,868份减至98份**，删除5,770份；296份`.exit`/`.txt`状态文件全部删除。
该轮共删除6,066个文件，释放约519MB实际磁盘占用；随后继续清理重复说明与源码副本。
结果目录当前共**680个普通文件**：

| 内容 | 文件数 |
| --- | ---: |
| JSON关键汇总与必要元数据 | 411 |
| 保留的Parquet | 98 |
| 结果说明Markdown | 70 |
| CSV分析表 | 69 |
| 24个基础模型与6个校准检查点 | 30 |
| 校准/质量历史NPZ | 2 |

98份Parquet包括：90份Motivation Q/H合并评分，7份当前校准/质量面板、
用户特征与匹配合并分数，以及1份唯一的历史Max用户名单。
删除内容是Full/Reuse的逐rank评分分片、旧请求分片和探针，以及已完成训练评价的原始输出。
15条Full/Reuse边的原始最终汇总、模型准入、全部局部重算和Q/H点、正负结论继续保留。
没有将已删分片另行打包或合并成新文件。

原始/处理数据、30份权重、当前历史包和全部保留JSON与评分未改写。
清理后当前面板、校准、匹配对照、完整成本和90点Motivation读取检查通过。
已删Full/Reuse分片的路径和哈希仍是原运行记录，不能再据此执行逐请求重分析或历史续跑。

## 2026-10-07代码与文档再精简

进一步删除30份退休代码、3份旧启动器、5份分支专用测试、22份重复或过时文档，
以及85份执行源码复制和107份Python缓存，共252份文件；全仓从1594份减至1342份。
22份文档中20份位于结果目录，主要为重复的逐模型说明和已空的源码归档说明。

旧producer-summary、Item/Response、query-compensation及已完成队列退出；
当前校准、Q/H数值实现和RecFlow子进程调用链仍需要的函数保留。
7份旧校准文件和历史汇总/状态文件收紧为共享函数，保留函数的AST与原文一致；
生命周期检查改为直接使用当前full-item KV方案。文档中的旧“当前选择”和待执行描述已更新。

411份结果JSON逐一哈希不变；数据、30份权重、98份保留Parquet与2份历史包未改写。
原执行源码哈希仍留在结果记录中，85份源码副本已删除，不承诺逐份历史重跑。
CPU检查176通过、7项CUDA检查跳过；当前校准/评测入口、冻结面板、完整成本与Motivation来源检查通过。

## 2026-10-07 JSON结果文件再精简

上次统计的1006份只包含名称带summary/analysis/report的JSON；加上过程元数据，
`results/`实际共有12,036份JSON。按用户要求进一步收紧后，当前共保留**411份JSON**：
其中**199份summary/analysis/report**，其余212份是必要配置、输入与用户角色绑定、
选定模型封存、校准元数据、成本或区间分析。删除11,625份JSON，释放约1.38GB磁盘占用。

保留单位是完整结论和必要依赖，不是每次运行、每个分片都留一份：

- 当前方法及匹配对照保留配置、最终质量/成本、区间、原校准和冻结输入。
- Motivation保留局部重算总表、15条Full/Reuse边和90个Q/H点所需的逐边汇总；匹配原始分数不动。
- 已结束分支保留最终聚合结果和独有的不利结论；用户角色名单仍用于后续排除，不能因清理改变划分。
- 删除已被最终汇总代替的分片/逐用户/逐epoch报告、旧canary和pilot重复结果、运行状态、日志元数据及重复来源清单。

没有新建大JSON或压缩包来存放已删文件。保留的411份JSON内容与清理前逐一哈希相同；
清理后当前4096用户/51,731请求面板、三次发布校准、匹配控制和完整成本仍可读取，
Motivation的90点来源与原始评分哈希检查通过。

## 2026-10-05精简

按用户授权清理三类内容：证实不work的专用分支、已有明确结论的实验中间产物，以及确定不再使用的旧资产。
删除约26.3GB实际磁盘占用（24.49GiB），包括旧候选权重/历史包/原始输出、逐用户断点、
日志、旧执行源码复制、旧预览图、过时launcher和分支测试；源码依赖按当前入口保留。
当时保留约1006份summary/analysis/report JSON及相关设置；2026-10-07已按上面的范围再次精简。

历史JSON中的路径与哈希是当时来源记录，不表示文件当前仍在。
只有“必要运行资产”明确保留的原始评分和权重可用于当前重分析/执行；
其余旧实验保留结论，不能据此声称仍可完整复跑。
精简没有重新训练模型、重新打分、重新选择用户或改变任何实验指标。
