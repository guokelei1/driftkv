# CUDA Graph 开发预筛：待128人正式系数核对

这份预筛将真实30k用户的CPU调度形状与已经完成的32人完整图服务系数组合，用于判断瓶颈；没有新增GPU实验，也不是30k实测或最终性能结论。服务两边都使用同级融合追加与CUDA Graph，仍保留短历史eager。系数来自旧32实现，未包含后续共享图输入缓冲区优化。

| 条件执行方式 | 方法固定项+服务+状态搬运 | Rebuild对应时间 | 方法多用时间 |
|---|---:|---:|---:|
| 每发布128人驻留窗口，允许跨用户bulk重排 | 3478.97秒 | 2668.90秒 | 30.35% |
| 全frontier，每operation页入及必要页出 | 10032.79秒 | 8807.48秒 | 13.91% |

上述固定项分别包含Relay和Native实测图准备6.358/2.556秒；Relay另含历史离线校准、来源生成、原起点及共享准备。约382–397秒的S范围来自原起点前缀计时边界，是构成口径范围，不是测量置信区间。主表尚缺30k新增图shape的冷准备，详见下文；纯JIT、额外backing初始化等未测项也不能据此视为零。

## 优先检查哪里

窗口场景的归一服务时间为Relay2907.15秒、Native2500.31秒。追加多431.98秒、评分多70.12秒；转换34.27秒替代重建129.54秒，维护仅省95.27秒。按原始同步阶段拆分，方法额外scatter约305.51秒，额外compute约131.13秒，pack约14.69秒；这些是端到端API阶段墙钟，不能把compute直接解释为GPU忙时。图减少了两边共同launch开销，但没有解决码随追加搬运和额外读取路径的代价。

在相同系数模型下，即使假设方法追加和评分与Native同速，保留转换、分页和现有S后，30k仍约慢11.54%。这只是当前轨迹/实现的乐观反事实，说明只修一个小copy不能保证总收益；不能当作方法的普遍上限。该反事实在固定Medium模型、目录与活动分布下约13.75万人才有条件盈亏，不能说Large/Max已证实。实际预测的C仍大于R，扩大人数不能消除它。

只把V4校准换成新实测59.353秒，窗口场景约3463.88/2668.90秒，仍慢29.79%；没有把joint阶段倍数应用到其他发布。此前source_available审计说明线上已维护来源可能减少S，但必须另测必要teacher、snapshot与idle物化，不能直接删除88.52秒历史source耗时。

## 图覆盖与空间边界

| 调度 | Relay所需/已测图 | Native所需/已测图 | 所需而未测 |
|---|---:|---:|---:|
| window128 | 278/122 | 192/104 | 156/88 |
| 全frontier | 336/122 | 173/104 | 214/71（2个已测Native图该轨迹不用） |

每个图绑定版本模型、操作、批大小、输入宽度和有无码；正码前缀作为动态输入共享。workspace不随UID数直接增长，但会随所需形状增长。旧32实测全版本累计live增量约6.85/5.93GB，不能当完整30k图集合的空间。按每图均值粗估，window缺图另需约8.13/2.16秒准备和8.76/5.02GB live空间；这只是诊断代理，不是实测或上下界。真实发布可释放旧版图，而当前探针保留所有版本；尚未测得按版释放峰值。后续共享输入实现另有配置，不能追溯加到旧32结果中。

服务邻桶始终限定同backend，短历史eager不会借graph系数。window大批整次耗时代理仅用于Native缺失的B1/B2图重建：167次调用、245个人操作、0.466原始阶段秒，占Native阶段时间0.0179%；Relay为0。GPU不同batch延迟不严格单调，因此它不是保守上界，高估Native会使方法相对收益偏乐观。其他前缀/形状邻桶仍是更大的预测不确定性，须等待128人系数复核。

冷图算术只计三次实际warm前向；第四次Python调用处于capture，只记录kernel，后续replay已属于服务。旧4次调用元数据保留，新实现另记3次执行和1次记录。图准备秒数本来就是实测，无需重跑来修改它。

## 证据与继续入口

- [窗口CPU预筛](/home/gkl/work/evokv/results/system_study_2026_10/service/time_model/graph32_window30k_development/attempt1/summary.json)
- [全frontier CPU预筛](/home/gkl/work/evokv/results/system_study_2026_10/service/time_model/graph32_frontier30k_development/attempt1/summary.json)
- [32人实测](/home/gkl/work/evokv/results/system_study_2026_10/service/batched_session32_graph_full/attempt1/summary.json)
- [来源状态口径审计](/home/gkl/work/evokv/results/system_study_2026_10/service/time_model/source_accounting.md)
- [时间组合入口](/home/gkl/work/evokv/scripts/system_study_2026_10/time_model.py)；12项CPU检查通过。原fair128负结果未改写。

本任务没有自有GPU进程。等`batched_session128_graph/attempt1/summary.json`完整且数值检查通过后，可执行以下CPU命令；入口会拒绝未完成结果、正式计时中的新capture或缺失backend证据：

```bash
python scripts/system_study_2026_10/time_model.py --output results/system_study_2026_10/service/time_model/graph128_window30k/attempt1 --session results/system_study_2026_10/service/batched_session128_graph/attempt1 --batches results/system_study_2026_10/service/time_model/window128_30k/attempt1/summary.json --transfer results/system_study_2026_10/service/batched_session32_fused/attempt1/state_transfer.json --v4-calibration-report results/system_study_2026_10/calibration/joint_fit_comparison/attempt1/summary.json
```

全frontier使用`batches30k/attempt1/summary.json`，写入独立`graph128_frontier30k/attempt1`。128仍需枚举与30k图集合的差集，不因样本扩大就称所有冷准备已测。原S、仅替V4的S及未来live-source S分别保存；不修改旧结果。
