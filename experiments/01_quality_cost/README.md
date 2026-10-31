# 01 质量—完整计算成本

对应Figure 6和Table 2。比较CacheTide与合理Rebuild、Reuse及强学习／重算基线，回答同预算恢复多少质量、相近质量需要多少完整FLOPs。

2026-10-22最新：Yambda-Large五条独立相邻边全部24点完成，共120点；GPU 1后四边队列08:07正常结束，实际12小时27分。[完整逐边报告](runs/yambda/large/baselines_all_edges/seed_17/attempt_1/report.md)含全部点AUC／成本／恢复率／UID95%区间，同目录有CSV／TeX及绑定原摘要hash的JSON。末边为E14_partial；两类学习基线的后期退化全部保留。以下启动与首边描述为历史记录，最新状态以[STATUS](../STATUS.md)为准。

- 范围：Yambda／RecFlow × Large／Max；只比较独立相邻版本，主前沿原计划为V2→V3、V3→V4、V4→V5及等权聚合。每组／边独立校准，不以跨版连续适配为基线前置。
- 输入：冻结的真实反馈角色／请求、实体模型、因果历史；每条边所有方法从共同的旧版原生状态出发，记录该边origin和来源。不同边不传递纠正后的cache、码或其他学生状态；边内各方法正常因果追加／FIFO。
- 前置：H10／72维、共同首次历史消费触发、强持久静态映射；RecFlow最终链另行准备。
- 输出：配置、绝对AUC与原始增益、逐边指标、完整操作账及成本。02可沿用选定设置，但必须另外生成连续轨迹，不能直接拼接这些独立边。
- 资源：质量、校准与FLOPs可共享GPU；运行耗时只作进度，校准秒数另行受控测量。
- 当前：已有Medium开发证据；本目录已具首轮基线配置及提速证据，本轮Yambda-Large V0→V1的24点全量结果已完成。

配置放 `configs/`，实际输出放 `runs/`；统一约定见[总索引](../README.md)。

2026-10-21当前首轮：用户要求先测对比方案，暂缓CacheTide质量；先Yambda-Large V0→V1。Reuse／Rebuild为单点，学习纠正各3点，四种部分重算各4点，共24点。最新要求先取自然参数再报告真实费用，不反推参数凑15／50／80%：Layer取2／4／6／8层，Tail取10／20／40／80%尾部，Deviation与Query取10／20／40／60%历史位置。不同方法的点数／成本坐标不要求相同。

配置见`configs/baselines_yambda_large_v0_v1_201.json`；使用201物理GPU 0。完整合法测试人口75,807，其中34,190人有345,630条真实反馈，另41,617人没有评分反馈；质量覆盖与费用绑定同一人口；512人仅用来提速验证，30k仅作用户指定的耗时估算规模。[提速报告](runs/yambda/large/baseline_speed/report.md)最新24点排期：30k约70–100分钟，全量约3–3.5小时；旧20点证据单独保留。本轮Layer点均从第一层开始，无需额外持久旧hidden边界缓存；此前通用45%内存表述已更正。非前缀区间如需旧边界，必须另核来源／重构费用。输出逐点AUC、恢复率、完整费用及配对用户bootstrap区间，附CSV／LaTeX；本轮只支持这条边，不当作跨边／多训练种子结论。

按4096人提交必要分数Parquet和小型完成标记，正常约8–10分钟，不额外落盘历史缓存或重复的合并分数；`--resume`在同一attempt上检查冻结输入／代码／权重并跳过完整块，已完校准也不重训。断点保存是本轮用户明确要求；不覆盖不同设置的旧attempt。完整配置与角色在各attempt的`configuration.json`，恢复命令随运行记录保留。

全量结果已完成：75,807名用户、345,630条反馈、24点，实际3小时06分。完整结果与表格在`runs/yambda/large/baselines_full_v0_v1/seed_17/attempt_1/`，包含AUC、恢复率、完整新增FLOPs、1000次配对UID置信区间以及CSV／LaTeX。Layer2恢复84.88%（UID95%[78.01,90.87]），费用21.04%×一次Rebuild。早期分块预览保留，仅作历史描述。

2026-10-21 19:39最新授权与运行：201物理GPU 1开始顺序评测V1→V2、V2→V3、V3→V4、V4→V5，每条边沿用24点和75,807名合法用户，各自独立校准／选择层段。队列配置在`runs/yambda/large/baselines_v1_v5_gpu1_201/attempt_1/configuration.json`；每边配置为`configs/baselines_yambda_large_v{旧}_v{新}_201.json`，输出为`runs/yambda/large/baselines_full_v{旧}_v{新}/seed_17/attempt_1/`。预计12–16小时，进度及恢复命令见[STATUS](../STATUS.md)。V4→V5的[287,301)仅有13个完整源数据日和部分日300，必须标为E14_partial；其余三边为完整14日窗。此次不启动CacheTide质量或底座训练。
