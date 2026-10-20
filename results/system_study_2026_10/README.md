# 本轮结果与过程索引

范围见[总计划](../../docs/system_study_2026_10/README.md)，当前运行状态见
[STATUS](../../docs/system_study_2026_10/STATUS.md)。本目录不覆盖历史结果。真实GPU会话、到达测试与优化诊断的汇总入口是
[服务实验报告](service/README.md)。

本批主线是Medium真实30k完整成本／时间和拟合人数选择。不同基础模型各自拟合，
原生多模型计时不构成纠正迁移证据。最新工程探针只用于支撑最终实现选择。

| 子目录 | 证据范围 |
| --- | --- |
| [inventory/initial_2026_10_19/summary.md](inventory/initial_2026_10_19/summary.md) | 74条资产、24个基础权重存在；未加载权重，不是完整性或质量验证 |
| [service/trace32/summary.md](service/trace32/summary.md) | 真实32用户两种协议、29,860条行为的 CPU 事件检查；不是 GPU 服务性能 |
| [checks/summary.json](checks/summary.json) | 初始10项CPU测试通过，包含命令与输出 |
| [checks/gpu_stage_cpu/summary.json](checks/gpu_stage_cpu/summary.json) | 本轮25项CPU检查通过：事件、队列、会话快照、预算选择和成本口径 |
| [service/gpu_smoke4/attempt2/summary.json](service/gpu_smoke4/attempt2/summary.json) | 两协议×三方法，10,041个候选评分及72个边界状态与原路径一致；attempt1保留封装检查失败记录 |
| [baselines/rebuild_smoke4/attempt1/summary.md](baselines/rebuild_smoke4/attempt1/summary.md) | 四用户预算重建；0%/100%退化检查，名义预算与实际使用量分开 |
| [checks/control_match4/summary.md](checks/control_match4/summary.md) | 独立GPU入口的6,694个原生对照评分与任务身份一致 |
| [cost/initial/summary.md](cost/initial/summary.md) | 当前成本逐项复现与Q1/Q2形状预筛；没有新质量或速度结果 |
| [cost/scaling/attempt1/summary.md](cost/scaling/attempt1/summary.md) | 固定行为分布／模型／目录条件下的人口盈亏点，含来源、公共训练及容量边界 |
| [cost/workload_mix/attempt1/summary.md](cost/workload_mix/attempt1/summary.md) | 30k真实工作量形状占比；与原生三组件FLOPs逐版精确匹配 |
| [service/model_group_probe/attempt1/summary.json](service/model_group_probe/attempt1/summary.json) | 三模型规模×V2/V5、86个原生算子单元；平铺B8仅为硬件形状，不是独立用户服务 |
| [service/batched_conversion/attempt1/summary.md](service/batched_conversion/attempt1/summary.md) | 同8个真实用户合批减少码转换调用，数值与原地存储检查通过；不含共享准备和完整请求周期 |
| [inventory/calibration_scale.md](inventory/calibration_scale.md) | 实际部署池、历史拟合角色及可评价人口；固定人数与人口比例不可混淆 |
| [service/fused_conversion/README.md](service/fused_conversion/README.md) | 转换融合的慢点、改进和通用长度版本；原生KV不变，补齐工作另计 |
| [service/shared_timing/attempt1/summary.json](service/shared_timing/attempt1/summary.json) | 全目录准备、实际共享状态搬运与历史校准时间边界 |
| [calibration/fit_size_v4/attempt1/evaluation159_attempt2/summary.md](calibration/fit_size_v4/attempt1/evaluation159_attempt2/summary.md) | 同一固定开发面板上的16/32/64拟合人数比较，仅V4增量设置 |
| [calibration/fit_size_v4/combined_curve/summary.md](calibration/fit_size_v4/combined_curve/summary.md) | 五档16/32/64/128/256完整结果：64主AUC最高，更大档teacher MSE更低但主质量下降 |
| [service/time_model/window128_30k/attempt1/summary.md](service/time_model/window128_30k/attempt1/summary.md) | 真实30k按128人驻留窗口的批次、内存与迁移账；离线/饱和调度，不是在线时延 |
| [cost/window128_dot_tiles/attempt1/summary.md](cost/window128_dot_tiles/attempt1/summary.md) | 数学有效FLOPs与转换、读取dot补齐分列；不把逻辑dot当全部GPU机器指令 |
| [service/time_model/fair128_comparison/attempt1/summary.md](service/time_model/fair128_comparison/attempt1/summary.md) | 公平128实测与真实30k完整时间预测均未获得收益；负结果分解及零额外读写的乐观边界 |
| [service/graph_compute/README.md](service/graph_compute/README.md) | 双方同等图复用的共同模型执行诊断；8个真实源状态，固定形状，不代替会话或人口收益 |
| [service/batched_sessions.md](service/batched_sessions.md) | 公平实际会话、码追加融合及双方同等图复用；评分/状态验证和交错完整服务时间 |
| [calibration/joint_batch_probe/attempt2/summary.md](calibration/joint_batch_probe/attempt2/summary.md) | 原8人joint目标真合批，loss/梯度一致、单批约7.87倍加速；不是完整校准S |
| [calibration/joint_batch_adam/attempt1/summary.md](calibration/joint_batch_adam/attempt1/summary.md) | 原8人副本一次Adam一致性检查，权重未保存为候选；新函数保留原训练目标和批次 |
| [calibration/joint_fit_comparison/attempt1/summary.md](calibration/joint_fit_comparison/attempt1/summary.md) | 完整V4同目标校准：joint含验证64.75→12.26s，单分支111.89→59.35s；原执行复现冻结权重，合批最大参数差1.19e-7 |
| [service/time_model/source_accounting.md](service/time_model/source_accounting.md) | 核对离线来源重放与线上已维护状态的口径；不能直接从S扣除包含teacher的88.52s |
| [service/time_model/graph32_development_review/attempt1/summary.md](service/time_model/graph32_development_review/attempt1/summary.md) | 旧32图执行系数的30k开发预筛与瓶颈分解；不是最终128/30k性能证据，缺图准备和经验代理均单列 |
| [service/scatter_accounting_cpu/attempt1/summary.json](service/scatter_accounting_cpu/attempt1/summary.json) | 追加账本统计改用精确整数计数；CPU微测与真实几何一致，不代替新GPU会话计时 |
| [calibration/maintained_source_v4/cpu_prepare/summary.json](calibration/maintained_source_v4/cpu_prepare/summary.json) | 新准备入口的80人/因果历史/2名idle核查通过；全部teacher和idle物化的GPU计时尚未执行 |
| [service/arrival_smoke4/attempt1/summary.json](service/arrival_smoke4/attempt1/summary.json) | 6组真实到达pilot完成；22,542事件/19,776评分；发现首读与发布开销，未证明负载优势 |
| [baselines/rebuild_pilot32/attempt1/summary.md](baselines/rebuild_pilot32/attempt1/summary.md) | 32人五预算、双协议质量／实际费用；原反馈反例保留，匹配既有relay质量 |

[分阶段诊断](service/phase_probe4/attempt1/summary.json)已完成128/512/1024行块与原生对照，
确认小行块转换耗时和约4MiB工作区权衡。
[1024行块无插桩确认](service/arrival_smoke4_chunk1024/attempt1/summary.json)也已完成，
首读／发布改善但尚无总体服务优势；接续会话优化与拟合规模运行状态见STATUS。

后续结果按 `quality/<candidate>/<attempt>/`、`baselines/<policy>/<attempt>/`、
`service/<setting>/<attempt>/`、`transfer/<model_or_task>/<attempt>/` 组织，实际运行才创建。
每个运行目录保留实际配置、日志和最终摘要；失败或中断不覆盖原目录。
固定输入引用原路径，不复制基础模型、旧评分或历史数据。
结果各自限定证据范围；四人正确性不能替代人口质量验证，有限到达回放不证明生产吞吐。
