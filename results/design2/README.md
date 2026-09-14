# Design 2 结果

第一轮检测证据保存在本目录。设计与执行范围见
[docs/design2](../../docs/design2/README.md)。

最新：[选择性状态更新完整结论](sparse_01/analysis/conclusion.md)、[全请求参考评价](sparse_01/analysis/full_reference.md)、[真实质量和完整费用](sparse_01/analysis/report.md)。
`sparse_01/`保留所有计算候选、canary、真实连续回放和全请求FreshCurrent；`evidence_01/`保留原512/1024 UID的廉价观测校准与评价。
规则在新增全请求教师前冻结，确认集未读。完整框架已写入论文，未满足的正常费用、短历史和失败样本量目标没有删除。

此前：[Benchmark分层审计](analysis/benchmark_01/report.md)及[真实请求FreshCurrent评价](analysis/benchmark_01/fresh_current_report.md)。
规则与工作目标见[Benchmark定义](../../docs/design2/benchmark.md)。新增离线参考单列评测成本，冻结策略未改。

此前：[真实闭环与两个固定备选](analysis/lifecycle_variants_01/report.md)。
原1024UID真实M0—M5连续回放，主方案及仅费用、延迟执行各完整完成。
四边AUC：D1 .582077、立即D2 .583210、仅费用 .581822、延迟D2 .583353。
保留延迟候选：320次实际重建，服务增量比立即方案少43.24%；相对D1的区间下界贴近零，
不称独立确认，M1损失保留。完整准备摊在1024人口仍贵，规模外推与实际质量严格分开。
主实验详见[第一版闭环](analysis/lifecycle_01/report.md)，未用面板归零作为收益。

前轮：[共享源编码的精确225维状态二次型](analysis/conditional225_01/report.md)。
512/1024UID全部完成，判定零差异、零回退；开发检查2.318605TFLOPs，
含四目标全部新增准备3.847219TFLOPs，比仅公式短路＋准确求解净省72.06%。
原质量与漏检不变；单候选更贵，准备/共享存储与输入恢复费用单列，尚未执行真实重建。

前轮：[固定32方向分级检查](analysis/tiered32_01/report.md)。
512/1024UID全部完成，判定零差异；开发90.39%状态仍需精算，准备＋检查的核实下界
较原准确检查高7.66%。停止32方向候选，保留无需谱计算的公式短路。

前轮：[目标/长度条件残余尺度校准](analysis/scale_calibration_01/report.md)。
512UID交叉校准→1024UID开发，主AUROC校准几何.9262、仅背景.7450，预定推进条件通过。
保留80.20%状态的严重率.0406%，仍有2UID/3状态漏检；M5整组拒绝代价保留。
准确检查26.12%的成本不变，未实施压缩、分级检查或重建调度。

前轮：[几何增量与完整检查费用复核](analysis/budget_audit_01/report.md)。
1024开发UID的CPU条件置换支持主尺度的增量排序信号：5%重建预算原法100%覆盖，
纯费用64.49%，两种打乱平均48.19%/55.90%；0.1尺度的5%点原法略差。
全量检查占Exact-All的26.12%，加5%重建与已知准备后总量下界32.10%。
原始续用可信度路径仍停止，尺度校准、压缩与真实重建未实施。

前一轮：[中规模检测报告](analysis/detection_01/report.md)。
512残余校准＋1024开发UID完成，另128历史诊断及256拟合用户单列；
原始全局分数未达到预定续用判别条件，分数/费用有条件正面信号。
三角求解与存储费用高，尚未进入在线压缩或真实重建。

每次新实验使用独立 run_id；分析产物放 analysis/。
报告、紧凑配置和摘要按现有 .gitignore 规则保留，矩阵、权重、raw、日志留本地。
引用 results/design 的既有证据，不复制或覆盖。

- geometry_01：256原拟合UID的四目标H恢复、冻结预测一致性与分解检查。
- canary16_01：16名预定残余校准UID的完整四目标数值/资源探针。
- medium_01：中规模四GPU队列配置、日志及exit状态。
- detection01_*：各互斥UID块的原始状态/查询、源回放和检测费用。
- timing16_02：修复current-device同步后的匹配计时和逐元素输出核验；不重复计入样本。
- analysis/detection_01/：完整阈值固定评价、曲线、低分失败与结论。
- analysis/budget_audit_01/：冻结条件置换、配对UID复核、选中数量和检查费用；无新增前向。
- analysis/scale_calibration_01/：冻结12组残余校准、折外预测、匹配背景与开发评价；无新增前向。
- tiered32_01/：固定32方向谱准备、16UID探针、512/1024UID原输入回放与分级判定；有模型前向，无新教师输出。
- analysis/tiered32_01/：完整分级数量、判定等价、费用下界及候选停止结论。
- conditional225_01/：共享逆块与残差核验、16UID探针、四GPU原512/1024UID回放及完整退出记录。
- analysis/conditional225_01/：候选分数/区间核验、判定保持、1/2/4/16候选费用与净节省结论。
- lifecycle_01/：原1024UID四分支M0—M5真实闭环；8UID探针、共享准备账本、四GPU队列、实际请求/动作/费用。
- analysis/lifecycle_01/：主闭环逐边AUC与配对区间、后续服务及完整生命周期成本；没有用面板归零证明成功。
- lifecycle_cost_only_01/、lifecycle_deferred_01/：两个预先固定的后续策略，分别保留canary和原1024UID完整回放。
- analysis/lifecycle_variants_01/：主方案与两个备选的共同参考核验、真实反馈和统一费用比较。

初版非0号GPU组件计时的[修订记录](medium_01/timing_invalidation.md)保留；
相关原账本不删除，不能用于完整GPU时间合计。数值结果和理论计算不受影响。
# 此前：Benchmark先行

[Benchmark基线审计](analysis/benchmark_01/report.md)复用旧六方法原始输出与原512/1024历史，
盘点困难状态/背景状态、真实质量、FreshCurrent证据缺口及规模成本曲线。新增模型前向0，
费用机会是待验证假设，不是已经实现的节省；旧实验与失败结果全部保留。

# 此前：2048UID中规模复核

[本轮结论](analysis/scale_followup_01/conclusion.md)：自然2048UID，短压力各512UID，冻结检测器的实际连续服务。
scale_followup_01/保留六个队列、六个策略canary、三个参考canary、分片日志/exit、真实反馈/动作/费用及评价教师。
analysis/scale_followup_01/保留完整分组、配对区间、失败案例、两套准备费用口径、报告与verification.json。
费用接近用户目标，正常M4与cap32任务质量尚未通过。guard为一次付费核验备选，不是最终验收版本。

# 当前：计算机制优先

compute_01/保留真实读取复用、预付预算、FIFO和费用主导路径的canary/独立单元日志与余额轨迹。
analysis/compute_01/包含[计算报告](analysis/compute_01/report.md)、完整费用与预算漏检；质量不足原样保留。
整体判断见[阶段结论](analysis/compute_01/conclusion.md)：计算骨架可保留，有限额度内的检测增量尚未证实。
2048队列的前四个单元链接到已完成512队列，无重复前向；独立预算单元与GPU完成顺序无关。
