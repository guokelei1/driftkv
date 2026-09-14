# Design 2 开发配置

小实验配置独立放在这里。detection_01.json冻结第一轮UID、参考、分数、指标、
对照与研究判别；thresholds_01.json只从512残余校准UID确定，开发评价不重选阈值。
budget_audit_01.json在新置换结果前固定预算、误差尺度、近长度分组与增量判别，
并保留既有状态输入和分析源码的SHA256；只做CPU复核。
scale_calibration_01.json预先固定一次非负残余校准的分组、稀疏合并、5折UID规则及推进条件；
scale_calibration_01_fitted.json只由原512UID产生，保存折外阈值、最终参数及每折拟合，
开发评价检查其来源哈希，不重选组或阈值。
tiered32_01.json固定32谱方向、公式短路、数值界与回退规则；沿用完整80%阈值，
保持512/1024UID及输入面板。谱准备/回放证据另放results/design2/tiered32_01/。
conditional225_01.json在数值探针前固定共享源编码精确收缩、逆矩阵准备、数值回退和成本口径；
不改C/H/校准/阈值，复核原512/1024UID，不重搜维数。结果放results/design2/conditional225_01/。
lifecycle_01.json固定原1024UID的M0—M5四分支真实回放、发布点检测/费用规则和独立重建语义。
lifecycle_cost_only_01.json与lifecycle_deferred_01.json在各自输出前固定两个后续策略：仅费用重建对照、
发布决定后首个真实请求执行重建。保留主结果，不改检测器，开发比较不称独立确认。
设计范围见 [docs/design2](../../docs/design2/README.md)。

复用既有模型、数据、准入和 UID 证据路径，不复制不可变合同。
小开发探针保留必要设置即可；长训练、正式人口运行仍遵守仓库的前瞻约定、
资源估计、canary 与显式启动要求，不继承历史阶段的启动许可。
# Benchmark v1

benchmark_01.json 固定下一轮评测的工作负载、校准侧稀有活动边界规则、困难/背景分组、
质量与规模成本工作目标。它在旧总体结果已知后制定，不是旧结果的事前注册，也不修改冻结检测器。
解释见[benchmark.md](../../docs/design2/benchmark.md)。

scale_followup_01.json及scale_followup_01_uids.json冻结2048自然UID、两个512UID短压力、首次消费检查与哈希30%对照。
scale_guard_01.json是在risk正常M4负结果后、guard输出前固定的一次付费提交核验；不改C/H/阈值。
阶段结果见[中规模结论](../../results/design2/analysis/scale_followup_01/conclusion.md)，不是正式确认。

compute_01.json在新输出前固定真实读取复用、6%预付预算、FIFO及分阶段费用Benchmark。
compute_min_01.json固定最后一个按当前实际计算费用选择执行路径的候选；质量阈值及预算均不搜索。
