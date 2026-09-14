# Design 2 诊断与实验

当前论文材料：[选择性状态更新](../../docs/design2/paper_framework.md)及[完整结论](../../results/design2/sparse_01/analysis/conclusion.md)。
本轮新增入口（`PYTHONPATH=src:scripts`）：

- `evidence_probe.py` / `report_evidence.py`：恢复冻结C的实际特征，校验嵌套几何下界，512 UID交叉校准后评价1024 UID；不新增拟合教师。
- `run_sparse.py` / `run_sparse_observe.py` / `run_read_observe.py`：首次使用的源、背景、真实读取候选及付费提交消融。
- `run_excursion.py`：当前225维首次风险触发方法，以及完整1281维/背景/匹配首次使用对照；`replay_requests.py`仅提供因果写入后、评分前的状态动作接口。
- `run_exact_demand.py`：无检测、首次真实使用就重建的Exact对照。
- `queue_sparse.py`：已通过canary的候选分128 UID块，四GPU开发回放；自然2048及短窗口512，保留日志和退出状态。
- `full_reference.py`：冻结策略后，为全部真实请求计算合法Current参考；评测教师单独记账，不回传策略。
- `report_sparse.py`：全部真实反馈、正常组、配对UID区间及完整理论费用；其中旧检查点残余单列保留。
- `report_full_reference.py`：全请求主/辅助残余、后续写入/下一发布、剩余失败和旧参考重叠核验。

以下入口保留此前实验。不要按历史说明重新启动旧拟合或重新选择阈值。

冻结C检测、预算复核、残余尺度校准、固定32方向检查及精确225维状态收缩已完成。设计与实施问题见
[docs/design2](../../docs/design2/README.md)。

后续小型检测诊断、参考回放和结果汇总放在本目录，复用 scripts/design 的冻结函数
和模型/数据原语，显式输出 results/design2/<run_id>/。不调用硬编码写 results/design
的旧 main() 来启动新阶段，不复制整条旧管线。
先检验准确约束指标，再讨论敏感度、便宜计算和预算重建。
2026-09-08用户已明确授权第一轮中规模检测实验。

在仓库根目录设置 PYTHONPATH=src:scripts：

- freeze_protocol.py：仅用UID元数据固定512校准/1024开发及两个审查组，拒绝覆盖。
- common.py：复用旧数据/写入原语恢复原源状态日程，读取冻结C，不构造旧teacher。
- run_detection.py：mode=recover恢复原256拟合H；mode=evaluate评价指定role/start/users块。
- run_medium.py：通过数值与16UID探针后，在GPU0–3串行排队各自互斥UID块。
- freeze_thresholds.py：只读取512校准组，冻结全部预定续用阈值。
- report_detection.py：汇总完整UID集合，报告残余检测、对照、低分失败及描述区间。
- audit_budget.py：只读既有开发状态，比较几何/费用、纯费用和两种近长度条件打乱；记录全量检查FLOPs。
- report_budget_audit.py：只读预算复核summary，重新生成报告，不重复置换或模型执行。
- scale_calibration.py calibrate：仅原512校准UID，按冻结规则合并组、5折交叉拟合，再冻结参数和折外阈值。
- scale_calibration.py evaluate：核对冻结哈希后评价原1024开发UID；留存匹配背景、漏检、组覆盖损失和费用。
- report_scale_calibration.py：只读本轮证据生成报告，无新拟合或模型前向。
- tiered_check.py prepare：从只读Cholesky分解得到固定32方向、余项端点及残差/正交误差预算。
- tiered_check.py replay：恢复原C输入，依次执行公式短路、谱界与准确回退；不生成新教师输出。
- run_tiered.py：16UID探针通过后按原块边界在GPU0/1覆盖512/1024UID，保留日志与退出状态。
- report_tiered.py：汇总判定保持、分级数量、输入恢复与准备/检查费用，不重新执行模型。
- conditional_check.py prepare：从冻结Cholesky显式生成完整逆块及FP64残差预算，不拟合或获取教师输出。
- conditional_check.py replay：恢复同一C输入，按状态生成临时225维检测矩阵，逐候选求值，数值不确定时回退原求解。
- run_conditional.py：原16UID探针通过后，在GPU0–3按原UID块完成512/1024复核；保留日志和退出状态。
- report_conditional.py：汇总分数/区间/判定一致性、完整新增准备费用、输入恢复与多候选成本边界。
- run_lifecycle.py：从共同M0到M5的四分支真实回放；发布时完整费用判断、225维检测、真实KV重建，随后独立native服务。
- queue_lifecycle.py：原8UID canary通过后，四GPU覆盖原1024开发UID，按64UID分块。
- prepare_lifecycle_costs.py：只读512校准历史与既有证据，核算源回放、Current教师、校准及共享矩阵准备；不获取新校准教师。
- report_lifecycle.py：汇总实际反馈AUC、配对UID区间、发布动作、后续请求及生命周期FLOPs/人口外推。
- lifecycle_cost_only.py：在相同回放编排中明确替换发布决策为仅费用规则；较便宜的动作对照，不是同预算对照。
- lifecycle_deferred.py：发布时沿用检测判断，将已决定的重建延迟到当前版本首个真实请求前执行；下一版本取消未执行意图。
- report_lifecycle_variants.py：核对两个备选的共同参考分支逐元素不变，比较完整真实质量和费用；不再执行模型。
- audit_benchmark.py：仅读原512/1024历史与已有输出，按预定结构/活动分组重评分六方法，盘点FreshCurrent样本与完整成本曲线；不运行新策略。
- benchmark_reference.py：prepare固定真实请求检查点与合法历史；8UID canary后按目标四GPU计算离线FreshCurrent参考，完全不进入服务策略。
- report_benchmark_reference.py：对齐原六条轨迹输出，报告真实请求检查点的严重残余、原正常变严重、持续收益与评测教师费用。

默认run_id指向本轮证据，不直接重跑覆盖；新实验使用新ID并先更新研究问题与资源估计。

- run_scale_followup.py：复用冻结连续回放，支持预定首次请求demand/risk/hash30与32/128保留上限；冻结新增UID，不重拟合。
- queue_scale_followup.py：匹配canary后每次启动一个有限队列，四GPU、128UID分片，保留日志/退出码。
- run_scale_guard.py：risk提议后付费重算，对实际首请求核验才提交；所有尝试均收费，含独立canary/队列。
- scale_followup_references.py：固定2048自然及两个短压力的合法真实请求参考与原Benchmark分组，评价教师不进入决策。
- report_scale_followup.py、report_scale_benchmark.py：逐边/半组UID区间、失败与正常组、完整费用、原准备和理论分块准备两套口径。

本轮入口见[实施记录](../../docs/design2/scale_followup.md)，结果见[中规模结论](../../results/design2/analysis/scale_followup_01/conclusion.md)。

- run_compute.py：真实C首请求复用、H直接求解、128UID单元预付预算与FIFO；所有请求和各自KV沿原回放执行。
- run_compute_min.py：在同预算下比较实际几何与动作费用，保留免费公式放行，不引入新的长度/质量阈值。
- queue_compute.py：8UID canary后按固定128UID单元运行；2048扩容复用已完成的前512单元，不重复前向。
- report_compute.py：对齐旧共同参考，核对逐单元预算、完整费用、未处理严重状态和单元bootstrap；复用已有FreshCurrent。

计算机制的研究边界见[compute_mechanisms.md](../../docs/design2/compute_mechanisms.md)，不以费用下降代替保护验收。
