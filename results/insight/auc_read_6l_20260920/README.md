# 真实反馈 AUC：已被统一实验替代的开发记录

本组5872用户、两条边的请求时刻诊断没有完成AUC评价。两次pilot启动失败均保留：
[首次日志](pilot_attempt_01.log)因未设置包含`src`的`PYTHONPATH`，导入
`hstu_kvcache`失败；[第二次日志](pilot_attempt_02.log)在50.1秒完成历史加载，
随后完成V1→V2的两种共享规则拟合，但真实请求的Arrow `uint64 item_idx`未经转换即进入
GPU embedding，因索引要求`int64`或`int32`而失败。
[失败summary](pilot/summary.json)记录总耗时56.394秒和完整错误。
当前代码已将候选panel显式转换为`int64`；失败源码不回写覆盖修复后的代码。

当前活动实验是[统一四版本、10000用户AUC诊断](../unified_auc_10k_20260920/README.md)：
四条边共用发布时刻的1024事件快照及真实反馈，用五个panel展示统一质量和计算口径。
其完整四GPU评价已完成，212773条真实反馈、四边27条路径，运行498.896秒；
输入用48线程准备，完整结果与所有预算见上述统一实验入口。
以下保留本组原协议及最初资源估计，
不将它们解释为已完成的质量结果。

2026-09-20 前瞻协议。使用当前固定六层 Medium 的 V1→V2、V2→V3，
分别评价完整 E14 `[245,259)`、`[259,273)`。配置见
[auc_read_6l_20260920.json](../../../configs/insight/auc_read_6l_20260920.json)。

模型、256校准用户、128 pilot 用户、seed17、ridge=.01与每人16个无标签校准候选
保持固定。扩大评价采用原 development6000 排除128 pilot后的5872人，与校准及两个
保留确认组互斥；短历史保留。只读请求元数据得两边24887／26029条known反馈，
对应2629／2707个有反馈用户，均超过256校准人数的十倍。另有4399／4941条OOV请求；
无known反馈用户仍在人口账本中，不补造标签或按结果换人。

每条真实反馈请求使用实际候选item和timestamp，取严格早于该时刻的最近至多1024个
真实listen事件。Parent／Current分别编码完全相同的prefix；Reuse与修正规则都用
Current读取Parent缓存，Exact读取Current缓存。同一timestamp的全部事件排除在prefix外。
这是每个请求独立构造的固定状态读取诊断，不是从发布点持续append的rolling迁移。

新两臂在每事件响应率上拟合共享偏移或共享仿射响应：目标为`(teacher-native)/N`，
其中N为实际历史长度。服务时分别应用 `delta_r=N*b` 与 `delta_r=N*b+Tr`；
所有参数只在256校准用户上拟合，后者额外使用本层原生历史响应内容。
两臂均已有历史计数，不将第一组称为完全没有用户信息；两者均不输入q或摘要，
不获得评价用户教师来生成修正。校准N均为1024，评价N至多1024，短历史保留。
两臂同预算、FP64求解、FP32读取，依次拟合各层。原计划用独立128人pilot完成数据、模型数值
与资源检查后冻结校准参数并用于扩大评价；该步骤未完成，扩大评价没有启动。

已有q-only／q+r规则按原 `.pt` 文件与hash读取，作为历史比较完整报告；
它们不属于本次共享偏移／额外用户响应两臂对照，也不根据本次AUC重新校准。

主指标是全部已观测known反馈请求上不加用户权重的pooled ROC-AUC，与已有二分类
推荐任务一致；同时报告log-loss、Brier和相对Reuse的AUC百分点变化。
恢复率 `(AUC_method-AUC_Reuse)/(AUC_Exact-AUC_Reuse)` 仅在正gap严格大于1e-4时展示，
其他情况仍报告绝对值与带符号差值。两边、所有方法与失败均保留。
评分先读取不含标签的fidelity表，保存raw后仅给已评分请求关联quality表中的真实标签。

[历史资源估计](resource_estimate.json)：原计划GPU0单A40、完整执行少于20分钟、
显存预算12GiB。第二次pilot只测得历史加载50.1秒及失败前总耗时56.394秒，
未测得完整评分吞吐或整组执行时间。没有本实验的完成AUC结果。

数据入口为当前训练合同绑定的
`data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet`
和配套`requests_quality.parquet`；使用当前Medium数据manifest及item mapping。
本组来自开发人口，未开放final／confirmation或theta3；此前固定候选诊断完整保留。
