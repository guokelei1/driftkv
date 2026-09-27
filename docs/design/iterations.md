# 适配与诊断决策记录

本页保留改变实现、实验解释或证据范围的决定。详细指标、运行时间及完整失败统一链接
各结果目录；当前Design尚未实现和验证，历史探索不替代新验证。
[适配计划](plan.md)维护下一步，[模型训练](../unified_training_2026_09/README.md)
维护版本状态，[图表索引](../../figures/README.md)维护当前输出。

## 历史六层适配依据

native读出在冻结六层开发模型上有可测质量信号，但不证明native特有信息或源条件必要。
[质量报告](../../results/design/analysis/native_base_quality4091_01_report/report.md)
保留主方法、同容量摘要与无源状态对照；[FLOPs报告](../../results/design/analysis/native_flops_01/report.md)
与实际eager时间属于不同账本，理论人口摊销不等于实际加速。
[查询留出](../../results/design/analysis/mechanism_query_holdout192_report_01/report.md)、
[native输入](../../results/design/analysis/mechanism_native_input192_report_01/report.md)、
[解码闭环](../../results/design/analysis/mechanism_decoder_closure192_report_01/report.md)
用于检查机制，不构成持续多次迁移结论。
单边重初始化、同步视图、教师干预与真实状态寿命分别解释。

## 2026-09-16：对比原语与模型绑定

用户先要求调研三个方案，随后缩小到基本原语正确性，不做性能优化或最终流水线：

- LR采用连续层区间、真实入口hidden与部分重算；附加存储和继承误差单列。
- TR在保留旧前缀下真实重放尾部，不是target Full的精确尾部替换。
- KT采用top-k源层、跨head特征与K/V独立centered ridge；HSTU无RoPE，
  校准和全人口重写均计费。方法来源和公式见[baselines](../../src/hstu_kvcache/baselines/README.md)。

基本原语CPU检查通过；另以旧Medium V0/V1权重、seed17合成输入检验
k=2/ridge=.01拟合、1024历史完整端点、局部保留、淘汰后append及只读query。
该检查没有真实质量标签，源层排名为in-sample，不是性能或推荐收益。
首版输入仅无padding等长batch，正式人口、producer谱系和划分由调用方负责。

独立功能探针初版误绑定旧Insight 1模型与3000人口，用户指出应使用本轮6L/10L链。
随后改为读取当前规模manifest、词表、父子hash和admission，显式传入三组UID；
candidate bank由全部指定evaluation UID构造，不随batch/max-users变化。
Medium五边与Large前四边可按既有准入执行；Large V5@1的原失败保留，
不自动改选V5@2。接口只证明单边功能差异，不建立AUC或连续适配结论。
入口与使用方式见[脚本索引](../../scripts/design/README.md#独立-motivation-2-功能指标接口)。

## 2026-09-19：后端与完整训练步测量

用户授权优化训练/评价后接入Triton pointwise attention，
保留原权重格式、KV布局和PyTorch对照。修复reference/exclusive单token路径的
除数/self边界，legacy/inclusive快路和普通RecFlow decoder不变。
[数值与性能证据](../../results/backend_acceleration/2026_09_19/README.md)
区分backbone、生成头与完整候选评分，不能把局部数倍加速外推到完整训练。

[Max四卡归因](../../results/backend_acceleration/max_profile_2026_09_19/README.md)
在V2训练完整结束后、原评测前插入短诊断；未改正式源码或读取评价质量，
结束后恢复原流程。通信、重复大embedding梯度、rank等待和编译预热仍是成本；
fused AdamW的阶段节省不等于整步差值全由其造成。
这些测量不建立长期训练等价或当前Design有效性，方法成本比较应统一执行后端。

## 2026-09-20：共享读取问题的收缩与扩大

按用户要求先在本轮Medium最早两条新训练边V1→V2、V2→V3上做小型诊断，
固定seed17、完整1024历史、256校准/128开发评价，未见仅指本次拟合隔离。
最初比较每用户offset+rank1教师、共享query/response规则及加入均值PCA32摘要的规则。
[摘要试验及快照](../../results/insight/shared_read_6l_20260920/README.md)完整保留。

用户随后明确Insight只检验共享规则及个体信息，不检验具体摘要：
改为candidate item embedding与再加入native历史响应的纯线性规则；
校准UID独占归一化/教师/拟合，评价UID教师仅作测量。
[128人试验](../../results/insight/user_information_6l_20260920/README.md)
同时报告U（逐用户恢复均值）与P（平均误差恢复），负尾和失败用户保留。
逐用户近99%教师及旧动态93.39%不作为可部署方法结果。
读端线性变换可与部分Value映射等价，不能靠移动计算位置声称不同数学机制，
也不能从输入对照推断具体摘要必要性。正文改为直接报告设置与数值，不另选结果。

随后用户要求评价人口至少为校准的十倍，并调整共享对照：
固定 `b+Aq` 与 `b+Aq+Tr`，保持两边、256×16校准、32评价候选及ridge=.01。
q来自各自下层修正路径，高层q已包含用户历史，不能称第一臂完全无个体信息。
128用户作一次pilot；按day217成熟历史及既有无标签顺序选2560评价UID，
排除校准、pilot及保留确认群体。pilot拟合参数原样用于扩大评价，不据pilot重调。
分batch释放缓存避免整个人口双缓存驻GPU。
[完整结果与成本](../../results/insight/query_read_6l_2560_20260920/README.md)
保留两种指标、所有负值和原始分数；这是静态保真度，不是推荐AUC或连续适配。

## 2026-09-20：统一真实反馈 AUC

用户要求Motivation/Insight共用AUC、用户、历史、计算横轴和完整版本曲线。
[原配置及结果](../../results/insight/unified_auc_10k_20260920/README.md)
在运行前固定6L四条准入边、完整E14、10000用户和发布时1024-event快照。
评价UID由3970旧development与6030未用calibration-reserve构成，
排除原256/全部512拟合、128pilot及两个确认组；不读取final。
这是开发角色调整，不是新的独立qualification。

方法包括Parent/Exact/Reuse、校准选层的DroidSpeak、尾部重算、256教师/1–4源层KT，
以及32/64/128/256教师的 `N*b` 与 `N*b+T*r`。
候选bank固定来自原256人的发布前历史；选层与拟合不读取评价标签。
主指标pooled AUC，恢复只在Exact−Reuse >1e-4时定义；
成本包含教师、拟合/profiling、状态处理及真实请求额外读取，
分母为每用户一次Exact重建。固定快照不解释为rolling连续状态。

执行成本调整停止了首个单GPU历史载入，尚无校准或分数，保留serial_attempt。
48线程准备替换大IN查询后与原loader对照，四GPUcanary与单GPU分数/AUC一致；
改为同边四卡、边串行，未改用户、batch64、预算或质量定义。
原资源估计及修订记录均保留，完成结果不抹去失败或旧预算。

用户随后扩展教师到512/1024/2048/4096/7144：
新增3329名Medium成熟用户和3559名共享历史外部用户，总计3585名训练人口内/
3559名外部教师，与评价/pilot/确认隔离。固定原256候选bank、首256数组和评价人口；
只增加五预算×两臂，不改变原27路径。参考参数/logit和Reuse复算均核对。
[九预算结果](../../results/insight/unified_auc_10k_teachers7144_20260920/README.md)
保留全部预算；V1→V2负值及扩大教师没有单调提高恢复的结果不删。

## 2026-09-20：十层五边诊断

用户授权在冻结10L完整复测五种方法、全部五边。
[配置与结果](../../results/insight/large_unified_auc_10k_20260920/README.md)
逐条匹配6L的用户/时间/物品/来源，再用Large自己的映射与request ID；
末边13天。7144教师顺序、1024历史和预算不变，DroidSpeak扩展为十种宽度。
V5仍采用用户选定的1epoch端点，原bootstrap准入失败不改写。

独立96人pilot保留ridge=.001/.01/.1全部结果；主10000人固定.01，不据pilot改选。
初始串行拟合因成本调整停止，无评价分数，保留serial_baseline_attempt；
后改四边初始校准分卡、第五边排队，后续每边四卡。数值、选层和参数对照通过。
主样本上的用户历史分支只在前三边改善，pilot的4/5改善不能替代主样本3/5。
显示均值先逐边clip到0–100再平均；原始负值、超100和未截断均值继续保留。

## 2026-09-21：十六层诊断与失败修复

[Max配置与结果](../../results/insight/max_unified_auc_10k_20260921/README.md)
事前固定V0→V1 epoch1→V2 epoch1两边，使用Max自己的20万人口：
pre217成熟历史+固定SHA顺序选7144教师、96检查、10000评价，避开旧适配角色。
保持与6L/10L相同历史长度、真实反馈AUC、预算和理论成本定义，人口不同须说明。

首次正式任务在Droid选层Exact读取时OOM：8候选展开了256份完整KV，
此前8人profile canary和batch64评分检查未覆盖这次分配。尚无完整评价分数，
[失败目录](../../results/insight/max_unified_auc_10k_20260921/failed_initial_attempt_20260921_032712/README.md)
保留日志和原执行源。修复只改为每次2候选，仍为32教师×16候选、全部136区间；
新chunk8/2对照、正式256教师两边初始预检通过后复用其产物，未改选择目标或FLOPs。
用户先要求跟进出图，后允许后台交接；任务最终完成并从raw核验94组AUC。
增加教师的收益有限、共享修正的负恢复与各方法成本全部保留。

## 图表与记录维护

历史6L/10L合并preview按用户选择前三条边，先逐线clip再平均；
该选择均值不代表所有边。随后三规模15子图恢复已有全部边、取消均值；
当前五方法选边/显示规则仅由[图表索引](../../figures/README.md)及selection/ledger记录。
排版改动不重跑实验，也不替换原始恢复率。

2026-09-22将重复结果、启动进度和纯排版流水账从活动文档移除；
独有的协议决定和失败原因保留于本页及各结果目录。
[修改前全文快照](../../results/README.md#2026-09-22-文档精简)
保留原文字与历史预算，不作为新的执行指令。
