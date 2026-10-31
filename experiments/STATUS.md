2026-10-24 A+B本轮已完成、无GPU/后继：candidate_strength_direction/attempt_1 固定8929居民/edges2、3；A共同写/read-only九λ新算280块717489分数、4032/4073反馈UID。主候选45/38覆盖913/798UID，near0 19/13、strong1.5–2 20/25，bootstrap stable29/45、20/38，split一致14/36、12/29；同context支持仅47/31UID，标签比例/共同分数偏移明显，不能称候选身份作用或gate必要已证明。B全部100head、两口径、r1–32完整，每head7577143/7718811行；Δr6中位98.38–98.81%，native亦98.51–98.78%，uncentered Δr6超过两native仅K/V8/8及41/34head，不支持Δ普遍更紧凑。原绝对cache门失败经真FP64/逐head谱敏感性证实为数值差，新门和directSVD均过，旧失败保留；一次外部显存高峰OOM留档后原配置重跑成功。completion_audit=passed、GPU1 only、论文SHA未变；主报告experiments/docs/ch3_candidate_strength_direction_2026_10_24.md。原7000→8929扩展发生在任何新loss/谱前，人口和plan SHA未改。不恢复旧队列。

2026-10-24 simple_config_sweep/attempt_1本轮完成（无本轮GPU/后继）：width32/64/128×steps256/1024六组×target3/4全12点，旧W64/256复用；另W64/256 native-pair输入控制2点。共12个新quality/384块498288分数，949/932真实反馈UID，GPU1 only、论文不改。W128/1024 Direct−Chain两终点+.4306/+.5241pp逐点UID区间排0，反例W32/1024 target3−.7662pp排0，均开发探索非独立确认；target4 nativegap负，只比两path不讲恢复。nativepair KV MSE增10.09/11%但AUC协议差混合且CI跨0，不能讲overwrite必损。候选新增逐反馈loss分布/CDF全7000纯CPU：正/负/近零并存，主要随label和共享logit下移；同context正label805/813UID内差异小，不能称强candidate特异性。CPU/旧W64权重逐位/actual8/全块/端点/论文audit通过。主报告experiments/docs/ch3_simple_config_sweep_2026_10_24.md，两图及全部正反CSV在新run。

2026-10-24 用户新授权继续第二组简单配置（覆盖前条已完成范围，仅新simple_config_sweep/attempt_1）：固定原2048人口/target3、4/cached-prefix-gap，width32/64/128×steps256/1024，旧W64/256质量复用，5新配置；只Direct/Chain+原生参照，无新增模块，GPU1 only、论文不改、正反保留。先缓存实际fit行复核旧配方一致性，再actual8及>=512真实反馈UID质量。候选新steer为同一校准逐反馈Δlogloss分布，不给单候选算AUC、不以单次标签噪声宣称稳定候选规律。计划在新run/plan.json。

2026-10-24 新收缩实验已完成，无GPU/后继队列：lineage_candidate/attempt_1候选只复核旧7000固定Linear256按见过/未见过分组（commonUID1700/1629），不加候选输入；原生直接vs链式仅四路，在固定2048居民949/932反馈UID上完成原实时gap+唯一cached-prefix-gap两视图，4run/128块166096方法分数。Direct−Chain原两边−.001668/+.007239，cached-gap−.001464/+.008869，四个AUC区间均跨0；V4 native gap负，不能讲高恢复或多次必然退化，亦不证明持久化必要。原anchor/short已拟合但不评分，candidate ridge从未GPU跑；GPU1 only，论文SHA未变。主报告experiments/docs/ch3_lineage_candidate_2026_10_24.md，completion_audit.json。

2026-10-24 最新用户收缩：lineage_candidate/attempt_1只做两项简单对照。候选复核已完成7000固定Linear256按history_seen/unseen的AUC/恢复率，不新增候选模型；state final3/4拟合完成，只评reuse/direct/chain/rebuild，取消anchor及short评分。固定2048居民949/932反馈UID，GPU1 only，准备实际8数值检查与完整评分。scope_revision.json覆盖原计划范围，全部正反结果保留，论文不改。

# 正式实验状态

2026-10-24 05:16 新用户明确授权小规模GPU检验两条motivation，覆盖前条完成状态，仅限新lineage_candidate/attempt_1：①固定base写轨迹下额外候选描述符ridge/盲/洗牌真实AUC；②普通同预算MLP比较chain/direct/original+same_chainprev，源V1最终3/4，target4补同目标2/3跳。2048固定hash人口，e2/3/4反馈921/949/932、reserved零交；GPU1 only，先一组小配置，再据诊断调整保留全部结果。state为固定cache真实候选/gap机制case非serving，anchor额外两份KV字节如实报；论文不改，不新foundation，不恢复旧取消队列。plan.json与experiments/docs/ch3_lineage_candidate_2026_10_24.md为入口。


2026-10-24 00:42 两轮7000有界补测均完成，无GPU任务/后继：candidate_robustness与candidate_strength各2边、220块504272方法分数，共440块1008544分数；每边3168/3190真实反馈UID。6静态map+6档Linear256全局alpha均保留，主3属性同UID>=512、原4000/新增3000切片齐。alpha1→2整体AUC两边+.013626/+.026029且UID95%排0；alpha4→8两边共同退化；候选反向取舍不跨边/新人群复现，不能称gate必要/有效。Rebuild非真实AUC上界，e3 alpha2超其+.017265且区间排0，为开发结果非独立确认。完整三方法旧新分数逐位一致，GPU1 only；高强度FP32差已FP64及原scoretol论证，旧失败保留，新core未改，CPU最终2pass。两completion_audit=passed、图/CSV/全部反例已保存；主报告 experiments/docs/ch3_candidate_robustness_2026_10_24.md。后续应纳入调强全局映射为强基线，候选动机仍待更直接控制。本轮完成，不启动旧阶梯/nativecross/E6，不改论文、不重训foundation。


2026-10-24 00:25 candidate_strength完整7000评分已启动：PID467869/467870，两边同物理GPU1，alpha .25/.5/1/2/4/8，base为已完成linear256_rows。原actual8在近零cache元素超1e−5；全8对照score≤7.15e−7原门槛全过，sameinput FP64≤1.78e−14；仅新runner本进程cache_atol3e−5，旧core/math/scoretol不改，失败源/日志保留。fresh diagnostic8_validated、alpha1/Reuse/Rebuild旧新对照均passed，CPU后继已耐久等待两边完整summary。不要另开重复任务或改冻结源。文档ch3_candidate_robustness_2026_10_24.md，论文不改。


2026-10-24 00:06 新7000鲁棒性全完成：两边3168/3190反馈UID、220块504272分数；六种静态map/三属性同UID分析与原4000/新增3000切片齐。history e2线性map收益差区间排0，e3变弱且新增3000连同native-gap反向，热门/重复无稳定跨边规律。不能称统一map候选失灵。用户已授权更多配置；追加限定candidate_strength/attempt_1，同7000/两边、已完成linear256_rows权重alpha .25/.5/1/2/4/8，真实AUC对强度与相邻步取舍；全部alpha保留、不按结果选人/阈值、GPU1 only、论文不改。原鲁棒性完成、旧阶梯/nativecross不恢复。


2026-10-23 23:16 用户新授权补配置实验，candidate_robustness/attempt_1 active：原结论仅部分支持candidate motivation，保持原cost证据，不恢复旧阶梯/KV/原生cross。固定mismatch5000+sensA2000=7000，confirmation和全部fit/val/dev零交；前4000重叠，属敏感性非独立确认。原top20%热门不改，history common1700/1629、popular common801/811，secondary出现1次/重复>=2 common825/728均>=512。静态map配置原linear100row+新100user-balanced/256row/256user-balanced(可带diag参照)，F1/F2保留较强静态基线；同head64dim，val16选lambda，无test标签拟合。GPU1 only，新合batch无oracle/KV误差额外开销，源码/实际8数值检查后完整两边新评分。论文不改，不重训foundation；新plan.json及人口/属性已冻结，旧结果保留。

2026-10-23 21:42 本轮已完成并收敛：固定4000两边完整反馈(cap1024+128)4run/252块703920GPU分数、candidate-class AUC/恢复与CPU FLOPs均完成；completion_audit.json=passed，论文SHA未变。主候选linear，预发布history seen/unseen同UID990/913恢复点27.71/41.50%、57.68/71.55%；AUC增益异质性仅后一边区间排零，F2可改变比例关系，不能称所有静态map失灵或gate有效。热门20%规则未调，sharedUID465/451不足512仅诊断。固定Large1024row，F一次5.99GF；F+fullpair h在Q16=107.39GF=5.93×一次Rebuild（解析新增forward，排共同nativequery/fit），图/CSV已保存。当前任务完成，无GPU队列；nativecross/Q64从未启动，不要启动；E6/旧阶梯仍不恢复。主简报experiments/docs/ch3_candidate_motivation_2026_10_23.md；完整探索/反例另文保存。

2026-10-23 21:23 最新用户收敛覆盖前条nativecross：K/V不拆，**不要启动native_cross_quality.py或Q64补测**；它们仅代码/CPU ready，从未GPU运行，不是结果。当前重点为(1)原完整4000真实scores按预发布热门/长尾、本人cutoverhistory见过/未见过分组，简单Reuse/候选无关map/Rebuild AUC与恢复率，配对同UID并检查分母；(2)固定实际Large和历史快照，只变候选数，比较一次map与逐候选逐行修正的已有FLOPs。两项仅CPU，无新模型训练/GPU工作；原已完成第一批结果及所有反例保留。用户要最简单、直观、与Design有关的1–2条，不再堆rank/能量新指标，不force候选gate必要性或阶梯。见experiments/docs/ch3_real_quality_angles_2026_10_23.md。

2026-10-23 21:09 真实反馈第一批已完成，新增原生K/V端点诊断继续：real_quality_angles/attempt_1 四run(cap1024与128各两边)全exit0，252块703920原始GPU分数行、四份500UID分析及两份candidate-drift分析完成。cap1024 Reuse/F1/F2/Linear/Rebuild AUC边2=.671848/.676521/.678804/.676149/.681184、边3=.704639/.715270/.725577/.725307/.734606；不能把F1当静态映射上限。FIFO后gap淡出、不支持越追加越坏；候选embedding漂移未找到稳定异质性。原论文SHA不变。小cap成本原分母后端错已用独立cost_reference_corrected附件修正，不改原分数，主1024成本无变化。现在采纳用户外部建议前两项，准备固定4000两边native old/current K/V四格、同cutover复用64候选读，不修改已冻结旧源、不做row-rank。GPU1 only；旧阶梯仍不恢复。详见experiments/docs/ch3_real_quality_angles_2026_10_23.md及initial_batch_completion_audit.json。

2026-10-23 20:36 新真实反馈多角度探索进行中：用户要求不限单一映射精度例子，围绕Design寻找多个insight。real_quality_angles/attempt_1固定main2000+sensA2000=4000，fit/val/dev/confirmation零交叉，两边1838/1834反馈UID；只用GPU1，论文不改。先actual8核对合批native路径，再完整反馈AUC/logloss、强度/K/V/简单平移线性/时间与预发布分组，后续cap128/512检验FIFO后的上下文影响。数值检查首尝试allclose失败，诊断中，不能当质量结论。人口/覆盖已冻结，原实验和旧暂停队列不恢复。见experiments/docs/ch3_real_quality_angles_2026_10_23.md。

2026-10-23 20:08 拓展机制分析完成：insight_extensions/attempt_1新增6个main2000运行（query-gap、候选panel、校准源2x2各两边）均exit0，GPU1 only，当前本轮无GPU任务；相关35项CPU检查通过，权重/输入/源/角色与论文SHA核查通过。两个候选正向角度：broad item描述器解释25.07/10.49%中心化read残差，但仅占总read0.01905/0.004927%；near teacher次序不一致0.673→2.658%、0.775→3.403%。973/1027固定哈希组方向均保留（同权重敏感性，不是独立确认）。固定query gap后绝对中心误差增长消失，teacher尺度归一化也下降；不能讲历史越长越坏。源匹配无一致优势，near/logit可退化；全部反例保留。候选read输出rank1约99%仅oracle几何，不能推导每行code秩。完整文档含动机链、费用与图：experiments/docs/ch3_insight_mechanisms_2026_10_23.md；新root completion_audit.json及candidate_context_angles.png/pdf。原论文和上一轮原结果未改，不恢复旧阶梯。

2026-10-23 新机制实验已完成：`insight_mechanisms/attempt_1`中发布内两边与固定历史版本两案例各2000人全部exit0，共128块；500次配对UID分析、artifact/source/角色/论文SHA核查通过，CPU18项检查通过。仅GPU1，当前无实验进程。主发现：旧映射状态在新版本下相对各自source基准残差18.51→36.31%、29.49→104.82%，配对增量区间均正；新目标refresh另减误差32.51/62.30%，有新增预算/容量混淆。发布内总体误差反而下降，teacher新行对照额外减5.62/12.82%；不支持单调退化。candidate总体能量0.10–0.19%，不足证明gate必要。完整结果、成本、限制与3.3建议见`docs/ch3_insight_mechanisms_2026_10_23.md`，汇总与审计在run根`combined_observations.json`/`completion_audit.json`，两组PNG/PDF已保存。论文原文未改。不补五边、不恢复旧阶梯；本次有边界的机制探索目标完成，未宣称最终Design有效性。

2026-10-23 新目标已启动：用户授权重新做3.3机制研究，两条边有清楚案例即可，不要求五边一致。旧暂停梯队不恢复，论文不改，GPU1 only。新目录`ch3_motivation/runs/insight_mechanisms/attempt_1`，tmux `ch3-kv-insight-fit-gpu1`已启动两边新F-only拟合（full_item按头独立W64/N100/256步/lr1e-4，原main2000/val16）；拟合进程输出fit_progress/fit_exit与每边日志。教师强制新行、trajectory分叉和同学生q候选残差新模块已实现，相关CPU11项通过。计划见`docs/ch3_insight_mechanisms_2026_10_23.md`，人口质量评分须待实际Large数值/速度检查；当前未声称机制结果。

2026-10-23 14:30 用户明确暂停3.3：用户要求先停止实验，反思失败现状、路径假设及对Design的影响；后续一两条边明显失败即可暂停诊断，不再默认必须跑满。已停止本轮协调223543及实际评分229923，两PID均absent，未触碰他人任务，未启动新配置。`simple_trend_round/attempt_2/user_pause.json`记录暂停及13/15已完成；第五边F已完整2000用户、恢复−109.1727%，第五边FG/H未完成，不补造全五边摘要。全部权重、已提交块、负结果保留。目标暂停，未经用户明确恢复不要重启本轮、旧队列或新探索；下一步仅分析现有证据，论文不改。

2026-10-23 第四边退化诊断（尚未跑完、不改变当前队列）：`simple_trend_round/attempt_2/interim_diagnosis.json`保存实际F/FG负点、五边F仍选128步上限，以及容量问题。当前layerwise f共享64维隐层输出10头640维修正，h按头独立，存在容量混杂；它是待检验假设，不是已证实退化原因。只读核对发布准备/FIFO/新行写入路径，未发现重复准备f或把新原生行再修正的问题，不能将该检查当完整正确性证明。先齐15点，再优先用预发布验证检验廉价按头f、优化预算/时间覆盖；不因负边换人口、不改当前权重。

2026-10-23 14:14：简单5%一轮完成11/15点，第四边出现重要负结果：F/FG恢复−178.9111/−233.4789%，AUC .69585928/.68925193，同人口Reuse .71752272、Rebuild .72963122；不是仅分母导致的负值。第四边H已评首256人，评分229923及协调223543实查live，继续H及第五边，不因首个坏边改人口、换权重或中止整轮。全部F最佳步均为128上限，验证loss仍下降，不能称饱和。只用GPU1，先齐15点后检查拟合不足/时间覆盖/行映射容量与状态处理，再按预发布验证选择修正配置；当前未启动新分支。

2026-10-23 13:57：简单5%一轮完成9/15点。第三边完整2000用户F/FG/FGH恢复38.6662/41.1123/46.8025%；H−FG为AUC+.00240187139、恢复+5.6902pp，低于第二边+11.2487pp。首边H增量0仍保留。三个点同人口/端点均核对通过，不能用这三边替代全五边平均或三seed结论。PID229923实查live，已进入第四边重算端点；GPU1、配方/人口/权重不变，先完成全15点。

2026-10-23 13:35：简单5%一轮已完成6/15点。第二边完整2000用户F/FG/FGH恢复56.8622/66.8224/78.0712%，H相对FG的AUC增量+.00156036616（恢复+11.2487pp）；同人口匹配端点通过。第一边仍为21.9592/34.1505/34.1505%。只是两边单seed点估计，尚不宣布五边或统计确认。实际评分PID229923继续第三边，只用GPU1；不改配置、人口或权重，不启动额外分支。

2026-10-23 13:05：简单5%一轮已完成首边全部3点（各2000用户）：F/FG/FGH恢复21.9592/34.1505/34.1505%。首边H在验证集选择step0；完整10590条FG/H分数及事件字段逐条一致，记录在`e5/simple_trend_round/attempt_2/edge_1_zero_h_consistency.json`。评分PID229923实查live，3/15完成，继续后四边；不据首边调参或更换人口，仍只用GPU1。未完成整轮或最终目标。

## 2026-10-23T12:39:01.160612+08:00 上一512完整对照已收齐：h点增量15.58pp，但费用不合格；简单2k一轮继续

上一轮是实质进展（全五边cheap拟合/预算/数值门槛通过，2k实际评分开始）；本轮完成了旧单配方512收尾及独立核查，未新增配置。旧PID214313真实exit0，15/15 complete及原1000配对UID分析齐。

`query_centering_selected_analysis/attempt_1`是先前pre-release选择的full_item W128/N256/2048/cutover/center_std、固定原512、seed17全五边。f/fg/fgh恢复54.91940882/49.65953567/65.24259877%；g−5.25987315pp，h相对FG+15.58306311pp（mean AUC+.0022569104，UID95%[-.0011411268,.0049082962]）。h正点边2/3/4，边1/5为0。保留g负值与h区间跨0，不能称严格单调、稳定显著或三seed确认。此配置F完整230新增费1331.2319%Rebuild，H14069.6158%；gradient fit256/test512=50%，所以费用明显不适合作低成本主方案。正的h点趋势可作机制线索，不能据测试结果直接选它或把提升全归因于中心化。

已输出`development_comparison_aggregate/attempt_1/comparison_v3.csv`，旧v2/原摘要不改；现9套完整配置共135点。独立核查新摘要131项原权重/配置/score artifacts与源SHA，及新旧执行sidecar/legacy block绑定，通过。`comparison_v3_root_verification.json`记录；不同活跃人口那一行仍分开，不混作同人口重复。

当前用户优先的`simple_trend_round/attempt_2` **保持同一5%/N100/W64/128步/seed17配置**，2k实际评分PID229923与协调器223543实查live。首个edge1 F已完整2000：AUC .6133751871，恢复21.95921613%，是本边单点，不能据此宣布整组趋势好/坏或改配置；FG已提交首512块，其余继续。matched Reuse/Rebuild已本轮重算、分块保存，后续同边共享。原旧512已结束，GPU1资源回到这一轮，不恢复撤销的广泛队列，不增加准备/seed，先跑满五边15点并解释。完整目标仍未达成，goal active；论文不改。

## 2026-10-23T12:30:00.131102+08:00 5%简单配方五边已拟合齐、预算全过，实际2000人口评分已启动

上一轮为实质进展（按用户纠正取消广泛昂贵队列、启动低成本一轮）；本轮已完成**全五边15权重**，错误0。`simple_trend_round/attempt_2/cost_gate.json`真实passed_all_five_actual_budgets。fit100/val16/test2000、5%梯度拟合比例、W64depth1、每阶段128步、seed17的单配方不变。

实际cost总表`cost_preflight_points.csv`15行已输出，来源每边完整实际calibration+完整因果事件ledger、230读取解析费用；F完整新增FLOPs五边约27.36–27.44%Rebuild，FGH累计拟合五边约53.21–57.76%Rebuild（以CSV精确值为准），所有F完整成本及所有阶段累计拟合都低于Rebuild。H完整在线交互费很高，表内保留，不把拟合费当总费。尚无本配方全2000质量，不以费用或训练loss冒充趋势。

本轮实际训练edge2 H的原`check_projected`8UID核对完整通过，`numeric/summary.json` statuspassed、约30.03秒，原分数/读取/动态mask/因果prefix检查通过；是数值诊断，不是质量。没有扩大检查矩阵。随后自动进入原计划15点2000人口评分：协调器223543、实际GPU评分子进程229923实查live，command为run_quality_batched.py、固定quality_plan.json、output attempt2/quality、physicalGPU1。根progress.json目前仍写fitting completed5是旧阶段状态，不应据此认为评分没运行；以实际子进程和quality.log/评分块为准。

旧单配方512收尾PID214313实查live，13/15完整；旧大预算formal/E6/N256队列保持撤销。当前不新增配置、seed或准备工作，优先完整五边15点结果并解释能否支撑后续Design。全部只用GPU1，GPU0其他用户进程不触碰。论文不改，goal active/incomplete。

## 2026-10-23T12:24:23.510872+08:00 用户纠正优先级：先完成低成本五边趋势；5%拟合简单配方已运行

用户要求反思并优先一轮完整结果：用户组只需看出Reuse/Rebuild差距，拟合必须简单且费用可控，先观察基本趋势再决定哪些结论能与后续Design接洽；不先堆精度/预算/准备。保留人数/隔离/因果/全FLOPs/负结果等既有约束；3seed和独立人口留在基本趋势后的确认，**不是取消最终要求或宣布goal完成**。

已落实资源：按用户新优先级，显式撤销旧formal2k135点协调器213176/当前217341、E6协调器156641/当前217698，以及W128/N256/2048配对协调器178606/当前216123；`simple_trend_round/attempt_1/superseded_queues.json`记录全部命令。已提交weights/score blocks/完整native anchors全部保留，不把中断进程改成成功。旧大预算120拟合权重已完成，可作后续对照，但暂不再扩展开销。旧OS停止父153714的唯一旧子进程已经Z，本轮已TERM+CONT安全退役，防旧formal重复启动。仅保留新batched512 PID214313把已有单配方15点收尾，目前11/15完整；它不再依赖旧CPU父。

**当前唯一新配方**：`e5/simple_trend_round/attempt_2`，session ch3-simple-trend-5percent-cap128-gpu1，协调器223543，子进程随edge更换。每边100 fit +16独立validation、原固定main2000测试；梯度拟合占5%，教师准备含val共116/2000=5.8%，全部计费。layerwise_kv、一层宽64MLP、Q16、lr3e-5、每阶段固定128步、每32步验证、不作复杂earlystop（patience0）；同一f→fg→fgh人数/预算/目标，seed17，反馈候选75%+均匀25%、cutover、center_std、balanced初始H。共享模型/教师准备，原first-affine与batched append等价执行。不是新增损失、低秩瓶颈或底座训练。

先拟合全五边并输出实际完整cost，三阶段累计拟合均需<对应Rebuild，f完整230新增FLOPs也需<Rebuild；随后只在本轮实际训练H上跑一次8UID原projected数值核对，再同一原2k人群15点完整评分/原1000配对UID区间/逐点CSV和完整FLOPs（analyze_simple_trend.py使用已核对layerwise重复归一化扣除，不改旧账目）。单边拟合错误会记录并继续其余边，之后修正，不因第一错误就收工；全部15点跑齐后先解释趋势，再考虑重复与E6。H在线全部候选×历史交互的完整成本仍可大于Rebuild，是原动机的一部分，不能混同于拟合费。

**预算已按实际修正而非只承诺**：attempt1原256步第一边F完整30.5947%、FGH累计拟合100.6437%略超用户预算；已保留实际账目，按费用把所有三个阶段共同降为128步，在attempt2重跑，人口/宽度/目标均不改，不参考新质量来选。当前edge1/2均拟合完整且通过cost：F完整230为27.3558/27.3768%；F/FG/H累计拟合分别15.8751/19.5270/57.1185%、15.8281/19.4714/57.0607%，都低于Rebuild。新协议第一轮仍未有完整2k质量，不能提前说有递进。正在edge3，错误0。

**人口无需继续搜索**：同一原main2000的E3当轮fresh基线五边gap为+.01463772/+.01387147/+.04221114/+.01210874/+.01787775；人口文件和各边输入SHA与本轮逐一匹配。旧formal刚完整重算edge1 main2000的anchors也证实Reuse .6101608707、Rebuild .6247985355，gap .0146376649。与E3 gap差−5.49e−8，原始独立数值保留，不断言bit-identical；`attempt2/population_choice.json`记录来源。继续原固定2k即可看差距，不按f/g/h输赢选人口。新简单quality仍重算自己的matched anchors，不把E3分数直接当新方法分数。

先完整基本趋势、保留不同/不理想结论、评估其与Design的逻辑联系，是最新执行主线。不要恢复旧大预算广泛队列；全部任务仍只用物理GPU1，GPU0可能由他人使用。3.2不扩，论文不改，goal active。

## 2026-10-23T11:51:19.501338+08:00 优化路径已在512实际评分接续；新增seed随机性与负分母解释

上一轮有实质进展（性能profile、真实FG/H bit-exact证据、正式优化后继）；本轮新增实际接续及分析，仍不是目标完成。仅物理GPU1，GPU0不派发。

`e5/query_centering_batched_development_queue/attempt_1` 已耐久启动，session ch3-query-centering-batched-development-1，PID214313。在原edge1 f/fg/fgh均完整提交后，按规则SIGINT旧153897，旧进程已Z；保留的原3点通过summary/config、全部512UID覆盖、score block fingerprint及Parquet SHA核验，写preserved_original_point.json。`handover.json`记原completed3，旧日志KeyboardInterrupt是预定执行迁移，不是数值失败。新进程已完整复用edge2 F（第4/15点），并在新batched helper下提交edge2 FG首256块（138.68秒共享执行，只作进度，不能直接与异人口时间相除）。后续固定512权重/人口/端点不变；每点先绑定helper/proof/wrapper和旧committed block SHA，仅剩余用户走新路径；新执行记录额外prefetch行/总字节/瞬态内存，保留旧硬件与source，不改写旧块。当前4/15完整，半个人口不作质量结论。全15点后仍调用原1000 pairedUID分析器，并由新driver写原SINGLE/development_exit，明确completion_driver，不把旧中断PID记为exit0。原CPU父153714仍T以抑制obsolete formal；新formal协调器会在此完整marker出现后清理它，不要手动CONT。

**第一条边512已完整，需正确解释恢复率**：`e5/query_centering_selected_analysis/partial_edge1/summary.json`，固定512、243反馈用户、2710反馈，Reuse AUC0.7233709273、Rebuild0.7186466165，分母−0.0047243108。F/FG/H AUC0.7286299081/0.7282957393/0.7282957393，恢复−111.317/−104.244/−104.244%；这里方法相对Reuse的AUC点估计为正，不能把负恢复率简单说成降质。FG与H逐事件原始分数最大差**0**，h无增量确实成立。没有全五边或显著性主张；不据此换人口/删边/改端点。

**固定配置的seed差异已保存**：`e5/formal_seed_training_diagnostic/attempt_1/{points.csv,summary.json}`，绑定实际pt/JSON SHA，15行且保留未完成项，13/15完整job可读；这是own-validation训练清单，不替代正式完整审计或人口质量。edge1同N128/mix50配方，seed17/23 Hbest0且输出层非零参数0，seed29 Hbest80且825600输出层参数非零。FG/H loss17为1.7533369930/1.7533369930；23为1.8035547128/同值；29为1.6515294965/1.6456841249，对照等额延长FG为1.6514128088。说明不能把该边h零收益直接归为确定性实现失效，随机初始化/训练轨迹会改变验证选择；FGH本身联合继续训练f/g，故有extended FG控制，不把联合loss下降全归因于h。没有正AUC或机制普遍成立主张。

迁移even-main PID151411已真实exit0（SINGLE/even_main_exit.json）；原GPU1 main7141仍live继续最后两seed，curve45已完成。正式2k优化协调器213176和E6 Gram协调器156641实查live等待全15实际拟合；不重启、不复制不完整weight。layerwise配对178606仍live，edge2正在拟合；全五边90共同验证cells齐后再选配置。3.2不扩，论文不改，goal active/incomplete。

## 2026-10-23T11:35:26.528063+08:00 定位并消除追加输入逐条H2D热点；正式2000人口队列已绑定等价优化

上一goal轮为实质进展（新增layerwise实际数值门槛和五边配对拟合），本轮继续推进而非停留在等待。**真实CPU profile证据**`e5/quality_cpu_profile/attempt_1` exit0：固定development前8有反馈UID、只作执行诊断，无AUC；37.970秒中torch.tensor 6231次累计34.937秒（约92%），主要是原FastProbeTrajectory.advance逐事件item/behavior/delta的GPU小张量构造和传输。非模型MLP主要耗时，不能把慢运行误认为h数学必须如此昂贵。正式512旧任务保留，FG512已完整提交；H首256块也已提交，剩余仍live，不重启丢弃。

**独立优化**`scripts/ch3_motivation/batched_append_inputs.py`：每用户把本发布窗口的item/behavior/clamped delta数组一次传到GPU，原顺序逐事件取视图；原注意力/MLP/KV、因果追加/淘汰/成本ledger全部不改。输入数组含可能未消费尾部，额外传输/瞬态内存单列，不冒称方法持久状态。CPU7tests通过（F/FG/H、短/满窗口、FIFO/继承清零/异常恢复）；加执行绑定wrapper2tests，共9passed2.53s。

**实际FG/H验证已完整通过**：`e5/batched_append_readiness/attempt_2` exit0，edge1实际训练FG、edge2实际非零H，每项固定8UID，逐事件score key、最终完整K/V/cursor/source长度/ledger及全部操作shape核对；49/236行raw score最大差均**0**，最终K/V最大差均**0**，所有操作counts/shapes一致。两组共享GPU短测原/优化45.974/11.323秒、67.019/26.037秒；仅诊断，不是受控端到端倍数或人口质量。attempt1原先只在32GiB准入等待、未加载模型/无任何结果；基于现有实际projected H约8.5GiB峰值，将这项有界证明改为16GiB余量并在attempt2执行；attempt1waiting_attempt_superseded.json保留143，**正式人口准入仍32GiB**。

**正式后继已具体替换**：`e5/formal_quality_batched_queue/attempt_1`，session ch3-formal-quality-batched-gpu1，PID213176 live，当前waiting_formal_fitting。原135个预声明2k点/相同三seed与人口、同一f/fg/fgh及extended控制、原实际weight审计、原分析器保留；新output `formal_quality_batched/attempt_1`，analysis `formal_quality_batched_analysis/attempt_1`。必须原main GPU1 exit0+迁移even-main exit0+两curve exit0才开始，不把95权重当全部完成。20个源/实际proof绑定；`run_quality_batched.py`先提交额外execution configuration再运行原sweep，拒绝给旧无绑定sweep默默加优化，每次记录新helper SHA、proof SHA、实际prefetch行/字节/瞬态峰值，原source SHA不伪装包含新helper。wrapper/kernel变更在resume时拒绝。

为防旧后继重复派发，旧CPU父协调器153714仅在OS层SIGSTOP（stat T）冻结它的obsolete formal分支，**不是暂停goal**；其原development子进程153897实查仍R并继续提交512分块。新协调器在其development_exit真实出现后终止旧父进程，保留旧阶段记录。不要手动CONT旧父进程，否则可能重复启动原慢formal队列。

**layerwise五边配对仍进行**：edge1两组F/FG参数精确相同，通过paired_fg_audit。18共同cells齐：F2.3258431753、FG2.3016631650、Hstandard2.2578195290/Hbalanced2.2558630410。原full_item center_std edge1 H2.1528394899更低，不能仅因新h出现正教师增量就宣称新配置更好；完整五边验证后再比较，不挑未来AUC。当前已进入edge2，协调器178606 live、fit子进程211272（查ps会继续换）。所有正式/诊断/新配置都只用物理GPU1，GPU0不派发本任务。

E5正式人口与E6仍未完成，所有负结果保留。3.2不扩展，论文不改，goal active。

## 2026-10-23T11:10:15.241209+08:00 单卡接续已实查；新增按层KV归一化/初始化配对，实际数值门槛通过

上一轮属于实质进展：已完成单GPU1接续并保留提交结果。本轮进程再次实查：原GPU1 main7141、迁移偶数边151411、512质量153897和E6等待协调器156641均live；原curve7148已自然退出。独立重新调用原`quality.audit_weight`和first-affine执行绑定，共**95个实际权重**通过（main50/curve45），完整job main9/15、curve45/45。清单`gpu1_only/attempt_1/current_weight_audit.json`；不是正式质量完成。现有512质量已提交edge1 f及FG首256评分块，另半未完成，不以半个人口作质量结论。原有负结果继续保留。CPU采样器py-spy因ptrace权限无法附加，未重启计算进程；已从实际新评分块和GPU进程确认仍在推进。

**新增配置的原因和实现核对**：full_item中心化后仍存在h best0边，不能宣称解决递进。已有layerwise_kv让f联合看全头KV、一次输出全头，比逐头f准备开销小；h仍得到与f同样的扩大行信息，原拼接MLP/读取+分数目标/预算不改。旧full_item query_centering明确只支持full_item；新增独立`layerwise_query_centering.py`及`run_probe_layerwise_centered.py`，不修改冻结源码。关键是row_mean/row_scale重估后，按真实各头K/V切片同步`kv_output_scale`，不能把全头拼接前64维当每头输出尺度。与原Layerwise.fit_input_statistics逐项比较通过；不据此声称旧full_item有这一错误。

**先数值再拟合**：`tests/ch3_motivation/test_layerwise_query_centering.py` 5 passed，覆盖原尺度、查询矩、导数、实际嵌套优化/序列化/算术、best0精确保留。实际Large真实edge2模型8fit+2val诊断`e5/layerwise_centered_readiness/attempt_1` exit0：materialized和factorized梯度及最终参数通过，raw score maxabs2.384185791e-7，read relativeRMS6.642908559e-8，两者完整算术sum通过，均best8。诊断非零h输出std.001仅用于压力测试，没有产出可用于质量的权重。materialized/factorized peak reserved13.598/6.617GB，共享GPU短测9.289/5.565秒；不是受控整体速度结论，8人不是论文结果。

**已具体耐久执行**：`layerwise_centered_queue/attempt_1`，session ch3-layerwise-centered-pair-1，协调器178606、edge1 fit178804。本轮实查已经进入标准初始化F训练step320。GPU1 only、32GiB余量，fit256/val16/Q16/W128/depth1/lr3e-5/2048cap/pat1024/seed17/反馈75%/cutover/center_std，逐边各比standard和balanced h初始化；每边原始教师准备在两配方间共享，F/FG参数必须精确相同。每边6权重×3共同时间context，总30权重/90验证cell；五边齐后按等边/等context Hteacherloss选，tie优先standard。仅拟合与共同验证，不读取未来AUC，不自动替换正式配方；后续人口必须先核对实际训练H的原projected执行并比较共同验证。选择、成本和负结果均保留。`launch.json`冻结13个数值/诊断/共同输入源绑定。

GPU0现有PID175134属于用户lty的cacheblend任务（非本任务），未触碰；本任务所有live GPU进程仍仅物理GPU1。正式2000三seed/独立复核/E6仍待完成，goal active。3.2继续已收尾，不扩、不改论文。

## 2026-10-23T10:58:14.714747+08:00 用户最新资源调整：只用物理GPU1，GPU0已释放

用户先提出GPU0、随后明确GPU1也可以；本轮已选择**仅GPU1，GPU0空闲**。nvidia-smi实查GPU0占用0MiB。停止旧双卡后继协调器14207/22635/55899；中断GPU0计算7137/140879，保留已提交stage和评分块，原exit1/历史硬件记录不改成成功。旧quality GPU1子进程142617也已退出，单卡新进程正在复用其已提交块。不是实验失败后的丢弃重跑。

接续入口集中在`experiments/ch3_motivation/runs/gpu1_only/attempt_1/`：`resume_even_main.py`（PID151411，session ch3-single-gpu1-even-main）按原计划接续偶数边，实际checkpoint仍在formal_fitting/attempt_1/main，独立migration ledger记录物理GPU1；已完成stage通过原audit复用，未提交stage重新拟合。原GPU1 main7141/curve7148继续；原GPU0 curve已真实exit0，不重跑。`quality_queue.py`（PID153714，session ch3-single-gpu1-quality；development子进程153897）先接续原15点512，逐点配置/评分块校验后保留旧硬件metadata、后续实际GPU1，然后等待原奇数边exit0+迁移偶数边exit0，运行正式main/confirmation/curve的2000人口质量。原数值源码与人口角色不变。32GiB余量准入，同卡可并行。

归一化五边30权重/90共同cell已完整，按预先H教师loss规则选center_std：equal-edge loss RMS1.3061924463、center_std1.2350491023。所选实际H已做原projected检查；512质量仍进行中，不称趋势成立，不替换global475正式配方。

E6 Gram两协议实际8UID全rank对比都完成：每项3100score rows最大abs2.384185791e-7，root证据`e6/sketch_svd_adoption/attempt_1/readiness.json`，CPU5tests通过。原actual8 guards/exact checks/serial checks保留。新`e6/formal_factorized_queue/attempt_2/run.py`及launch绑定准备并已耐久启动，session ch3-single-gpu1-e6-v2；目前等待全部正式FG/H，**所有19项只派发GPU1**，原两卡排程仅作reference保留，新排程元数据单卡。每项先绑定新kernel/source/readiness及原configuration/block，额外执行审计后才分析；不是修改原source manifest或把共享时间当受控速度。旧attempt1数值证明保留，人口从未启动，无重复人口运行。

最新实际资源及source hashes见`gpu1_only/attempt_1/resource_change.json`。E5/E6目标仍active/incomplete；3.2不扩，论文不改。后继不能恢复GPU0，除非用户再次调整。

## 2026-10-23T10:23:37.253829+08:00 E6双协议实际8UID门槛已通过，55个正式权重已审核；归一化512后继与SVD诊断已具体排队

本轮是进展：实际timestamp8完整通过并与release8合并审核；新增归一化全五边共同验证选型→原固定512质量→完整cost/paired UID后继；新增分解热点实测与独立数值原语/实际输入检查，原正式配置与所有正在运行数值源码保持冻结。

**正式进度重新实查**：formal_factorized_queue/attempt_1/current_weight_root_audit.json截至本轮实查55个实际已提交权重（main29、curve26）全部通过原quality.audit_weight及新first-affine执行绑定；完整job31个，其中main5/15、curve26/45。不是120权重或正式质量完成。四个原拟合PID7137/7141/7145/7148均live，已进入后续seed/edge。formal_quality_queue/attempt_1 PID14207继续等待。不要重复拟合或把曲线源当主阶梯权重。

**E6两项实际门槛都已通过**：formal_factorized_queue/attempt_1/selected_readiness_audit.json，所用真实edge2seed17 FG/H、原固定8UID、全部rank/4variants绑定通过。release8满秩logit0、peak reserved23,844,618,240bytes、181.223s共享诊断；timestamp8共91快照、36400矩阵全部guard通过、无需exact fallback、最差energy lower bound0.9968307074、full-rank error4.768371582e-7、peak reserved23,731,372,032bytes、1388.449s共享诊断。**不是人口质量，也不是受控跨协议时间比较**。E6原协调器PID22635现等全部15正式FG/H；须全部实际权重到齐后才生成计划/成本/19项质量和分析，继续保留负/零收益。

**归一化后继已耐久启动**：query_centering_selected_development_queue/attempt_1，session ch3-query-centering-selected-development-1、PID55899。等两个multiedge拟合队列实际exit0、全30权重90共同validation cells完整后，按五边等权/三context H教师loss最低选rms或center_std（精确tie优先rms），不读取人口AUC、不自动取代global475正式配方。所选每边实际H再通过原check_projected 8UID核对后，原development512全15点两卡执行，原sweep内部同边共享native anchors，完整费用与1000共享UID分析输出query_centering_selected_quality/attempt_1、query_centering_selected_analysis/attempt_1。代码/输入源冻结，32GiB实测余量准入。

已有归一化edge1两模式6权重/18共同cell完整：rms f/fg/h loss2.1771525483/2.1771525483/2.1771525483；center_std2.1771525483/2.1528394899/2.1528394899。两H best0，仍是零增量，**不能称已解决全五边h趋势**。edge2新共同H改善约9.55%仅本边教师验证信号。其余边继续，当前无新512人口结论。

**效率排查保留更正**：timestamp8的seconds是整轮累计时间，不是单用户耗时；输入only selected8_workload_census.json显示个别用户31/39时间快照解释长块。进一步读调用链verified Counter确认：额外exact SVD spotcheck只在初始快照完成，不是每快照都重复，先前怀疑已纠正，**原正式verify-svd-heads2不改**。逐快照分块timing显示主要耗在随机分解本身。timestamp_verification_overhead/attempt_1在GPU0另跑同8、只去额外exact spotcheck（all-matrix guard不变）作诊断对照，PID61930仍live；非正式配置、不以共享GPU时间挑有利性能。

独立新候选sketch_svd_gram.py仅对投影后的宽小矩阵以FP64 Gram/eigh求SVD，保留原随机sketch/QR/截断秩/主效应/allmatrix guard/exact fallback。**尚未启用正式E6**，normal-equations小奇异值精度风险须实际检查。CPU零/秩4/满秩16三case通过。sketch_svd_primitive/attempt_1的6个synthetic矩阵原语case完成：典型10/25×32×1024短测原/Gram时间比57–165倍，但高斯近邻奇异值截断重构relative RMS最高1.19e-4（不是bit-exact）；只作原语诊断，不推导整模型速度或质量。衰减谱相对差约4e-6；短方阵不采用该路径。

实际模型检查已排队sketch_svd_actual/attempt_1，session ch3-e6-sketch-svd-actual-1、PID75992，**等两个原timestamp8参考退出且32GiB余量**才运行同真实FG/H/8UID/4variants/全rank/原exact spotcheck2；只拦截宽小投影矩阵SVD，原数值源码不改，补记execution_optimization.json source/counts，不伪装旧配置包含新实现。比较后继compare.py（session ch3-e6-sketch-svd-compare-1，PID78641）核对完整8个committed blocks及SHA、全部事件/方法key和所有rank分数，固定最大误差1e-5；失败保留，不能按结果放宽或直接采用。原E6正式队列继续使用原实现，无重复人口测试。

两卡持续99%附近，显存随N/阶段变化（曾GPU1约67GiB仍有13GiB余量）；新增SVD实际probe先串行等参考结束，避免在同卡再叠一份E6峰值。正式E5的2k三seed/独立2k/人数曲线与E6人口仍未完成，goal保持active。3.2不扩，论文和底座不改。

## 2026-10-23T09:57:56.497785+08:00 新归一化配对完整拟合与共同验证完成，E6实际正式权重数值检查正在运行

补充09:59实际E6发布时刻诊断已完成，并独立调用原audit_readiness通过：release_root_verification.json。实际选中FG/H、固定8UID、3200矩阵全部guard通过且无需exact fallback；满秩logit误差0，364次串行/批量比较最大logit误差7.152557373e-7，peak reserved23,844,618,240bytes（约22.21GiB），共享执行181.223秒。相同检查中的serial/batched rank scoring25.824/13.076秒仅作诊断，不能作为端到端性能倍数。timestamp PID40687仍live、已完成前3/8UID，日志seconds为本轮累计时间（不是单用户耗时），实际负载差异需按快照数/分块timing判断，不因观察超时重启。E6人口测试尚未启动，必须等timestamp实际proof和全部15正式FG/H。

上一轮是实质进展；本轮不是仅等待：新增8组完整开发配置统一CSV/核对，完成新RMS/center_std配对的18个共同验证cell及实际H GPU推理检查，已依据预发布验证信号扩展其余4边，并实际启动E6对正式FG/H的双协议检查。

**开发结果如实保留**：development_comparison_aggregate/attempt_1/comparison_v2.csv共8组/120点，保留原comparison.csv；comparison_v2_root_verification.json绑定全部源SHA，并核对selected128十个逐边增量。前7组同original512，active为不同发布前固定场景512，不混合/不作同人口配对结论。新selected128五边f/fg/h恢复56.6623645961/56.1452933888/56.9494870633%，h+0.8041936745pp，均AUC差0.0001164721642、CI[-0.0007043466740,0.0009744434631]、仅边2/3正。小宽度h+0.4368620141pp、active−2.6883745503pp，CI均跨0。尚无稳定预期阶梯，不按好边拼接。完整15点/两口径费用/共享UID1000次新selected128汇总在selected_fit128_mix50/。

**查询条件具体新证据**：query_centering_pair/attempt_1两模式edge2/N256/Q16/反馈75/**cutover**/W128/seed17均完整自然exit0。RMS F/FG/H bestloss0.2882776929/0.2659466943/0.2498053170；center_std0.2882776929/0.2315548798/0.2060597391。cap2048、pat1024不改。query_centering_common/attempt_2实际exit0，原fit64尺度、val16、3时间上下文、6权重共18cell：RMS f/fg/h meanloss0.3657135076/0.3308232698/0.3217627290；center_std0.3657135076/0.3233861888/0.2910298277。H共同loss约降9.55%，不是AUC或跨边证明。两F完整参数/rowstats精确相同，明确排除F不用的query_scale（两模式定义不同）。attempt_1因sha工具须Path而传str在源码核对前失败，记录保留，attempt_2只修Path调用，未重训、未改变比较任务。

query_centering_actual_projected/attempt_1/oracle原check_projected实际通过两套**真实训练H**，8保留fitUID、raw/projected/动态mask/append、原容差不改，两H非零；耗时52.365s共享诊断。wrapper最终哈希行同样str/Path错误导致exit1，**不将wrapperexit1伪称exit0**。root_verification.json另外用正确Path重新核对全部weight/model/runtime SHA、configuration==result.identity和两个原oracle passed，实际GPU结果有效，无需重复数值跑；未改任何原失败记录或权重。少于512无质量结论。

**已实际扩展其余4边**：query_centering_multiedge_queue/attempt_1，sessions ch3-query-centering-multiedge-0-1/1-1，PID43177/43180。GPU0边1/3、GPU1边4/5，各边RMS/center_std同N256/W128/Q16/反馈75/cutover/cap2048/pat1024/seed17；每边raw准备共享但每point完整独立费用保留。8新阶梯24weights，逐边随后在相同旧三context任务比较72cells；边2复用刚完成实际配对18cells，形成90cells前不做全局选择/跨边结论。它是独立新探索，原正式配方不改、不偷换正式权重，不读取未来质量标签。

**正式E5实进展**：formal_fitting/attempt_1首个完整edge2seed17五stage已真实提交，F/F_extended/FG/FG_extended/H beststep1752/352/1728/1728/1720，completed2048/1376/2048/2048/2048。其余15main/45curve两卡继续，formal_quality_queue/attempt_1(PID14207)耐久等待main双exit0后审核2k主/独立与curve各自45实际权重；尚未跑完正式质量。

**E6已实际启动**：formal_factorized_queue/attempt_1(session ch3-e6-formal-factorized-1)，先等实际edge2seed17FG/H，现2个诊断子进程已live：release PID40691 GPU0、timestamp PID40687 GPU1；固定feedback_union_2000的前8UID，Q230、r0/1/2/4/6/8/16、4KVvariants、projected、分解headbatch25，release额外串行/批量比较。原progress missing字段只是上一等待状态；以ps和selected_*8.log为当前依据，**不要重复启动**。检查完成需原audit_readiness确认满秩、全部矩阵guard、绑定与显存；全15实际FG/H到齐后才生成可执行E6计划，再计算同人口完整CPU费用，随后19个正式/512敏感性命令和1000配对UID分析。队列源码及全部新依赖冻结在launch.json；当前仅数值诊断，不是E6人口结果。

两卡共享执行，最新实际各约40GiB、99%利用，另4正式拟合与2新归一化拟合及2有界E6诊断同时运行；formal质量以32GiB实测余量准入，共享时间无受控性能结论。目标保持active，3.2不扩、论文/底座不变。正式2k三seed、独立2k、人数曲线、E6均仍未完成。

## 2026-10-23T09:39:13.032947+08:00 3.3 等价训练优化通过，三seed拟合/人数曲线与查询条件配对已启动

补充09:41：已实查首批5个真实正式权重（main边1/F与F_extended、边2/F，curve边1/边2的N256/seed17/F），原audit_weight与新execution_optimizations双核对通过，记录formal_factorized_queue/attempt_1/first_weight_root_audit.json。正式拟合仍未完成。两卡各3个拟合进程实际约40GiB，99%利用，约39GiB余量。新增耐久formal_quality_queue/attempt_1（session ch3-formal-quality-1）当前只等main实际exit0，随后审核实际权重及加速源码绑定，再做主2000×3seed75点、独立2000seed17的15点；curve实际exit0后做2000人的45点；完整后统一配对分析。每卡quality须实际>=32GiB余量，失败明确exit1，不以waiting plan开跑、不自动重启。此后继不是已完成结果；E6仍待实际FG/H与8UID数值/显存门槛。

本轮不扩3.2。原475共同验证已完整完成，选择calibration_followup/fit128（full_item/W128/N128/lr3e-5/cap2048/pat1024/fb75/gap_mix50/standardH），依据教师共同验证loss1.2061332204922413。冻结正式计划formal_calibration_plan/attempt_2保留。selected_development_queue/attempt_3已09:28:45自然exit0：15点、每边512、seed17；h差边1..5为0、+0.0005692716、+0.0007654240、−0.0004299415、−0.0003223933，区间均含0；**尚未建立预期阶梯，不据单个好点宣称成功**。原活跃与小宽度汇总队列均exit0，保留其负结果。

训练加速scripts/ch3_motivation/probe_first_affine.py与factorized_probe_fitting.py将h首层仿射按q/row分开求和，保持原非线性拼接MLP函数/梯度，不施加低秩约束、不跨步缓存可训练特征。first_affine_readiness/attempt_2实际Large 8fit+2val原/优化4步全部raw梯度、最终状态、分数、读取、beststep及完整算术delta通过预设容差；分数误差0，读取最大相对RMS3.915e-8；两轮32步诊断原/优化时间比1.246/1.306，显存8.146/5.566GiB。共享GPU短测不能作正式性能结论，少于512无质量结论。CPU12tests通过。attempt_1路径错误在优化器前失败，原失败记录保留。

**真实耐久启动**：formal_factorized_plan/attempt_1由原正式计划派生，附实际GPU proof及新helper/runner SHA；原配置/目标/初始化/角色不改。run_formal_factorized.py复用冻结runner，仅进程内替换fitter，每个新/旧权重额外核对execution_optimizations绑定；新算术delta已计入，不得再次扣除。formal_factorized_queue/attempt_1，sessions ch3-formal-main-0-1/main-1-1/curve-0-1/curve-1-1，实际PID7137/7141/7145/7148。两卡各main+curve并行，15main阶梯含extended controls=75weights，45F人数curve=45weights，60jobs/120weights。输出formal_fitting/attempt_1，目前真实在执行首边/第二边F，**不是完成、不是正式质量结果**。此配置基于验证选取；并行探索仍独立记录，不能据未来主/确认结果倒选配置。

**具体机制探索**：8fit查询诊断共同均值能量中位85.773%，候选差异相对总能量较小。query_centering.py保留函数族/目标，以自己的预发布拟合用户query mean/std调整优化坐标并折回原第一层bias；旧实际Large folding/projected checks通过，当前等价firstaffine核对亦通过，CPU实际optimizer/resume tests通过。query_centering_pair_queue/attempt_1新配对center_std(GPU0 PID9709)/rms(GPU1 PID9712)，sessions ch3-query-centering-center_std-1 / ch3-query-centering-rms-1，各edge2/N256/val16/Q16/W128/fb75/**cutover**/cap2048/pat1024/seed17，无future标签，不是原gap_mix50的单因素对照，尚未选择/评质量。两组同输入/损失/预算，只改query归一化，后续先核对F精确一致、共同验证再考虑五边及>=512质量，不用少人口作结果。

两卡80GiB，启动前实际全空，4formal进入训练时合计各约25GiB、利用率97-98%；新pair可共享余量，持续核对实际峰值。当前运行源冻结，不重复启动原队列。正式E5三seed/主2k/独立2k/人数曲线与E6均未完成，goal保持active；论文及底座不变。

## 2026-10-23T08:20:46.207451+08:00 小宽度首两个完整F实际账与零交互参数核对

新W32首边/第二边F均已完整512，native端点复用标记verified_cross_process_reuse=true，输入/模型/源码核对实际通过。完整费用分别86.7559%/86.0142% onceRebuild，**不是16.68%**；后者只是发布准备下界，不含固定校准/setup。整体成本目标尚未达到，原始费用不改。

另在没有人口质量计算的情况下审核新10个FG/FGH checkpoint全部状态：`small_width_calibration/attempt_1/nested_null_audit.json`。W32边1/2的H末层权重及bias精确全0，完整FG共享状态也和对应FG点精确相同；W64边1/3/5同样如此。因此有限输入下这些H是恒等零增量，W32最多3边可有h正点、W64最多2边，**不能把这个小成本分支当成可达到至少4边h正的正式主配置**。这是参数/函数核对，不是少于512的质量结果。仍保留其真实完整开发数据以核对成本/容量权衡；下一步主机制判断优先看原PhaseB最终配置、活跃场景和必要的更充分拟合/输入条件检查，不因这些低预算H为0而跳选有利checkpoint或放宽验证。当前原活跃/原PhaseB与小成本质量均继续，正式E5/E6仍未完成。

## 2026-10-23T08:16:37.929783+08:00 小容量30权重与90共同验证全部完成；正式512开发质量已启动（不是2k正式结果）

`small_width_calibration/attempt_1` 两卡及 `small_width_common/attempt_1` 两卡均实际exit0：两宽度×五边×三stage=30个真实权重全部通过原quality.audit_weight，三上下文90个共同验证cell完整。共同H教师loss W32=1.68461617815，W64=1.67603064608；按事先写入common launch的最低1%内优先小宽度规则，选择W32作为**这个新成本配方内部的开发候选**。它不表示优于原75mix/global475，后者尚待完成，不据测试AUC换配置。正式记录 `small_width_selection/attempt_1/selection.json`。

`small_width_projected_readiness/attempt_1/exit.json` 也已exit0；原check_projected实际两项均通过，W32独立非零压力副本与W64实际训练H的h_output都非零，覆盖真实读取/追加、动态mask及raw/projected执行。8UID只作数值/性能，压力副本绝不能作质量/E6。共享计时不作受控性能声明。

新质量计划 `small_width_quality_plan/attempt_1/plan.json` 已逐一重新审核选中15个真实W32权重，固定原development512，五边同UID，fit64/val16/Q16/feedback75/发布时刻delta/lr1e-4/cap256/pat128/val32/seed17，自己的因果轨迹、完整校准/准备/读写/setup账。队列 `small_width_quality_queue/attempt_1`，worker4051666/4051669，sessions `ch3-small-width-quality-0/1`，GPU0边1/3/5、GPU1边2/4，32GiB实测余量准入。当前实际已开始，首边F已全512/2710事件提交；不以单stage/单边作阶梯趋势结论。

执行优化明确复用本goal先前重算的原development512 native端点，不复用不同人口/模型的旧结果：ShardLoader实际验证原F点、native分块/SHA，核对固定UID、输入SHA、20个runtime sources、backend、模型文件真实SHA和anchor key；验证后如已通过的初始化比较入口一样预置shared_context，三个新探针仍各走自己的真实轨迹并重算完整账。source明确verified_cross_process_reuse和原summary SHA。模型/数据/代码不匹配即失败，不降级复用。跨边释放旧GPU对象和allocator，避免自己的保留显存阻碍后继准入。

每边原分析run导出summary/increments.csv；完整五边后CPU队列 `small_width_aggregate_queue/attempt_1`（worker4054246，session `ch3-small-width-aggregate-1`）导出到 `development_comparison_aggregate/attempt_1/small_width32_fit64_cutover` 的质量、完整费用及配对UID汇总。当前它在等两卡质量exit0，尚不是结果。

原活跃场景及PhaseB后续继续；全局475共同选型和正式2k三seed/独立复核/人数曲线、E6仍未完成。不得把这些新单seed开发/诊断替代正式要求，goal继续active。论文及底座不变。

## 2026-10-23T08:09:22.155169+08:00 五种完整开发配置统一汇总；小宽度准备检查通过并真实拟合

本轮是进展：新完成5组统一开发CSV/冻结协议与初始化五边分析核对，识别实际f准备费用下界，完成6个新宽度GPU数值/性能case并启动10个实际拟合阶梯，而非只等待旧队列。

原三组CPU后继 `development_comparison_aggregate_queue/attempt_1` 已于08:03:04自然exit0。当前完整五组统一表在 `e5/development_comparison_aggregate/attempt_1/comparison.csv`（原75mix、75经验间隔、50mix、平衡初始化、冻结前级）；原development512/seed17，共享人口与部分权重，**不是5次独立重复**。h恢复增量依次+2.3624/+2.2303/+1.9673/+3.5095/−0.2499pp，均值UID区间均跨0、正点都是边2/3/5。平衡初始化h均差0.0005082879，CI[−0.0001979383,0.0011083851]；冻结协议h均差−0.0000361894，CI[−0.0009031498,0.0007578704]。没有建立预期稳定大增益，不能按正边拼接。两组各10个逐边g/h点与CI已和原单边analysis核对，root_verification.json保留；新f/fg/h与成本CSV均完整，未改原数据。冻结协议2×forward backward仍是保守估计，不宣称实际节省。补充README说明边界。

**成本原因有实际依据**：`f_preparation_cost_floor/attempt_1/{analyze.py,points.csv,aggregate.csv,summary.json,README.md}` 用原完整512五边prepare_shapes，逐边精确复现full_item128已提交准备FLOPs。仅发布准备/onceRebuild：full_item32/64/128/256为16.6823/32.8258/65.1129/129.6870%；layerwise128/256为24.1920/47.1987%。这是解析规划且不含校准/setup的下界，替代配置无新质量数据，不外推2k/30k。原128仅靠减少训练人数/步数不能将发布准备降至22%，因此引入小容量成本配置，不故意弱化同组对照。

**新小宽度实际准备检查已成功**：`small_width_resident_readiness/attempt_1/exit.json` exit0，原 `check_resident_probe_fitting` 保持原文件/数值内核不变，仅进程内配置WIDTHS32/64与lr1e-4；真实Large edge2、fit256/val16/Q16，6个F/FG/FGH梯度、最终状态、原report/RNG/counters与2轮32步计时对照全通过。各case实际peak allocated13.36–17.33GiB；共享GPU时间只作诊断。dispatch源码与实际参数记在oracle/summary.json，未写质量权重、未算AUC。

**新拟合已真实执行**：`small_width_calibration/attempt_1`，sessions `ch3-small-width-calibration-0/1`，协调器/worker4020944/4020949。先等上述真实exit0/6cases通过，再以32GiB实测余量进入拟合；GPU0边1/3/5，GPU1边2/4。两组完整阶梯：full_item宽度32/64，fit64/val16/Q16/反馈75%，**原reserved64候选池与发布时刻delta**，lr1e-4/cap256/pat128/val32，F/FG/H同人数/上限/目标，seed17。它是单独的成本配方探索，不是“相对原75mix仅改宽度”的对照，也不替换final475选择。10阶梯/30权重，完整原校准账保留。

run_sweep.py 调用原run_probe_variants.run，配置其fit_probe为已审核resident helper，metadata写入正式execution_adapter binding；输入reader使用同一个restore_pack并缓存一条边的不可变CPU包，不再解析原入口会丢弃的未来requests数据。模型/教师准备跨两个宽度共享，各point仍计完整独立准备费用；数值核、目标和原源码文件没改。每阶梯完成后对三权重调用原quality.audit_weight，执行dispatch文件绑定原driver、resident helper、配置来源与实际SHA。首边/第2边W32与W64均已完成且三权重audit通过；另有first_f_weight_audit.json独立核对首两个真实F。小步诊断不是质量证据。

后继共同验证已耐久等待：`small_width_common/attempt_1`，workers4024255/4024258，sessions `ch3-small-width-common-0/1`。各卡拟合全部成功后调用原三时间上下文/fit64归一化/val16评估，2宽×3stage×3context×5边=90cells；不读取未来质量标签。预先记录只在完整90cells后按等边/等context H教师loss最低1%内优先小宽度，作为后续开发候选，不是正式配置选择或成功门槛。当前未自动启动新小宽度512质量。

**新宽度quality内核也须单独检查**：`small_width_projected_readiness/attempt_1`，worker4036121、实际原check_projected子进程4036338，session `ch3-small-width-projected-1`，32GiB余量准入。用实际edge2/W64训练H及单独W32非零压力副本，8个保留fit UID，核对raw/projected分数/读取/追加与动态mask。W32压力明确diagnostic_only、quality_use_permitted=false，随机填最后H输出层，仅用于非零分支数值压力；原质量权重未改，**绝不能把该副本用于质量或E6结论**。固定原容差，不按结果放宽；实际W64权重保留原状态。共享计时不作正式性能声明。该检查还在运行，失败则先修执行，不直接扩质量。

新活跃场景两卡原质量继续，原PhaseB已到GPU0边4fit128/GPU1边5width256；最终475选型、原2k/三seed正式E5、独立人口/人数曲线及E6仍未完成。论文与底座不变；所有当前运行源码保持冻结。

最新追加实查：新活跃组第1边三端点已完整512，后继f/fg/h worker4004269实际运行；第2边三端点worker3999550继续。真实第1边subplan的UID-list SHA与cohort-manifest SHA已分别重算核对，actual anchor512成员与冻结开发组精确一致，见质量队列attempt_2/metadata_verification.json。两卡当次均100%利用，显存约14.4/20.5GiB。

## 2026-10-23T07:47:52.465357+08:00 候选50%五边开发汇总已核对，新活跃场景固定人口并实际启动

上一轮为有效进展：替换了仅等待的后继协调器、审核62个可复用点及514文件SHA。本轮进一步完成候选50%五边/15点开发汇总和独立root核对，并新增实际执行的场景检验；不是只重复状态。

候选50%两卡均自然exit0（GPU1 07:24:03、GPU0 07:42:18）。`e5/development_comparison_aggregate/attempt_1/feedback50_mix50` 包含15点CSV、30条两种口径完整成本、6条成本汇总、配对UID1000次报告和README/root_verification.json。五边f/fg/h恢复54.5260/54.1503/56.1176%，g−0.3757pp、h+1.9673pp；h只在2/3/5正点，平均AUC差0.0002849208，CI[−0.0005480221,0.0009060879]。不能认为候选50%解决了预期h增益问题。root核对10个逐边g/h点和区间与原单边报告一致、五边差后比值、费用先和后比值。固定校准费用仍高，不作低成本或正式结论。

新增CPU耐久汇总 `development_comparison_aggregate_queue/attempt_1`，协调器3976539，session `ch3-development-aggregates-1`。feedback50已实际完成，等balanced_h和new_term_only各自五边真实quality exit0后汇总；全组各15点，不用缺失或局部正边当结果。原数值源码未改。

**新独立场景已固定并执行**：`prepare_active_development.py` 只用既有第1边发布前特征（历史≥512、近14天行为≥32、反馈≥4），从原reserve中排除原fit/validation/development/main/confirmation/mismatch、两个人口敏感性组、原各边E6预留与feedback_union_2000成员，余16206候选。固定SHA256 e5_active_pre_v1_seed17:UID排序，先写cohort.json冻结512开发+2000补充主组+2000补充独立组，然后才读未来请求。五边使用完全相同UID，不交集未来活跃度、不按模型分数选人，原正式角色不变。这是单独探索场景，不能混称原主/独立人口的正式结果。

新输入 `active_pre_first_release_inputs/attempt_1` 已于07:36:23 exit0：五边各1040打包用户（512开发＋512原fit＋16val），只补载缺失UID，原fit/val历史逐元组签名一致。`cohort.json`冻结全部4512个互不重叠用户，当前只准备/评测开发512，未读取两个新预留组的方法分数。重复冻结结果精确一致，人口敏感性角色确实排除。五边均由原scenario loader实际审核输入/角色/校准SHA，记录 `active_pre_first_release_anchor_readiness/attempt_1/summary.json`。

新增 `qualify_active_development.py` 沿用已验证的fast baseline/三端点配对分析，仅明确核对这个512固定场景，允许负或不确定端点，不做positive资格筛选。质量队列**只运行 attempt_2**：`active_pre_first_release_quality_queue/attempt_2`，协调器3998826/3998830，sessions `ch3-active-pre-v1-quality-0-v2/1-v2`。GPU0边1/3/5，GPU1边2/4；按32GiB实测余量与原拟合/质量共卡。先测相同人口oldFull/Reuse/Rebuild，再用原75mix/fit256/W128/seed17已审核冻结权重跑f/fg/h，最后各边1000UID分析。此对照只改评测场景，不额外训练或自动替换最终配置。输出anchors/quality/analysis分别在 `active_pre_first_release_{anchors,quality,analysis}/attempt_1/edge_e`。

attempt_1启动后发现尚未发出的计划中development_uids_sha256错误地取整份cohort文件SHA；已停旧协调器、确认两个anchor子进程终止后建立attempt_2，将UID-list hash和cohort-manifest hash分开。旧attempt保留cancelled.json；原数值源码/输入/权重均未改变，同一native输出支持分块续算，没有重复活跃进程。**不要重启旧attempt_1**。新workers3999547/3999550已实际继续；目前只是部分256人进度块，不作质量结论。补充场景全五边完成后CPU汇总 `active_pre_first_release_aggregate_queue/attempt_1` 自动导出完整质量与费用，不与原场景混合平均。

正式E5三seed/原2k主与独立人口/人数曲线、E6仍未完成；PhaseB最终475共同验证选型及各后继继续。当前不修改论文、不重训底座。

## 2026-10-23T07:32:07.422132+08:00 后继评测复用优化已生效；初始化比较四边完成，保留混合结果

仅替换了仍处于 waiting_final475_selection、没有任何 GPU 子进程的 selected_development 等待协调器。旧 attempt_2 的取消原因和原来源保留，新队列为 `e5/selected_development_queue/attempt_3`，tmux `ch3-selected-development-3`，实际协调器3972164；未停止任何拟合或质量进程。最终475共同验证选型仍未完成，新的等待队列没有提前生成配置或启动正式训练。

新增 `prepare_selected_development_reuse.py` 在真实最终配置产生后，按每条边每个方法的完整 checkpoint SHA 查找本goal已提交点，用目录链接建立只读候选索引；选择过程不读取分数。之后仍调用原 plan_selected_development.generate 的完整权重/输入/代码/用户/分块/native anchor 审核，不匹配的点正常补算。CPU实际核对62个完整512候选点、514个来源/分块文件SHA全部通过，记录 `selected_development_reuse_readiness/attempt_1/summary.json`；这不是保证最终配置一定有62个可复用点，实际复用数需等最终配置确定。原数值与审核源码未改。新后继按32GiB实测空闲显存准入，可与其他任务共卡；保留实际准入时间/余量，运行时间不作受控计时证据。

初始化标准/平衡比较边1–4全512完成。balanced−standard H AUC差依次0、−0.00003503、+0.00007936、+0.00081249，后三边区间均跨0，不能认为初始化已解决h问题；边5等实际拟合源。候选反馈50%第2/4边完整512分析已出：g/h差第2边+0.00086179/+0.00058153，第4边−0.00097115/−0.00257965，区间均跨0。其余点仍运行，不能挑正边当全组结论。

实查PhaseB两卡原worker3647467/3647468继续，分别进入边4 cap1024与边5 reference；候选50% GPU0 worker3875110已有7/9点完整，GPU1六点完成。初始化和冻结协议第五边等待真实源提交。最新实测两卡均100%利用率，显存约15.8/29.9GiB，占用有余量，采用共卡而非强制空卡。

另对既有发布前场景定义仅做可行性计数：`pre_first_release_scenario_feasibility/attempt_1/summary.json`。只使用第一条边发布前的历史/活跃/反馈阈值与原reserve角色，不交集未来各边活跃度、不读模型分数、不冻结新人群或启动GPU；作为后续场景检验备选。原固定512、main/confirmation角色不变。E5正式三seed/2k主与独立人口/人数曲线及E6尚未完成。

## 2026-10-23T07:12:01+08:00 两套时间采样五边开发汇总与完整费用导出完成，初始化比较已真实启动

原时间采样quality两卡均自然成功：mix50 GPU0在06:36:12退出0，empirical100 GPU1在06:41:48退出0；每组15点、五份512人/1000UID逐边分析全部完成。没有把该单seed开发组当正式2k/三seed结果。新的独立 `analyze_development_ladder.py` 已对真实30点逐一审核source权重／模型／分块、角色与统一配方，五边共享UID multiplicity后先求等权AUC差再比值；不调用或绕过正式2000人/三seed/extended-control gate。

混合时间组五边f/fg/h恢复点55.1387%/54.1342%/56.4966%，g−1.0045pp、h+2.3624pp；h仅边2/3/5正点。h平均AUC差0.0003421508，CI[−0.0005555317,0.0009705126]。经验间隔组46.6752%/47.4257%/49.6560%，g+0.7505pp、h+2.2303pp，h仍仅边2/3/5正点；h平均差0.0003230215，CI[−0.0003979261,0.0008776007]。当前不足预期h≥10pp/至少4边正参考，不宣称成功。共同native平均gap0.01448310，CI[0.00423248,0.02348794]，但1000draw中5个非正，遵循既定规则不报告有界恢复率CI；绝对差区间保留。

输出 `e5/temporal_comparison512_aggregate/attempt_1/{mix50,empirical100}/`：各含15点points.csv、30条两口径原始/修正cost_points.csv、6条cost_aggregates.csv及完整summary。正确使用原共享行归一化校准修正，所有原权重/score/summary未改；完整准备、校准/验证、读取/行为写入和setup账保留。root独立核对两组20份逐边g/h点与CI和原单边analysis一致、恢复计算顺序正确、费用先求和再取比值，记录 `root_verification.json`。开发512固定校准费用很高：mix实际账f/fg/h约1395.20%/2419.53%/24483.41% Rebuild；不能套成2k或30k人口费用，也不能宣称当前f是低成本点。汇总说明在 `.../attempt_1/README.md`。正式人口必须另外完成完整账。

针对已有balanced h初始化，新增**只复用经过精确证明的点**的 `evaluate_probe_initialization_pair.py`：原normal audit逐个审核reference/balanced六权重、完整身份/角色/目标/来源；f/g所有参数/缓冲与完整校准ledger必须精确相同。旧fresh-goal逐点来源仅在架构/全部状态/实际step与完整校准账均相同后复用，否则重算；旧512分块/config/native anchor均由ShardLoader真实SHA审核。实际CPU检查边1/2/3均通过，每边3个reference点可复用，只新增balanced H一项，而不是重复跑两套完整阶梯。初始CPU边1的旧check输出保留，最终code额外冻结所有runtime来源并保留E14_partial。

新 H 评测复用本goal已重算的同人口native anchors：在原evaluate.run的shared_context预载经审核相同key/输入/源码/用户/模型的anchor，补足显式跨进程来源标记；不改原核心文件。校准及新H自己的发布/读/写/图账照常重算。每mode原生1000UID分析，直接计算H_balanced−H_standard及同FG下h增量之差的配对区间。独立身份/差值/负结果均保留，不自动改最终配置。

耐久队列 `e5/initialization_pair_quality_queue/attempt_1`，sessions `ch3-init-pair-quality-0/1`，协调器3888215/3888217；实际worker **3888339(GPU0 edge1)** / **3888272(GPU1 edge2)** 已在ps和nvidia-smi出现，native shared_anchor_reference两边均实际写出verified_cross_process_reuse=true。后续GPU0边3/5、GPU1边4等真实paired源提交后继续；无额外训练，源F/G不同则明确重算。输出 `e5/initialization_pair_development/attempt_1/edge_i`。实际消耗4个服务点（F/FG/sharedStandardH/BalancedH）中的1个新增点；当前尚无完整初始化比较人口结论。

候选50%后继两卡也已真实启动：GPU0 worker3875110、GPU1 worker3885945（原时间质量退出后接替槽）。原PhaseB3647467/3647468及补充第四边质量继续；实测GPU0约31.8GiB占用、47.5GiB余量，GPU1约18.2GiB占用、61.0GiB余量，利用率100%。共享时间仅作进度。E5正式配置、三seed/2k主与独立人口/人数曲线、E6仍未完成。

## 2026-10-23T06:26:57+08:00 冻结首边与原第四边完整负向点保留，查询尺度诊断未发现明显失配

补充首边完整512阶梯与直接配对比较已实际成功：f=fg0.7283939014202172，h0.7271428571428572；h差−0.0012510443，1000UID95%[−0.0036121262,0.0006734960]，区间跨0。教师共同验证改善约1.67%没有转成该开发人口AUC改善，见 `frozen_predecessor_development_analysis/attempt_1/edge_1` 和 `frozen_predecessor_protocol_comparison/attempt_1/edge_1`。与第二边h增量较大但总质量略低的结果一并保留，未据此替换正式协议。首边质量worker3776348正常退出，后继已实际完成第三边共同验证并开始第三边512质量。

新共同验证第三边：joint f/fg/h=3.0449333894/2.8414525302/2.7118425926；new_term_only=3.0449333894/2.9837811554/2.8408468491，冻结整体也不及joint，`frozen_predecessor_common/attempt_1/edge_3`。所有同f逐UID/context指标一致。补充GPU0现进入边4拟合；GPU1等边5实际source f。它们是独立协议比较，不分边挑有利结果拼接阶梯。

原mix50第四边完整512分析在 `temporal_comparison512_partial_analysis/attempt_5/mix50/edge_4_complete_ladder`：f/fg/h0.6288464656/0.6273396098/0.6252036384；g差−0.0015068557，CI[−0.0043629821,0.0027861190]；h差−0.0021359714，CI[−0.0062903105,0.0009312771]，均跨0。h点估计负向；当前不能宣称五边预期趋势已满足。原时间质量与候选50%后继继续。

为核对“f改变后g/h输入尺度失配”这一具体可能，实际运行8个保留fit UID的数值检查（不是质量实验）：`query_conditioning_check/attempt_1/{run.py,summary.json,exit.json}`，tmux `ch3-query-conditioning-check-1`，worker3849651自然exit0（06:23:34），peak allocated5.56GiB。与原GPU任务并发，固定实际Large/edge2、fb75 mix50/Q16、原f/fg/h权重；只读取发布前历史/候选与模型trace，没有读取未来请求标签、没有输出AUC。

在这8个数值样本，修正后的q RMS相对同UID nativeReuse的中位数均1.0，F范围0.920–1.124、FG0.906–1.143、H0.899–1.141，未见数量级尺度漂移。原固定q scale下normalized RMS中位约1.02。候选间变化相对整体q能量较小：坐标中位约2.2%；实际h首仿射q投影能量占q+row约6.4%–12.2%，其中候选变化部分约3.2%–13.0%。这些是8UID输入条件数/激活数值诊断，不代表人口质量、实际h输出贡献或低秩结论。它们不支持把归一化错误当作已确认根因；若后续检查条件信号，应优先比较已训练的balanced初始化和候选构成，当前未新增归一化或架构训练。

正式E5三seed/主与独立人口/曲线及E6仍未完成。论文不修改，少于512的诊断不作论文质量证据。

## 2026-10-23T06:12:50+08:00 冻结协议第二边完整512阶梯与直接协议配对比较完成

本轮先实查6个原CUDA worker仍活跃，并以45秒有界等待后复查同PID，未因观察等待而重启。期间GPU0补充第二边质量worker3776172自然成功退出，三点全部512人/2742事件完成；后继单边1000UID和CPU直接协议比较均实际成功。现在GPU0补充postqueue按计划等待边4实际拟合审核，GPU1首边h继续，无失败。

第二边new_term_only f/fg/h AUC0.4783090006218197/0.47814084655065203/0.4791900578905422。g差−0.0001681541，CI[−0.0008873290,0.0003752635]；h差+0.0010492113，CI[−0.0007324748,0.0020172755]。h恢复增量点估计14.1742个百分点，但端点bootstrap33/1000非正，保留点比值、**不提供有界恢复率CI或确认收益**。结果 `e5/frozen_predecessor_development_analysis/attempt_1/edge_2`（summary/increments.csv）。

完整六点直接协议比较已经通过真实数据、实际参数和输入身份审核：相同f参数/缓冲、同512UID与事件/label、完全相同native anchor、候选时间/人数/预算/seed全部一致。`e5/frozen_predecessor_protocol_comparison/attempt_1/edge_2/{summary.json,contrasts.csv}`：new−joint最终h方法AUC差−0.0005219783，CI[−0.0013961794,0.0004303045]；new−joint的h增量之差+0.0005044622，CI[−0.0004689491,0.0010413509]。各差值用同一1000组UID multiplicity直接计算，非独立区间相减。冻结协议h相对自身FG的增量点上约翻倍，但整套方法AUC点上较低，所有这些区间仍跨0；两个结论分开保留，不宣布胜者，也不只截取h增量作总体优越证据。

这说明额外项的归因与最终质量应分别看；继续其余四边、候选50%比较和原PhaseB共同选型，不能因第二边FG负值就停止新协议，也不凭此单边改正式配置。原时间质量已进入第四边h，双卡PhaseB继续；正式E5三seed/主与独立人口/人数曲线和E6尚未完成。

## 2026-10-23T06:07:10+08:00 h参数变化诊断完成，排除仍停留初始权重的误解

只读检查本轮已提交的27个h checkpoint（PhaseA三输入组、PhaseB reference/balanced/lr/width已完成组和冻结协议前三边），实际按同seed/config重建f→fg→fgh CPU初始化顺序，应用真实standard/balanced h初始化后比较。6份已知best0 h的完整参数与初始值精确相同，初始化重建通过真实权重oracle；非0步h确有变化，joint查询块相对L2变化约0.4%–5.3%，行块约0.4%–4.9%，包括边5best40的约0.4%。边2balanced查询块变化约1.25%，row约3.91%。独立结果 `e5/h_initialization_movement/attempt_1/{summary.json,points.csv,analyze.py}`，记录每权重SHA、best/actualsteps、学习率/容量、query/row运动和输出权重RMS。

这只是参数运动及真实初值核对，不是AUC、实际查询特征能量或“训练不足”的证明；较小运动也可能已经足够，best40/0由验证选择，不能只因步数少就强行改选checkpoint。后续若完整质量仍弱，可结合实际查询分布再判断归一化/优化配置，当前不新增学习率训练队列，不打断既有实验。

两卡现有运行在本轮ps和nvidia-smi实查继续：PhaseB3647467/3647468，原时间质量3685043/3685045，补充质量3776172/3776348，CPU协议比较3785093和候选50%后继3805607/3805609。旧时间质量前三边均已完整，第4边mix f/fg点已提交；补充边1/2 h各有256用户进度块，**尚未满512，不分析半块质量**。最新显存约27.8/26.3GiB占用、两卡利用率100%，仍有充足余量。正式E5三seed/主与独立人口/人数曲线与E6仍未完成。

## 2026-10-23T06:01:23+08:00 第三边完整分析与时间窗口诊断完成，新增候选占比50%开发比较已排队

原mix50第三边完整512阶梯已提交并实际1000UID分析：f/fg/h AUC0.5838809466/0.5844047506/0.5846904618；g差0.0005238040，CI[−0.0007085766,0.0018717077]；h差0.0002857113，CI[−0.0011955477,0.0014270178]。输出 `e5/temporal_comparison512_partial_analysis/attempt_4/mix50/edge_3_complete_ladder`。该边点估计递进但未确认；Rebuild−Reuse0.02615316的1000draw有125个非正，不输出有界恢复率CI。empirical100第三边h也已完整提交，尚未额外单边分析（原队列全五边后会自动分析）。

为检查完整E14是否稀释早期交接收益，先固定 `configs/e5_handover_windows_2026_10_23.json` 的1/3/7/14天四窗，再新增只读 `analyze_probe_handover_windows.py`，使用同512全UID及原因果轨迹/anchor，统一1000组UID multiplicity，所有窗口及0/负收益保留。前三边mix50实际完成在 `e5/handover_window_diagnostic/attempt_1/mix50/edge_i`（points/increments CSV和summary）。独立正负分数两两比较AUC的36点oracle与14天原配对增量6份完整报告均核对通过，见 `.../attempt_1/verification.json`。该诊断只算质量，不把原E14费用当短窗口费用；源partial状态保留，不是新正式口径或独立确认。

当前不支持缩短窗口能恢复预期h趋势：边1各窗g/h都0；边2h在1/3天分别−0.0001792/−0.0010152，7/14天+0.0005539/+0.0005447；边3h在1/3天−0.0039331/−0.0026549，7/14天+0.0000665/+0.0002857，区间均跨0。早期实际有反馈UID仅57–65，统计弱，全部512UID仍参加cluster抽样，计数明列；不以此宣称早期负效应或改正式窗口。暂保留E14，不挑有利日期。

下一项已用本轮PhaseA已完成权重准备（**没有新训练**）：反馈候选50% vs原75%，固定gap_mix50、fit256/val16/Q16、W128、lr3e-5、cap2048/pat1024、seed17、原development512。实际共同loss1.2621454535983503 vs1.2521462138276547，差约0.80%，在已有1%容差内；不读取main/confirmation方法分数、不换人。新创建器 `e5/candidate_fraction_comparison512_plan/prepare_actual_plan.py`，实际15权重已通过原audit_weight及sweep绑定，89份冻结依赖；完整计划 `.../attempt_1/fb50_mix50/plan.json` SHA f5e394b640053baaa7b909e500e40f35b496e66253e2a74535307dea941a2c6c。

新耐久 `candidate_fraction_comparison512_queue/attempt_1`，sessions `ch3-fb50-quality-0/1`，协调器3805607/3805609已在ps实查存活。GPU0边1/3/5共9点，GPU1边2/4共6点，等各卡当前时间采样质量正常完成后接替该槽，再以实测32GiB余量准入，与拟合/其他质量共卡。每边原生512三stage＋1000UID g/h分析。输出 `candidate_fraction_comparison512_quality/attempt_1/gpuG`、`candidate_fraction_comparison512_analysis/attempt_1/edge_i`。当前只是已审核待执行配置，没有此组AUC，不是最终选型。

补充冻结协议第三边拟合已自然完成（f/fg/h best1576/1584/1840）；边1/2质量继续，第二边FG完整512 AUC0.47814084655低于自己的f0.47830900062，首边FG等于f。尚无补充完整h，不提前判断协议胜负。两卡PhaseB仍活跃，已进入GPU0边2fit64、GPU1边3balanced_h_init；原质量继续边4。正式E5三seed/主与独立人口/曲线及E6尚未完成。

## 2026-10-23T05:50:20+08:00 新协议前两边同f执行等价已由真实512分数验证

新补充质量首个f点在边1/2均实际提交，全512人，分别2710/2742反馈事件。只读ShardLoader真实审核后，不仅AUC相同，而且所有逐事件f分数、完整native Reuse/Rebuild两套分数、事件键/标签/输入/模型/执行来源、f全参数与缓冲均与原mix50联合协议对应点**精确相同**。AUC分别0.7283939014202172/0.4783090006218197；证据 `e5/frozen_predecessor_development_audit/attempt_1/edge_{1,2}_f_exact.json`，含已核对来源及权重/块SHA。两个点均512人，不使用未满512的中途块作质量结论。这仅证明新入口继承与因果执行等价，不能推出g/h收益。

新增独立CPU分析 `compare_probe_training_protocols.py`，完整六个点才比较：同人口/事件/anchor、模型/执行来源、训练输入/候选时间/N/预算/seed必须一致，实际f参数/缓冲精确一致；只允许左joint、右显式new_term_only协议。直接在同一1000组UID multiplicity下计算每协议g/h增量、各stage协议差及g/h增量之差，保留无反馈用户，避免相减独立CI。报告条件于固定权重的绝对AUC差，不把负/跨0端点恢复率当成稳定改善；没有自动选协议、独立确认或三seed声明。配对公共扰动消去的数值oracle通过（1项/2.07s）。

CPU耐久后继 `e5/frozen_predecessor_protocol_comparison_queue/attempt_1`，session `ch3-frozen-term-protocol-compare-1`、协调器3785093，已在ps确认活跃；逐边等实际六份完整512 summaries后才运行，对应输出 `e5/frozen_predecessor_protocol_comparison/attempt_1/edge_i/{summary.json,contrasts.csv}`。当前pending五边，没有生成协议胜负结论或额外GPU负载；不会将缺失点或waiting文件当结果。

两卡原PhaseB、两套时间采样质量、补充边3拟合和补充边1/2质量仍实际运行。原时间质量第三边h都已提交256用户进度块，但尚未完整512，不分析其半块AUC。后续继续等待完整g/h、五边共同验证与最终选型，不重复已完成点。E5正式三seed/主与独立人口/曲线及E6仍未完成。

## 2026-10-23T05:44:45+08:00 补充协议第二边完成；按显存余量提前启动真实512质量

补充前级固定协议第二边完整拟合自然完成并审核：f/fg/h best1392/1384/1392，三stage各实际2048步；source f原权重保持精确绑定，前级及统计逐元素固定。GPU0补充拟合现仅等待边4实际source f，未判停止、未重跑；GPU1已进入边3 worker3760417。

第二边配对三context共同验证已实际完成，在 `e5/frozen_predecessor_common/attempt_1/edge_2/{summary.json,points.csv}`：joint f/fg/h mean0.36357625/0.32686754/0.29901220；new_term_only0.36357625/0.35956423/0.32534407。补充协议相对自己的FG改善，但整体不及joint；与第一边改善约1.67%的情况不同。两个完整结果都保留，不分边挑协议，不以val16代替人口AUC。

检查发现两卡补充postqueue仅在等原时间质量全15点退出，而GPU0补充拟合已等待下一源权重、实测显存余量充分。按用户同卡并发授权，将**仅等待、没有活GPU子进程**的postqueue attempt2两个协调器3764766/3764769写cancelled/exit143并停止，已经完成的共同验证、实际计划、模型和分块全部保留。原两卡PhaseB、时间质量和补充拟合均未动，不重启任何GPU实验。新调度依据实测free≥32GiB准入质量，不要求原质量全组结束或整卡空闲；共同验证可按16GiB余量执行。

实际运行后继现为 `e5/frozen_predecessor_quality_queue/attempt_3`，协调器3775361/3775363，sessions `ch3-frozen-term-quality-0-v3/1-v3`；复用已完成共同验证时逐一检查fitplan SHA、6权重SHA、3context和执行来源，未重复GPU验证。新补充512质量 **已真实启动**：GPU0 worker**3776172**跑edge2，GPU1 worker**3776348**跑edge1；root在ps和nvidia-smi均核对存在。输出仍 `frozen_predecessor_development_quality/attempt_1/gpuG/edge_i`，后继1000UID分析不变。当前尚无补充完整阶梯人口结果，不能把启动当完成。

最新实测GPU0约20.4GiB占用、58.9GiB空余；GPU1约46GiB占用、33.3GiB空余（GPU1四个CUDA任务）；两卡利用率100%。所有运行是共享资源质量/FLOPs工作，时间只作进度，不是受控性能。原时间采样质量继续第三边h；原PhaseB仍双卡正常。正式E5三seed/主与独立人口/曲线及E6仍未完成。

## 2026-10-23T05:39:25+08:00 补充协议真实权重审核、共同验证和512后继已接通

新增独立 `frozen_predecessor_quality.py`，先用原source f真实质量auditor验证因果输入/角色/来源，再逐一验证新协议完整身份、三stage实际配置、metadata/JSON/SHA、自己N的RMS与val16、best/budget、源f精确继承、FG/H前级及统计逐元素固定、resident binding、可训练参数数目和完整累计校准账。没有修改原sampling auditor或把新driver伪装成原driver。5项回归最终14.16s通过，使用**真实原审计器**，拦截前级参数、协议身份、source_f绑定和费用篡改。

首次对接发现原PhaseB execution job省略read_weight等固定字段，直接传旧审计器触发KeyError。已用其原expected_identity补齐audit job字段（执行job值仍覆盖同名字段），没有修改实际身份、损失或权重。此前5项仅mock旧审计器，覆盖不足；本轮已去掉mock、加入真实输入manifest/原审计全过程。原失败质量postqueue attempt1：GPU1自然exit1，GPU0仍等待且无GPU子进程，已写cancelled/exit143并停止该等待协调器；两卡所有拟合和旧质量均未动，失败日志保留。

新 `plan_frozen_predecessor_quality.py` 对实际首边已生成3个ready job、85份冻结依赖：`e5/frozen_predecessor_quality_plan/readiness_actual_edge1/plan.json`，f/fg/h best224/0/224，所有真实源与权重审核通过。不存在的边5用实际CLI返回waiting/0jobs（`pending_check_edge5`），没有冒充完成。规划只使用固定development512，与fit/validation及main/confirmation隔离；没有人口筛选或正式人口质量。

新 `evaluate_frozen_predecessor_common.py` 复用原固定fit64/val16、三context准备及原始loss计算，每边对joint和new_term_only全部6权重/18cells比较。首边已**实际完成**：`e5/frozen_predecessor_common/attempt_1/edge_1/{summary.json,points.csv}`。joint f/fg/fgh共同mean均2.1410691865409412；new_term_only f/fg同值且全部逐UID/context指标精确相同，h为2.105344270200779（约低1.67%）。这支持继续检验冻结h，不是16人AUC或人口质量证据，尚不能选最终协议。其余边仍须验证，包含可能的恶化。

修复后耐久后继 `e5/frozen_predecessor_quality_queue/attempt_2` 正在运行，64份冻结来源/输入；协调器3764766/3764769，sessions `ch3-frozen-term-quality-0-v2/1-v2`。GPU1首边共同验证worker3764807已正常完成，并已生成实际3点512质量plan；现在明确等待原empirical100完整质量成功退出后接替该质量槽，**不要求整卡空闲**。GPU0仍等实际第二边补充拟合审核。每条边执行共同验证→真实权重ready规划→固定512三stage质量→1000UID g/h直接增量，输出分别 `frozen_predecessor_common/attempt_1/edge_i`、`frozen_predecessor_quality_plan/attempt_1/edge_i`、`frozen_predecessor_development_quality/attempt_1/gpuG/edge_i`、`frozen_predecessor_development_analysis/attempt_1/edge_i`。当前没有补充协议512 AUC，不能把waiting或共同验证当人口结果。

两卡新补充拟合继续，首边已完成并进入边3 worker3760417（替代已完成的3746867）；边2 worker3746873仍训h。原PhaseB3647467/3647468和时间质量3685043/3685045继续。共同验证曾与其余任务并发实测总占用约38–40GiB/卡，余量约40GiB；共享wall只作进度。没有启动正式三seed或E6，目标保持未完成。

## 2026-10-23T05:30:40+08:00 前级固定补充协议已接入真实五边双卡拟合

新增独立持久化入口 `scripts/ch3_motivation/run_probe_frozen_predecessors.py`：精确复用原PhaseB reference已训f（显式source路径/SHA/原数值来源审核），不重训f；随后同一条轨迹仅训练g，再固定自己的FG训练h。输入仍fit256/val16、Q16、fb75 mix50、W128、lr3e-5、cap2048/pat1024、seed17。源f的完整独立校准费保留，准备只记一次；额外训练使用已验证resident输入。权重与metadata显式记录新的driver、training_protocol、source_f以及resident adapter，不与原联合训练身份混同。原2x全forward反向估计保留且明示为冻结协议的保守上界，不冒充实测反向计算。

两项真实CPU优化器/持久化检查通过（7.00s）：只执行FG/H优化、f逐元素继承、各阶段前级参数和统计逐元素不变、完整累计校准费、独立身份、resident元数据、部分h阶段重跑后的loss/history/权重精确重现，以及源SHA改动在模型/数据准备前拒绝。此前7项helper CPU及实际Large两stage数值诊断保持通过。root读了新增入口、源f审核/费用/resume代码与测试；没有改正在运行的原源码。

新耐久队列 `e5/frozen_predecessor_queue/attempt_1`，plan5条边/15份stage权重（其中f仅继承，实际10次FG/H优化）、52个冻结来源/输入。GPU0边2/4，GPU1边1/3/5；以后几条边等待原PhaseB实际f提交后绑定SHA，不能从预期SHA编造权重。source job仍逐字绑定原expected_identity；每条完成后独立审核新权重SHA、协议身份及前级逐元素不变。tmux session `ch3-frozen-term-fit-0/1`，协调器3746653/3746655，实际worker **3746873(GPU0 edge2)** / **3746867(GPU1 edge1)**。已产生真实validation进度（约step280/256），不是空等待计划。新输出 `e5/frozen_predecessor/attempt_1/edge_i/seed_17`。后续还有自己的共同验证和固定512质量，当前没有该协议AUC或正式3seed结果，也没有把它纳入原475权重选型。

启动时两卡原PhaseB与时间采样质量均保留；每卡现3个CUDA worker，实测总占用约26GiB（reserved随stage波动），尚有约53GiB余量。每10秒保存资源快照，所有shared-GPU wall仅作进度。

原联合协议第二边两套512完整f/fg/fgh及1000UID分析完成，在 `e5/temporal_comparison512_partial_analysis/attempt_3/{mix50,empirical100}/edge_2_complete_ladder`。mix g差0.0008582864，CI[−0.0005438594,0.0017629082]，h差0.0005447491，CI[−0.0010951767,0.0014576144]；empirical g差0.0007514385，CI[−0.0005592081,0.0015747572]，h差0.0006148133，CI[−0.0011626997,0.0015861882]。两套点估计均递进，但区间跨0，不称稳定/统计确认收益。Rebuild−Reuse0.0074022823的1000draw有33个非正，因此不输出有界恢复率CI。首边零/负结果均保留，当前继续第三边，尚无五边趋势结论。

E5最终主/独立人口、三seed、人数曲线与E6仍未完成；3.2不追加，论文不改。

## 2026-10-23T05:19:32+08:00 3.3新增前级固定补充协议，实际Large数值检查通过

本轮只读诊断8条已提交PhaseB阶梯、24份stage权重/metadata及历史，10个best0均为首边g/h。真实checkpoint SHA、metadata、最优验证step和budget全部核验。所有best0均逐元素保留前级状态，新增输出层精确为0；没有漏训或加载别的状态。训练损失下降而验证上升，例如reference首边g首/末epoch训练loss1.0262→0.7127，验证初始1.8069→最终2.0784，实际训练1024步后恢复step0；h相似。边2实际保留非0g/h并同时更新前级参数。证据 `e5/calibration_followup_progress/attempt_1/{summary.json,stages.csv,diagnose_committed_fits.py}`。这只是fit/validation诊断，不把16人loss当人口质量。

据此新增独立 `scripts/ch3_motivation/frozen_predecessor_fitting.py`，仅在补充协议FG训练g、FGH训练h，暂时关闭前级参数梯度并finally恢复flags；不detach学生查询、不改原始读取＋分数损失、不改结构或数据。不触碰正在运行的联合训练。metadata明确 `training_protocol`（科学变体，不伪装同数值执行优化）。原forward账保留；原2x全forward backward估计在冻结协议下为保守上界，不能声称实际反向工作或和联合协议费用完全相同。若后续采用，必须独立driver/身份/费用解释及完整人口质量，不能静默替换已跑结果。

7项真实CPU优化器检查全部通过（最终4.16s）：新项零初始化、非零旧前级、非零AdamW decay、手动冻结原fit_probe数值对照、resident两模式和失败后flags恢复。新增项有非零梯度且权重更新；前级无梯度、权重逐元素不变。

实际Large/edge2数值检查已耐久运行并自然exit0（05:17:57），入口 `check_frozen_predecessor_fitting.py`，queue `e5/frozen_predecessor_readiness_queue/attempt_1` / session `ch3-frozen-term-check-1`，worker3726885已正常退出。它曾与原4个GPU任务同卡并发，未停原任务；8fit+2val、Q16、实际历史/模型和已训f/fg，各4步手动冻结原fitter vs新增helper。fg/fgh的全部逐步梯度、最终状态、完整report/counters/FLOPs均精确一致（maxdiff0），前级没有变化或梯度。结果 `e5/frozen_predecessor_readiness/gpu_attempt_1/{summary.json,fg.json,fgh.json}`；峰值allocated约7.62GiB，启动free约65.52GiB。只作正确性诊断，**不产生AUC或正式权重，不是512人质量实验，也未证明冻结协议有收益**。helper SHA0b349fd5fd0cc9f49c952d166d0cc5c62e5e9f1418615ce6a778672cfbbfafac。

原两卡PhaseB和两套时间采样512质量仍活跃（最新GPU0 width256/edge2，GPU1 fit128/edge1）。第二边f/fg点已提交，原队列仍继续h与其他边。正式E5/E6尚未完成；下一步补充协议须有独立持久化driver/身份与真实校准探索后，才能进入至少512人的质量比较。

## 2026-10-23T05:12:06+08:00 3.3首边两套时间采样完整阶梯与配对分析

本轮实际完成CPU配对分析，未改变正在运行的模型/拟合/质量源码。固定development512首边共2710反馈事件：mix50 f/fg/fgh AUC均0.7283939014202172，g/h增量及其1000UID区间均为0；empirical100分别0.7229928989139516/0.7227401837928153/0.7225981620718464，g差−0.0002527151，UID95%[−0.0010833384,0.0003552708]；h差−0.0001420217，UID95%[−0.0006085186,0.0004219973]。完整单边分析在 `e5/temporal_comparison512_partial_analysis/attempt_2/{mix50,empirical100}/edge_1_complete_ladder/`，各含summary及increments.csv。没有把此单seed/单边称为五边完成或独立确认。该人口原生Rebuild−Reuse为−0.0047243108，signed recovery保留但不作为有效恢复解释。

另对两套时间采样使用同一1000组UID multiplicity直接比较f/fg，避免相减独立CI；真实事件、原生anchor分数、输入/人口/模型/执行来源、权重和分块SHA均由原ShardLoader审核，全部512用户参与抽样。可复算入口 `e5/temporal_comparison512_partial_analysis/attempt_1/compare_time_laws.py`，输出 `cross_law_edge_1_f_fg/{summary.json,contrasts.csv}`。mix−emp f差0.0054010025，UID95%[0.0002276170,0.0111442196]；fg差0.0056537176，UID95%[0.0004101396,0.0115795065]。这是时间采样对绝对质量的开发信号，不是g/h额外收益；目前首边不满足预期阶梯，保留并继续其余边及PhaseB容量/预算/初始化探索。

只读实查原PhaseA与PhaseB reference的边1/2共6权重：参数/缓冲逐元素一致，best/completed step、完整loss history、objective、RMS、operation_counts/arithmetic及累计FLOPs均精确一致。记录 `e5/calibration_followup_execution/attempt_1/reference_equivalence_snapshot_2.json`；旧snapshot1保留。不是新质量测试或身份替换。

当前四个CUDA worker仍真实存在：PhaseB3647467/3647468，时间采样质量3685043/3685045；最新显存快照总占用约16/12GiB（分阶段波动，先前峰值约19GiB），两卡利用率均100%。首边两套完整阶梯均已提交，后续边继续，不重启原进程。E5正式三seed/主与独立人口/人数曲线及E6尚未完成。

## 2026-10-23 04:57 用户授权同卡并发，已实际启动两套512时间采样质量

用户明确显存足够时可同卡并发，并允许为3.3积极调整配置、测试方式和实现。已同步AGENTS。当前每卡均为一个原PhaseB拟合进程＋一个新质量进程：GPU0拟合3647467、质量**3685043**；GPU1拟合3647468、质量**3685045**。不等待显卡完全空闲。04:56实测两卡总占用约11GiB、free约68GiB；新质量队列每10秒记录GPU总内存/利用率及compute PIDs，shared wall只作进度。

新配对探索仅改变时间采样：fit256、width128、75%反馈候选、Q16、lr3e-5、cap2048/pat1024、seed17、相同训练目标和原development512，五边每边f/fg/fgh。mixed与empirical100实际candidate IDs跨所有fit/val用户逐一一致，三stage真实权重/driver/身份通过现audit_weight及sweep实际绑定检查；每套15点、89个冻结依赖。完整原PhaseA权重直接评测，未新增训练或读取main/confirmation方法分数。计划在 `temporal_comparison512_plan/attempt_1/{mix50,empirical100}/plan.json`；创建器 `temporal_comparison512_plan/prepare_actual_plans.py`。它们是开发探索，不是最终475选型或独立确认。

并发耐久入口 `temporal_comparison512_queue/attempt_1/gpu{0,1}`，tmux session `ch3-e5-time-quality-0/1`，shell3685020/3685026；mix50在GPU0、empirical100在GPU1。启动要求实测free≥32GiB，并保留旧拟合；新的质量worker已在nvidia-smi实际出现，首个512人f点已完成。质量输出 `temporal_comparison512_quality/attempt_1/{mix50,empirical100}`，后续每边原生1000UID、seed17051直接g/h配对分析输出 `temporal_comparison512_analysis/attempt_1/{mix50,empirical100}/edge_i`。两套都保留全部五边、零/负增量；不改变原PhaseB和final475→selected512队列。

注意原development512的edge1 Rebuild−Reuse为负，signed recovery不能当作更新恢复或与预期目标直接判断。两套首点的同人口native anchor AUC已核对完全一致；后续主读绝对AUC及g/h直接配对差，保留原signed ratio和不稳定CI，不切换用户或删边。

## 2026-10-23 04:31 3.3共同验证完成，修复CPU汇总并接好512质量

04:33接续已实际成功：sampling selector attempt2在04:32:23正常exit0，完整370权重/1110cells通过身份与数值审核。全球370与新六输入组都推荐 `calibration_sampling/full_item128_fit256_fb75_gap_mix50_lr3e5_cap2048_pat1024`；这是阶段输入选择，并非正式3seed/AUC结果。共同均值loss1.2521462138276547；cutover/mix50/empirical三context分别1.2596182860434055/1.2294538526330143/1.2673665028065444，第一边h仍best0。已对实际summary运行 `analyze_probe_common_contexts.py`，输出 `common_context_analysis/phase_a370_attempt_1`：370权重、1110实际验证cell、74个完整method groups，保留所有原配置与零步。该选中配方g与h在其余四边均降低共同教师loss；不能解释为AUC收益。

PhaseB正式配置已生成 `calibration_followup_plan/attempt_2/plan.json`（35阶梯、105权重）。04:33实际双卡worker为GPU0 **3647467**、GPU1 **3647468**，均在 `sweep_probe_calibration_followup.py` 运行；不是等待队列或重跑PhaseA。两卡分别承担偶数/奇数边，原7配方设置保持。后继final selector attempt3、selected-development attempt2正在等待此新队列，最新PID/路径以下列条目为准。

04:40实查两worker仍活跃。首个edge1 reference阶梯已完成，GPU1进入下一初始化配置；GPU0仍在edge2 reference。只读真实Large检查保存在 `calibration_followup_execution/attempt_1/reference_equivalence_snapshot_1.json`：edge1的f/fg/fgh、edge2的f/fg共5份已提交权重，与原PhaseA同输入标准配置逐元素相同，best/steps、完整loss history、objective、RMS、operation_counts/arithmetic及累计FLOPs也精确一致；edge2 h尚未提交，明确pending。原始driver/init身份保持各自记录，未重写权重、未用GPU或计算AUC。E6入口已读码核对，新的execution_adapter通过实际权重audit传递到FG/H绑定；正式E6仍须最终实际选中权重的两协议8人检查，不能拿旧权重诊断替代。

PhaseA30阶梯90权重均已自然完成；old280与新90的五边×三context共同验证全部正常exit0（共1110cells），不需要重跑GPU。首次CPU合并在04:27被真实身份检查拦截：`select_probe_temporal_common._jobs`把legacy sweep未声明stages的f-only配置套用了独立CLI默认f/fg/fgh。已核实实际 `sweep_probe_variants.py`默认f、controlsFalse，在source jobs解析时采用该默认，显式阶梯仍原样保留，未放松实际权重／原计划比较。新增回归及原测试共15项通过（3.01s）；原失败日志和exit1保留。此修复只影响CPU汇总身份解析，没有改模型、输入、权重、共同验证数值或选择规则。

四个仅等待的后继因前驱失败自然退出，未开始PhaseB/质量GPU。新的耐久链（socket ch3_201）已启动：
- sampling选择 `calibration_sampling_selection_queue/attempt_2`，session `ch3-e5-sampling-select-2`，shell3644276／协调器3644279；CPU选择worker3644349正读取原已完成370权重/1110cells。输出仍 `calibration_sampling_common_selection/attempt_1`，旧失败未生成该结果。
- 双卡PhaseB `calibration_followup_queue/attempt_4`，session `ch3-e5-followup-4`，shell3644389／协调器3644392；仍执行35阶梯105权重及315context，数值计划/输出根不变。
- 最终475选择 `calibration_followup_selection_queue/attempt_3`，session `ch3-e5-followup-select-3`，shell3644620；输出仍 `calibration_followup_common_selection/attempt_1`及正式计划attempt_2，不自动启动正式拟合。
- 新固定512开发质量 `selected_development_queue/attempt_2`，session `ch3-e5-selected-dev-2`，shell3644626；等最终475，生成 `selected_development_plan/attempt_2`，GPU0边2/4、GPU1边1/3/5，新分数分别在 `selected_development_quality/attempt_1/gpu{0,1}`，每边paired分析在 `selected_development_quality_analysis/attempt_1/edge_i`。严格复用真实同SHA旧点，其余才执行；15点/原development512/seed17、g/h两增量均检查完整。1000UID、bootstrap seed17051显式固定，无五边共同区间或三seed结论，不自动进入正式拟合。

正式sampling runner已显式支持 `--resident-inputs`（默认reference），仅临时替换原driver的fit_probe并finally恢复；现有权重在任何准备前检查mode，实际adapter binding/metrics贯穿JSON/PT/audit，helper与core SHA冻结。6项真实CPU optimizer/resume＋24项旧审计/路由回归通过，root读完新增源码和测试。runner最终SHA6a0532edcc13580cd342d52d626d41bde4b6c9be70c9284dfca11b4cbb3fcc6a；quality auditor SHA7d55a46087e1613e8dea62fa97819fa95b75ab87276f3f1363592e33543e5176。原数值driver/helper/core未修改；此模式尚未用于正式GPU，也未改PhaseB。adapter每stage会reset CUDA peak，原driver peak不再代表whole-run峰值，已在formal summary注明。接入记录在 `resident_fitting_readiness/gpu_attempt_1/formal_integration.json`。

3.2已按用户要求收尾，不再追加E3/E4。3.3最终配置、正式三seed/主与独立人口/人数曲线、E6实际冻结h低秩质量仍未完成。

## 2026-10-23 04:10 用户接受3.2，双卡后续集中3.3

用户查看现有五边E3主/独立确认图后明确：3.2按当前结果完成，后续重点为3.3寻找合理配置和预期机制趋势。E3/E4资料与差异全部保留，不再扩展其测试；3.3可独立配置。AGENTS、原设计和ch3 README已同步。E5正式阶梯、人数曲线/三seed/独立人口与E6仍未完成，整个目标保持进行中。

已于04:06:50撤销**尚未开始GPU工作**的 `scenario_qualification/natural_active_queue/attempt_1`；同时停止依赖它的三个CPU等待协调器以重排。旧followup2、followup_selection1、resident_readiness1及natural1均保留 `cancelled.json`、原日志和非0退出；未改成完成、未停止两卡PhaseA。自然人口输入保留，后续若3.3确需场景补充，可单独安排原生端点与E5，不再要求附带E3全部24点。

新的耐久顺序已实际启动（tmux socket ch3_201）：
- `e5/resident_fitting_readiness_queue/attempt_2`，session `ch3-resident-fit-check-2`、shell3602066／协调器3602069：等GPU0 PhaseA成功，独占GPU0 compute槽做原定六case数值/效率检查；输出仍 `resident_fitting_readiness/gpu_attempt_1`。GPU0 PhaseA已自然exit0，诊断worker3602406已实际开始。
- `e5/calibration_followup_queue/attempt_3`，session `ch3-e5-followup-3`、shell3602164／协调器3602167：仍等PhaseA共同选型；GPU0额外等resident2，GPU1无需自然E3。35阶梯105权重、原数值源、plan/output根均未改。
- `e5/calibration_followup_selection_queue/attempt_2`，session `ch3-e5-followup-select-2`、shell3602258／协调器3602261：等followup3，仍合并475权重/1425context并生成 `formal_calibration_plan/attempt_2`，不自动启动正式GPU。

新三队列35/82/86项绑定，并集106文件约7.49GB经独立实际SHA检查全部一致、0冲突。04:09:16一次进程快照为GPU1唯一PhaseA worker3306973、GPU0唯一resident worker3602406；新PhaseB/selector仅等待，无重复。resident首个W128/f case完成，尚未六case全齐，暂不采用优化或承诺加速。

本轮另保留了27/30阶梯的同view拟合诊断 `e5/calibration_sampling_progress/attempt_2/{summary.json,stages.csv}`：81份stage metadata一致、54次嵌套连续性通过。8个best0集中于第一边cutover/mix的g/h，均实际训练1024步；出现完整epoch训练下降而验证恶化，支持进一步检查泛化。不能把16人校准loss当人口AUC。旧attempt1未改。

按用户接受前的工作生成了只读派生核对图 `e3_e4_review/attempt_1/`（E3汇总/逐边、E4中位与IQR，共三份PNG/PDF和source摘要），逐图视觉检查通过，保留全部方法/边、负值及actual/230区别。仅用于结果核对，未写入论文，后续不继续扩展3.2。

新增 `plan_selected_development.py`（SHA8b37b29683ab2595c8369eba1e878cae787fc3ca3b8d7c1699d7dfac37fad4e3），15项CPU检查通过，root已读码。最终475选型后，用真实选中seed17 f/fg/fgh在原development512检查teacher-loss是否转化为流式AUC；保留旧driver/controls/时间/init身份，完全同SHA结果可严格复用。当前 `selected_development_plan/attempt_1`仍waiting_final_selection/0jobs，不能启动sweep或当作质量结果。

04:12:14 resident2正常exit0，六个真实GPU case全部通过；root已复核并写 `resident_fitting_readiness/gpu_attempt_1/root_review.json`。原始报告浮点差、反向梯度和最终状态差均0，RNG、beststep与计数一致。两次32步且包含预放置的速度比（reference/resident）：W128 f1.350/1.366、fg1.079/1.111、fgh1.030/1.046；W256 f1.318/1.368、fg1.085/1.293、fgh1.020/1.097。最大resident allocated20.083GiB；10秒资源采样只见该GPU0 worker，GPU1/整机并非独占。该检查支持未来正式拟合显式接入常驻输入，尚未自动采用，不是2048步总时保证，不改FLOPs或校准费用；现有PhaseB仍保持冻结数值入口。

selected-development另以真实旧15点验证了复用路径：159个唯一文件SHA通过、30个probe块/10个anchor块完整，不读分数或重算AUC；实际PhaseA五边identity与selector settings投影一致。证据 `selected_development_plan/readiness_actual_inputs/`，没有构造最终winner。后续仍以实际475选型结果为准。

## 2026-10-23 03:46 常驻拟合GPU检查已排入空闲槽，上下文分析入口完成

独立 `check_resident_probe_fitting.py`（SHA3cc07774183568b131da2caf20b195081319f266bb76bdbf02fc5058d71e104f）已通过7项CPU检查，连同此前helper的8项对照，验证逐步梯度／损失／最佳权重／计数与费用。root读完入口、helper和测试；24份直接数值来源可读。CPU证据在 `e5/resident_fitting_readiness/attempt_1/cpu_validation.json`，没有GPU数值、速度或峰值结论。

耐久队列 `e5/resident_fitting_readiness_queue/attempt_1/launch.json` 已启动CPU等待（tmux socket ch3_201/session ch3-resident-fit-check；shell3558521、Python3558523）。必须等 `calibration_followup_queue/attempt_2/gpu0/exit.json` 成功，即GPU0现有PhaseB偶数边拟合和共同验证都完成，再确认GPU0无compute进程。随后仅执行Large edge2、fit256+val16、fb50/empirical100的两宽×三stage检查：数值4步，计时32步×2次成对重放，预放置计入wall，不写正式权重或AUC；输出 `e5/resident_fitting_readiness/gpu_attempt_1`。首计时已被数值检查预热，allocator保留，reserved不能当额外常驻内存；静态双K/V满窗上界13.28GiB不含模型/Adam/激活，实际峰值待测。队列每10秒采样GPU0其他compute进程并保留污染证据，GPU1/整机可并发；不是独占整机时延。完成后仍需root判断是否有收益并显式接入execution_adapter provenance，未自动采用。

新增纯CPU `analyze_probe_common_contexts.py`（SHAcce35e3d6e13b961f9f1bf55a90c399c8a6cfc08a01a7df4784d58da10a3bacf），11项针对检查通过，root已读实现。输入最终475 selector的**summary.json**，逐context导出五边loss/read/score/raw_score、FGH描述排名、同配方seed的f→fg/fg→fgh逐边与等权损失差（after−before，负为改善），保留step0、负增量和失败；未齐五边/三context不给均值或排名，不增加选型规则。`e5/common_context_analysis/readiness_1`只含370个missing记录、0个验证cell，明确pending，不能当结果。

共同验证的三个context总共48个query标签确为24个cutover＋24个经验gap，但候选×gap联合配对不同，综合loss一般不等于cutover与empirical两loss简单平均，更不代表完整真实流式联合分布。它使用固定cutover缓存、fit64 catalog/law与共同尺度，是发布前验证诊断；新sampling拟合catalog/law256另记。正式选型保留原冻结规则，root还会查看逐context的g/h增量与分歧。3.2与3.3无需共用配置，图内公平条件及各自范围继续保持。

03:44实查PhaseA24/30阶梯、72权重（GPU0十条、GPU1十四条）。03:45九个运行/等待队列150份唯一冻结文件全部通过，0冲突／0变更；新增队列尚未用GPU。正式E5主/确认、自然补充和E6质量仍未完成。

03:57接续：PhaseA26/30、78权重（GPU0十一条、GPU1十五条）。`sweep_evaluate_probes.py`又补齐实际已存在的plan绑定检查：准备GPU/创建output前合并source/frozen/formal-plan/actual-weight SHA、确认权重audit覆盖所有jobs；完成前复验，计划本身也必须未变。canonical路径去重，大文件前后各哈希一次，不按每个点重复。原development格式仍可用。24项CPU检查通过，root读码复核，最终SHA4accd49a0a23ae918761f061f30a2c5f3d5e37660f633da2403861c2103ab4c4。没有当前队列冻结此入口；以后必须重新生成natural ready plan引用最终SHA。

自然补充的独立 `analyze_natural_quality.py` 已准备（最终SHA3966d734bd2b2a12f55e5693b0359c8c28da45cb3c7e7f8a8732618f40c04cb1），8项合成2k marker/score/CI/费用检查通过；root review后再补E5已执行源码与计划入口SHA核对，3种stale-source拒绝＋完整seed17 pair/cost路径4项通过。直接复用原单边paired核心和E3审核，仅向后者提供已审计natural current缓存以避开旧union布局，不改原分析器。按各边固定2k用户导出E3的24点与E5的f/fg/fgh、g/h直接区间、raw/corrected actual/230费用；各family五边齐才写CSV，不做跨边同UID聚合。E5仅seed17补充，无extended控制，增量本身不能排除额外训练预算影响。`runs/natural_quality_analysis/readiness/attempt_1`是旧源码下的pending库存快照（0CSV），已保留；不等于最新源已做正式结果分析。最新代码尚无自然人口结果可读。当前三份新/修改的分析与sweep源均不在九个运行/等待队列冻结集合中。

## 2026-10-23 03:35 拟合输入常驻封装通过CPU对照，修复空质量计划入口

新增独立 `resident_probe_fitting.py`（SHA28e67369617a1e88bf85a7f123b18a4704de13728839cb28fbdadae18e44d24a），原样调用冻结fit_probe，仅预放置其实际读取的K/V与小输入，每stage结束释放，无全局patch／跨stage特征缓存。原metadata仅增加execution_adapter.binding/metrics，静态helper/core SHA和策略与动态bytes/wall/memory分离。8项CPU真实f/fg/fgh×raw/prepared对照通过：逐步损失/gradnorm、逐参数反向梯度、最终状态/beststep和全部operation_counts/arithmetic精确一致，输入未改、别名去重和跨stage新候选/时间隔离通过；root已读实现和测试。尚无GPU测速或加速结论，也未接入正式runner；独立Large GPU诊断入口在准备，当前运行/等待源不改。

重复搬运量用真实每UID长度解析计算而非实测PCIe：S2048/fit256每fit用户恰8次，val16含step0每8步共257轮；两份FP32 K/V每历史行51200B，五边仅训练+验证逻辑搬运约220.660/227.410/229.819/232.436/236.716GiB/阶段。未包括额外statistics/target/bind pass、协议开销；常驻方案的数学FLOPs及完整校准费用仍保留。是否采用须由GPU数值、含预放置wall和显存结果决定。

自然活跃人口检查发现质量sweep会把waiting_selection/0jobs写成complete。`sweep_evaluate_probes.py`现只接受实际ready状态、非空且唯一命名的jobs，并在GPU准备前检查权重/人口manifest文件存在；SHA b8afbcbf20f43f7b7e306ccba2862ecfda14f575ff9927b2f72535993fe5007b。7项入口检查＋5项natural planner检查通过（12项/66.15s）。只有已exit0的development_quality_queue旧记录引用原源；八个当前运行/等待队列135份唯一冻结文件再次0变更。原开发结果不重写。

自然人口五边UID不同，现配置只要求逐边1000次UID区间，不能套主面板同UID五边聚合。E3可交付各边runner的24点质量/配对CI及actual/230完整账；后续CPU锚点审核需读natural的edge_i/current/quality_shards布局。E5经sweep的Namespace执行natural_active（不能直接用evaluate_probes CLI的旧panel choices），未来计划必须ready且全15点后再按卡拆分。单边analyze_probe_quality的per_seed17可以给g/h直接配对区间，其默认三seed状态不用于该单seed补充场景；自然E5费用需逐点应用现有校准修正，并单独报告，不能混入main/confirmation费用grid。自然质量尚未开始。

03:33实查PhaseA21/30（GPU0九条、GPU1十二条），两worker3338132/3306973仍Rl+；共同验证、PhaseB、正式主/确认和E6继续按既有依赖推进。目标未完成。

## 2026-10-23 03:19 正式费用导出已接入，双卡探索继续

E5/E6最终导出已接入校准共享行归一化重复计费修正，保留原账和实际权重，不修改质量算法或冻结执行源码。E6 `account_lowrank_costs.py` SHA55d1362850f3899547b0b71579705b837ed1d554fc1407391206558e052bb373，7项CPU检查通过；主账actual_event/analytic230显式为修正账，raw_actual_event/raw_analytic230保留原值。`e6/complete_costs/attempt_3/summary.json`仍waiting_formal_weights、0bindings，无费用结果。

E5新增 `probe_cost_export.py`（SHA50fb4a1c34823f771e10c09a59d296a1a9e10c24921fdd65058c8942e851eca5），仅在已完成且审核通过的正式grid上，从原逐点summary.costs和实际checkpoint导出raw/corrected/calibration_correction；输出cost_points.csv和cost_aggregates.csv。实际事件／230账分别按同人口、seed、方法、N及配置先加五边FLOPs再求比例，不平均各边百分比。11项费用专项＋92项原主/曲线检查通过，root已读导出实现。主分析入口最终SHA894d961520645e089293edf20fbdc81aeda66f2b3c3096390166b4718e65791f，曲线入口SHAcd38aff5f506a2df97262414e91be25006d3f0e9c7432b4d3f17eafbfcd25892。未完整grid不会读费用权重或生成成本CSV。

`e5/formal_quality_analysis/attempt_1/`与`formal_curve_analysis/attempt_1/`均是pending/0点的真实CLI检查；仍引用旧formal_calibration_plan/attempt_1占位。正式执行必须读取最终 `formal_calibration_plan/attempt_2`及实际main/curve权重重建计划和分析，不能把pending当作交付完成。自然活跃场景另外汇总，不混入主/确认人口费用。

03:16实查PhaseA18/30阶梯、54权重完成（GPU0七条，GPU1十一条），worker3338132/3306973继续，八队列135份唯一冻结文件0变更。校准诊断快照在 `e5/calibration_sampling_progress/attempt_1/summary.json`，逐权重SHA通过；第一边cutover/mix的g/h仍best0，经验间隔h在各自输入的16人验证损失额外下降约3.67/3.79%。这是同view阶梯诊断，不作<512质量证据，不跨view按原损失排序；共同验证、PhaseB和正式质量继续等待既有队列。当前无正式E5/E6人口结果，目标未完成。

03:21更新：PhaseA19/30（GPU0八条、GPU1十一条），两worker继续。E6五边分析只补了E5费用导出的传递依赖，数学不变；最终SHA00d801c0574cff7ce9047fd4a375d96044c6847ab1c9ef1b67bcf77dd03dec49，`e6/multiedge_analysis/attempt_2/summary.json`为pending/0results、47份来源，来源核验与pending单测通过，旧attempt1保留。

后续正式拟合效率候选已只读核实：`fit_probe`每step和每次16用户验证均把CPU的parent/teacher K/V搬到GPU；已有target时每stage绑定尺度也先搬全部用户一次。五边fit256+val16实际历史总行数212293/217476/221861/225856/229357，两份FP32 K/V占10.123–10.937GiB，全272人满1024窗上限13.281GiB。cache只有k/v/seq_len，无持久hidden；查询hidden为瞬时张量。最小候选为仅常驻冻结K/V与小输入，避免parent/parent_cache别名双拷；不能跨optimizer缓存f输出、学生q依赖teacher读取或h激活，全池缓存normalized readfeatures还会额外占约39.844GiB，当前不采用。尚未实施、无GPU测速；待正式拟合前空闲卡用小数值/性能探针核对同seed顺序、损失/梯度/beststep/权重、预放置与完整wall/峰值后决定，不改现有队列或冒用旧执行来源。数学FLOPs与standalone准备费用保留，实际传输和复用另记。

## 2026-10-23 02:57 补齐正式统计，并核对费用量级

E5五边汇总发现并修复真实接续问题：旧`_recipe`把各边自然不同的sampling/gap manifest与发布时间当成配方不同，实际新PhaseA边1/2必被误拒。现在只在正式计划、实际权重、人口及逐边输入审核通过后允许剔除这些边特有字段；采样规则、N、catalog/law、初始化和driver仍严格匹配。`analyze_probe_quality_multiedge.py`最终SHA029b3ebd98d6ae21763f00f4f16185f888a677734bce91072bba2726817de33f，multiedge＋f人口曲线90项CPU检查、最后3项专项通过；两份原始新权重已实际核验，step0保留。未改冻结拟合或质量入口。

E6新增`analyze_lowrank_multiedge.py`（SHA5920eca38ec4e247c3d0c2b1c420582bc988f296002ef7680797e0ccb969806d），21项CPU检查通过，root已读关键加载／统计路径。五边反馈UID集合不同，跨边按各family的UID并集共享multiplicity，先等权AUC差后比值；限制到单边后的重采样人数可变，已明确口径。主both_norm三seed分报，seed17四KV及边2/5的512协议对照分开，0/负H收益保留。实际pending在`e6/multiedge_analysis/attempt_1/`，0结果、无GPU。预计正式长表约516万行，逐块读取；20k合成事件×1000draw的CPU数值核约0.55s，整体估2–4min另计I/O，不是正式总运行时间保证。

实验设计正文已同步本次授权，移除旧“E3不新跑／直接复用旧端点／执行者不处理差异”指令；历史数字保留并标明。E3汇总README补充真实事件账与230次解析账区分：主Layer1/2/4/6/8的230次费用21.27/33.52/56.18/76.39/94.16%，不能将原表真实事件费用当作此口径。主Layer2真实34.28%由校准13.27＋选择修复21.01个百分点组成，小人口固定校准摊销必须说明。

新256人长预算的校准费已核对实际边1/2/3：f累计66.028/105.969/107.554 TF，单是校准约为主2k一次Rebuild的241/376/379%；fgh累计937.650/1730.967/1923.018 TF。各阶段累计与一次teacher准备未大额重复，每8步验证16人产生约两倍训练前向次数，验证约占阶段拟合账39%。另发现fgh共享行归一化在旧拟合FLOPs公式重复计一次，约多计累计0.032–0.033%。部署费用路径已缓存行特征，不含此重复；主2k费用分母不变。

03:03修正已完成：独立`calibration_cost_correction.py`（SHA939714a9a8b1822ad376e7e79bafc76f47673188f101b5b90dd2f604f1891f4c）16项CPU检查通过，root读码复核。实际9权重审计在`e5/calibration_cost_correction/attempt_1/summary.json`，对应fgh修正累计937.348336211932/1730.408270026352/1922.390385597852 TF；f/fg及extended扣减为0。按原FLOPs模型与相同平均居民历史，f仅校准摊至Rebuild20%需要约2.41/3.76/3.79万居民，这是未计部署工作的线性解析下界，不是人口实测，也不替换主2k分母。旧源码、权重、metadata、数值结果均未重写。

**正式费用交付接续仍须完成**：当前E5/E6冻结执行入口及E6 sidecar输出仍是原账。最终取得实际权重和完整费用后，调用`checkpoint_correction(path, expected_sha256=actual_sha)`，再`apply_to_costs(raw_costs, record)`，显式导出`raw`／`corrected`／`calibration_correction`；对实际事件账及230账只修正校准项及其总额／比例，所有部署项与分母保持原值。E6拆分账可按helper说明接合；不能漏用修正，也不能对已修正账再次扣减。

02:57 PhaseA完成15/30阶梯（GPU0边2六条全完，GPU1九条），两实际worker继续；E5主/独立确认质量和正式E6仍未启动，目标未完成。

03:03实际复核：PhaseA16/30（GPU0六条、GPU1十条），worker3338132/3306973均Rl+；八个运行／等待队列135份唯一冻结文件仍0变更。E5共同验证、PhaseB、正式三seed及独立质量按既有接续继续。

## 2026-10-23 02:41 E6接续入口与完整事件费用账已准备

E6正式计划器`plan_lowrank_formal.py`已完成11项CPU检查，最后readiness/路径别名3项复核通过；独立timestamp guarded入口25项检查通过。实际权重齐备后绑定15组正式FG/H及每边原定2000反馈用户，生成15条release任务（seed17共享四种K/V口径）和边2/5各512用户的4条协议对照；两卡按输入工作量分配。新的发布/逐时刻入口各需实际所选权重8人数值与显存检查，不能复用旧权重检查代替。所有随机分解矩阵都检查精度，正式结果只保存聚合覆盖计数，不输出逐头巨量日志。

当前`e6/formal_handoff_plan/attempt_1/plan.json`仍为waiting_selection/0jobs；这是旧attempt1正式计划的等待占位。最终需读取`e5/formal_calibration_plan/attempt_2/plan.json`和真实正式runner的`main`权重根重新生成，尚未启动E6质量。正式planner SHA e9fc8ebbcc7fc35fa2f45ac0364e9e835475328614c50b968fbf59f145c44e8f；timestamp入口SHA e1ee6f9ec6a9ecccbc05f44966a410e480ec55bbecafe1de0dc9f362eeb06800。

费用sidecar `account_lowrank_costs.py`已通过7项CPU检查，按实际E6人口、模型结构及FG/H权重的累计校准费用算完整事件账；研究中的SVD、数值oracle、多rank读取开销另列，不能冒充可部署低秩算法费用。`e6/complete_cost_input_evidence/attempt_2/input_evidence.json`已核算五边完整行为935429/971919/942165/1008485/916012，其中末次反馈后尾部229619/220291/232899/260025/237348，占22.67–25.91%，全部必须计费。当前`e6/complete_costs/attempt_2/summary.json`为waiting_formal_weights/0bindings，尚无最终费用。实际最多2070/306/288/1058/177条反馈对应9/2/2/5/1个230候选panel，GPU每次仅保留一个H panel；8人检查不代表长用户CPU快照和总运行时上限。

02:41实际核查：PhaseA已完成12/30阶梯，GPU0为4/12、GPU1为8/18；两worker3338132/3306973均继续。八个运行/等待队列135份唯一冻结文件实查0冲突、0变更。后续共同验证、PhaseB和自然活跃2000队列保持等待，无重复启动。正式E5三seed及主/确认质量仍待选型。E6计划器与费用sidecar已独立只读复核，未见接口阻断或人口误绑；成本完成标志由sidecar独立给出，不自动回写planner。

## 2026-10-23 02:27 E4时间结构诊断完成

新增CPU分析`ch3_motivation/runs/e4/temporal_diagnosis_5000/attempt_1/`，保留原五边各5,000用户、全部平方偏差，25,000行逐用户CSV与summary。原始分块hash、严格发布前历史顺序和重算top10占比逐用户一致性通过；时间首行0按实际编码处理，关联统计不把它当作观测间隔。无新GPU或未来反馈读取。

“长间隔跨会话行主导集中失配”的假设在固定人群的典型用户上未获支持：间隔≥3600秒的行比例中位5.27/5.27/5.37/5.37/5.42%，其绝对平方失配份额中位仅1.11/0.80/1.25/2.21/1.60%；逐用户间隔—绝对失配Spearman中位−.422/−.529/−.363/−.355/−.372。固定history≥256且间隔中位≤300秒组仍有4317–4549用户，top10份额仍51.26/57.24/55.92/45.90/46.22%；不能靠这个自然条件得到均匀偏差。相反组仅3–5人，全部组统计已按<512规则抑制。相对平方口径单独保留，不套用上述绝对口径数值。这个结果缩小原因范围，不是AUC重要性或时间特征的因果证明，也没有据此更换人口。行为行间隔与候选查询间隔不同；PhaseA三种查询间隔的配对验证继续。独立只读复核已确认切片、原始binding与<512抑制正确。

## 2026-10-23 最新澄清：按图探索配置，不要求全章共用

用户明确3.2与3.3属于不同机制的motivation，可在同一模型下采用不同的合理人口、历史/场景与校准配置。已同步`AGENTS.md`、原实验设计和ch3 README；E5五边共用配方仅约束自己的阶梯图，不强加给E3/E4。图内对照/端点保持同条件，跨图若要比较定量优劣另需同条件证据；E6仍分析它所绑定的实际冻结h/FG。当前队列和原始结果均未改写。

02:23实际核查：PhaseA30阶梯已完成9条（GPU0边2三条，GPU1边1六条），两worker3338132/3306973继续；尚无最终E5配方或主/确认质量。已完成边1的经验间隔配置出现fg/h非零验证改进，cutover/mix的fg/h仍选初始状态；这是校准信号，跨输入的原验证损失不能直接横比，也不是AUC结论。继续共同验证与PhaseB配对对照。

自然活跃2000用户的E5质量planner已完成5项CPU检查，输出`e5/natural_quality_plan/attempt_1/plan.json`为waiting_selection/0jobs；以后用实际正式main权重根生成15点，只复用同人口/window/models的原生三端点，不绑定E3的Layer选择或预算。E6正式planner及自身2000反馈人口的完整CPU费用账正在补齐；还发现旧逐时刻敏感性入口未对所有矩阵启用能量筛查，正在以独立新入口补上，不能拿joint协议8人检查替代。未启动未校验的正式E6质量。

## 2026-10-23 02:10 已接好PhaseB双卡及最终共同选型

当前两卡仍由PhaseA新256人采样校准占用（GPU0 worker3338132、GPU1 worker3306973），未停止或改写运行配置。新增`e5/calibration_followup_queue/attempt_2`已实际进入等待，tmux `ch3-e5-followup`／shell3382021，81项源和输入绑定：等PhaseA共同选型成功，生成`calibration_followup_plan/attempt_2`的35阶梯105权重；GPU1跑奇数边，GPU0等`natural_active_queue/attempt_1`完成后跑偶数边，各卡随后验证自己的21权重/边×3共同上下文。拟合/验证均已通过真实CPU数值和身份检查，reference/balanced的f/fg必须逐张量相同。

该后续队列attempt1在GPU启动前因`queue.py`遮蔽Python标准库而导入失败，失败日志及exit记录保留；attempt2使用`orchestrator.py`，已实际确认等待进程、progress与标准库导入路径，当前PhaseA两卡未受影响。不要重启attempt1或重复启动现有等待队列。

`e5/calibration_followup_selection_queue/attempt_1`也已实际等待（tmux `ch3-e5-followup-select`／shell3383565，85项绑定）：PhaseB完成后在完全相同上下文上合并旧370+新105＝475权重、1425单元，保留所有配置/零步/负向结果；遵守原固定1%集合选型规则，准备`formal_calibration_plan/attempt_2`，此队列不启动正式三seed GPU。新formal sampling执行入口支持main15阶梯含controls、curve45个f-only，main/curve路径隔离，4项真实优化/恢复CPU测试通过；实际最终配方、三seed质量、独立确认、active2000的同权重质量及E6仍待完成。

## 2026-10-23 02:00 后续比较接续

两张卡仍执行新256人校准，当前实际worker为GPU0 3338132／GPU1 3306973。CPU合并选型队列`e5/calibration_sampling_selection_queue/attempt_1`已在tmux `ch3-e5-sampling-select`（shell3359134）等候90个新权重的共同验证成功，随后合并旧280与新90；root复核selector的11项CPU测试通过。全局370配方建议保留为外部基准，下一阶段的输入只按新六种catalog/time-law256配方的共同损失选取，避免把旧law64配方静默换成law256。

PhaseB将比较七个匹配配置（人数64/128/256、学习率、宽度、预算、h输入平衡初始化），五边共35阶梯；其独立入口正在完成CPU验证，尚未启动。三seed正式计划入口已完成10项CPU检查，仍是等待最终选型的0任务计划，不能据当前单seed校准称趋势已验证。

## 2026-10-23 01:46 两卡均已进入新256人拟合

旧fit64时间对照10阶梯40权重已自然exit0。GPU0于01:45接上新sampling边2/4（worker3338132），GPU1继续边1/3/5（worker3306973，已完成2阶梯、第三条运行中）。两卡当前都是新的30阶梯/90权重探索，不是只排了等待。其余接续队列、共同验证和natural cohort见下。

补充输入证据：`e5/implementation_audit/attempt_1/own_feedback_coverage.json`。50/75%候选配额来自“自身＋其他拟合用户”的发布前真实反馈池；fit256中仅84–90人/边有自身可用近期反馈，自身反馈占全部Q16约8.7–9.9%／10.5–12.4%。这是已记录的候选来源语义，不把75%误写成每用户自身反馈75%；暂不据此改人口或扩实验矩阵。

## 2026-10-23 01:43 新共同验证已接入队列

新增90权重/270上下文的独立共同验证入口已通过13项CPU数值/身份检查，并实际核对首5份新权重，0失败。`e5/calibration_sampling_common_queue/attempt_1`（shell3332225、tmux `ch3-e5-sampling-common`）等待旧280共同验证attempt3以及GPU0新拟合成功后，使用GPU1执行；69项源/输入/计划绑定冻结，每边要求18个权重全部完成并与旧共同验证的上下文逐字段一致。旧280+新90合并后共370权重/1110单元；合并选型工具正在准备，尚未选最终配置。

新sampling-v1校验已接入f人口曲线、FG配对、质量计划：cutover也绑定真实manifest，catalog/time-law256与实际N教师/梯度分开，init standard/balanced显式身份不可混配。112项CPU回归、270项真实输入身份检查通过；当前运行39/43/43/27项冻结清单未变。E6旧诊断证据保持原源码身份，新身份检查仅进入后续正式运行。

## 2026-10-23 01:38 实际双卡启动与接续

新30阶梯/90权重已于01:29启动：GPU1 `ch3-e5-sampling-1`（shell3306858、worker3306973）承担边1/3/5共18阶梯，已完成首条，当前第二条mix；GPU0仍在`ch3-e5-temporal`运行旧fit64时间对照，结束后`ch3-e5-sampling-0`（shell3306863）自动承担边2/4的12阶梯。启动证据与43项冻结源/输入见`calibration_sampling_queue/attempt_1/gpu{0,1}/launch.json`；新完整CPU集成及helper共11项通过。上述均为校准探索，无未来反馈质量结论。

原280权重/840上下文共同验证数值计划不变，只有空闲等待进程被停止重排（attempt2 exit130明确未启动GPU），当前`common_temporal_validation_queue/attempt_3`（shell3307353）等GPU1新拟合结束；新90权重使用单独共同验证入口，与原280逐上下文输入指纹匹配。

固定长历史活跃2000场景已通过五边真实输入/角色/hash检查，`natural_active_comparison.json`与`natural_active_queue/attempt_1`（shell3315791）安排在GPU0新拟合之后执行新的OldFull/Reuse/Rebuild和全部24个E3点；不按端点正负删边，不改变main/confirmation。其未来查询中旧行存活中位约72–76%、清空约7–8%，这是输入特征分析而非质量。

新实现诊断在`e5/implementation_audit/attempt_1/`：真实小范围fit输入的h首层候选投影能量约7%；独立PhaseB标准/平衡初始化helper/driver已过18项CPU检查，保持MLP形式、零输出初始功能、f/fg前驱、损失不变。`probe_calibration_followup.json`列明后续N64/128/256、lr、width、cap和初始化的匹配对照，等待共同验证选定输入，不据人口AUC选配方。

## 2026-10-23 01:30 新增推进（以真实进程/exit为准）

用户再次明确两张GPU都用于主动配置探索，目标是五边共用配置能复现预期趋势，并完成三seed和独立人口复核。当前目标仍未完成。上一轮已新增实际输入/实现证据及运行结果，属于推进；不把等待/计划当完成。

- 512人开发15点已自然完成，分析`ch3_motivation/runs/e5/development_quality_analysis/attempt_1/`：h三边正、一边0、末边负；末边直接AUC增量−.001109，UID95[−.001929,−.000028]。其余非零g/h差区间均跨0。只有seed17，主/确认学习方法尚未读取。
- GPU0继续10条fit64时间输入阶梯，`ch3-e5-temporal`；原240权重cutover共同验证已完整，最优是fit256/lr3e−5，但该建议只是当前输入条件下的阶段性结果。新统一三时间上下文比较仍须完成。
- GPU1的E6能量筛查诊断已自然exit0（`release_guard_readiness_queue/attempt_2`）：三种batch10/25/50各3200矩阵全部通过，最坏条件下限约.99647，0精确fallback；全流程82–83秒，batch25峰值约18.96GiB。仅8人正确性/效率，不作质量结果。attempt1等待旧common不存在的exit.json而失败，已保留解释并在attempt2核对真实completed.txt和240点summary，未重跑已完成权重。
- 新输入`calibration_sampling_fixed256/attempt_1`五边×6view（反馈50/75%×cutover/mix50/empirical100），90次N64/128/256真实load/apply检查通过。固定fit256来源，validation16不贡献候选池或时间法；无删人/无fallback。经验反馈间隔等UID中位数70–140秒，发布时刻间隔约9.2万–11.4万秒；这是需要检验的输入覆盖差异。
- 新拟合计划`configs/probe_calibration_sampling.json`为30阶梯90primary权重：fit256、width128、lr3e−5、cap2048、patience1024、Q16、同一read+score目标。新driver共享同edge/N的原始旧/新缓存，逐view重算targets/statistics。三阶段断点加载现在保留CPU RNG；旧fresh运行未触发该resume缺陷。新helper/driver真实tiny教师与optimizer、角色/费用/恢复数值检查已通过。
- 拟将空闲GPU1先承担边1/3/5新拟合，GPU0完成当前阶梯后承担边2/4；原三上下文共同验证仅重排等候顺序，数值计划保持冻结。新拟合完成后在相同共同输入上与原280权重比较；最终三seed仍待配置冻结。具体启动证据在`calibration_sampling_queue/attempt_1/gpu{0,1}`，没有launch/started记录时不视为已运行。

## 当前目标：第三章重新测量（2026-10-23 00:56更新）

用户已授权g201物理GPU0/1重做Yambda-Large五条独立相邻边E3–E6；旧结果仅作历史对照，不改论文或底座。少于512名评测用户只作数值/性能诊断，不能交论文质量结果。新入口、协议和限制见 [当前README](ch3_motivation/README.md)。

新测三端点五边全完并审计310块。原定main/confirmation各2,000人均五边点上符合“新版整体获益、缓存复用损失收益”；五边等权新旧差为+.024806/+.013864，Rebuild−Reuse为+.020141/+.009446，对应四个共享UID95区间均正。沿用原定面板，未按方法分数删人；其他固定抽样的相反边也保留。逐边显著性按各正式run原定bootstrap报告，不能在临界区间中挑有利seed。

固定人口代表性CPU补核完成：`ch3_motivation/runs/scenario_qualification/panel_coverage/attempt_1/`。严格发布前特征覆盖75,807×5行，原分层/hash角色精确重建；main/confirmation五边长度×活跃单格比例最坏差1.671/1.160个百分点，反馈次数CDF最大差2.754/1.831个百分点。覆盖支持主要输入范围接近总体，不能由此保证总体AUC或方法排名；没有据方法成绩改人口。

E3全部14项任务于00:41自然完成，main/confirmation十组240点及80分块审核通过，正式汇总在 `ch3_motivation/runs/e3/main_confirmation_bridge_queue/attempt_1/analysis/`。Layer1/2五边等权恢复主样本36.39/41.00%、确认48.22/53.42%，成本21.89/34.28%；主样本末两边、确认末边有负值，不能用早两边代表整体。绝对AUC及配对差值区间完整保留；两个人口各1000次bootstrap有1次参考差非正，依原定严格规则，恢复率区间标为不稳定。边1/5各512人once/request-local语义桥完成，9个方法的直接语义差区间均跨0，不能据此解释正式曲线。23:15出现的外部GPU0作业已自然退出，未终止他人任务；共享负载期间时间不作受控性能证据。

GPU0当前 `ch3-e5-dev-quality` 执行固定512开发人口的15点因果质量检查。原定主/确认的学习方法分数尚未读取。随后 `ch3-e5-temporal` 按冻结计划执行五边×mix50/empirical100共10条拟合阶梯40点；见 `ch3_motivation/runs/e5/temporal_calibration_queue/attempt_1/launch.json`，等候上一队列成功后才启动。新时间特征仅取fit64发布前真实反馈间隔（val16不贡献分布），候选IDs及缓存不变，21项CPU检查通过；这是拟合输入探索，不是人口质量结果。原发布时刻间隔的240权重共同验证及选型保留为cutover-only对照；最终三seed配方还须经过统一时间上下文验证，不能自动拿该旧选型启动正式主测。

E4五边各5,000人已完成交付：`ch3_motivation/runs/e4/final_aggregate/attempt_1/{points.csv,summary.json,README.md}`，1,420行原完整精度数值，444文件/100主测块/30敏感性块审核通过。绝对平方偏差top10行中位占比约51.41/57.49/55.59/45.42/45.98%，与约16%的预估不符；固定512独立人口×256/512/1024窗口、fit64固定层头RMS、相对平方和Frobenius口径均分别保留。不能把数值偏差份额解释成Layer质量贡献。

GPU1耐久顺序：25条E5开发阶梯已完成；三点等价h满窗GPU检查和E6满尺寸8人数值/性能检查通过（无AUC）。20条低学习率/64与256拟合对照已完成；当前 `ch3-e5-feedback-pool` attempt2运行15条因果候选配方，随后 `ch3-e5-common-val` attempt2统一比较240份最终权重，`ch3-e5-select` attempt2做CPU选型。attempt1在GPU启动前拦截了等待进程缓存旧plan校验值的问题，失败记录保留，未重跑完成拟合。旧名义50%近期反馈候选实际仅10–13%，新配方只用拟合用户发布前已观测物品补足，评测人口/历史/事件不变。不同配方原验证损失不能直接横比：共同验证固定同16用户、Q16/50%候选、fit64共享尺度。正式3seed预先固定17/23/29；目前开发仅17，尚无正式E5质量/seed稳健性结论。

E6保留预定每边2,000反馈用户，不再按结果搜索人口。该反馈cohort边4新旧差为负点且UID区间跨0，未证实更新收益；缓存重建减复用差为正且区间高于0，仍可检验压缩性质。当前逐时刻矩阵协议保留；另实现每发布230查询实例的joint矩阵（重复item不同时间q区分、old全列固定、FIFO只在读时裁剪），避免重复大矩阵分解。00:43真实8人全窗口数值/性能检查通过（无AUC）：满秩logit误差0，串行/七rank合批最大差9.54e-7，峰值保留20.20GB；81.40s含41.25s精确SVD检查及1.35s额外执行对照。这些8人不是论文人口结果，也不足以直接承诺全量耗时。所有rank只作fullH轨迹上的事后读取干预，不能冒充各rank独立演化/可部署编码器。廉价能量精度筛查已完成独立CPU helper，尚待真实矩阵验证后集成；正式E6质量未启动。

剩余：固定人口更新前特征代表性补核；E5发布时刻与fit-only时间配置在共同验证条件下选型后完成同配置3seeds、主/独立确认质量及f拟合人数曲线；E6精度/效率确认后完成正式低秩结果和必要语义对照。全程分块保存、保留不符合预期的结果。

以下保留本轮初始安排与历史正式实验状态。

# 第三章重新测量：2026-10-22 晚间启动

用户已授权 g201 两张 A800（物理 0/1）重新完成 E3–E6，替代旧“复用 E3/端点”的安排；少于 512 人仅作数值和速度诊断。新入口与状态见 [ch3_motivation](ch3_motivation/README.md)。Large V0–V5 已存在，无需传输或重训。

五边 CPU 输入已按发布前历史长度/活跃度冻结：fit512/validation16；开发512；E3/E5主2000（嵌于E4的5000人口）；独立复核2000；两组额外2000人口抽样（固定种子20261023/24）；每边另绑定2000名合法反馈用户用于E6。所有角色、请求与历史pack有hash，尚无新质量结论。

新的Layer逐连续区间校准、参考执行和E4/E6统计已通过CPU针对检查。E3 GPU0 8人数值/性能检查通过：24点912分数，优化与参考最大差9.54e-7，不计算AUC；精确64人34区间校准16.09s，参考14.21s、优化无图2.54s/有图3.11s（含首次捕获），只作执行诊断。E4 batch16/8的KV绝对差约1.23e-4略超预设1e-4，保留失败记录，正在采用已很快的scalar路径核对，尚未形成E4主结果。

旧结果及其状态在下文保留；不能据旧“无GPU任务”段落判断当前目标运行状态，应检查新目录launch与实际进程。

更新日期：2026-10-22。Yambda-Large五条独立边基线全量已完成；最终四配置质量确认尚未完成。资产核查见[准备度报告](docs/experiment_readiness_2026_10_21.md)，当前恢复点如下。
更新日期：2026-10-22。已从目录核查进入基线准备及提速探针；最终四配置质量确认尚未完成。资产核查见[准备度报告](docs/experiment_readiness_2026_10_21.md)，当前恢复点如下。

用户最新明确：01为独立相邻版本比较，各方法每条边从共同旧版原生状态重新开始；只有02研究我们方法的连续跨版状态。旧文档中要求01所有基线连续、或02直接复用01连续轨迹的安排已被替代。201最初分配GPU 1，随后改为后续任务使用物理GPU 0；可从g202按需获取数据和已有V0–V5模型。

随后优先级更新：先测01全部对比方案，暂缓我们方法的质量训练／评测。H10已做代码改动只完成必要验证；先推进Reuse、首次历史消费前Rebuild、四类选择重算及两类学习纠正。最新论文源在202的`/home/gkl/work/paper/main.tex`，201暂未有该文件，可经SSH参考；正文中“所有方法连续”和模拟数值结论不能覆盖用户最新协议。

## 全量完成：Yambda-Large五条独立边，共120点

**2026-10-22 08:07（上海）：后四条边顺序队列正常结束，exit_code全部为0；实际12小时27分14秒，符合启动前12–16小时预估。加上原有V0→V1，五条边每边24点／75,807名合法用户，共1,671,042条反馈、40,105,008个分数。95个分块的分数SHA、UID身份、Parquet行数，以及配置／请求SHA和费用／恢复率公式均已核对。本任务没有运行中的GPU进程。**

[完整逐边报告](01_quality_cost/runs/yambda/large/baselines_all_edges/seed_17/attempt_1/report.md)包含全部120点的AUC、完整新增FLOPs、恢复率、1000次配对UID置信区间；同目录有`table.csv`、`table.tex`及`summary.json`，摘要绑定五份原结果hash。没有重新评测或改写原始汇总／权重／分数，没有平均恢复百分比，也没有将末边部分窗口混成统一完整E14聚合。

主要结果：Layer2约21.04–21.05%成本下，五边恢复84.88／85.48／55.92／47.28／60.46%；Query重算60%点约79%成本下恢复88.43／92.02／88.54／91.20／114.08%，末边UID95%为[78.72,249.66]，跨100%，不能宣称显著优于Rebuild。Query-only全部15个点估计均负；History256恢复76.64／58.64／56.72／4.57／−40.42%，末边三点区间均低于0。应保留不利结果并另行诊断，不能据全量测试结果追调后当作独立确认。当前均为seed17开发探索，未测CacheTide，不等于最终四配置结论。

以下为原启动配置、输入覆盖及恢复记录。
## 已恢复运行：Max V1→V2、V2→V3，GPU0/1＋NUMA1，64 GiB内存上限

2026-10-23 11:29（上海，准确启动时刻见launch.json）：用户要求恢复原队列并确保大批量内存使用NUMA1，随后明确少量共享／系统页可接受。已在独立tmux `yambdamax202/max-baselines-v1-v3-numa1`＋systemd用户服务`max-v1-v3-numa1-1.service`恢复；先V1→V2，再自动V2→V3。GPU0/1，CPU及新分配业务内存从Python启动前严格NUMA1绑定；整组MemoryMax=64 GiB、MemorySwapMax=0、OOMPolicy=kill、memory.oom.group=1，父／子进程共同受限。

先完成[真实Max小面板](01_quality_cost/runs/yambda/max/numa1_real_probe_202/attempt_1/README.md)：两份实体模型加载、实际历史读取、24策略，16用户双卡，32 GiB硬上限，峰值21.56 GiB，144次采样匿名N0=0，无OOM／触限。全量恢复后再次监测启动加载，内存峰值46.28 GiB、监测匿名N0=0；主队列、调度及两个GPU worker均为bind:1，其所有线程CPU affinity仅NUMA1。此前34块（139,264人，70.9%）已各卡17块校验跳过，六点同边拟合直接验证复用，开始剩余14块质量；V2→V3尚未启动，将独立校准。

- 新启动／接入／停止／恢复命令与NUMA／cgroup核查：[numa1_resume_1/launch.json](01_quality_cost/runs/yambda/max/queue_v1_v3_gpu01_202/attempt_1/numa1_resume_1/launch.json)，同目录startup_check.json、startup_memory_samples.json及service.log。主队列PID1616672，调度PID1616732，当前GPU worker PID1617015/1617016。
- 当前恢复入口：`/home/gkl/miniconda3/bin/python scripts/resume_max_v1_v3_numa1_2026_10.py --host-memory-gib 64`。仅在现有队列／服务已退出、GPU空闲且用户要求接续时运行；不要使用会强制NUMA0的旧--resume-manifest入口。
- 原配置／数值源码／校准权重／分块不改，保存原指纹；每边execution_placement_override.json单独记录本次CPU／内存放置变化、历史NUMA0分块来源及新服务路径，避免把旧配置NUMA0元数据误读成当前执行。
- 全量参数仍每边196,363人、24自然点、append256/query16、每卡4096人原子保存、完整费用与1000次配对UID bootstrap。新增资源包装不改变方法、拟合或质量协议。GPU2/3其他任务继续，未操作。
- 下节“全量停止”及旧停止记录均为历史；本节恢复授权覆盖它们。改变NUMA后耗时需用新分块更新，旧结束时间不再适用。

## 最新NUMA要求：GPU0/1使用NUMA1内存，小试已通过，全量仍停止

2026-10-23用户说明此前停止是为了避免Max加载峰值与NUMA0的大CPU程序争内存，要求先做极小双卡验证，后续GPU0/1进程的CPU／内存均绑定NUMA1。已完成[受限小试](01_quality_cost/runs/yambda/max/numa1_gpu01_probe_202/attempt_1/README.md)：两个随机微型HSTU在GPU0/1执行，父／子进程MPOL_BIND:1与NUMA1 CPU affinity均通过；4 GiB总主机内存硬上限、swap0，实际峰值1.663 GiB，max/oom/oom_kill事件均0。普通张量、CUDA锁页、Parquet缓冲、checkpoint保存／加载／mmap与GPU回传张量逐页查询全部在NUMA1，33次cgroup匿名内存采样N0最大0字节，恢复分数完全一致。原GPU2/3任务及NUMA0程序未改变，全量队列保持interrupted。

**后续不得直接使用旧launch.json的resume_command：旧入口会写NUMA0启动脚本。** 新入口`scripts/resume_max_v1_v3_numa1_2026_10.py --host-memory-gib <明确的总上限>`通过独立tmux＋systemd MemoryMax/MemorySwapMax/OOMPolicy限制整组，从Python／模型导入前numactl绑定CPU1/membind1，验证后直接执行旧队列--run-manifest；旧数值源码、配置、权重与分块不修改。新入口NUMA/cgroup检查与被拦截的exec命令已在1 GiB受限CPU测试中验证。小试不等于完整Max／196k用户加载验证；正式内存上限需先通过受限的真实模型小用户加载确定，4 GiB仅属于本次小试。没有启动完整模型或恢复人口队列。

严格绑定覆盖新分配业务内存；已有共享库／文件页缓存不会因membind自动迁移，不能承诺系统／驱动每一页都在NUMA1。GPU0/1物理靠NUMA0，改NUMA1后的完整速度需重测。后续以本条新要求覆盖下节NUMA0历史启动／原恢复命令描述。

## 用户已停止：Yambda-Max V1→V2、V2→V3，202 GPU0/1

**2026-10-23 10:40（上海）：用户明确要求kill当前GPU0/1任务，已正常终止队列及两张卡的所有对应子进程。** 队列PID1508077、边调度PID1508137、质量worker PID1509547/1509548均已退出；GPU0/1显存均为0MiB、无计算进程。GPU2/3其他任务未触碰。队列状态为`interrupted`（SIGTERM退出143），这是用户主动停止，不能自动重新启动或执行下一条边；后续等待用户恢复指令。

断点保留：V1→V2完整提交34/48块、139,264/196,363人（70.9%）、14,457,360个分数，六个独立校准点及层profile已完成并保留；V2→V3尚未开始。未完成块恢复时重算。未删除或改动分数、配置、模型、纠正权重和冻结源码；仍可使用原`launch.json.resume_command`断点接续。停止前PID／命令证据在队列目录`stop_requested.json`，退出、GPU释放与保存覆盖在[stop_completed.json](01_quality_cost/runs/yambda/max/queue_v1_v3_gpu01_202/attempt_1/stop_completed.json)。此前完成时间预估已失效。

以下启动记录为历史，当前以用户停止状态为准。

2026-10-23 04:31（上海）：用户明确要求继续这两条边、GPU0/1、同V0→V1全量协议及后台运行。已在独立tmux socket/session `yambdamax202/max-baselines-v1-v3`启动顺序队列：先V1→V2，再V2→V3，每条边同时使用两张卡。启动前GPU0/1空闲；GPU3另有其他进程，未触碰。当前首边已进入独立校准，尚无本轮质量结果。主PID1508077，首边调度PID1508137，初始拟合worker PID1508199/1508200。

| 独立边 | 窗口 | 合法质量／费用用户 | 反馈用户 | 反馈事件 | 角色／校准 |
|---|---|---:|---:|---:|---|
| V1→V2 | [245,259)，E14 | 196,363 | 89,406 | 865,122 | 固定Max fit512／validation16，重新拟合六点及选层 |
| V2→V3 | [259,273)，E14 | 196,363 | 90,387 | 872,163 | 同一UID角色，独立重新拟合六点及选层 |

数值入口完整复用已完成V0→V1的`scripts/max_baselines_gpu01_2026_10/dual_gpu.py`，该包及共享数值源码未修改。仅新增顺序启动器`scripts/launch_max_v1_v3_gpu01_2026_10.py`，其自身及全部139份数值源码hash随队列冻结。三份实体V1/V2/V3 checkpoint hash／16L/H320/10头/1024结构已核对，Max角色与上一条边相同且测试与拟合／验证互斥。仅复用该边已准备的请求与Max固定角色，没有复制其他边的权重、层profile、缓存或分数。

每边仍24个自然参数点（Layer2/4/8/12及其余原点），append256/query16、选择器候选固定16、NUMA0。每条边从对应旧模型原生缓存重新开始，发布前教师校准、首次历史消费前一次重算／映射、Query-only查询纠正与原生写入、完整事件FLOPs和附加内存账均沿用；每卡4096人原子保存分数及标记、每校准点单独存权重，最终原分数合并AUC及1000次配对UID bootstrap，保留负恢复率。CSV／LaTeX／报告随每条边完成生成。

- 正式输出：[baselines_v1_v3_gpu01_202/attempt_1](01_quality_cost/runs/yambda/max/baselines_v1_v3_gpu01_202/attempt_1/README.md)，子目录`edge_v1_v2/`与`edge_v2_v3/`；输入及模型检查见`preflight.json`。
- 启动／恢复：[queue launch.json](01_quality_cost/runs/yambda/max/queue_v1_v3_gpu01_202/attempt_1/launch.json)，同目录`progress.json`保存当前边和PID，`queue.log`记录切换，`v1_v2.log`／`v2_v3.log`及各边fit／quality日志分别保留。
- 原启动命令：`/home/gkl/miniconda3/bin/python scripts/launch_max_v1_v3_gpu01_2026_10.py --edges 2 3`。接入`tmux -L yambdamax202 attach -t max-baselines-v1-v3`。客户端断线无需重启；中断后仅在原队列及子进程全部退出后使用`launch.json.resume_command`，跳过完整边／完成校准／完成分块。运行期间不要修改被冻结数值包、共享依赖、启动器、配置、数据或模型。
- 新启动器Ruff F/E9通过，顺序两边、已完成边恢复跳过、首边失败不启动下一边三项有界CPU调度检查通过；数值推理沿用上一轮完整核查通过的同一源码，没有为新边改变方法或训练batch。
- 以V0→V1实际8小时54分加上本轮逐边新校准预留18–20小时，预计10月23日22:30至10月24日00:30（上海）完成两边；共享机器进度估计，不作受控论文计时。不重跑V0→V1，不排V3→V4/V4→V5或CacheTide。

## 已完成：Yambda-Max V0→V1，202 GPU0/1双卡

**2026-10-22 23:01（上海）全量正常完成**，本条边队列已退出，GPU0/1已释放；未启动其他Max边。196,363名合法质量／费用用户，89,121名反馈用户、891,487条反馈，24点、48个分块、21,395,688个分数齐全。双卡质量阶段8小时32分05秒，后续合并／统计约22分钟，总历时约8小时54分11秒，落在此前8–9小时排期内；复用了同边已完成校准，为共享机器执行进度，非受控论文计时。

2026-10-23完成CPU核查：139份冻结源码、输入／配置、两份模型、六份纠正权重及全部48块hash一致；每个方法的事件键／标签与冻结请求吻合，角色互斥，逐点全量AUC重算差为0，恢复率／FLOPs公式通过。1000次配对UID bootstrap的参考差距均为正。证据在[completion_audit.json](01_quality_cost/runs/yambda/max/baselines_v0_v1_gpu01_202/attempt_1/edge_v0_v1/completion_audit.json)；[完整报告及24点表](01_quality_cost/runs/yambda/max/baselines_v0_v1_gpu01_202/attempt_1/edge_v0_v1/report.md)、同目录summary.json/table.csv/table.tex已保留，原分数／summary／权重未改。

Reuse/Rebuild AUC=0.678035877/0.690710286，差距0.012674409。Layer2：13.21%×一次Rebuild新增FLOPs，40.59%恢复，UID95% [34.55,47.12]%；Layer4：25.61%成本／58.23%恢复。Deviation40：55.01%成本／87.58%恢复；Deviation60：81.38%成本／105.27%恢复，但区间[98.48,113.74]%包含100%，不能声称确定超过Rebuild。Tail80：87.25%成本／97.33%恢复。History256：70.88%成本／65.18%恢复。Query-only三个点均明显退步，恢复−258.79/−309.32/−249.12%，保留失败结果，不据测试表现改拟合。以上仅一条独立边和固定校准权重，未产生跨边稳定性证据。

以下启动、预估及首块进度文字为历史记录，以以上全量结果为准。

2026-10-22 14:07（上海）：用户明确要求开始V0→V1双卡评测、不使用GPU2/3。本条边已在g202物理GPU0/1的独立tmux后台启动；启动前确认RecFlow底座链已完成、GPU0/1空闲，未终止其他程序。14:14核查两名质量worker运行、各占约30–32GiB显存；GPU2/3保持空闲，日志无启动错误。尚无全量质量结果。

保持196,363名Max合法质量／完整费用用户（89,121名用户贡献891,487条反馈）、24个自然参数点、append256/query16和固定16候选选择器。复用已完成的**同一Max V0→V1边**六点校准及层profiling；原GPU2/3准备来源保留，不复用Large或其他边权重。实际执行绑定GPU0/1、NUMA0，源码隔离为`scripts/max_baselines_gpu01_2026_10/`；6项针对检查与Ruff通过。每卡4096人原子提交分数及标记，恢复校验配置／源码／模型／角色／权重／分块hash；最终合并原始分数计算AUC、恢复率及1000次配对用户bootstrap，费用覆盖全部用户与完整事件。

- 正式输出：[baselines_v0_v1_gpu01_202/attempt_1](01_quality_cost/runs/yambda/max/baselines_v0_v1_gpu01_202/attempt_1/README.md)，逐点结果生成于`edge_v0_v1/`。
- 队列及命令：[launch.json](01_quality_cost/runs/yambda/max/queue_v0_v1_gpu01_202/attempt_1/launch.json)，启动检查见同目录`startup_check.json`，进度见`progress.json`。tmux socket/session：`yambdamax202/max-baselines`；主PID1021789，质量worker PID1022028/1022029。客户端断线不影响任务。
- 本次只启动V0→V1，未排其他四条边或CacheTide。执行命令为`python scripts/max_baselines_gpu01_2026_10/launch.py --edges 1`。中断后仅在原主进程及子进程均已退出时使用`launch.json.resume_command`，不要重复启动。运行期间冻结本包、配置和输入。
- 根据先前同型号A40双卡实测，排期8–9小时，预计10月22日22:10–23:10（上海）完成；为共享机器上的进度估计，不是受控论文计时，首块后可据实际速度更新。当前仍在第一个4096人块内。

下节保留此前GPU2/3准备期记录；正式运行以本节GPU0/1为准。

## Yambda-Max：202 GPU2/3准备完成，等待正式启动指令

2026-10-22：用户确认使用g202物理GPU2/3（A40），本轮只跑通、优化和预估，尚未启动完整人口质量或五边正式队列。隔离入口在`scripts/max_baselines_2026_10/`，原Large执行源码hash未变；GPU0/1的RecFlow保留原运行。完整记录见[Max双卡准备](01_quality_cost/runs/yambda/max/readiness_2gpu_202/attempt_1/README.md)。

Max独立冻结196,363名合法质量／完整费用用户，512 fit／16 validation；五边UID角色一致，每边重新拟合和选层。实体16L/H320/10头/1024，24点采用Layer2/4/8/12，其余沿用自然点。V0→V1六点已独立校准并保存；其余四边只准备输入。V4→V5仍为E14_partial，保留部分源日300。

选定append256/query16、选择器固定16、NUMA1与expandable segments。2048人／9216反馈、24点双卡热重放291.01秒；实测每卡保留显存约31GiB。按正式分配人数／反馈外推纯质量7.53–7.76小时，加入准备、校准、保存和1000次UID bootstrap后，每边排期8–9小时，五边约40–45小时；为共享机器的进度估计，不是受控论文计时。最终32人链路、表格、四块新进程恢复及六点复用全部验证；6项针对检查通过。每卡4096人一块，预计18–20分钟。

后续用户下令时执行`python scripts/max_baselines_2026_10/launch.py --edges <1..5>`；会在独立tmux `yambdamax202/max-baselines`顺序运行指定边，每边同时使用GPU2/3，保存启动／恢复命令并防重复。此命令本轮尚未执行。输入与权重位于Max准备记录下各`edge_v{old}_v{new}/`；正式运行前保持当前源／配置／数据，不能拼接成连续发布状态。

## 当前运行：GPU 1，V1→V5四条独立边

2026-10-21 19:39（上海）：用户明确授权继续在物理GPU 1评测V1–V5内所有相邻release。已启动V1→V2、V2→V3、V3→V4、V4→V5顺序队列，每边沿用24个自然点及75,807名合法质量／费用用户；V0→V1已完成，不重跑。每边独立从旧版原生状态开始，六个学习点和层选择全部重新校准，只沿用fit512／validation16的UID身份。暂缓CacheTide质量，不重训底座。

| 边 | 窗口（天，左闭右开） | 反馈人数 | 反馈条数 | 数据身份 |
|---|---|---:|---:|---|
| V1→V2 | [245,259) | 34,080 | 318,504 | 完整E14 |
| V2→V3 | [259,273) | 34,266 | 323,362 | 完整E14 |
| V3→V4 | [273,287) | 34,756 | 335,971 | 完整E14 |
| V4→V5 | [287,301) | 35,925 | 347,575 | E14_partial：13个完整源数据日＋部分日300 |

以上覆盖统计现已与最终summary核对一致。最后一条保留全部可用数据，其中26,677条反馈来自部分日300；不能声称完整14天结果。完整窗口／部分窗口信息随配置进入summary及报告，禁止无说明混合成统一E14结论。

- 队列根：`01_quality_cost/runs/yambda/large/baselines_v1_v5_gpu1_201/attempt_1/`；`configuration.json`冻结四条边配置／源码hash，`preflight.json`记录覆盖，`launch.json`记录总PID和原命令，`progress.json`记录当前子任务及PID，`queue.run.log`记录切换。
- 队列PID：370084。入口：`scripts/experiments_2026_10/run_baseline_edges.py --manifest <队列configuration.json>`。恢复只能在原队列及其子进程都退出后执行`launch.json.resume_command`；跳过完整边，当前边由`--resume`校验输入并接续。遇错误停止并保留证据，不静默改参数。
- 各边配置：`01_quality_cost/configs/baselines_yambda_large_v{旧}_v{新}_201.json`；输出分别为`01_quality_cost/runs/yambda/large/baselines_full_v{旧}_v{新}/seed_17/attempt_1/`，同级`.run.log`／`.launch.json`。每4096人提交分数和完成标记；已完成校准复用，未完单块／单个拟合点重做。
- 以已完成V0→V1的3h06m为参考，排期每边3–4小时、四边12–16小时，预计10月22日07:40–11:40（上海）全部完成；这只是进度估计，后续按实测更新，不是论文受控时间。GPU 1为`GPU-942a1fcc-bce5-31ce-94a2-9b71db2ff8b8`，子进程`CUDA_VISIBLE_DEVICES=1`，内部`cuda:0`。
- 启动验证：5项因果／断点／图权重生命周期测试通过，Ruff F/E9通过。数值路径和训练batch沿用上一轮；仅增加窗口标记及顺序调度。队列运行期间不要修改被冻结的执行Python文件。

## 已完成：Yambda-Large V0→V1基线全量

**全量完成（2026-10-21 18:46）：75,807名合法用户，其中34,190人有345,630条反馈，24点全部完成；19个分块均已保存。实际执行3小时06分08秒，符合启动前3–3.5小时预估。后续四边队列见上节。** 结果在`01_quality_cost/runs/yambda/large/baselines_full_v0_v1/seed_17/attempt_1/{summary.json,report.md,table.csv,table.tex}`。Reuse/Rebuild AUC=0.634243/0.648931；Layer2完整新增FLOPs=21.04%×一次Rebuild，恢复84.88%，配对UID95%区间[78.01,90.87]%。H256=72.77%成本／76.64%恢复；Q-only三点恢复−33.77/−8.68/−8.15%。完整原始分数合并后算AUC，再以差值计算恢复；1000次用户配对bootstrap，参考差距全部为正。单条独立边、单个纠正训练seed17，不代表全部四条件／跨边／训练稳定性结论。

**内存口径更正：** 本轮Layer四个点均为起点0的前缀重算，从原始事件即可重算，无需额外持久旧hidden边界缓存。此前通用区间实现45%的说法不适用于这些点。原型为了共享profiling／评测仍做临时边界捕获，不能作为已选基线的必需持久内存。H10 CPU探针确认有／无旧边界时前缀重算K/V完全相等。仅修正摘要内存元数据与报告说明；AUC、FLOPs、置信区间、分数与权重均未改，修正记录及原摘要hash在`summary.json.reporting_correction`；执行时源码hash仍保留。后续源代码报告器已修正。

以下保存的启动与中间进度描述为历史记录，以以上全量结果为准。

## 当前RecFlow恢复点：Large／Max训练及Max Recall补测均已完成

**2026-10-22下午：** 主队列已正常结束，状态`complete_large_and_max_v0_v5`；Max完成95,082个优化步、六版各两轮，纯训练约6.67小时，十二份永久epoch checkpoint均保留。用户要求的Max Recall@100补测也已完成，仅用202物理GPU2/3，未动GPU0/1上的其他实验。两进程均退出0，总历时20分43秒，最高保留显存7.15GB。输出`foundation_recflow/runs/chain_2gpu/attempt_1/max/sampled_retrieval_full/`，原命令／计划在`plan.json`；已校验1,484个完整rank块、八份冻结数据hash、十份训练源码及六个最终checkpoint的提交身份／当前大小。没有后续队列。下文Max训练进度及等待指令属于历史记录。

**Max口径与结果：** 使用自身350万视频目录独立采样，每请求1000个随机干扰项＋可覆盖观测正例，未复用Large候选／预测。实际冻结数据与Large共用3,210名final用户／94,809个未来日请求，保留全部无正例及OOV零分，没有临时换人口。五次Recall@100相对变化依次为 **+10.56473／−2.60571／−4.66152／+7.06526／−2.28181%**，仅2/5正向，**未满足至少三次正向发布的要求**。五次等权平均+1.61619%，配对UID95% [0.37400,2.93966]%；新版为匹配随机期望的1.924–2.180倍，各边新版减随机的区间均为正。两条正向及V1→V2／V2→V3退步的单边名义区间排除零，V4→V5区间跨零。平均为正不能替代发布一致性。Recall@100在查看Max结果前选定，但与Large共用用户／窗口，仍不是新人口上的独立确认；属于采样检索而非全目录或自回归生成证据。

全部七项指标及不利结果见[Max配对统计](foundation_recflow/runs/chain_2gpu/attempt_1/max/sampled_retrieval_full/paired_uid_analysis/summary.json)和[逐发布CSV](foundation_recflow/runs/chain_2gpu/attempt_1/max/sampled_retrieval_full/metrics_by_release.csv)。Recall@50与MRR各3/5点估计为正、均值分别+1.21532／+1.42498%，但均值区间均跨零；不据此事后切换主指标。NDCG@50／100均值+1.37143／+1.72509%，各仅2/5正。未改训练、数据、冻结数值源码或挑选checkpoint。

**2026-10-22 02:51全量指标核对完成：** [结果报告](foundation_recflow/runs/chain_2gpu/attempt_1/large/sampled_retrieval_full/report.md)，全部3,210名合法final用户人口、94,809个请求，含无正例／OOV零分。**当前推荐Recall@100：** 五次相对提升+1.050／+1.022／+0.346／+1.594／+0.722%，等发布平均+0.94667%，配对UID95%区间[0.32784,1.58641]%，新版是匹配随机期望的2.107–2.272倍。三条点估计超过1%，但仅V3→V4单边区间排除零，不能说三条或五条均统计显著。**NDCG@50全量推翻小面板判断：** -0.718／+1.215／-1.329／-2.403／-4.594%，仅1/5正，平均-1.56585%，区间[-2.47884,-0.59904]%；MRR也平均退步。NDCG／Recall全部cutoff与不利结果均保留。结果是候选覆盖的温和平均收益，不是全面排序改善或原生生成证明；完整人口含此前探索面板，非独立确认。

两份2/3卡补测均退出0，原0/1训练及`recflow-foundation`未停，完成时Max V0第二轮累计40,200步。`completion_audit.json`核对全部冻结数值源码hash未变、完整请求覆盖、保留原训练PID；FP64统计与原FP32汇总最大舍入差1.70e-9。新推荐是当前现有模型链上的探索性指标选择，没有追加训练或推理队列。以下保留CPU筛选、小面板和迁移经过。

**2026-10-22 指标探索补充：** 用户希望常见指标在多次发布有约1%–3%量级的相对提升，同时明显超过匹配随机；候选限定AUC／NDCG／MRR／Recall等，不选冷门指标。既有反馈分数的CPU扩展见`large/metric_screen_cpu/summary.json`：MRR五次均正但平均仅0.302%，请求AUC均正但平均0.680%，NDCG@1平均0.569%。已保留的固定FPR点属于之前的探索，5%点虽五次正、均值1.405%，其配对UID95%均值相对增益区间[-0.227,2.094]%仍跨零，1%／10%点各仅三次正，不能称稳定收益；最新用户要求也不以此作候选主指标。`large/operating_point_analysis/summary.json`保留所有不利结果。

大候选池1024请求／日诊断已完成（286秒，峰值保留5.00GB）：`large/sampled_retrieval_1024`。NDCG@50五次相对提升+3.053／+2.234／+5.037／-2.413／-4.134%，均值+0.755%；新版是匹配随机的2.78–3.34倍，但各边增益UID95%区间均跨零。Recall@100五次均正（+0.790／+4.922／+1.024／+2.965／+3.869%），均值+2.714%，均值UID95%区间[0.283,5.224]%；随机倍数1.79–2.24。MRR三正两负、均值+0.992%，均值区间跨零。不能将此小面板当正式全量或独立确认。

用户随后明确只需至少3/5次有明显现象，RecFlow更侧重生成／检索任务，指标应常用自然。已启动`large/sampled_retrieval_full`覆盖全部94,809个合法请求，保留无正例请求零分；首选NDCG@50、补充Recall@100，所有cutoff／MRR／退步项仍保留。初始与训练共享0/1，**用户明确只停这轮找收益的补测，保留原训练**；02:39已只终止补测PID702518／702522，将177个已提交评分块连同原硬件／hash来源迁移到GPU2／3，继续`gpu2`（V0→V1、V2→V3、V4→V5）与`gpu3`（V1→V2、V3→V4）。tmux socket仍为`recflow202`，补测会话`recflow-metric-full-2/3`。原训练PID628534／670134／670150／670151和`recflow-foundation`均未停止，0/1仅运行原训练。迁移记录`migration_completed.json`保留旧block绑定和实际原GPU，数值源码／模型／面板／精度不变。

补测使用初始220万目录中1000个固定均匀干扰视频及已知观测正例，未知正例漏检留在分母；未曝光干扰项没有真实负反馈身份。工作标准、任务匹配和随机对照见该目录`plan.json`。**现有Large／Max的训练目标仍是effective_view BCE，标量反馈头；本次是现有模型的候选检索诊断，不能称原生自回归生成或全目录自由生成证据。** 新用户生成式方向优先于旧任务规划，但尚未授权改变正在运行的训练配方或重训底座。

**2026-10-22 01:51（北京时间）检查：** [运行健康与速度快照](foundation_recflow/runs/chain_2gpu/attempt_1/health_2026_10_22/summary.json)。Large V0–V5完整79,236步已结束，12个永久端点和两份滚动点齐全，五条相邻评价均完成；训练含保存3.81小时，含评价约3.92小时，原预算4.5–5小时。Max当前V0第一轮约70%，最近5000步0.233秒/步，与V0短测0.234秒/步一致；两份滚动点正常。日志无报错／非有限值／OOM，tmux独立运行，GPU0／1利用率99–100%、各约24.8GB，剩余磁盘约534GB。冻结源码／设计／配置hash未变。按当前速率，更新完成预估为当日06:45–07:30；保留原13–14小时为启动前保守排期，不作为实际耗时。没有重启、改参或新增训练队列。

- 最新用户范围为202物理GPU0／1，**先Large V0–V5，再Max V0–V5**。单入口`scripts/recflow/run_foundation_chain.py`，配置`foundation_recflow/configs/chain_large_max_2gpu_202.json`。用户已明确授权启动，`training_authorized=true`已冻结；202的独立tmux socket `recflow202`／session `recflow-foundation`已运行，无客户端连接时继续执行。不从旧只跑V0/V1入口启动。
- 正式attempt：`foundation_recflow/runs/chain_2gpu/attempt_1`。`launch.json`保存实际命令、PID、会话、日志与恢复命令；`launcher.log`记录阶段，`large_train_v0_v5.log`记录当前Large训练。接入：`tmux -L recflow202 attach -t recflow-foundation`。保持数值源码／配置／数据冻结；只在旧进程确认退出后使用`--resume`，断网无需重启。
- 启动确认：tmux服务器PPID为1、会话attached=0，训练在客户端未连接时继续推进。Large V0第1轮已进入正式optimizer更新，首份第101步完整恢复文件已提交（约8.76GB），其后进度继续增加。当前PID与时刻以launch.json为准。
- 每版两轮，独立从头初始化各档V0；V0拟合D1–18，V1–V5分别只拟合D19–23。每档12份完整模型／AdamW／RNG／进度永久保存：`fresh → v01 → v02 → v11 → v12 → … → v51 → v52`，版本间继承实际优化器。小时滚动恢复保留两份，永久epoch端点同样可恢复。
- [运行准备](foundation_recflow/README.md)、[预算](foundation_recflow/planning_chain_two_gpu.json)、[模型设计](docs/recflow_model_design_2026_10_21.md)。Large为10L/H320、220万独立视频、730,211,201参数；Max为16L/H320、350万独立视频、1,149,285,121参数。各有65,536个稳定哈希行，非embedding参数与对应Yambda匹配；旧六层链继续叫Medium，原NDCG证据不改。
- 两档映射已准备，均与CPU规划的已知视频hash一致；Large目录嵌套于Max。角色维持development26,277、admission3,291、final3,210，根训练实际23,668人；每档六版两轮共7,606,044次请求访问、39,043,674条反馈目标访问。既定admission分工保留，评价含所有合法哈希目标。
- 两档完整六版速度和显存／保存／新进程恢复检查已完成。Large全局96／每卡48；Max全局80／每卡20分两次累积，较每卡40耗时只增约4.3%。当前纯训练预算Large3.69小时／Max6.40小时；含保存、启动、评测排期分别4.5–5小时／8–9小时，串行约13–14小时，属短测外推，非完整链实测。
- [显存与恢复证据](foundation_recflow/runs/memory_readiness/summary.json)：两档各覆盖15种形状，初始／恢复后各120步混合重放，图缓存最多15、稳态显存未持续增长。最大保留显存Large31.43GB／Max29.12GB每卡；完整恢复文件8.76／13.79GB，实测保存24.5／38.4秒。新进程恢复、空图缓存重建后两步对照loss差均为0。启动器检查GPU空闲、剩余磁盘、防重复运行与同源恢复；这些是有界检查，尚无整夜正式质量结果。
- Large质量初析（CPU读取已保存评分，无新推理）：V4→V5在D24的整体AUC下降0.002970，配对UID95%区间[-0.004151,-0.001810]；LogLoss／Brier／正负PR-AUC也退步。用户平均AUC和曝光请求NDCG点估计上升，但V5这条边补算的UID区间均跨零，不能称稳定正收益。五条边的请求NDCG@5均上升，等发布平均差0.001670，事后配对UID95%区间[0.000988,0.002400]，支持局部请求内排序的有限收益；不替代预定整体AUC主指标。见`foundation_recflow/runs/chain_2gpu/attempt_1/large/update_analysis/summary.json`，分析脚本`recflow_feedback_update_analysis.py`。未用final结果改训练配方或选择checkpoint；V0与V5在同一D24的累计收益尚无直接测量。
- 五次发布的相对提升与NDCG口径（CPU解析同一批保留评分）：NDCG@1／3／5均五次正向；@5逐边相对提升为0.1495／0.2578／0.6173／0.2832／0.2618%，等发布平均0.3139%，不能跨不同未来日复利累计。五条@5边仅两条的逐边名义UID95%区间排除零。条件于每条实际曝光请求的候选数与正例数，均匀随机排序的解析期望@5为0.4968–0.5079，新模型为0.5228–0.5450，分别比匹配随机基线高4.95%–7.84%；每请求约5.2个候选。约24%–25%的请求无正例且固定记0，故理想排序的全请求平均上限约0.75–0.76。0.5本身不等于50%准确率，也不代表强排序能力。计算公式、逐边绝对值／百分比和评分hash见[口径与相对收益](foundation_recflow/runs/chain_2gpu/attempt_1/large/slate_metric_context/summary.json)。此分析未启动推理、改动冻结训练文件或更换主指标。
- 按用户追问补算曝光请求NDCG@10／20／50／100，直接使用现有分数、未改模型或冻结评测源码、未用GPU。每请求候选最多11–12个，因此@20=@50=@100；@50五次相对提升0.05494／0.18470／0.35124／0.15665／0.12829%，等发布平均0.17516%，绝对平均差0.001025，配对UID95%区间[0.000566,0.001525]。见[扩展cutoff分析](foundation_recflow/runs/chain_2gpu/attempt_1/large/extended_ndcg_analysis/summary.json)。旧Medium的uniform1000 NDCG@50五次提升32.83／45.46／15.56／21.68／14.93%，见`results/recflow/README.md`；旧任务为生成式观测正例检索，正例加1000个随机干扰视频，不能将其与当前小曝光候选反馈任务视为仅cutoff不同。保留当前checkpoint也能另做大候选池的观测正例检索诊断，但需新候选评分推理；未曝光干扰项不能冒充真实负反馈。本次未启动此类GPU评测或改变当前Max训练。
- 每档训练后，在D20–24做五条相邻版本同人口、多指标配对评价；ROC-AUC主指标，附正负PR-AUC、LogLoss、Brier、用户AUC、曝光请求NDCG及known／hash分层、配对UID区间。完整结果尚未产生。24份永久端点加两档各两份滚动文件约315.8GB，计划至少350GB余量。

## 当前恢复点：GPU 0全量基线已启动

- 用户收窄为先Yambda-Large V0→V1。最新要求选自然参数、各方法约3–5点，不反推费用凑点：Layer 2/4/6/8层，Tail 10/20/40/80%，Deviation/Query 10/20/40/60%；两种学习基线各3个fit人数点，Reuse/Rebuild各1点，共24点。真实费用逐点计入，不把名义token／层比例当FLOPs。
- 用户要求质量覆盖全部合法测试用户；撤销最初自行设置的512人质量面板。保留既有fit/validation及开发角色排除，普通历史test用户不因其质量结果被排除。当前完整合法测试人口75,807，质量与完整费用绑定同一人口。其中34,190人有345,630条评分反馈，41,617人本窗口无评分反馈；不能说75,807人都贡献AUC。Large用户池79,681人，V0实际训练62,035人、V1增量训练36,975人，来自实体checkpoint记录。
- 用户要求先提速并在运行前报告30k耗时，旧人口任务停止；提速、自然点和保存／恢复验证完成，已报告最新预估，现已启动75,807人质量／费用任务。没有其他版本或CacheTide训练队列。
- 数据同步完成：D004/D005/D006/D007/D008/Y002/Y003，共221文件、55,482,807,946 bytes，全部SHA256匹配；201已有Large/Max V0–V5全部12权重。见`results/environment_201_2026_10/formal_yambda_inputs/attempt1/summary.json`，本机`sync/201`已更新。
- 提速完成：各策略状态合批、相同Query-only写入共享、History映射／层前缀准备复用、CUDA Graph，以及仅对独立边质量省去无影响的GPU尾部重放。完整CPU部署账仍计入所有事件。真实32人4220分数核对最大差5.25e-6；同输入57.43→4.59s。H10相关核心验证79 passed/1 skipped；新的因果／图权重生命周期验证另见tests。
- 24个自然点的最新速度证据：完整人口独立hash的512人、2086条反馈合计66.58s。30k纯质量约65.0分钟，完整75,807人约164.3分钟；含校准、保存、1000次配对bootstrap，排期分别为70–100分钟和3–3.5小时。数值核对最大差5.36e-6。见`baseline_speed/natural_points_1/summary.json`；旧20点结果仅保留作历史，不直接套用。
- batch补测完成：同128人／20点、6组设置各2次，append band 16/32/64/128与query tile 4/16/64；band128＋query16最快，13.11s对原band32的14.06s，再省6.7%。所有设置10,560个分数一致至最大2.15e-6。选择输入固定为同时间戳前16个候选，独立于计算tile，避免改变方法。训练batch与已完校准权重不变。见`baseline_speed/batch_sweep_1/summary.json`。
- 已补逐点配对UID bootstrap（1000次）及CSV／LaTeX／Markdown表格输出；区间条件于当前权重，不充当多训练seed方差。若Rebuild−Reuse分母在bootstrap中跨零，保留AUC／增益区间并标记恢复率区间不稳定。
- 用户进一步要求及时保存、容忍小段浪费而不能全程丢失。用户随后认为无需1–2分钟频繁保存，现改按4096人写`quality_shards/part_*.parquet`，分数写完后原子提交小型JSON标记，正常约8–10分钟；仅保存必要分数，不保留逐用户执行统计／历史缓存／重复的合并分数。`--resume`验证配置、模型、角色、请求、代码与每块SHA，跳过完整块。已完成校准直接复用；中断的未完拟合点重做。质量全部完成后的统计也可从分块恢复。中断模拟与真实8人恢复验证通过（全部块resumed=true、0次GPU建图），5项针对测试通过；小验证不作质量结论。
- 六个学习点的校准现已全部就绪：复用Q128/Q256/H64，新完成Q512/H128/H256并保存权重；当前已进入全量质量评测。进入评测后的连续30秒GPU0平均利用率89.8%（67–98%）；此前较低利用率出现在校准阶段，不能将速度探针稳态利用率套用于校准。原旧人口attempt在质量阶段前停止，失败／中断attempt摘要保留；性能探针不是质量结论。

配置：`01_quality_cost/configs/baselines_yambda_large_v0_v1_201.json`；入口：`scripts/experiments_2026_10/baselines.py`。当前新attempt为`01_quality_cost/runs/yambda/large/baselines_full_v0_v1/seed_17/attempt_1`，同级`attempt_1.launch.json`保存PID、原命令、恢复命令及日志路径。只在原进程退出后使用`--resume`，不启动重复进程；不从旧512人或next-item队列接续。结果尚未完成。

最新实际进度核查（2026-10-21 15:55）：第一个4096人块已完整保存，覆盖20,244条反馈、24点共485,856个分数，纯评测601.81s（10.03分钟），保存标记也已提交。完成用户占5.40%，第二块运行中。正式首块平均每人比512人速度探针慢约13%，但反馈密度高约21%；按人数／反馈量分别外推纯质量约186／171分钟。当前排期收敛为启动后约3小时10分–3小时30分（约18:50–19:10），仍在原3–3.5小时范围内；仅有一块实测，后续继续校正。此处是执行进度，不作论文受控时间。

用户请求的首块预览已保存到当前attempt的`first_block_preview.md/json`，仅CPU读取已提交分数，不干扰GPU任务。首块Reuse/Rebuild AUC=0.658286/0.666291；Layer前2层恢复点估计98.45%，其全量费用账为21.04%；此前45%边界状态表述已按上述更正。仅查询纠正的低fit点为负收益。配对UID bootstrap500次中14次参考差距非正，参考差距95%区间跨0，因此首块恢复率只作描述，不作最终质量结论；全量固定点继续运行。

累计8192人预览：3759人贡献41,073条反馈，24点按原始分数合并重算（不平均块AUC／恢复率），详见当前attempt的`preview_8192_users.md/json`。Reuse/Rebuild AUC=0.648084/0.657275，差距UID95%区间[0.002950,0.016948]。Layer2恢复102.18%、H256恢复80.98%、Q512恢复39.38%；Q128首块−74.57%、第二块+29.07%、累计−15.58%，波动尚大。第二块评测581.90s（9.70分钟），全量继续，预估结束时间维持18:50–19:10。以上均为中间描述，非最终论文结果。

累计12,288人最新预览（2026-10-21 16:17）：5586人贡献59,501条反馈，用户覆盖16.21%。详见`preview_12288_users.md/json`，由三块原始分数合并重算。Reuse/Rebuild AUC=0.645401/0.655390，差距UID95%区间[0.004148,0.016348]，500次bootstrap参考差距均正。Layer2恢复95.25%、H256恢复75.45%、Q512恢复7.26%；Q-only随着新增用户仍大幅波动，不据中间结果调整固定点。第三块580.28s，速度稳定，预计18:50–19:10结束不变。仍为中间结果。

## 已有基础与主要缺口

- Yambda-Large／Max各V0–V5六个实体权重、数据与版本记录在，无须重训两条底座。
- 当前方法及准备复用、融合读取、原地转换、合批已有Medium原型；本轮H10／72维核心参考／融合／转换及校准形状已扩展并通过针对验证，完整origin训练入口和正式候选质量仍待后续落实；现在优先基线。
- 当前Medium的4096人真实反馈质量面板恢复约65.89%；独立30k费用账下，工程口径完整新增成本约23.72%。两者不是同人口联合测量，不当作最终四配置结果；晚期退化及完整受控时间收益待验证。
- RecFlow旧六层生成式链现称Medium。Large／Max反馈入口、冻结数据、双卡速度与显存／恢复检查已完成，正式权重尚未训练。恢复入口见本页上方foundation_recflow，遵守等待启动指令。
- 当前论文实验图仍为模拟。01已有本轮基线配置及性能探针，其他正式配置尚未冻结；不能从图中数值或旧配置自动恢复执行。

## 下一步及依赖

1. **可并行准备：** 反馈角色／事件与共同触发；H10参考实现；强基线适配；RecFlow反馈入口与目录选择；CPU完整成本／字节。
2. **先行验证：** 极小真实H10执行闭环→Yambda-Large独立相邻边的origin与校准→强静态映射比较→晚边及Max独立复核。先用结果确定默认配置；01不等待从V0开始的连续链。
3. **默认稳定后展开：** 01正式逐边质量／前沿与全部对照；02按默认设置另外建立我们方法的连续链。RecFlow canary通过后可准备底座；05／06可在真实H10输入就绪后开展，不必等四组质量齐。
4. **补足解释与边界：** 03／04必要机制分支、07完整路径时间与存储、08长期追加及读取量边界；默认轨迹、权重和事件账尽量复用。

三项先明确的设置：Figure 8(b)从人数扫描改为固定fit的训练路径归因仍需统一正文；Figure 10稳态V4→V5与首版初始化分开；五版固定特征origin须在V1或更早建立，不能使用V3产物倒测早期版本。

## 资源安排

01／02／03／04／08的质量与FLOPs可合理共卡，训练量、有效batch和精度保持一致。05／06及07的正式时间单独安排无同卡计算干扰窗口，搬运阶段还需检查CPU／I/O／PCIe竞争。先共享生成权重与轨迹，再冻结输入重放计时；共享耗时不填论文时间结果。

实际启动前检查负载，不结束他人程序，不从旧Medium、next-item或到达回放队列自动接续。每次真实运行后，在对应实验的 `runs/` 留配置／summary，并在本页追加完成结果和下一步。旧研究长日志见[历史STATUS](../docs/system_study_2026_10/STATUS.md)。

2026-10-22 23:40 — Chapter3 GPU1：25条E5开发阶梯、三点等价h满窗GPU检查、E6满尺寸8人数值/性能检查已完成。当前20条低学习率拟合，随后`ch3-e5-feedback-pool`等候运行15条因果反馈池配方；见`ch3_motivation/runs/e5/feedback_pool_queue/attempt_1/launch.json`。<512诊断无论文质量用途。原名义50%近期反馈候选实际仅10–13%，新配方只从拟合用户发布前反馈物品池补足；评测用户未改。正式跨配方选型待共同验证口径。E3主/确认队列继续，外部GPU0负载已自然退出。
