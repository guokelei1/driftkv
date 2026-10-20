# 本轮状态与恢复入口

**最新时间与存储主线（2026-10-20）：** 6.4保留读取结构与校准投入；6.5用单栏2×2阶段堆叠图说明相同资源、同一事件账下的完整时间，再以四行小表列原生K/V总量、新增存储及比例。存储不作方法对比柱图，显存不单列；新增量收全码、共享表／参数及交接旧表。Base→+P→+F工程优化在6.6，与工作负载敏感性组成补充分析。当前为七图三表；新增时间和存储均是明确声明的模拟预览，无新测量、GPU运行或论文编译。

**最新章节顺序（2026-10-20）：** 6.4为读取机制与校准投入，6.5为执行成本与资源占用，6.6继续检查计算敏感性；主线改为整体与持续收益→设计选择与拟合投入→执行与空间→条件边界。LaTeX标签保持，下一次编译后校准图为Figure 8、实现优化图为Figure 9，读取结构／路径费用／辅助空间分别为Table 3／4／5。仅调整顺序和衔接，数据与图像不变。

**最新长续写趋势图（2026-10-20）：** Figure 7改为单条成本—恢复率轨迹，横轴累计附加FLOPs／一次Rebuild，纵轴恢复率，稀疏点标签表示追加量/N。取消上下独立线图、阴影和持平点主线，仅展示长期不更新时费用积累、恢复收益可能减弱；成本平台及质量非单调均保留。原模拟数据和完整费用账不变，Figure 5／6保持，无新训练或GPU运行。

**最新Figure 6细化（2026-10-19）：** 用户重新加入Large，现为四配置2×2、各V1—V5，顶部共用Recovery／Added cost／Δ AUC gap(pp)图例；caption不再解释Δ。Max五点不变，Large V3—V5沿既有默认值，新增Large前两点仍是模拟。Figure 5比较方法逐版前沿，Figure 6跟踪固定默认结构持续收益；各模型链独立校准。下方“只Max”属于旧设置。本轮无训练/GPU或真实结果修改。

**最新连续图更新（2026-10-19）：** 当前Figure 6已按用户明确要求改Max两组×V1—V5、同轴成本柱＋恢复率线，Gap以灰色Δ点旁标注；Large、四面板与图下Cost行退出此图。V1前有可继承V0；新加两个早期点仅为模拟，重叠V3—V5与已接受Figure 5前沿不变。正式目标至少五发布，额外真实检查点仍待准备，本轮无训练/GPU或实测。此前“三版足够／五版只是例子”已被本轮明确要求覆盖。

**最新Figure 8/9预览（2026-10-19）：** 用户授权重写模拟数据以突出预期成本—恢复差距，并将Figure 8扩为双栏2×6：四配置×现有三发布。Figure 9改四配置2×2，横轴发布、恢复率曲线，配同版Gap(pp)/Cost(%)；删除static。两个图共用同一默认档和完整费用账，首次初始化仅V3入账。E2/E4/E5共享模拟默认值同步。没有新增模型版本、训练或实测；下方“全部单栏／逐版分组点”等属于旧版。

**最新实验主线审读（2026-10-19）：** 已从编译稿重审证据和重复，调整为 E1→E3→E2/E6→E4/E5→E7，五图五表。E3 改逐版总体 AUC 方法比较，并接入原追加监督表；E2 人口曲线/构成条退出正文，改人口阈值与路径时间表；E6 删除未定义 All。连续链明确各方法/seed继承自己的状态，组合/顺序与独立直接生成对照仍待填。下方六图及配对追加点描述属于旧布局。详情见 paper/new_eval.md 和 evaluation_rationale.md；无训练或 GPU 执行。

**更新：按比较目的选择图形（2026-10-19）：** E2 为摊销曲线＋满人口绝对成本堆叠条，E3 为每版追加前后的配对点，E5 为离散质量—拟合时间散点；E1 的前沿、E6 的分步柱图和 E7 的简单敏感性折线保持。删除原 Table 2 的重复费用定义，原 Table 3 配置表增模型宽度／头数及窗口并填满单栏。只改呈现与解释，模拟数据未改，无新实验；旧布局记录以本项为准。

**当前成文预览（2026-10-19）：** 最新实验计划已同步至 `paper/main.tex`。实验图均为单栏，E4 用表，E2／E5／E6 的路径时间、监督路线和空间用紧凑表；E7 改为两幅简单成本敏感性折线。六幅图及四张对照表的 caption 均标记模拟，另有一张单栏配置表；图源在 `eval_figure/`，整篇预览为 `paper/main.pdf`。本轮没有训练或 GPU 测量，原 Motivation 及历史结果未改写。以下此前的“尚未编译／未改正文／复杂组合图”属于旧阶段记录。 图内主题、脚注和模拟水印已删除；E2/E6 改成紧凑双面板，原资源口径表并入设置正文，结果示范文字以趋势及关键数值为主。

更新日期：2026-10-19。总目标和共同口径见 [README.md](README.md)。
服务测量与诊断的完整汇总见[服务报告](../../results/system_study_2026_10/service/README.md)。

## 当前状态

**最新用户范围：取消端到端及有限在线验证。** 不新增到达回放、排队、吞吐、p99 或全周期墙钟实验。已把 `paper/new_eval.md` 的 E2/E6/E7 改为真实模型路径计时、完整 FLOPs 规模核算、总辅助空间与适用边界；可选时间估算与实测明确分开。完整成本没有删项。并行只读审计补出了强静态复用对照、D3 直接生成对照、任务／人口／触发口径及全目录容量问题，处理与依据见 [evaluation_rationale.md](evaluation_rationale.md)。基础链不重复多 seed，旧结果不改名。此次只修改文档；没有启动训练、GPU 测量、图形重画或论文编译。下方服务平台及 Medium 30k 目标均为历史状态，不是当前接续队列。

**实验组织与资源边界更新：** 已重整 `paper/new_eval.md`：三条主证据（E1—E3）、收益解释（E4—E6）和适用范围（E7）；设置正文约半页，详细协议放作者备注。用户明确固定基础模型链，不做多 seed 底座重训；建议只重复最终纠正训练及关键学习型基线，初步三 seed，复用连续轨迹。模型条件、拟合用户抽样、训练随机性与计时波动分别报告。RecFlow 正式反馈条件及统一历史消费触发等未落实项保持显式。此次只查文献、修改实验蓝图与说明，无新 GPU 实验，旧模拟图和所有结果不变。

**最新展示设置：** 用户确认 Yambda／RecFlow × Large／Max 四组，Medium 仅为开发背景，覆盖以下早期主结果人口设置。已更新论文 `new_eval.md` 和仓库 `eval_figure/` 的模拟图、独立数据／绘图脚本与预览。E3、E7 为较丰富的组合图。尚未产生新实测，也未启动新模型或 GPU 实验。

**当前阶段：讨论实验章节蓝图。** 用户明确第四、五章主体无需大改，优先按论文问题组织
实验、说明每组意义与乐观预期。已在[研究计划](README.md#实验章节草案2026-10-19-最新讨论)
整理7组实验、6个小节的第一版；预期均为待验证假设。本轮未启动训练、GPU验证或扩大人口
评测，也未修改第四、五章。下面保留的实验命令与待办是历史工作状态，不是自动接续队列。
原反馈主任务不变。

**最新任务决定（2026-10-19，覆盖此前同日口径）：** 用户确认以真实like/dislike原反馈为主任务，
与Motivation衔接。原反馈已有4096人／51,731次评分和独立30k人口／393,521次评分的事件账；
后者是工作量统计，不是30k方法质量已测。Next-item全部结果保留为独立探索证据，以下既有
成本、时间与拟合人数结论须按其原任务解释，不能直接移用于原反馈。未执行的next-item配置
不再默认启动；此次澄清没有启动新训练或人口实验。

**最新进展：** 公平的32人图执行已使两方服务时间约减半，但Relay仍慢11.37%；完整V4
同目标校准的joint从64.75s降至12.26s。最终128人图执行尚待稳定空闲GPU，主结论仍是
“数学FLOPs节省成立，完整时间收益未证实”。当前权重、用户角色和请求比例不变。

真实模型会话已接入并通过小规模正确性验证；成本预筛、预算重建小验证也已完成。
四用户真实到达队列、32登记用户的五预算点重建均已完成。
分阶段诊断与无插桩确认均已完成：1024行块减少首读／发布的码转换耗时，
评分和缓存边界检查通过。规模理论、六模型原生探针和八真实用户码转换探针也已完成，见下。
前批已完成；2026-10-19用户授权接续真实会话合批、完整时间及拟合规模研究，现已重新推进。
GPU使用仅限本任务启动的进程，不停止其他人的实验。
当前参考仍为 `A_itemgate6_rms_roll4`；本轮未重训基础模型。V4新增16/32人校准及固定开发评价
已完成，128/256也已完成；五档结果支持继续保留64人的增量配置，不选替代质量候选。
用户最新强调：各模型分别拟合，不研究Medium纠正参数向Large/Max迁移；实验只服务论文主线与
最终设置。本批集中完成Medium真实30k全周期时间核算和拟合人数选择，不为覆盖面新增分支。

此前口径（已由本页顶部最新任务决定覆盖）：next-item为主任务、原反馈为辅助边界。总体效率按最终部署规模的完整
发布周期FLOPs判断，再在代表性规模验证真实时间收益；四人延迟仅为工程诊断。
四人arrival实际上未计训练／目录生成，因此局部慢点不是训练摊销不足造成的；共享固定
成本与逐用户／请求成本分开处理。保留所有测量，不以四人慢于Rebuild否定完整规模的潜力。
基础模型训练仍如实记为like/dislike，不因此自动重训；同模型链、候选和因果历史上的
next-item比较继续作为主评测，绝对质量和Rebuild增益与恢复率一并呈现。

## 历史接续记录：规模收益与多组覆盖

本轮新增执行（历史结论与原输出不覆盖）：

- 真实会话合批：同版本／同长度分组，以组内最大纠正前缀计算，额外零码工作另收费，保留每个用户的因果顺序；
  Rebuild首次请求也合批，先用原32人、V3/V4/V5与逐用户oracle核对。
  饱和批处理与真实到达延迟分开；pack/scatter与传输必须计入。
- 转换后端：新增`fused_conversion.py`，两次Triton调用覆盖所有层的共享／私有分支；
  FP32状态及原数学顺序不变，使用TF32x3矩阵乘，先验证实际误差。
  attempt1合批性能不佳；attempt2按64坐标分块改善转换，attempt3将N/stride改为运行时参数
  减少真实会话的形状编译。三轮数值核对通过、原生KV不变，证据及补齐FLOPs边界见
  `service/fused_conversion/README.md`，实际会话核对另执行。
- 原Inplace、fused转换和零尾合批三个32人真实会话均已完成，评分/状态核对通过；
  逐轮结果见对应`service/batched_session32*`目录。projection attempt1因mask stride长度特化
  在已耗291s预热后主动中止，失败证据保留；改为runtime stride后attempt2已完成，
  全部评分/边界核对通过。交错batch8两轮relay13.821/13.707s、Rebuild12.847/12.875s，
  这组32人局部仍略慢。`batched_session128_final/attempt1`已退出：relay的92,800评分与512状态边界
  通过；native评分差≤9.54e-7，但末drain状态2/1,179,648元素超原容差，最大6.0e-5。
  未生成完整两法计时，失败保留，不静默放宽阈值。
  独立核对发现原生score已融合、append仍用旧HSTU后端（短前缀Torch、长前缀BM32），
  而relay追加使用BM16/vector核。此128轮保留为追加优化前证据；接着为原生追加接同级
  `code_count=0`融合读取。`batched_session128_fair/attempt1`已完成，
  以原native batch8作为参考，同名单、同容差，两方法各512边界及92,800评分通过。
  两轮交错计时Relay26.568/26.419s、Rebuild22.927/22.878s，Relay慢15.7%。
  不把不同追加实现的速度差归为避免重建的机制收益。
- 共享时间：核对已有校准边界，GPU1实测完整固定物品表与V3/V4/V5发布准备、实际搬运；
  共同基础训练、起点来源与多副本准备逐项标注，不将缺项计成零。
  `service/shared_timing/attempt1/`已完成（PID3332810已退出）：resident共享生成0.4264s，
  D2H＋4次串行H2D实测7.2081s。后者不是四卡并发墙钟；独立模型加载不可重复叠到旧校准中。
- 完整规模：主参考先用真实Medium 30k部署人口；成本／时间依真实操作账组成，
  Large/Max多端点先报告已有原生覆盖。已精确核对Medium历史fit/val并集1280，质量eligible
  28720，原4096是开发面板；Medium严格未用角色余量为0，不能称新盲确认。
  Large79681、Max200000是实际池；前者包含Medium，后者不包含全部Large，不能当同一递增抽样。
- 用户新增要求：研究拟合人数与服务人口的关系。固定评价角色，预先嵌套拟合规模，
  比较饱和曲线与扩大用户范围后的质量；不能仅凭固定拟合成本外推1亿人的质量。
  已选V4单边16/32拟合用户、同16验证、local8/joint4/rollout4/seed17，从原64拟合有序前缀取人；
  V3起点与source_v4固定，64模型复用，原source80用户费用不假装按fit数同比下降。
  先原登记32开发人检查，再原4096开发前128固定评价；不是新盲确认。不根据旧确认结果选档位。
  16/32拟合流程分别42.40/57.84s，原64=74.44s；继承同一256人structured起点与64人V3，
  只能解释增量拟合规模。固定epochs也增加了大样本的optimizer步数，不能归因纯样本量。
  固定159并集评价已完成，结果在`calibration/fit_size_v4/attempt1/evaluation159_attempt2`。
  原评价前128人next-item AUC依fit16/32/64为0.51742/0.52485/0.53282，尚未饱和；
  初次评估的adapter复制失败保留，训练未重跑。128/256已完成，同159人评价也完成，GPU1已释放。
  五点完整曲线在`calibration/fit_size_v4/combined_curve/summary.md`。
  新增192人来源只捕获一次，144/272行source及代价分别保存，不改原80行来源。
  主128人V4 next-item AUC的128/256档为0.52898/0.51616，低于64档0.53282，
  相邻差的UID区间为负。teacher MSE却持续下降，不将此解释成样本饱和或过拟合证明；
  当前保留64，不再扩大角色。128/256记录校准流程197.87/368.53s，均独立收费。
  已有分数的query内排序也支持64：pair-AUC/NDCG@10为0.74148/0.62523，
  128档为0.73348/0.61743，差值UID区间为负；见`combined_curve/ranking.md`。
- 融合读取投影：`query_projection/attempt1`完成7个合成数值形状，最大差4.77e-6；
  单层callback在N1024/Q16/B8约0.194→0.131ms。只属数值/算子探针，真实会话正在核对。
  GPU2短时使用后已退出；此计时不含跨层gate准备，不称总服务收益。
- 零尾合批32已完成：查询分组由1250降至837批，与原生对照一致；额外零码工作1.1452GF。
  不同运行间Rebuild基准有明显波动，当前改为同运行交错测量，不据此前单点声称加速。
- CPU调度已逐批复现32人、逐用户操作精确复现30k。时间模型并列每operation分页与
  每release驻留128人的bulk分组，后者峰值KV+code约1.295GB、B8用户操作占比61%，
  不套用全30k驻留时94%的填充率。128窗口含idle、首次页入和最终页出。
  该执行顺序允许发布期跨UID重排，不能称在线到达时延或上线吞吐结论。
- 128窗口的一次共享准备数学有效新增/重建为17.3022%，含来源18.5368%；加转换dot补齐
  为20.6812%，再加读取dot补齐为37.1533%。后两者是已知方法dot敏感性，分母仍为
  数学Rebuild，不能称全部机器执行FLOPs；`cost/window128_dot_tiles/attempt1`完整保留。

### 公平时间结果与当前优化方向

`service/time_model/fair128_window30k/attempt1`使用公平128实测阶段系数与30k真实联合形状账：
一次共享准备S=382.30–397.90s；128用户驻留窗口含完整host页入出时，30k方法预计7686.33s，
Rebuild6292.65s，慢22.15%。这是人口条件预测，不是30k完整GPU实测；方法逐用户项已较高，
扩大人口不能消除当前劣势。global逐operation分页亦为负收益，见对应time_model目录。

按未归一阶段分解，完整Rebuild维护156.10s、码转换30.37s，确有维护节省；但Relay追加多
744.67s、评分多177.64s。即使额外读写全部消除，保留S与页差，30k仍慢4.35%，故仅优化
码转换或继续扩大统计样本无法完成整体时间主张。当前不修改质量配置和请求比例。

接续两项最小工程探索：一是批内融合码追加/截断，消除每用户zero/cat/contiguous小调用；
二是对两方法同等应用CUDA Graph，诊断共同完整query/append路径的框架分发开销。后者先用
8个不同真实校准源状态做固定形状操作探针（`graph_compute_probe.json`），不称真实会话或
人口时间结果。校准的逐用户joint forward/backward与source捕获也做只读检查，尚无新训练。

接续结果：`batched_session32_code_append/attempt2`同进程旧/新Relay与公平Rebuild对照完成，
融合追加码使完整Relay服务仅快0.715%，仍慢于Rebuild约11.66%；保留为小实现收益，不扩128。
`graph_compute/attempt2`完成双方法同等图复用诊断，`graph_dynamic/attempt1`确认动态图前缀
1024/511/7/0以及改换ID后输出逐元素一致。图输出不得充当用户持久KV，B1也需独立复制。
`batched_session32_graph/attempt1`已完成两法各27,200评分/128边界核对及eager/graph
双方法两轮交错：Relay均值11.876→8.608s、Rebuild10.708→7.740s，Relay仍慢11.21%。
所有正式pass新增capture=0。剩余满窗B2–7和追加宽2/4/8仍占约4s compute；当前
`batched_session32_graph_full/attempt1`已完成这些真实形状的图复用：Relay12.070→5.413s，
Rebuild10.884→4.861s，两方均约快55%，Relay仍慢11.37%。短N保持原执行、不填充历史；
实际226图、额外live allocation Relay6.848GB/native5.928GB，图准备6.358/2.556s。
全部评分/边界一致，计时新增capture=0。128实际形状预计会产生更多图，不能两法同时常驻。
下一`batched_session128_graph`采用单方法图驻留、每pass先单独完成预热再计时，两法交错；
模型/表共享，图准备费用和峰值另报，不把完整预热轨迹时间都当作capture费用。
早期窄形状方案为50条图；上述226条方案的空间和准备费应单独使用，不混用两轮系数。

`calibration/joint_batch_probe/attempt2`验证同一V4原8人Adam批的等目标用户合批，仍保留四步
续写。loss差2.79e-9、logits7.15e-7、梯度整体相对L2差7.04e-7；交错forward/backward
1.804→0.229s，验证forward0.772→0.102s，额外峰值74.5→842.7MB。无optimizer step、
无权重改变。这只证明执行方式可用，不按7.87倍折算完整S。已提取可复用joint拟合函数，
保持原批次、目标和选epoch规则。`joint_batch.py`已完成，临时副本的一次clip+Adam更新
后参数最大差7.45e-8、moment检查通过，证据在`calibration/joint_batch_adam/attempt1`。
`joint_fit_comparison/attempt1`已完成完整V4校准实现对照（PID3506314已退出）：
原64fit/16val、local8/joint4/seed17，local一次，两支从相同local终点独立拟合，目标及
epoch选择不变；模型加载/来源/teacher/local/joint/统计/保存分阶段计时。新权重仅为实现
对照，不替换当前候选；不使用评价用户选结果。joint含验证64.748→12.255s，完整单分支
111.889→59.353s；两者均选第4轮，最终参数最大差1.19e-7，理论操作账一致。
原执行分支与冻结V4完整权重逐元素相同。旧V4单独测量74.445s，与本次原路径的运行条件
不同，因此S只能将V4一项替换为新实测59.353s，不能从旧S直接扣本次差额52.536s，
也不能把joint倍数推广到V3/V5；外部来源/teacher生成、origin与共享发布准备继续另计。

最终128图执行尚未完成：外部任务在四卡反复启动。attempt1仅加载约9秒后停止自己的
PID3510190，保留`resource_conflict_before_measurement`，没有有效计时；随后启动保护再次
发现占用，未启动attempt2。只在持续空闲的单卡执行，绝不停止他人任务。
当前没有自有GPU进程或自动排队。短暂空闲后外部任务会再次占满四卡，不能把episode间隙
当作稳定测量窗口。恢复命令保存在`service/batched_sessions.md`。

`time_model/graph32_development_review/attempt1`已完成CPU开发预筛：旧32图执行系数
结合真实30k窗口操作账，方法约3479s、Rebuild2669s；仅供定位，不是最终人口实测。
追加差额主要落在scatter；S也仍显著。30k窗口需要Relay278/native192个图，32实测仅
122/104，缺失图准备和空间另列，不能沿用小样本图集当作全量准备已完成。
新128配置进一步共享同形状K/V及码的图输入工作区，按张量尺寸可分别减少约7.96/6.21GiB
重复输入，private pool与输出保持独立；CPU存储/别名检查通过，实际峰值与正确性仍待GPU。
另已将追加中的实验账本dense-mask求和改为等价整数计数，保留原Counter与数值核对。
全部N=0..1024、实际width档、coded=0..N与两种因果对角的CPU计数核对通过。
`service/scatter_accounting_cpu/attempt1`在原32人追加几何上测得Relay账/metadata
160.5→61.3ms，Native88.7→36.0ms；净缩小差距约46.5ms，只解释原scatter差的一部分。
这部分此前也进入无阶段插桩计时，旧32结果不能替代两项修正后的真实128测量；
不从旧人口预测中直接扣除CPU微测秒数。

成本核对发现另一个需要明确的口径：时间S含source_v4/v5从初态重放旧版本、生成当前
teacher及保存的88.518s，而数学主账假设校准源状态已由服务维护。这80个fit/val用户均在
30k部署人口内，源状态可在旧版drain后、反锚前截取；各版2个idle用户仍需按需物化。
现有88.518s没有将源重放与必要teacher/快照分时，不能直接删除。保留离线重放S作为
保守实测，接着明确已有源状态下的准备入口和实际费用，使时间与数学账可按同一场景比较。
审计见`service/time_model/source_accounting.md`；新`maintained_source_probe.py`及
`maintained_source_v4.json`已完成CPU核查（`calibration/maintained_source_v4/cpu_prepare`）。
GPU运行将实际重建80人当前teacher，并重算2名idle用户的旧当前码；旧teacher只作数值
对照。此入口尚未GPU执行，不据CPU准备结果减少S；须在最终128之后、稳定空闲时测量。

- CPU规模曲线已完成：`scale_cost.py`复现30k三分支、通过6项成本检查。
  `cost/scaling/attempt1/`给出current条件盈亏点1613人，总服务省20%／30%分别需4323／27131人；
  30k净省30.181%，含校准来源重建29.731%。固定目录／四副本／行为分布，不能称实测时间阈值。
  共同基础训练B和初始V2建缓存未计入，两边若加回，绝对节省不变、百分比降低。
- CPU工作量权重已完成：`cost/workload_mix/attempt1/summary.md`。实际30k中1024历史占
  query FLOPs 96.89%、append 98.67%、rebuild 90.32%，不能把长短小样本等权平均成总体。
- GPU0多模型探针已完成：Medium/Large/Max×V2/V5，86个算子单元，2预热／5实测；
  原生重建／query／可用append的B1/B8。B8为同历史平铺，不是8个独立用户。
  `service/model_group_probe/attempt1/report.md`保留真实历史、模型绑定和波动；
  零码融合评分最大差7.15e-7，2项CPU因果选择检查通过，不宣称方法质量迁移。
- GPU1码转换探针已完成：8个不同真实满1024历史fit用户，V4反锚／V5物化。
  `service/batched_conversion/attempt1/summary.md`：单阶段逐个处理69.25／70.68ms，
  合批8.19／8.84ms；设备事件3392→424，scratch约4.7→32.6MiB，码误差≤4.77e-6。
  原生K/V精确不变、持久码同一buffer。组合交接波动大，不把10.589倍当稳定加速；
  不重训，不把单个转换算子的耗时称为整个发布周期的C。
- [多组矩阵](cross_group.md)已核查三规模×早中晚9组及RecFlow缺口；当前H6/48代码不能
  直接用于H10/72。Medium已有质量证据可复用，其他组适配与校准另做，不能借权重声称通过。

本批目标是给出有条件的FLOPs盈亏点和时间测量覆盖，并找到C能否真正低于R的实现条件。
最终规模总时间尚未验证；不以4人或4096人局部延迟作结论，也不因扩大N就假定时间必然获益。
四副本全驻留30k状态加共享表的下界已约73.5GB/副本，超过48GB A40；后续时间路径必须
计入主存／GPU缓存及搬运。批处理对照须同时优化Rebuild，不能只给方法组摊薄调用。

| 工作线 | 当前结果 | 下一步 |
| --- | --- | --- |
| Q 拟合规模 | 五档及query内排序完成，保留64；完整V4合批校准保持原目标和选轮 | 不扩拟合角色，不替换质量候选；只按实测更新V4校准时间 |
| B 强基线 | 32人五预算、双协议完成，27,527评分与既有relay身份／权重绑定匹配 | 固定政策扩开发人口；原反馈反例与预算偏移须保留 |
| S 服务 | 公平128合批与32图执行完成；30k形状/分页账就绪 | 空闲GPU完成128图测量，再含校准、图准备及搬运预测30k；另核对已有源状态准备口径 |
| X 跨条件 | 24个权重在库，六端点原生算子已测 | H10方法仍需码形状适配与独立校准；RecFlow需适配SiLU、位置偏置和三token评分 |
| R 相关工作 | 审读已完成 | 按related_work.md定义局部层重算对照 |

## 已完成证据

结果路径均相对于 `results/system_study_2026_10/`；完整索引见该目录README。

| 输出 | 结果及限制 |
| --- | --- |
| `inventory/initial_2026_10_19/summary.json` | 74条资产路径、24个基础checkpoint存在；未重验全部大文件哈希 |
| `service/trace32/summary.json` | 三发布、29,860行为；next-item1,700轮／27,200评分，feedback286轮／327评分；CPU因果检查通过 |
| `service/gpu_smoke4/attempt2/summary.json` | `passed_gpu_global_replay_correctness`；4UID×三版×两协议×relay/reuse/rebuild，评分、最终K/V/码及操作账与逐用户oracle一致 |
| `baselines/rebuild_smoke4/attempt1/summary.json` | 0/20/100%小验证；0%等于Reuse、100%等于首次推荐Rebuild；融合读取对Torch最大误差7.15e-7 |
| `checks/control_match4/summary.json` | 两个独立入口的6,694个原生对照评分最大差0，完整候选身份一致 |
| `cost/initial/summary.json` | 当前三版校准组件、来源费用、双协议30k总账逐项复现；Q1/Q2为形状预测，无新质量结论 |
| `checks/summary.json` | 初始10项CPU因果／队列测试；后续检查单列，不改写历史记录 |
| `checks/gpu_stage_cpu/summary.json` | 本轮25项针对性CPU检查通过，GPU证据在各运行目录单列 |

GPU小验证用户为380、4520、130、1010，来自原32人登记的四个发布前分层首位。
next-item每方法3,296个评分，原反馈每方法51个；多版模型共驻仅用于正确性，不是容量证据。
S的attempt1在完成首个relay单元后遇到封装元数据错误：将NativeTrajectory原始行为数组
当作session事件日志。修复快照类型区分后attempt2通过；attempt1配置、日志、失败摘要保留。

四人预算重建的20%名义点只选中短历史用户380，实际next-item重建费用仅为完整首次重建
预算的1.3211%，原反馈为0（该用户未发请求）。因此不能称为同20%成本质量比较；这也是
扩到既有32人面板的原因。选择只用发布前14天活动数／估计重建费用，未使用未来请求或标签。

成本预筛（next-item16/16／原反馈，新增FLOPs÷真实首次重建预算）：

| 方案 | 主口径 | 含来源重建敏感性 |
| --- | --- | --- |
| 当前3＋3读取、64fit、local8/joint4 | 17.32610%／23.71512% | 18.56072%／25.96605% |
| Q1：读取6＋6，其余不变 | 26.43227%／32.75046% | 27.68527%／35.03490% |
| Q2：读取3＋3、128fit、local16/joint8 | 20.22440%／28.99923% | 22.15680%／32.52235% |

Q1主要增加每次服务的读取工作；Q2主要增加发布前校准，不能由此推断质量一定改善。
20%为设计目标而非掩盖边界的筛选线；两个协议分开报告。

## GPU批次与接续诊断

启动前四张A40均空闲；只使用物理GPU0和GPU1，GPU2/3保留空闲，未停止任何他人进程。

| 批次 | PID / 物理GPU | 配置与日志 | 完成依据 |
| --- | --- | --- | --- |
| 六模型原生探针（完成） | 3151313（已退出）/ GPU0 | `configs/system_study_2026_10/model_group_probe.json`；`service/model_group_probe/attempt1/run.log` | 86单元、六模型绑定、评分／K/V检查通过 |
| 八真实用户转换探针（完成） | 3185700（已退出）/ GPU1 | `configs/system_study_2026_10/batched_conversion.json`；`service/batched_conversion/attempt1.log` | 三操作×B1/B8，5次交错，码／KV／持久空间检查通过 |
| 1024行块到达确认（完成） | 2866701（已退出）/ GPU0 | `configs/system_study_2026_10/arrival_smoke_chunk1024.json`；`service/arrival_smoke4_chunk1024/attempt1/run.log` | 两速率完成、评分核对通过 |
| 分阶段诊断（完成） | 2842079（已退出）/ GPU0 | `configs/system_study_2026_10/phase_probe.json`；`service/phase_probe4/attempt1/run.log` | 同目录summary成功、四分支评分／边界核对 |
| 真实到达pilot（完成） | 2765203（已退出）/ GPU0 | `configs/system_study_2026_10/arrival_smoke.json`；`service/arrival_smoke4/attempt1/run.log` | 同目录summary成功、6单元评分核对通过 |
| 预算重建32人pilot（完成） | 2763412（已退出）/ GPU1 | `configs/system_study_2026_10/rebuild_pilot32.json`；`baselines/rebuild_pilot32/attempt1.log` | attempt1/summary成功、两协议全部预算完成 |

到达pilot使用4UID、next-item三版、三方法；每方法完整预热后重新建会话，分别以
1e-5／1e-6缩放墙钟到达间隔，逻辑跨度压缩为36.288／3.6288秒。GPU同步后停止计时；
初始V2构建、共享加载和预热单列，首推荐／热推荐／发布／drain分开。
这是有限合成负载诊断，不能用四人观测p99证明生产尾延迟或稳定吞吐。

32人pilot使用原工程登记全部用户，0/10/20/30/100%五预算、两种任务；每条政策独立保留
重建后的缓存。原生路径不维护无关support flags，并使用融合零码读取。复用已有relay32
评分前须检查候选身份与权重／面板绑定，不重复跑旧relay质量实验。

已完成到达pilot共22,542条计时事件、19,776评分；relay评分差0，原生优化对照≤7.15e-7。
在36.288秒跨度下，relay/Rebuild推荐平均执行时间约14.68/8.86ms、发布约149.33/14.65ms；
发布包含旧版历史推进及交接，尚不能全部归因于码转换。在3.6288秒跨度下出现更多突发排队。
这些差异触发新的phase_probe4，而非据此宣称FLOPs节省已经转化为延迟优势。

32人五预算已完成并与retained relay32匹配：

| 协议 | 名义10/20/30%实际使用量 | relay / Reuse / Rebuild平均AUC |
| --- | --- | --- |
| next-item16/16 | 10.180/20.888/31.595% | 0.513449 / 0.501210 / 0.501943 |
| 原反馈 | 14.669/29.744/40.186% | 0.803979 / 0.815458 / 0.845028 |

原反馈的反例保留；三版负样本只有25/8/14，不能推广总体，不能基于这32人调选政策。
此前4096人原反馈的正增益仍是原结论；这个32人反例提示异质性／抽样波动，不覆盖旧结论。
扩大比较时保持14天活跃度／估价政策，报告配对区间；不从本面板挑有利用户或预算。
完整曲线、实际重建／idle事件和绑定核对见baselines/rebuild_pilot32/attempt1/summary.md。

phase_probe4已完成相同四人轨迹，拆advance、publish、score及嵌套materialize/anchorize：

| 行块 | 11次物化累计ms | 7次反锚累计ms | 最大转换临时空间 | 最大评分差 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 622.34 | 432.65 | 0.50MiB | 0 |
| 512 | 223.55 | 140.25 | 1.98MiB | 4.77e-7 |
| 1024 | 123.87 | 68.35 | 3.95MiB | 4.77e-7 |

512/1024各16个边界检查通过（元数据、操作账精确一致，K/V及码在数值容差内）。
128下publication_switch的436.96ms中432.65ms来自反锚；此处旧历史推进已单列。
算术工作和持久码结构不变，收益来自减少小算子调用，代价是较大临时工作区。
1024行块的独立无插桩确认也已完成，所有6,592评分最大差4.77e-7：

| 到达尺度 | 行块128→1024推荐p95 | 首推荐平均执行ms | 发布平均执行ms |
| --- | --- | --- | --- |
| 1e-5（36.288秒跨度） | 42.68→28.76ms | 57.59→15.87 | 149.33→29.47 |
| 1e-6（3.6288秒跨度） | 331.37→183.36ms | 62.93→19.75 | 154.60→38.31 |

热推荐没有同量级改善，且1024下仍慢于本次原生Rebuild；不能称已取得端到端服务优势。
128与1024是固定顺序、独立时段的单次四人测量；128时GPU1基线曾并行，各卡独占但
宿主机负载未受控。结果支持继续探索1024工程参数，正式延迟比值需要交错重复和更大面板。
不将此调整称为理论FLOPs节省；原128配置及所有原结果保持原样。

实际启动命令（不向已有输出目录重跑）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u scripts/system_study_2026_10/arrival_replay.py --config configs/system_study_2026_10/arrival_smoke_chunk1024.json --output results/system_study_2026_10/service/arrival_smoke4_chunk1024/attempt1 --gpu 0
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u scripts/system_study_2026_10/phase_probe.py --config configs/system_study_2026_10/phase_probe.json --output results/system_study_2026_10/service/phase_probe4/attempt1 --gpu 0
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u scripts/system_study_2026_10/arrival_replay.py --config configs/system_study_2026_10/arrival_smoke.json --output results/system_study_2026_10/service/arrival_smoke4/attempt1 --gpu 0
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/system_study_2026_10/budgeted_rebuild.py --config configs/system_study_2026_10/rebuild_pilot32.json --output results/system_study_2026_10/baselines/rebuild_pilot32/attempt1 --gpu 0
```

已完成的核心命令（输出已存在）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u scripts/system_study_2026_10/model_replay.py --config configs/system_study_2026_10/gpu_smoke.json --output results/system_study_2026_10/service/gpu_smoke4/attempt2 --gpu 0
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/system_study_2026_10/budgeted_rebuild.py --config configs/system_study_2026_10/rebuild_smoke.json --output results/system_study_2026_10/baselines/rebuild_smoke4/attempt1 --gpu 0
CUDA_VISIBLE_DEVICES='' python scripts/system_study_2026_10/cost_screen.py --output results/system_study_2026_10/cost/initial
CUDA_VISIBLE_DEVICES='' python scripts/system_study_2026_10/compare_controls.py --service results/system_study_2026_10/service/gpu_smoke4/attempt2 --baseline results/system_study_2026_10/baselines/rebuild_smoke4/attempt1 --output results/system_study_2026_10/checks/control_match4
```

## 接续边界

S保留on-demand历史推进：行为只记到达，推荐、发布、drain才编码严格过去；行为标记延迟
不是写入完成延迟。初步全版本共驻之后，正式容量测量只保留活动及交接所需状态。
B的发布前估计预算不等于实际首次推荐预算；idle入选者、未用预算和实际重建必须单列。
按层局部重算仍需说明入口隐藏状态的来源和代价，不能把完整重算叫作局部修复。

Q从原Structured V3起点与冻结origin_v3.pt特征继续；固定A、item门控、RMS、rollout4、
seed17、LR0.001，V4/V5必须capture本候选前版状态，不借当前候选来源。质量探针使用原4096
开发角色固定前128人，不使用两批已揭晓确认作新盲测。原Q1/Q2均未启动，当前只推进V4拟合人数实验。

## 中断后恢复

1. 读本页与总计划，查看git状态，保留其他人的修改。
2. 检查活动PID命令和日志、progress与summary；路径存在或进程退出不等于成功。
3. 成功结果直接读取；失败保留原attempt，新输出目录只续做未完成工作。
4. 从各工作线下一项继续；旧rank8、结束的partial-refresh和旧校准不是默认队列。

本轮原型不承诺任意事件断点恢复；持久配置和结果用于确认哪些工作完成、避免重复训练。
