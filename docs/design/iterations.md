# Design 迭代记录

本文件与 [plan.md](plan.md) 配套：计划保存当前方案和下一步，本文按时间追加探索与决定。
不删除失败尝试；纠正旧结论时追加修订并指向证据。完整配置与数值放在对应 `results/design/<run_id>/`，
这里说明研究判断，不复制日志。新一轮开始与结束都更新当前状态。

## 当前状态

- 探索状态：**已按2026-09-07用户最新指令收束本轮goal**：完成当前6000用户实验和两份总结后停止所有相关工作。方法尚未通过质量与完整成本联合验证；结束goal不表示方法已达标。
- 固定基础：六层 Medium、H192、6 heads、context1024、seed17、D14 模型 M0–M5；缓存服务阶段记为C0–C5，保留真实混合producer。
- 2026-09-07用户调整范围：重点优化和评价M1/M3/M4/M5；M1→M2就是历史E14滚动AUC +0.1279756个百分点的小变化边，M2设为native No-op，不再为它调校Translator。仍经过M2真实模型发布/追加/淘汰并维护摘要，后续缓存谱系不跳版本；该范围来自用户明确决定，不称为自动选择规则。历史M2负结果保留。
- 最终实验：方案15四个适配目标M1/M3/M4/M5恢复96.60%/28.95%/90.17%/37.29%，M2为native No-op；同6000用户、136610条五阶段请求，893.09s完成。相对方案9两种设计的四条配对区间均含0，不宣布稳定提升。
- 已交付：[论文设计总结](research_summary_2026-09-07.md)、[代码与实验工程总结](engineering_summary_2026-09-07.md)。6000人确认输出仍未读取；后文的历史计划和未来建议不构成自动继续执行指令。
- 路线状态：四组件和五次真实生命周期已实现；质量仍不均衡，等待期覆盖、完整人口成本与独立确认尚未完成。M5保留E14_partial诊断边界。上述缺口按用户要求留在总结中，不在本轮继续扩展。

记号按用户最新约定：方法用“方案N”，模型用M，缓存服务阶段用C。旧记录中小写v对应
方法编号、大写V对应同号模型M；旧run_id、配置和封存证据不重命名。五次模型更新不是
五次方法实验；任何单段恢复率均不能读作跨五段平均。

## 2026-09-06：建立首版计划

**输入与约束。** 用户要求遵循论文四条主线，先完整闭环再联动迭代，所有模型实验只用六层，
记录尝试与失败；局部问题自行修复，根本推翻设计的结果及时报告。长于 30 分钟的任务先估时，
使用 tmux detached；保持论文仓库的最小实现与必要检查。

**检查范围。** 阅读论文 Cross-Version Cache Adaptation、文档地图、现有实验草稿、Medium 资产
记录、当前 Motivation/Insights 说明，以及 HSTU attention/block、cache、滚动追加和诊断 reader。
确认六个 D14 checkpoint 与 seal 文件存在；未重新校验权重内容 hash、读取新方法结果或运行模型。
本轮开始工作树干净。

**发现与安排。**

- 四条主线已有论文描述，代码只有基础模型和诊断路径；新方法还不能运行。
- 原实验草稿中的“先测摘要再翻译”“固定 S1 schema 后训练”等推进约定与本轮完整 v0 的要求不一致。
  将它们调整为诊断维度；真实摘要和学习式闭环可以同轮研究。
- 原生 HSTU 是 SiLU pointwise 聚合、按模型 `max_seq_len` 缩放；读时 slot 权重不能误作 softmax。
  backbone 冻结后，Translator 的训练仍需要可求梯度的 reader 路径。
- 旧 oracle per-user fitting 不能直接搬作 shared Translator。建议用独立 calibration 的共享监督，
  具体边界列在计划讨论项中，尚未建立新校准合同或启动训练。
- 现有 Insight 1 覆盖全部 3,000 用户；原 Insight 2 confirmation 的禁读边界继续保留，
  不把整个用户集称为全新盲测。建议提前预留对方法未读的独立确认用户。
- 连续原型从第一版纳入追加、淘汰、混合 producer 与第二次发布；独立单边结果不能代替它。
- 代码拟集中在 `src/hstu_kvcache/adaptation/` 与 `scripts/design/`，输出在 `results/design/`；
  本轮只建立文档入口，没有创建占位代码或移动现有实现。

**结果与限制。** 首版计划已经落盘；这里没有质量、耗时或成本达标结论。
计划中的用户数和网络选择是起步建议，首次 canary 后补真实预算。
已检查本轮文档的本地链接和 diff 空白格式；没有运行模型测试或编译论文。

**下一步。** 完成计划讨论，按 v0 整体联调推进；实际训练前落实一次共享校准边界，
小样本结果同时用于数值检查、误差定位和估时。

## 2026-09-06：用户澄清与实施前 review

**用户决定。** 摘要拟合和 K/V 衍生学习可直接用于设计，撤销过时的 target-KV fitting 笼统禁令。
方法仍沿摘要、翻译、后续读取与连续维护展开，各部分按实践调整。连续优化指同一份 K/V 从产生到
跨多个版本、新旧 producer 混合、淘汰及长期保留的演化。少量计算恢复大量差距从初期就是方向要求，
不能以“90% 计算、10% 恢复但代码能跑”作为完整成功交付；初期精确阈值保持灵活。
以上决定取代前一条记录中的待讨论状态与监督授权前置项，历史记录本身保留。

**Review 方法。** 逐项对照论文四条路径、当前计划与实验定义，并复查 HSTU 可微追加接口和
`state_transition.append_with_rolling_cap` 的淘汰顺序。检查完整接口、训练/推断一致性、连续谱系、
监督和评价集合、效果/成本验收及过时规则。未启动模型或新实验。

| 发现 | 为什么影响实施或结论 | 已落实的处理 |
| --- | --- | --- |
| 旧 fitting 禁令和额外监督许可仍在多个活动入口 | 新方法会被自己的 reconstruction loss 卡住 | 更新 AGENTS、计划、实验设计与相关导航；小校准按配置推进，旧 sealed 合同只解释旧运行 |
| 旧 v0 完成标准允许“质量弱也完成” | 把连通性误当作研究目标达成 | 区分接线完成与 v0 初步有效；第一张表就同时看低计算与实质恢复，高成本弱恢复继续改 |
| 连续版本只有概括，没有写入/淘汰操作顺序 | 可能在淘汰后读到不匹配视图，或使用已经应当淘汰的历史 | 明确淘汰→扣除→必要刷新→修正前向→保存实际新 K/V/source，并保留严格 prior 请求语义 |
| 全局 source revision 容易被无关 current 追加触发 | 默认 Translator 只读旧段，过度失效会造成多余刷新 | 按目标译者的实际输入判定变化，稳定时间/事件序号在读时映射到窗口坐标 |
| 满窗口淘汰可能每个事件都触发剩余旧段刷新 | 发布转换很便宜仍可能被持续刷新吃掉收益 | 首次 canary 同时测刷新单次成本和真实出现频率，作为 v0 方向判断的一部分 |
| 单边全历史 response 教师直接套到混合状态 | 旧段修正可能被要求无说明地吸收 current 后代误差 | 初版 summary/response 目标限定到实际翻译的旧段，后代及全历史误差另测；联合补偿保留为可比较变体 |
| 冻结 backbone 容易连同 reader 梯度关闭 | Translator 无法学到实际逐层修正的反馈 | 明确当前请求六层可微，持久轨迹 detach；每个 optimizer step 重新计算 translated payload |
| 改早期 Translator 后继续用旧混合输入 | 下游训练和测试不再对应新方法实际生成的 cache | 固定最终译者后重放受影响的后续轨迹；schema 变化同样更新整条运行所需的摘要 |
| 同步转换的轨迹容易被写成完整服务覆盖 | 等待期请求及其后代状态可能被漏掉 | 首轮标记就绪后机制测量并计准备/刷新成本；完整服务评价纳入等待期 Reuse 与实际写入 |

**保留的合理选择。** 六层冻结 backbone、独立 fitting/development/confirmation、实际 query 的 paired
read、固定事件归属和源 producer、再次发布从原 source 转换、当前版本后代不自动视为 exact。
这些用于定义可解释的比较，不要求摘要或某个网络先做到完美。关于原 3,000 Insight 用户的已有接触
范围继续如实记录；数据用途分离不等于恢复一条“KV 不准拟合”的禁令。

**Review 结论。** 当前计划可以进入六层 v0 开发，没有尚需用户回答的原则问题或必须先解决的
计划级阻塞。尚未证明摘要在预算内可提供足够响应信息、共享译者能否泛化、长期误差是否可控；
这些是接下来整链小样本实验要回答的问题。没有把论文 oracle 数字或用户目标写成已验证的方法结果。
长训练/正式人口评价仍保留既有的资源、canary 和明确启动要求；本轮没有启动这些作业。

**下一步。** 同轮接通四组件，准备固定的小校准/开发划分和初始配置，运行六层最小代数/梯度/资源
canary 后进入短校准和真实状态轨迹。局部迭代直接推进，结果写回本记录；路线级反证按计划及时报告。

## 2026-09-06 / 方案 v0 / v0_canary_01：完整原型开始

**授权与起点。** 用户授权持续完成整个六层 Design 探索，包括该范围内长校准和正式评价，
无需逐次确认；长作业仍须前瞻配置、估时、canary 与 detached tmux。本轮初始工作树已有用户
文档变更，保留；没有存活的 Design 进程，GPU 0–3 空闲。

**实现。** `src/hstu_kvcache/adaptation/{summary,translator,reader,state}.py` 实现固定64事件分段、
两个32事件 slots、FP32增减、共享跨层/跨段残差网络、逐层 paired read 与真实追加/淘汰/再次发布。
`scripts/design/run.py` 同轮包含 identity、逐事件摘要参考、梯度 canary、独立 UID 校准、真实目标
摘要诊断、单边开发、真实反馈连续轨迹与成本记录。混合来源由此前最终译者的真实回放产生，
训练持久状态 detach，当前六层 query 保留梯度；每 step 重建视图，并纳入发布前末尾真实事件重放。

**配置纠正。** 实际加载 V0 payload 后发现 `block_variant=legacy, activation=elu_plus1`；
初稿声称使用 SiLU/max_seq_len 不适用于这些冻结权重。读取改为 checkpoint 原算子（ELU+1、
无序列分母、attention projection 后 SiLU gate），保留参数。后续每版加载都验证这一事实。
这是实现前假设的纠正，不能把当前结果描述为 SiLU-native 架构验证。

**划分。** 首次方法输出前保存 `data/manifests/evokv_design_medium_v0/split.json`，SHA-256
`bc44ca5ea6cac5dcfaa729c0ea78983f4b79c0572885c5974117b5e7e0d53d87`。
development 保留原 `[0,512)`；原其余2488用户仍不读。排除整个3000 Insight cohort 后，
用 `n_theta0>=1024` 的历史计数保守保证长度，按既有 label-free selector_rank/UID 排序，
先预留2048新 confirmation UID，剩余15516为可按顺序扩容的 calibration。未读确认模型响应或标签。
成本外推人口预先固定为既有 Medium 30000，外推不冒充实际人口测量。

**首轮预算。** 2 calibration / 2 development / 1 trajectory users，V1、V2各6 steps，
hidden64 / Adam lr0.001 / response+0.1 summary reconstruction。首估固定10–120s +
0.1s/step + 0.04s/真实事件，取数后按实际事件数更新，预计小于30分钟直接监控。
本 canary 只回答数值、梯度、资源和完整连通性，6 steps 不用于判断共享估计器是否可行。
发布使用原 Full-only admission seals；所有新输出独立保存至 `results/design/v0_canary_01/`。

**已有必要检查。** 两个纯张量参考测试通过（4.08s）：固定事件归属的部分淘汰/混合 producer 摘要
与保留 cache 重算一致；count-weighted slot read 与重复事件读取一致且可微。未运行全量测试。
下一步运行真实六层 canary，然后按测量预算短校准；不把已实现接口当作阶段完成。

## 2026-09-06 / 方案 v0 / canary 与短校准结果

**检查。** `v0_canary_01` 19.15s、峰值2.34GiB，identity最大误差2.38e-7、逐事件精确差分
1.19e-7；末层query loss对第一层 translated payload 的梯度范数0.00261。真实追加/先淘汰与
原滚动算子一致，逐事件writer增删检查通过。实际V1→V2轨迹保留381个V0和643个V1事件。
两拟合用户、每目标6 steps 的正数仅说明连通性，不能据其宣布方法有效。

**短校准。** `v0_short_01` 固定前32 calibration、前8 development、前2 trajectory users，
各目标200 steps，预测651.64s，实际96.47s、峰值3.50GiB；输出、源码快照和译者均保留。
V1/V2用户等权未裁剪概率gap recovery分别为 -2.918 / -3.260；绝对平均概率gap分别由
0.005081→0.004898、0.008673→0.012959。前者只有很弱绝对改善，且多个小gap用户被过修正，
后者变差。真实目标均值摘要诊断也不强（-2.476 / -0.345），不是 learned 表示的严格上界。
训练response loss下降至约0.02–0.35，不能代替未拟合用户的预测恢复。

**连续性与成本。** 两用户共2485次真实追加；一人仍有旧V0，一人V0已全淘汰而V1后代误差仍在。
V2切换的连续概率恢复分别0.808 / -0.393。只有56条真实反馈，V1 AUC三分支相同，V2仅1条，
任务质量仍未验证。旧事件淘汰1799次，触发逐次刷新，累计转换18214个segments。
learned写入10.239s vs Exact写入7.639s（+34.0%），读取0.169s vs0.119s。
初始摘要backfill均值3.57ms/用户、转换0.64ms，Exact-All约5.14ms；仅backfill已约69%，
不能凭单次转换约12%宣称总发布成本达标。教师cache/replay合计1.49s、拟合14.16s、
后续校准lineage生成41.82s均计入方法开发/发布准备，不能藏到诊断账本。

**判断与下一步。** v0 是高成本且不稳定的完整原型，尚未有效；属于可定位的局部失败。
先以 `v0_score_01` 加一个校准-only输出蒸馏loss，保留实际query的response loss、固定32/8/2
用户和200 steps，检验低S4误差是否不足以约束最终预测。使用同一可见历史的Current reference
完整输出作为额外teacher，明确允许旧段修正吸收该校准形态的少量current后代误差；不混合最终分数，
不替换paired response的实际query。权重1、尺度取calibration固定Reuse–teacher RMS。
预计100–130s，直接监控。随后按结果联改局部Translator/刷新依赖，不能扩展为正式评价。

## 2026-09-06 / v0_score_01 → v1_local_01：预测目标与刷新依赖

**输出蒸馏对照。** `v0_score_01` 99.63s，峰值3.50GiB。相同32/8/2用户、两目标、200 steps，
加入校准用户的完整输出teacher后，V1/V2用户等权概率gap recovery为 -0.220 / -0.544，
平均绝对gap为0.003944 / 0.009096。比v0明显改善，但仍不稳定，第二边绝对误差仍略高于Reuse。
连续V2两用户仅恢复0.128 / 0.152；不能据V1个别高恢复停止迭代。保持原负结果。

**下一假设。** 全用户累计context使一个旧事件淘汰触发所有剩余segments重翻译。
`v1_local_01` 只改变Translator与维护的这一依赖：仍编码同segment两个跨层slots，
累计context置零，位置改为segment内坐标，每次只翻译改变的segment；完整删除只移除其view，
其余view复用。reader、loss、UID、200 steps、lr和真实轨迹不变，与v0_score配对比较。
这将牺牲跨段条件信息，质量是否受损必须测量；所有四组件仍实际工作。
新增纯张量回归检查局部刷新与全量重翻译在部分/整段淘汰后相等，避免用过期context省成本。
预计90–110s，直接运行；仍属开发，不读取确认用户。

## 2026-09-06 / v1_local_01 结果与 v1_cal128_01

**局部刷新结果。** 同一32/8/2用户、200 steps，局部Translator把两条开发轨迹的转换段数
从18214降到1837（约90%减少），但实际learned写入10.84s vs参考8.00s，仍约35%额外墙钟；
小矩阵与Python发射/维护开销抵消了算术节省。V1/V2用户等权机制恢复 -1.530 / -1.399，
比累计context+输出蒸馏更差。此变体不作为当前首选，保留为维护成本对照。
后续不能只报转换段数下降，而隐去质量损失和墙钟未改善。

**下一问题。** 32用户、单用户一步的校准梯度波动很大（已见norm超过19），且200步末端loss
仍波动；目前不足以断言摘要输入本身不可预测。`v1_cal128_01` 恢复累计context和输出蒸馏，
按固定次序扩到128 calibration / 128 development / 8 trajectory users，512 optimizer steps、
每步累计4个独立场景的平均梯度、lr=0.0003，每遍重新排列calibration场景。hidden64、
summary权重0.1、score权重1、候选生成和两条模型边保持。检验更充分且更稳定的共享校准
是否改善泛化，同时保留前8开发用户的配对结果。全128用户均报告，不挑人、不读确认。

**预算。** 根据v0实测0.035–0.041s/场景、约0.005s/calibration事件、0.010s/三分支服务事件，
向上取0.06 / 0.007 / 0.016，加120s固定开销；取数后按真实事件数细化，预计约10–15分钟，
直接监控。按授权继续；这仍是开发优化检查，不是正式确认或方法有效性验收。

## 2026-09-06 / v1_cal128_01 结果与 v2 功能摘要 canary

**更稳定校准结果。** `v1_cal128_01` 预计778.37s，实际451.61s、峰值7.56GiB。
128开发用户的V1/V2平均用户恢复0.0728 / -0.9511，中位数0.6413 / 0.1033，
受损用户15.6% / 43.0%。绝对平均gap从0.008324→0.002309、0.004912→0.004410。
前8匹配用户仍为 -1.915 / -0.135；不隐藏该反例。连续V2八用户平均恢复0.283、中位数0.578。
仅74条真实反馈，V1 learned AUC 0.87305 vs reference/Reuse 0.89648；没有质量差距恢复证据。
训练变稳、较大gap用户有所恢复，但共享预测仍不稳定，不能扩到正式确认。

**实现与成本检查。** 初次writer backfill改为一次分段张量归约，保留固定归属和FP32 sums；
纯张量增删/局部刷新参考3项通过。此修改在上述运行启动后，未影响其已快照代码与计时。
发现早期summary只列writer和translated view、未单列materialized source read-view。
`scripts/design/report.py` 补计已知完整1024快照的295680 bytes，保留旧summary不静默修改。
该配置三部分合计919296 bytes，约普通K/V的9.74%，不含Python容器；新runner显式记录此字段。
bulk writer的共享底层tensor即使部分段淘汰仍存活，新的storage函数按实际底层分配计一次全额。

**v2假设。** 真目标均值K/V摘要的差分也不强，继续逼近这种payload未必逼近所需历史响应。
`FunctionalTranslator` 先用一个轻量全用户网络：输入仍为真实旧source的加权跨层K/V均值、
producer混合比例和old count，输出每层共享响应变化。source读取view保留sum(V)常量项，
translated view在该项加入预测变化；ELU+1零key基的count加权paired read得到该响应差。
K均值仍作为估计器输入，实际event K/V完全不改。普通读取和后续层query仍实际参与闭环，
校准使用同一实际query的旧事件response teacher及完整输出teacher；不使用逐用户oracle网络。
摘要重建loss在此表示下置0，以response+score监督功能payload。新的目标发布仍从原实际source
转换，current追加和old淘汰仍实际维护；old输入变化触发一次共享网络计算。

**首轮。** `v2_functional_canary_01` 用相同2/2/1用户、两目标各6 steps，预计20–30s直接监控。
同时验证新常量基代数、真实六层梯度/identity/追加和bulk writer；保留均值K/V诊断作为旧表示
对照，不称其为功能摘要oracle上界。通过后再做固定开发用户短校准；这不是方法有效性结论。

## 2026-09-06 / v2_functional_canary_01 结果与 v2_functional128_01

**新表示canary。** 18.28s、实际两次迁移；identity/逐事件精确差分仍在2.4e-7内，
末层query到低层payload梯度3.39e-6，真实追加检查通过。四个必要纯张量测试通过（4.04s）。
6 steps恢复约零，不能据此判断表示；paired read约2.91ms vs参考2.59ms，643次learned追加
2.38s vs参考1.98s，额外墙钟由v0约50%降至约20%，仍需继续测量完整账本。
bulk初始化已生效，平均初次writer backfill由3.6ms降至0.63ms。

**紧凑writer与下一运行。** 功能译者只需要加权全局均值与producer count，原64事件×2slot
对该网络属于冗余存储。新的同方案运行从V0起统一使用1024事件上限、每段1slot，发布仍封段，
所以混合producer与部分淘汰继续精确保留，global mean及count可加可减。ordinary cache不改。
这改变source的物理压缩和calibration normalization，因此全链重新生成；不复用旧译者权重或
旧中间状态。对应的真实目标均值K/V诊断也使用此粒度，并非原64×2 oracle或功能响应上界。

`v2_functional128_01` 固定128 calibration / 128 development / 8 trajectory users、两目标、
512步×4场景、lr0.0003、hidden64、response+score、summary loss=0。
预计约6–10分钟（按真实事件数和旧计时保守估算后再细化），直接监控；运行入口再次检查新writer
粒度的真实六层canary后执行。需要检验共享功能预测的泛化、连续后代以及实际成本；确认仍未读。

## 2026-09-06 / v2_functional128_01 与 response probe：转向共享闭式校准

**v2完整运行。** 354.88s、峰值7.09GiB；V1/V2平均用户恢复 -0.582 / -2.927，
中位数0.596 / -1.034，受损25.0% / 75.8%。原128人v1对照仍更好，不能采用该神经功能译者。
持续learned写入14.196s vs Exact 12.529s（+13.3%），读取0.174s vs0.155s；writer backfill
平均0.429ms、转换0.570ms、Exact 5.111ms。完整快照的辅助payload/逻辑metadata为60488 bytes，
不含Python容器。计算/存储方向改善，但质量失败；教师/拟合120.56s、校准谱系161.71s仍单列保留，
没有宣称完整发布/服务成本已达20%。

**最小诊断 `v2_response_probe_01`。** 固定128 calibration和128 development、前两条单边，
不用标签、不读确认。共享ridge（相对正则0.01）从source跨层均值预测CurrentReuse实际query下的
目标响应均值，在开发上的分层相对MSE均值分别0.0171 / 0.0427。把calibration随机16候选换成
history-derived64候选后为0.0172 / 0.0424，基本不变；加入实际Reuse query均值后改善有限。
这不支持“候选协议差异是主因”的假设。对应逐层shared-response oracle闭环概率恢复约
0.9428 / 0.9449，表明常量响应基在这个单请求条件下仍有用，但oracle不是方法。
源摘要含有比现有MLP利用得更充分的可预测信号；这些proxy ridge误差尚未证明推荐质量或连续性。

**下一实现 `v2_ridge_canary_01`。** 不继续加大MLP，使用一个共享低秩ridge Translator，
输入为按producer分别累计、按old总数加权的跨层实际K/V均值及producer质量占比，输出每旧事件
响应变化率。原生聚合的count乘回总响应，避免只在1024/1020个旧事件上拟合总量后向少量旧历史
外推。每次校准pass先以当前译者实际闭环产生query，再计算同query的teacher response，重新拟合
共享权重；下游轨迹仍由最终译者真实生成。固定rank32、ridge相对正则0.01；source增删、paired
常量基读取与多版本状态保持。闭式拟合不做optimizer反传，但reader的payload→末层query梯度
仍由canary验证；没有用独立Exact query替代实际query。小canary 2/2/1、两pass、两目标，估计20–30s。

## 2026-09-06 / v2_ridge_canary_01 与 v2_ridge128_01

**检查。** 共享ridge的解析单方向收缩参考通过，连同摘要/局部刷新/常量基共有5个必要张量
测试通过（4.15s）。真实六层identity、精确差分、payload→query梯度及真实追加继续通过。
canary 17.92s，译者两目标拟合合计0.24s；两拟合用户严重不足以约束跨用户线性映射：
V1/V2开发均值恢复 -71.39 / -15.63，一名用户出现很大过修正。保留该失败；另一个连续用户的
V2恢复0.953不能抵消它，更不能称方法有效。128用户独立response probe的预测误差较小，
因此按原固定顺序补足校准样本是有依据的下一步，而不是选有利用户。

**完整开发运行。** `v2_ridge128_01` 固定128 calibration / 128 development / 8 trajectory users，
前两目标、每目标3次共享重拟合、rank32、正则0.01；per-old-event rate按count还原响应。
每轮使用本轮修正后实际query，Teacher只来自calibration同一可见历史。无summary reconstruction
或score优化，头部质量由完整闭环评价检验。保持新source粒度和整链生成，不复用MLP后代状态。
预计约4–7分钟，按1536场景和真实事件数保守上界约10分钟，直接监控。短校准与开发，不读取确认。

## 2026-09-06 / v2_ridge128_01 结果；补齐校准状态类型

**完成。** 上一goal turn属于实质进展；本轮重新核实PID，原作业正常完成，没有重启。
`v2_ridge128_01` 预计624.77s，实际263.15s、峰值7.04GiB。V1/V2平均用户恢复0.427 / -0.740，
平均绝对概率gap 0.002048 / 0.004497（Reuse 0.008324 / 0.004912）。V1进一步改善，但V2仍不稳定。
校准拟合22.89s，低于神经功能译者113.14s；持续写入14.07s vs Exact12.73s。
实际反馈仍只有74条，V1 learned AUC0.88867 vs Exact/Reuse0.89648，任务效果未成立。
全部数值、权重及负结果保留，不以响应proxy的低MSE宣布成功。

**状态覆盖假设。** V2校准源来自实际V0→V1轨迹，大多数用户仍含V0，只有少数用户全部换成V1；
而单边开发完整初始化Parent V1缓存。当前shared map在这个状态类型上的校准覆盖不足。
`v2_ridge_adjacent128_01` 仅增加目标2的每校准用户一个独立Parent-Exact源缓存场景，使用同一
可见前缀和同一个Current teacher；原256个实际混合/追加场景保留，成为384场景。
这不是拼接Exact K/V冒充混合轨迹；每种输入分别对应其真实生成过程，连续分支不重置。
V1译者、rank32、三pass、UID划分、reader和维护不变。预期约4–5分钟，保守上界约11分钟，
直接运行，复用已通过的数值和ridge canary。先判断对单边/连续两类输入是否有帮助，再扩大真实反馈评价。

## 2026-09-06 / v2_ridge_adjacent128_01；共享decoder闭环微调

补入Parent-Exact校准场景后，实际261.41s、峰值8190.61MiB。V1不变；V2平均用户恢复从
-0.740升至0.114、中位数0.612、21.9%用户受损，绝对概率gap从0.004497降至0.002208
（Reuse 0.004912）。连续8人V2平均恢复0.0646，仍很不稳定；真实反馈仍只有72/2条，
不能据此确认质量。它支持补齐来源分布的局部假设，尚不支持方法达标。完整负结果保留。

下一轮固定source投影和rank32，用校准用户的最终logit误差微调共享decoder。固定目标为
`score_MSE/global_calibration_score_scale² + 0.1*normalized_old_response_MSE + 1e-4*residual_weight_MSE`；
不混合最终分数，不读取开发用户教师用于优化。归一化latent和每层rate尺度后学习一个线性残差，
结束时折叠回原decoder/offset，推理表示、paired read和维护均保留。先运行4/4/1用户、4个
微调step、2目标的`v2_ridge_score_canary_01`检查梯度、折叠后状态与真实追加，预计20–40s；
通过后`v2_ridge_score128_01`保持128/128/8用户、3轮岭回归、adjacent场景，新增256step×4
累积、Adam lr0.0003。按相同路径保守估计约13分钟，预计实测5–7分钟，直接监控。
这是输出敏感方向的局部拟合检查，不能把训练损失下降当作泛化成立。

## 2026-09-06 / v2_ridge_score128_01；暂不采用输出微调

实际329.82s。V1/V2平均用户机制恢复0.391/0.126，绝对gap0.002185/0.002143；相对
未微调adjacent版本0.427/0.114、0.002048/0.002208没有稳定改善。增加约68s运行，故当前
保留更便宜的闭式版本。小canary实际21.40s通过数值/梯度和真实追加，4人拟合的严重泛化失败
完整保留，不能把canary当有效性证据。

先用`v2_ridge_rank128_probe_01`检查rank32响应压缩是否忽略输出敏感方向：只改rank128，
128 calibration / 128 development、3 pass、V0→V1、无服务尾段；所有四个模块仍用相同
实现，复用已经跑通的连续canary。这是单边定位实验，不作为连续性结果。预计30–50s，
资源公式保守估计低于3分钟，直接运行。若无改善再考察固定calibration次序的样本量。

## 2026-09-06 / rank压缩定位；扩大独立校准样本

`v2_ridge_rank128_probe_01`实际40.47s，V1恢复0.42650、绝对gap0.00204950，与rank32的
0.42696、0.00204845几乎相同。因此不增加rank。下一项`v2_ridge_cal512_probe_01`仅把
calibration从固定次序前128扩到前512，保留原128 development、rank32、3pass、V0→V1和
零服务尾段；检验当前共享估计器是否受独立样本不足限制。预计60–100s，资源公式保守约5分钟，
显存按128样本实测外推约25–32GiB，使用GPU0直接运行；不读取确认。

## 2026-09-06 / v2_ridge_cal512_probe_01；定位真实生命周期误差

512人校准的单边检查实际111.78s、峰值20586.26MiB；V1平均用户恢复0.5355、绝对gap
0.0017686，相比128人0.4270/0.0020484有改善，但4倍校准样本仍未达标。它只建立样本量
的局部正信号，不替代真实质量与连续性。

下一项`v2_lifetime_probe_01`固定该译者，在前8个开发用户的真实day231–245轨迹上，按
0/64/256/512/1024/1536次真实追加及期末观察冻结64个无标签候选。独立诊断分支分别将
旧事件或current后代替换为同轨迹Exact分支的K/V，定位旧段估计与后代继承误差；另保留
shared-old-response oracle。全部替换只是干预，不进入方法状态、校准或可执行动作集合。
预计50–110s，直接运行；不把这些无标签快照当真实请求质量。结果决定先改估计还是生命周期。

五边128 calibration / 128 development全轨迹的CPU事件计数已完成：校准跨四段196,362次
追加，开发全服务段261,545次三分支追加；单卡预计67分钟、保守105分钟。该预算表保存在
`results/design/analysis/five_edge_resource_probe.json`，**尚未启动**长作业。先解决当前局部瓶颈，
再按授权落实适用canary、前瞻配置和tmux；不能以长作业替代方法诊断。

## 2026-09-06 / 真实写入的token类型差异；v3 query-only correction

生命周期诊断实际43.76s。UID1930在512次追加后，learned恢复-0.449，即使把全部剩余旧
事件精确替换也只有-0.559；把current后代精确替换、保留learned旧段修正则达到0.874。
这说明当前误差已进入普通新K/V；旧段离开后也不能按producer tag宣布Exact。

`v2_write_response_probe_01/02`分别实际15.89/16.35s。固定128 development用户，各取
day231前最后timestamp组的第一个真实事件，排除整组构建prior prefix，按真实先淘汰语义
比较CC候选与实际write token的response。候选query训练的常量修正不能跨这两种token类型：
第2–6层新K/V的用户等权相对MSE为0.503/0.666/0.490/0.474/0.479；不加修正的原模型
写入只有0.0566/0.0462/0.0582/0.0583/0.0657。response相对比例在近零分母上可极大，
不拿这些极端比例作为主要证据；上述新K/V误差对照更直接。它是当前常量view的局部假设
失败，并非原HSTU算子错误或整个四组件路线的反证。此前负结果和原始状态均保留。

最小联改`--write-mode reuse`：真实事件使用Current模型+实际缓存的原读取路径；只对已知
CC查询类型加paired correction，普通source仍逐次精确更新。在此模式，淘汰旧事件只标记
view dirty，下一次CC读取前从实际保留source刷新；发布仍转换，追加不再消费陈旧view。
这是由输入token类型确定的修正权重，不混合输出分数，不使用未来标签。只纠正写入假设，
尚不保证current继承误差已解决。`v3_query_only_canary_01`用4/4/1用户、两目标、2pass检查
实际追加等价、dirty刷新时点和连续切换，预计20–40s；通过后用128/128/8、3pass、adjacent
的`v3_query_only128_01`与v2原始闭环对照，保守估时约11分钟、预期约4分钟，直接监控。

补充CPU proxy在已有响应raw上比较固定线性+二次kernel，V1相对MSE0.0196比线性0.0171差，
V2 0.0399只略好于0.0427。未增加此预测器复杂度；结果保留在analysis/kernel_response_probe.json。

## 2026-09-06 / v3_query_only128_01；扩大真实质量评价

v3 canary实际19.14s通过实际追加和lazy刷新检查。完整128/128/8运行实际229.55s；V1/V2
平均用户机制恢复0.4268/0.2594，绝对gap0.002049/0.002112，V2较shared-write的0.1138
改善，V1不变。校准谱系耗时由156.2s降至127.1s；方法真实追加11.56s，对照Exact 12.88s，
paired read（含按需刷新）0.209s，对照0.160s。这些是同一路径同步墙钟，未作人口成本资格声明。
8人轨迹V1 AUC 0.88086比Exact/Reuse 0.89648差；V2仅2条正反馈，仍不可估AUC。该负结果
保留，不能用机制改善掩盖任务质量。正式确认尚未开启。

下一步固定这些Translator、扩大至原定128 development用户的两段完整14天轨迹，避免只在
72/2条反馈上反复判断质量。新增`--evaluation-from`直接加载完成运行的译者并引用其配置、
权重hash和全部校准成本，不重新拟合，也不把准备成本记成零。先用8个cutover用户和1条
真实链的`v3_frozen_eval_canary_01`检查与原始运行数值相同，预计20–35s。CPU资源探针
单独统计固定用户全部事件，随后按估时选择直接监控或tmux；当前尚未启动128人长评价。

## 2026-09-06 / v3_quality128_01 启动

冻结加载canary实际15.89s，48行单边和真实链发布点的概率gap与原运行差异均为0。
128用户两段14天需106,477次三分支真实追加，预计1158s、保守1824s。因此已依据用户
本阶段授权、[前瞻运行配置](../../configs/contracts/evokv_design_v3_quality_development_v1.yaml)
及通过的canary，在`evokv_design_v3_quality128` tmux detached中启动GPU0评价。
日志、退出码、配置和逐边raw均在`results/design/v3_quality128_01*`，启动引用hash另留
`launch_record.json`。本次只扩大开发质量评价，仍非confirmation或人口成本资格结果；
所有原校准成本继续引用v3_query_only128_01，不能因本次只加载权重就将其省略。

128人结果汇总使用既定AUC主指标及全部绝对指标；补充1000次seed17的配对UID bootstrap，
只描述单个backbone seed内的用户抽样不确定性。AUC比例仅在Exact−Reuse>0.0001时显示，
此为读数保护，不是模型准入或用户调度规则；零/反号gap仍报告所有绝对量。旧状态量分层为
0、1–256、257–768、769–1024，主结果始终包含全请求。新汇总已用保留的72/2行raw核对，
两目标全部原始绝对指标与既有summary完全一致。

成本解释补充：当前functional reader把self contribution单独加到历史聚合，native baseline
沿用已有append primitive；两者canary数值等价，但Python/张量拼接开销不同。方法追加墙钟
偶尔低于native baseline不能直接归因于摘要策略节省。正式成本资格前需用相同基础kernel
或同时列出这一实现差异，当前只保留实际墙钟、尚不做低于baseline的系统收益声明。

## 2026-09-06 / v3_quality128_01 完成

tmux退出码0，实际957.58s（约16分钟）、峰值5582MiB；完整回放106,477次真实追加。
V1为649请求/79有反馈用户，Exact/Reuse/方法AUC分别0.754236/0.745439/0.753839，
正gap上的点估计恢复95.49%；log-loss为0.234958/0.235793/0.234756，但dislike PR-AUC
0.274579/0.273671/0.258264，方法次指标较差，不隐藏这一取舍。
V2为495请求/66用户，AUC为0.630104/0.633982/0.624406：Exact–Reuse反号，方法也低于
Exact，不能计算正gap恢复或宣布成功；log-loss亦较差（0.224446，Exact0.222061、Reuse0.219572）。
原始全请求、source状态量、支持率及实际成本完整保留。正反方向都属于开发证据；还未覆盖
五个目标、正式确认、按相同基础kernel计量的完整成本，也尚无路线级反证。

## 2026-09-06 / 严格rolling-band回放加速

128人质量配对UID区间较宽：V1 Exact−Reuse AUC的95%区间[-0.00526,0.02785]，方法−Reuse
[-0.00497,0.02785]；V2分别[-0.02104,0.01317]、[-0.02732,0.00885]。因此V1的95.5%只是
点估计，不是已通过的统计结论。全请求平均概率gap由0.003773/0.006027降至0.001089/0.002953。
123条old=0请求上方法与Reuse最大logit差4.8e-7，证明native写入分支按预期继续同一实际状态；
这些No-op请求全部计入主结果。106,477次追加只引发1,012次实际转换/刷新。

为覆盖剩余版本并公平比较基础写入成本，给已有native attention/cache block增加可选
rolling window mask，默认None不变。批量新token的每个query只看自身及前1023位置，
不是先无界append再裁剪。仅在没有中间候选请求的真实事件区间使用，所有分支共用该kernel；
writer按固定slot批量扣除与添加，保留跨过空窗的open segment原始write coordinate。
新增一个纯张量回归覆盖这一成员关系，6项相关检查通过。无随机/小层backbone。

`v3_rolling_band_canary_01`在两个真实用户的643/1710事件上通过逐事件native、原functional
写入、批量写入及前缀不受后续token影响的比较。最大K/V差4.2e-6、CC logit差7.2e-7；
1710事件native逐次4.65s，批量含writer0.047s。实际canary19.14s。这是离线回放吞吐，
不声称线上单请求延迟加速百倍，也不把三条分支同时需要的批量优化记成方法独有收益。

`v3_banded_quality128_01`冻结同一v3译者，用`--replay-chunk 128`重新执行128人完整两段
质量，验证全部请求和实际状态数值与逐事件结果一致；保守估计约227s，直接运行。
新增所有models源码到运行快照，确保可复现本次共享kernel改动。之后才用该路径扩展五目标。

`v3_banded_quality128_01`实际53.48s完成，同一1144条请求、标签、时间、old count与逐事件
结果完全一致，全部AUC和PR-AUC不变；逐请求logit差通过2e-5限值，明细留在
loaded_reference_check.json。此加速使五版本检查可保持小预算，也让所有写入分支使用同一基础kernel。
接下来`v3_five_edge_canary_01`用4 calibration / 4 development / 2 trajectory、两pass和
五个目标，先覆盖V3–V5模型、混合producer及V5独立Full-only与extension manifest入口。
预计25–50s、保守约3分钟，直接运行；微型拟合只作接口/数值检查，不作效果判断。

五目标canary实际20.69s完成，V1–V5及V5独立数据入口均正常；4人拟合的较差泛化结果仍
保留，只把它作为接口检查。下一项固定同一rank32/3pass/128 calibration与native写入，
将development和连续轨迹扩大到原定完整512人，全部五段14天（V5为E14_partial），
先统计真实事件数再启动。不是挑选有利版本，独立confirmation仍未读。

范围澄清：rolling-band数值canary实际前向的是V0 setup/V1回放；其中`pack(2)`只检查
第二目标的writer成员关系，旧configuration的targets=[1,2]不应解释为运行过V2模型。
原配置/hash不改写，脚本已改为分别标记forward_target=1与writer_target_probe=2；
后续五目标canary才实际加载全部V1–V5。

`v3_five_edges512_01`的前瞻CPU计数为246,225个校准回放事件（估时保守包含末段）和
1,053,816个开发三分支回放事件；同一128 calibration、rank32、3pass、adjacent场景、
native写入与chunk128，全部512 development用户、五段14天，V5标为E14_partial。
预计456s、保守1619s（27分钟），显存预计22–32GiB，使用GPU0直接监控；超过预期则
按实际进度重估，不默认延长成未记录长作业。完整canary已通过，原确认用户仍未读取。
资源依据保存在analysis/v3_five_edges512_resource_probe.json，方法/划分/指标没有按V1正数选边。

补充诊断`v3_lifetime_probe_01`预定复用已冻结的`v3_query_only128_01`，仍取前8名开发
用户，在真实V1轨迹0/64/256/512/1024/1536/end快照比较learned、exact-old、
exact-current及old/full shared-response oracle。所有替换只发生在诊断副本，不改方法
输入或持久cache；确认未读。native回放使用已验证band128，预计25–110s，主运行后直接执行。
此项检查v2中发现的后代误差在native写入下是否仍限制old-only修正，不作真实AUC结论。

`v3_five_edges512_01`实际552.12s、峰值23338MiB，全部五边14164条真实反馈。逐边AUC
恢复为97.5%/反号未定义/69.3%/111.1%/57.4%。V5的Exact−Reuse区间为[.01131,.06255]，
learned−Exact为[-.02574,-.00399]；V5仍显著落后Exact，且Brier劣于两对照。其余各边
也保留PR-AUC、logloss与Brier，不以V4 AUC超Exact掩盖该边PR-AUC下降。
单边用户等权未裁剪概率恢复为.281/.160/.011/.683/−.319，部分小gap用户仍严重过修正。
服务方法/Exact实际102.35/96.71s，另有82.15s教师拟合、8.92s校准谱系、4.15s校准源构建
和.305s初次writer backfill；未完成同人口20%成本验证。完整bootstrap/分层在analysis同名quality文件。

`v3_lifetime_probe_01`实际8.35s。UID1930在512次追加后，learned恢复.313，exact-old
诊断仅.239，exact-current+learned-old为.935，full shared-response oracle为.942。
UID543930在1024次追加后旧producer已空，当前方法恢复0，full shared oracle仍为.998。
这确认native写入减轻但没有消除后代继承误差；只扩大old-only校准无法修复其表示/监督范围。

方案v4因此联改为整个实际保留cache的producer摘要及full-history响应目标，包含Current
后代；依旧native真实写入、CC paired常量基读取。任何真实追加使view变dirty，下一次CC
前刷新；不恢复逐事件翻译。附加输入是实际writes-since-release（上限六个1024窗口），
不是未来事件或质量标签。发布时仍从实际原source转换，producer从不改写为理想版本。
校准在原128用户中按固定前缀取32名，为每目标增加发布前64/256/1024/2048/4096/6144
事件的历史回放：独立Parent-Exact源和Current-Exact教师在同一历史时点开始，以Current
原生写入真实事件到cutover；与既有实际混合谱系场景并列。历史不足则使用实际可用前缀、
去重，不造事件。另加同用户Current-Exact初始化的零修正控制。新增教师与回放计入方法成本。
先运行`v4_full_history_canary_01`（4 calibration/4 development/2轨迹/2 lifetime用户，
两目标两pass，预计30–90s，保守约3分钟，GPU0直接监控），检查四个组件的新接口和因果数值。
不把oracle恢复率或微型canary当作方法质量；确认仍未读取。

v4 canary实际12.27s、峰值2588MiB，完整六层身份/paired参考/梯度/真实追加检查及
跨第二发布接口通过；微型泛化不作效果结论。随后仅补齐校准writer计时，并去掉native
历史回放不使用的临时identity译者，不改变其实际K/V。`v4_full_history512_01`保留相同
128 calibration/512 development、五边14天、rank32/3pass，只联改上述full-history表示、
目标和32人lifetime场景。前瞻合同为`evokv_design_v4_full_history_development_v1.yaml`；
按上一轮实测预计650–900s，保守2694s，额外回放用真实事件数的上界估算，显存24–40GiB。
按既有用户授权在GPU0的tmux `evokv_design_v4_full_history512` detached启动，保留日志
与退出码，结束后继续分析。资源/launch hash在analysis下；没有读取确认或改backbone。

`v4_full_history512_01`实际621.92s、峰值26866MiB、exit=0。相同14164条请求与对照，
AUC恢复102.9%/反号未定义/81.3%/119.6%/58.9%；V3有所改善，V5仍明显不足且Brier
继续劣于Reuse，不能宣布达标。Current后代纳入修正只解决了部分问题，完整绝对指标/区间
继续保存在analysis同名quality文件。额外教师回放及source回放是真实成本，未作20%声明。
实际lifetime覆盖另见`v4_lifetime_coverage_probe.json`：各目标有140–159个独立回放场景，
每个目标5–9个场景达到6144次真实追加，另有32个Current-Exact零控制；不足历史未补造。

为降低共享校准的CPU发射开销，新增`--calibration-batch`：native写入不依赖译者，因此
真实校准状态/特征可在三轮之间复用，仍逐轮重新计算修正后的实际query和教师响应。
只按真实cache长度分组批量计算，不跨用户注意力、不补造历史；paired常量差可以等价融合
为每用户每层delta。预定`v4_batch_canary_01`与原v4 canary使用同4/4/2用户、2lifetime、
两目标两pass，batch4，预计15–40s直接运行，逐项比较原配置的cutover与真实请求数值。
七项相关张量检查再次通过。此项只验证同一计算的执行优化，不增加模型表达能力。

`v4_batch_canary_01`实际11.61s；8个user-edge与56条真实请求的全部元数据相同。
learned cutover最大logit差9.54e−7、真实请求5.96e−7，Exact/Reuse相同；拟合.681→.261s。
该差异在实际浮点容差内，批量只优化同一计算，参考检查保留原结果。
`v4_response_v5_probe_01`实际63.38s，在固定512 calibration/128 development上比较
128/512共享响应预测：source-only相对MSE .03782→.02583；候选分布匹配几乎无变化，
额外加入Reuse query仅小幅改善。V5 shared-response oracle用户等权恢复.9384；这是诊断，
不是可执行方法或质量结果。该证据支持先补共享校准样本，不立即增加网络复杂性。

`v4_cal512_01`按相同v4方法扩至512 calibration，lifetime仍固定前32，batch16；五边
仅在前128 development测单边，暂不持有全部512人的三分支cache，以控制显存。
实际校准谱系773152事件，按同一公式保守2871s，实测路径预计350–600s、28–44GiB。
前瞻合同`evokv_design_v4_cal512_v1.yaml`引用已通过batch canary和原方法范围；按既有
整阶段授权在GPU0 tmux `evokv_design_v4_cal512` detached运行，保留日志/退出码。
如果方向改善，再冻结该组权重回到相同512人五段真实轨迹。确认没有读取。

`v4_cal512_01`实际338.65s、峰值35468MiB、exit=0。与原v4配对的128名开发用户，
五边平均绝对概率误差均下降：.001867→.001741、.002068→.001593、.002427→.001676、
.003904→.003236、.002279→.001905。用户等权未裁剪恢复变为.591/.476/.021/.753/.047，
还没有真实AUC结论。全部校准开销保留，不把batch提速当作已达成成本目标。
因此启动`v4_cal512_quality512_01`，冻结这组五目标权重，在同一512 development用户
执行全部五段14天真实轨迹。直接复用已计数的1053816实际事件，保守1174s、预计400–500s，
单GPU0直接监控，日志和退出码保留。原512校准的完整教师/拟合/回放成本通过reference
继续计入，不是零准备成本；确认仍未读取，所有边和原负结果均保留。

`v4_cal512_quality512_01`实际435.08s、峰值17222MiB、exit=0。相同14164条真实请求，
AUC恢复115.3%/反号未定义/56.3%/71.3%/66.7%。V5 Brier降至.076157（Exact .075921、
Reuse .076990），但V3/V4 AUC较cal128变差；不能凭单边绝对概率误差全面下降就宣称
任务质量更好，也不按目标混用两个校准量挑选有利结果。两个完整版本和全部边均保留。

接下来检验已有共享decoder的闭环输出蒸馏：仍固定rank32 source投影，在真实校准用户上
优化最终CC logits并保留0.1响应约束；不混合最终预测分数。v2曾试过同一损失但当时CC
修正污染真实write token，且只覆盖旧producer；当前native写入、full-history与512用户
改变了该检验的前提。先在4 calibration/4 development/2 trajectory/2 lifetime用户、
两目标、score_steps=4、batch_users=2上运行`v4_score_canary_01`，预计15–45s直接监控。
仅在接口/梯度正常后才扩大，不把小canary效果当作泛化结论。

校准准备也去掉ridge从不使用的teacher logits（有score蒸馏时保留），并把发布前短尾
native写入复用已验证band路径；同一输入/目标，新增canary会核对。未来正式成本须区分
离线重建既有Parent服务状态与实际新增summary工作，并避免把同一native服务回放在
校准和人口服务中重复计数；新造的Parent-Exact场景与Current历史回放仍全部计方法成本。
现有保守完整ledger不改写，不能静默删掉已有开销或假设取数免费。

发布计算探针`v4_release_batch_probe_01`实际19.66s，使用前32名开发用户真实V0→V4
native写入形成的混合状态，测V5。批量features与逐用户参考最大差4.77e−7，最终logit
差9.54e−7。原逐用户state.release共21.98ms；直接writer sums投影在batch1为8.56ms、
batch16为1.17ms，相同batch的Exact-All分别143.06/117.69ms。该快路径尚未安装进
人口发布执行器，且不含全局校准和history-weighted人口/I/O；只说明该计算可显著加速。

`v4_prepare_canary_01`实际11.49s；去掉未用teacher logits和批量短尾后，cutover最大
logit差1.43e−6，真实请求7.15e−7，原元数据均相同。`v4_score_canary_01`实际12.58s、
峰值2630MiB，full-history共享decoder的输出蒸馏梯度与有限数值正常；微型拟合依然
严重过拟合，全部负数保留，只视作接口检查。
随后`v4_score512_01`固定512 calibration/32 lifetime/128 development、五目标，
rank32岭回归后按原v2对照的256 steps×4 users、lr=.0003精炼decoder；不增加推理网络。
预计400–650s，保守3179s，28–44GiB。合同`evokv_design_v4_score512_v1.yaml`和资源
记录已在启动前保存，按既有授权在GPU0 tmux `evokv_design_v4_score512`运行并留退出码。
只有在同一开发控制上解释结果后，才决定后续完整轨迹；确认未读。

`v4_score512_01`实际438.17s、峰值35418MiB、exit=0。配对128用户的绝对概率误差
仅有小幅混合变化：V1 .001741→.001698，V2 .001593→.001622，V3 .001676→.001656，
V4 .003236→.003219，V5 .001905→.001950。新增拟合没有稳定方向，暂不为它扩大真实
质量评价，也不认定输出蒸馏已经解决问题。保留权重、loss与所有负结果。

批量发布现已接入`--release-batch 16`：实际writer统计生成并安装融合paired常量差，
普通K/V、producer与真实写入不变，下一次source变化仍走相同刷新。Exact也按同一batch、
实际prefix长度重建。新增状态参考检查通过；预定`v4_release_batch_canary_01`冻结
`v4_cal512_01`权重，4 development/2 trajectory、V1完整14天与V2首日，预计12–30s，
与已保存完整轨迹的相同用户/时间范围逐项比较。

`v4_release_batch_canary_01`实际10.21s，相同56条真实请求及全部元数据一致：方法/Reuse
logits完全相同，Exact最大差2.38e−7。批量发布安装通过实际连续轨迹核对，仍不代表人口成本合格。

`v4_query_time_probe_01`实际26.28s，固定128开发用户、V5同一Parent-Exact前缀与16候选，
只把无标签查询时间放在最后真实事件与cutover之间。并未扰动K/V或使用未来事件。
发布点固定的真实shared-response oracle在cutover恢复.946；移至last+1s、min(60s,gap)、
min(3600s,gap)后仅恢复.520/.540/.687，而按各次实际query重新计算的oracle仍为
.930/.947/.939。因此“候选间共享”不意味着“查询时间变化后仍是同一常量”；v2–v4常量
view即使在发布点得到真实响应，也受此限制。这推翻该局部简化，尚不是四条主线的反证。

方案v5保留实际全历史writer和native写入，将功能view扩为intercept加原模型32维
sin/cos时间基的系数。时间系数按各producer的实际保留事件占比条件化，避免所有
来源、包括健康Current，都被迫共享同一个时间修正。共享rank32译者发布/刷新时从原source
生成系数，CC读取按其实际time delta加权；时间变化本身不使source失效，不逐请求重新拟合。
修正仍在每层原projection/gate/residual前，保持实际下层修正到上层query的闭环。
校准同一历史前缀上的gap/1s/min(60s,gap)/min(3600s,gap)四组查询，均不晚于发布、
没有中间被遗漏的事件；各组保持原16候选，按各自实际query监督完整历史响应。
先运行`v5_temporal_canary_01`（4 calibration/4 development/2 trajectory/2 lifetime、
两目标两pass、calibration batch4、release batch16），预计15–60s直接监控。
公式检查覆盖时间系数分解中的centering与producer条件；新增系数的实际内存单列计入。

`v5_temporal_canary_01`实际11.67s、峰值2648MiB；代数/真实追加与八项张量检查通过。
`v5_temporal_scalar_canary_01`冻结同一权重，改为逐用户发布，实际10.08s；56条真实请求
全部元数据一致，方法/Reuse logits完全相同，Exact最大差2.38e−7。小样本质量有过拟合，
不作效果判断。新增时间系数使末个用户translated view为156672逻辑字节，照实计入。

启动`v5_temporal512_01`：128 calibration、固定前32 lifetime、同一512 development/trajectory，
五目标各14天、三pass/rank32、四组时间查询、calibration/release batch16、native band128。
前瞻合同`evokv_design_v5_temporal_development_v1.yaml`引用上述检查；依据已有真实事件计数
和时间canary预计650–1100s，保守4266.02s、26–40GiB。按整阶段授权在GPU0 tmux
`evokv_design_v5_temporal512`运行，保留日志和退出码；全部五边与此前负结果均保留。
V5仍是E14_partial诊断，没有serving admission或人口成本资格；确认未读。

`v5_temporal512_01`完成，实际567.43s、峰值31108MiB、exit=0；同一14164条真实请求，
AUC恢复111.3%/反号未定义/48.9%/91.1%/80.1%。V5方法AUC .713991（Exact .721309、
Reuse .684559），方法−Exact的配对UID区间[-.014038,.004934]；其Brier .076505，仍高于
Exact .075921。V3弱于v4，不能凭V5改善宣称全局稳定成功或按边混用权重。
`v5_query_time_probe_01`实际26.95s，同一128人Parent-Exact静态时间诊断的用户等权恢复
仍为负（cutover/1s/1min/1h：−.407/−.109/−.610/−.240），与连续任务提升不是同一指标；
全部结果保留，不用AUC改善掩盖该机制弱点。确认未读。

`v5_publication_cost_probe_01`实际23.63s、2848MiB：32名真实混合来源用户的批量发布
含安装为1.540ms，相同batch16 Exact为117.984ms；标量/融合最大logit差4.77e−7。
固定30000人口按真实长度、32事件桶和batch16外推，五次Exact计算457.29s，方法全局
教师/校准/来源/长回放112.25s，加发布7.22s，比例26.13%。该代理尚缺人口I/O、初次
摘要backfill和持续服务，并对相同架构复用V5形状计时，不是20%资格测量。
512人真实服务107.81s vs Exact 91.27s（+18.1%，不含全局准备），因此成本也未达标。

小诊断`v6_linear_response_probe_01`检验ELU在零点的一阶响应矩：实际query读取
delta(sum V)+Q×delta(sum K outer V)。实际18.94s；零干预和旧reference一致。
即使矩来自真实教师，四组查询的用户等权恢复仍为−13.9/−11.2/−21.9/−10.8，尤其第一层
响应相对MSE为3.37–8.75。放弃这个零点线性化，不为它训练译者；这说明不能忽略ELU非线性，
不是实际query加权路线的反证。教师矩干预从未进入持久cache或可执行方法。

成本实现先合并同一请求所有层的常量/时间基计算，并让dirty CC刷新复用已验证的writer
批量特征与融合安装，保留实际release age。方法和权重不变；八项公式/状态检查通过。
`v5_runtime_canary_01`冻结原小canary权重，4 development/2 trajectory两目标，预计10–25s，
逐项核对旧真实请求、生命周期和预测后再测开销。不把运行实现优化当作新的质量变体。

`v5_runtime_quality512_01`冻结v5完整权重，实际426.18s、21155MiB；14164请求元数据与
Exact/Reuse预测相同，五边方法AUC也相同。99%方法logit差≤4.77e−7，两个样本超过首次
3e−6检查界限，最大1.54e−5（概率3.05e−6）；来自等价求和/时间GEMM的浮点次序变化，
未改变任一AUC，原失败检查及实测差异保留在loaded_reference_check。服务97.26s vs
Exact90.48s（+7.49%），仍需进一步降低写入及准备费用。

writer的计数/序号/时间改为CPU整数，生成view时再物化原dtype；K/V sums仍为FP32。
固定slot的淘汰区间连续，改用slice而非GPU高级索引。校准长尾按实际cache/chunk长度
batch16执行同一native rolling kernel，再维护各自真实writer；source与teacher费用仍分列。
`v5_preparation_canary_01`实际11.25s、2709MiB；八项张量检查通过，实际六层两个不等长
canary尾段的批量/逐用户K/V最大差0。重新拟合的小canary与旧56请求元数据相同，方法
最大logit差1.19e−6。小样本的既有过拟合仍保留。
启动`v5_efficient512_01`重测同一128 calibration/32 lifetime/512开发的完整v5。
合同`evokv_design_v5_runtime_development_v1.yaml`记录相同方法与拟合场景、上述canary，
预计450–600s、保守4267s、28–42GiB，在GPU0 tmux `evokv_design_v5_efficient512`
按整阶段授权运行并保留退出码。没有更改质量配置、选择用户或读取确认。

`v5_efficient512_01`实际487.34s、31077MiB、exit=0。所有14164请求元数据和baseline
logits相同；重新批量校准使方法权重出现浮点差异，最大logit差.004127（最弱的既有
过修正样本），p99为6.73e−5、中位4.77e−6；不能宣称重新拟合后预测完全等价。
五边AUC变化均≤1.1e−5，仍约111%/反号/49%/91%/80%。全局校准准备降到63.47s；
教师长尾30.50→9.48s、source长尾34.37→7.92s。服务额外耗时约4%，累计成本仍待优化。

编译执行仅作同一数学的系统探针：`v5_compile_probe_01`因默认g++不支持安装版Torch
要求的C++20而失败，日志/失败记录保留。系统已有g++13，后续仅在该命令设置CXX/CC，
没有更改全局环境或安装依赖。`v5_compile_probe_02`对相同真实V1 CC同时编译native/paired：
原生约2.413→.352ms、paired约2.713→.345ms，最大logit差2.38e−7；初次编译分别37.69s、
6.41s，另有约.375ms的dirty刷新，不能漏计。此为单CC计算探针，不是完整服务提速结论。
`v5_compile_probe_03`把实际source统计到view再到读取的GPU计算一起编译，包含source
输入组装约.609ms，编译8.48s；旧kernel缓存使native/ready的二次编译更快，不能当冷启动。
继续`v5_compile_probe_04`，只让小型metadata传输不阻塞前一GPU工作，预计10–35s；
所有编译探针保持TF32关闭，尚未接入正式runner或读取确认。

`v5_compile_probe_04`让metadata非阻塞传输后，包含实际source组装的dirty read约.392ms，
native编译读取.364ms；最大logit差2.38e−7。该循环是吞吐探针，还需真实逐请求计时。
已将相同计算接入可选`--compile-reads`，仅用于冻结v5的评价；source固定8行padding只为
这六个producer的已知容量，真实空行权重为零。所有可保留的图输出显式clone，避免后续
调用覆盖另一个用户的视图；两类candidate尺寸和实际cache stride先热身，费用分列。
`v5_compiled_canary_01`冻结`v5_efficient512_01`权重、4 development/2 trajectory、两目标，
预计45–180s（执行器保守另加600s启动余量），直接监控。八项现有张量检查已通过；
本次重点检查多用户真实请求、版本切换及已发布视图的独立性，确认仍未读取。

`v5_compiled_canary_01`完成，183.17s；56条实际请求相对冻结参考的Exact/Reuse/learned
最大logit差分别7.15e−7/4.77e−7/4.77e−7，已发布视图未被后续图调用覆盖。编译/热身
121.19s，实际服务中仍出现额外编译，不能把该次计时当作稳定服务成本。
`v5_compiled_canary_02`在编译入口把滚动cache的K/V view转为contiguous，实际115.88s，
相同56条请求和数值差异，热身59.03s；服务读取仍额外耗时约46.49s，因此仅规范cache
stride没有解决问题。该小样本包含41组单候选、6组双候选和1组三候选，热身只覆盖前两种。
下一次`v5_compiled_guard_probe_01`保持同一输入/权重/计算，启用本地Torch重编译原因日志，
预计60–180s，GPU0直接监控；先查实际guard原因再调整执行形状，不扩展质量样本或读确认。

`v5_compiled_guard_probe_01`的实际guard日志定位到candidate张量的batch stride：热身
使用NumPy `[None]`产生stride0，实际请求是stride1/2，`contiguous()`和默认`clone()`
均保留这个大小为1的维度。改为flatten后显式view统一metadata。此前关于修正视图stride
的怀疑没有被该日志证实，未据此更改view数学或持久状态。
固定开发请求共有31种group长度，因此编译入口用1/2/4/8/16候选桶，超过16分块；padding
重复第一项暂态候选并丢弃其输出。候选之间无attention，所有三条分支用同一规则，不变更
事件、标签或history。热身逐桶覆盖并检查3/17候选的padding/分块、dirty后ready及输出所有权。
`v5_compiled_canary_03`复用相同冻结权重和56请求，预计90–300s，GPU0直接监控，保留
重编译日志，以实际请求一致性和服务中是否仍有重编译判断此执行修复。

`v5_compiled_canary_03`通过数值和图输出所有权检查，实际59.29s；显式view修正了stride，
但日志指出slice的`_base`尺寸仍导致单候选请求再编译。入口因此复制极小的candidate整数
张量为拥有独立存储的contiguous tensor；该复制计入三分支读取费用。
`v5_compiled_canary_04`沿用4 development/2 trajectory并覆盖全部五目标，预计60–240s、
约3GiB，GPU0直接监控。逐段记录服务前后Dynamo图数，区分数值通过与编译成本已解决；
现有Inductor缓存仍可能复用，所有启动/热身照实保留。

`v5_compiled_canary_04`完成，100.95s、2312MiB，五段服务的新增Dynamo图数均为0。
83.11s编译/热身费用单列；全部实际请求与冻结参考逐项核对，padding/分块及视图所有权
检查均通过。此处只有两用户，不能据其读取计时宣称完整服务已合格。
下一次`v5_compiled_publication_probe_01`复用32名固定开发用户的真实混合producer状态，
给Exact-All也启用相同Torch编译，保持原batch16/32-event成本分桶；每种测量形状先核对
全部K/V。预计90–600s、约10GiB，GPU0直接监控；启动编译与steady kernel费用分列，
固定30000人口计算外推仍明确缺少人口I/O/backfill及服务，不是正式20%成本结论。

`v5_compiled_publication_probe_01`通过所有34种Exact形状的K/V检查。匹配编译后，同一固定
人口的五次Exact计算外推降至203.67s，方法发布6.71s加原校准准备63.47s，占34.45%。
它仍未计人口I/O/初次backfill和启动编译；所以旧eager Exact分母下可能低于20%的估计
不足以支持成本达标，后续还需降低校准费用。这是匹配基础kernel后出现的实际成本不足。
`v5_compiled_quality512_01`保持`v5_efficient512_01`全部五目标权重，在原512人、14天尾段
和14164请求上执行编译读取对照。canary_04已覆盖五目标与输出持久性；预计350–650s、
约24–36GiB，保守事件公式1773.82s，GPU0直接监控。各目标启动调用按native/paired/dirty
分项计时，保存服务中新图数，所有校准成本仍引用冻结来源；确认未读。

`v5_compiled_quality512_01`完成，411.41s、21171MiB，五段服务新增Dynamo图数均为0。
读取费用降至method8.23s / Exact6.62s，五段编译/热身45.26s另列；真实写入仍有约4.58s
的writer维护增量。Exact的实际轨迹发布仍用eager重建，不能把其9.67s直接作为匹配编译
后的节省。下一步先用`profile_calibration.py`在原128 calibration/32 lifetime、V1和3轮
拟合上分离响应生成、线性求解与SVD时间；`v5_calibration_profile_01`预计30–150s、18GiB，
GPU0直接监控，不变更输入、目标、训练轮数或读确认。

`v5_calibration_profile_01`完成，23.47s、11392MiB、428个V1校准场景。三轮拟合3.19s中
响应生成/设置2.37s，岭回归.82s（其中SVD.75s、线性求解.054s）；因此当前瓶颈不在
求解器。不增加样本或改loss，先把已有`cache_at_many`用于相同的初始/teacher/Parent
历史前缀，并在一次make_scenes内复用同UID同时间的不可变teacher；Current-Exact零控制
继续保留。每个返回cache拥有自己的连续存储，避免deepcopy把整批底层allocation重复复制。
跨发布的校准谱系也按已验证的native batch replay执行，仍在严格相同timestamp边界留快照。
`v5_batch_prefix_canary_01`用原4 calibration/2 lifetime/4 development/2 trajectory、两目标
两轮配置，预计10–30s、4GiB，GPU0直接监控；核对相同56请求及实际维护canary后再扩大。

`v5_batch_prefix_canary_01`在7.70s停止：新增批量校准谱系包含真实无事件用户，旧的
非空tail时间差组装用strict zip报错；尚未产生轨迹质量，不能解释为方法质量变化。
修复空tail的时间差构造（不增加事件），在现有逐用户/批量真实K/V canary中加入空tail
不变性检查。`v5_batch_prefix_canary_02`同范围重新执行，预计10–30s，保留首次失败记录。

`v5_batch_prefix_canary_02`完成，11.41s、2755MiB；空/64/129事件尾段的批量/逐用户
实际K/V最大差0。相同56请求元数据和baseline logits完全相同，重拟合方法最大差1.67e−6；
原四用户过拟合负结果仍保留。继续`v5_calibration_profile_02`，同一128/32/3轮V1校准，
预计20–60s、18GiB，直接计时新批量前缀路径；先看完整校准规模的费用变化再扩大五目标。

`v5_calibration_profile_02`完成，22.41s、13517MiB；同428场景、三轮fit MSE与原配置
仅有浮点差异。V1校准准备（不含IO/model加载）11.29→9.79s：初始source1.64→1.25s，
teacher前缀2.21→1.50s，lifetime source构建.95→.65s；拟合3.19→3.13s。代价是临时
prefix批量常驻使峰值显存增加约2.1GiB。该节省尚不足以满足完整成本目标，不立即为此
重复五目标质量评价。下一步将已通过34种K/V形状核对的编译Exact kernel用于同一校准
前缀，分别计启动和后续计算，再检查真实谱系及拟合响应生成的剩余费用。

`v5_calibration_profile_03`只将同一Exact前缀计算接入可选编译执行；参数、128名校准用户、
32名lifetime用户及三轮拟合保持相同。用实际Parent/Current前缀的batch1/16核对全部K/V
及后续调用不覆盖snapshot，启动/热身和诊断reference分项记录。预计60–240s、18GiB，
GPU0直接监控；完整准备费用仍保留，已有磁盘kernel缓存可能复用，不能当冷启动测量。

`v5_calibration_profile_03`完成，46.30s、13512MiB。实际Parent/Current、batch1/16的
最大K/V差6.68e−6，snapshot所有权检查通过。编译/热身24.60s单列，后续V1准备9.79→8.10s：
初始source.55s、teacher前缀.68s、lifetime source构建.39s。没有把启动开销记成零。
下一次`v5_calibration_profile_04`让全历史响应目标复用本次前向已经读出的source heads，
并编译同一教师响应生成。旧公式作为数值对照，在原冻结译者的非零修正、真实early replay、
lifetime后代和Current-Exact零控制上核对；逐层实际query不变。容差为绝对1e−8、相对5e−4，
总体相对RMS差<1e−4，输出拥有独立存储。预计90–300s、24GiB，GPU0直接监控，启动/热身
与实际拟合分列；仍只读原V1校准场景，不读取确认或新增质量样本。

`v5_calibration_profile_04`通过prefix检查后在第一个响应batch停止：1/4608元素超出
预设逐元素容差，绝对差1.53e−7、相对差8.71e−4；未开始新拟合。保留该失败。
下一次`v5_calibration_profile_05`先要求复用source heads的eager实现与原双读公式逐元素
一致（atol1e−9/rtol1e−5），再记录编译结果原阈值的全部不通过元素数。编译接受尺度
改为与校准目标一致的逐层RMS：全局及每层相对RMS均<1e−4；不隐藏原逐元素失败。
这区分公式改错与FP32执行误差，若仍不通过则保留eager响应路径，不继续放宽。
同一场景、GPU0，预计60–240s；所有对照结果在断言前保存，确认仍未读。

`v5_calibration_profile_05`完成，77.01s、13512MiB。复用source heads的eager响应与旧公式
最大差0；编译每层相对RMS约6.6e−8–1.7e−7，通过按拟合尺度的检查。原逐元素阈值仍有
1/4608和1359/73728处不通过，全部记录；最大绝对差为3.66e−4/5.49e−4（部分目标坐标
量级很大），不能只引用首次异常中的1.53e−7。前缀/响应启动分别26.42/28.83s，三轮拟合
3.17→2.53s，增益有限。因此编译校准暂留诊断选项，不立即接入新完整校准；默认eager
路径仍复用已经读取的source heads，公式不变。

下一项主要假设是校准预算：从128 calibration/32 lifetime同比降到64/16，保持三轮、
rank32、四个时间组、相同loss和各类场景比例，按原冻结UID顺序取前缀，不根据开发标签
挑用户。`v5_budget64_01`先校准五目标并测原128 development用户的cutover机制，未运行
连续质量前不判断成功；随后冻结同组权重在原512人全请求轨迹上比较。预算减少只有与
原v5实测质量并列才有解释，V2反号和各边负结果均保留，确认不读。
另将一次性writer backfill的NumPy时间戳转Python整数集中执行，单slot事件归属用C迭代器
生成，序号和用等差公式；GPU sums与每事件归属不变。先跑已有摘要/维护公式检查及原
微型六层canary，再启动上述64/16校准。校准预计90–240s、约15GiB；按原保守公式约
1665s，GPU0直接监控，不需要扩大模型或数据权限。

`v5_budget_canary_01`完成，11.44s、2755MiB；八项既有摘要/维护测试通过，同56请求的
三分支logits与`v5_batch_prefix_canary_02`完全相同，空/64/129尾段K/V差0。
据此启动`v5_budget64_01`：64 calibration/16 lifetime、128 development、无质量轨迹，
五目标、三轮，使用eager校准以保留实际启动费用的诚实比较；通过后冻结权重测原512轨迹。
运行配置和源码在新目录先保存；没有更改确认划分、教师访问边界或任何方法超参数。

`v5_budget64_01`完成，72.98s；五目标校准准备26.19s，较原128/32配置63.47s明显下降。
128人cutover机制仍有负恢复，不能由费用下降推断任务质量保持。启动
`v5_budget64_quality512_01`，冻结本次全部五目标译者，原512人、完整五段、同14164请求，
使用已通过的编译读取和原生写入。预计350–650s、24GiB，保守公式1773.82s，GPU0直接
监控；全部准备费用引用本次26.19s，不记作零。完成后在同一质量表比较128/32和64/16。

`v5_budget64_quality512_01`完成，408.64s、同14164真实请求。五边方法AUC为
.727870/.636411/.713317/.696383/.717598；V1/V3/V4/V5恢复约133%/41%/80%/90%。
V2 Exact−Reuse反号，方法低于两条基线，不报恢复比例。与128/32相比，V5提高、V3/V4
下降，保留这一质量取舍；不能只凭26.19s校准准备宣布方法达标。
`v5_budget64_publication_probe_01`沿用已通过的34种编译Exact形状检查、固定32名混合状态
用户、batch1/16及30000人口元数据，另外实测初次source/writer backfill并明确外推。
预计90–600s、约4GiB，GPU0直接监控；启动和warmup分列，仍不代表人口I/O或服务成本合格。

下一项结构诊断`v6_clearance_probe_01`只取已开放前128名calibration UID中，按原顺序
前四名n_theta0≥7168的用户。V0 Parent与V1 Current分别在相同真实历史前缀建cache，
随后都用V1 native路径追加相同6144个发布前真实事件；每1024次写入比较全部六层K/V。
预期第j层（从0计）在(j+1)×1024次写入后相同：本层K/V先于attention投影，下层依赖
经过真实窗口逐步退出；这不是把旧producer清空误认为所有后代已Exact。只检验此结构
上界，不人为扰动K/V、不读取候选标签、不改变方法；四条真实轨迹通过后才试读取策略。
预计15–90s、4GiB，待当前GPU0成本探针结束后直接运行，保留完整逐层差异和失败情况。

`v5_budget64_publication_probe_01`完成，55.41s、2704MiB，34种编译Exact形状检查通过。
固定人口五次Exact计算外推203.02s；方法发布12.03s加校准准备26.19s占18.82%，再加
初次source backfill的4.59s全窗口外推为21.08%。此前不含backfill的比例不能当完整费用。
发布微批计时较前次有波动（6.71→12.03s），不选取较低一次计时判成功；人口I/O、存储、
实际全人口执行仍未计量。启动编译30.10s、测量warmup .96s另列，未宣称20%达标。
低预算质量的完整绝对指标、所有旧状态分层和1000次UID配对bootstrap已保存在
`results/design/analysis/v5_budget64_quality512_01_quality.json`；V5方法−Exact的AUC区间
[-.0180,.00834]，仅为本seed下用户抽样不确定性，不构成模型seed重复或正式确认。

`v6_clearance_probe_01`完成，7.66s、2148MiB；四个用户在1024/2048/3072/4096/5120/6144
次真实写入后，分别有前1/2/3/4/5/6层K/V逐元素完全一致。1024时其余层仍有最高6.33
的绝对误差，证明旧producer退出不足以立即关闭所有修正；4096时未清层仍有最高8.94e−6。
据此接入可选`--layer-clearance`：Ridge常量项和时间系数使用同一逐层结构mask，达到
6144次写入后直接native读、停止当前view刷新，但继续维护真实source；下一实际发布重置
年龄并重新翻译。该规则只依赖真实写入数、窗口和层数，无标签、teacher或拟合调度。
先用冻结`v5_budget64_01`权重做`v6_clearance_canary_01`：4 development/2 trajectory、
五目标、14天尾段、native128和release16，预计15–45s、3GiB，GPU0直接监控；不重拟合。
新增真实写入数/已消退层数记录用于解释全部请求。通过后才测同一512人连续质量。

`v6_clearance_canary_01`完成，20.00s、2312MiB，199条真实请求与冻结参考元数据完全
相同；两基线最大logit差≤7.16e−7。187条尚无整层消退的请求方法最大差7.16e−7，
另12条请求已关闭第一层修正；该两用户canary没有6144次写入后的请求，未声称覆盖。
九项现有/聚焦公式检查通过，新增检查覆盖常量/时间mask一致性、完全消退后跳过view
而保留writer，以及下一发布重新计龄。四用户真实K/V诊断另支持6144次的结构上界。
启动`v6_clearance_quality512_01`：冻结同一64/16译者、原512用户五段全部请求，只有
读取策略改变；全链native普通K/V不依赖读取修正。预计350–700s、24GiB，保守公式
1773.82s，GPU0直接监控；逐桶编译检查和启动费用仍保留，确认未读。

`v6_clearance_quality512_01`完成，520.41s、18666MiB，14164条请求与v5低预算元数据、
两基线logits完全相同；没有整层消退的请求方法logits也完全相同。V1/V3/V4/V5 AUC
恢复约137%/42%/80%/90%，V2方法.636583仍低于两基线。编译/热身118.84s单列，五段
服务均无新增Dynamo图；读取method9.10s/Exact7.18s。结构mask只小幅改善质量。
其中12678条请求尚未完成一个写入窗口，1423条清第一层、63条清前两层；没有请求达到
三层或6144次全部消退。因此完全消退后的No-op只由四条真实诊断支持，当前质量轨迹
没有这类覆盖，刷新数仍12474，不能声称获得刷新节省。主要质量误差仍集中在早期窗口。
保留无标签结构mask，但不围绕这个小收益继续扩展。下一假设是v5时间系数缺少用户条件：
当前仅由producer占比产生时间系数，实际query的缓存依赖仍可能需要source×time交互。
拟在同一64/16、rank32、三轮校准中，用校准source的低维无标签投影给时间项补充条件，
先做小样本数值与成本检查，再看五边完整质量；不增加网络网格或读取确认。

v7将校准source标准化后取8个PCA方向并白化，仅使用拟合用户的输入统计，不用响应或
标签选方向。把这8维与原producer占比一起乘原32维时间基，仍由同一rank32岭回归
预测常量和时间系数；per-user view大小不变，新增投影、系数生成及校准计算全部计费。
保留v6逐层结构mask，拟合仍在实际修正query上做三轮；不是逐用户网络或新的读时调度。
`v7_source_time_canary_01`先取16 calibration/4 lifetime、4 development/2 trajectory，
五目标、14天尾段、batch16/native128，预计20–60s、约5GiB，GPU0直接监控。
通过分解公式/白化核对与真实六层canary后，再按同一预算64/16测五边；确认不读。

`v7_source_time_canary_01`完成，29.12s、4460MiB；九项公式/维护检查通过，真实六层
identity、Exact差分和query反馈检查通过，空/64/129真实native尾段K/V差0。两用户
反馈不足以判断质量，多条边AUC未定义；不据此筛选结果。时间项source投影已经进入
实际拟合和五次连续调用，全部校准准备约7.40s（16/4），没有隐藏PCA或教师费用。
下一步顺序运行匹配的`v6_refit64_01`与`v7_source_time64_01`：都用64 calibration/
16 lifetime、128 development、无质量轨迹、五目标、三轮、rank32及逐层mask；唯一
方法差别是是否增加8维source×time条件。v6也重新校准，以免把修正query改变后的
重拟合收益误归因于PCA。每项预计90–240s、约15GiB；v7在既有公式上增加30s拟合
余量，仍预计小于30分钟，GPU0顺序直接监控。随后先看同一128用户的真实五段质量。

`v6_refit64_01`完成，75.33s、9174MiB，准备26.35s；`v7_source_time64_01`完成，74.41s、
9277MiB，准备26.41s。PCA加入后的五目标normalized fit MSE由
.01110/.01774/.01359/.01564/.02156降至.00552/.01028/.00860/.00974/.01637；
但128人cutover机制仅V4明确改善，V1/V2/V3/V5未改善，拟合误差下降不是泛化成功。
因此不立即扩大至512；先顺序执行`v6_refit64_quality128_01`与
`v7_source_time64_quality128_01`，冻结各自权重，使用相同前128 development UID、
五段完整真实请求、native128/release16。小样本质量比较采用同一eager reader，避免为
非成本探针支付两套启动编译；每项预计90–240s、12GiB，GPU0直接监控。各边全部保留，
不以切换点机制或训练MSE代替真实AUC结论，确认未读。

用户澄清了模型、缓存与方法编号：后续统一使用模型M1–M5、缓存C1–C5，并另列初始
M0/C0；方法使用“方案6/方案7”等文字。此前“v6、V5约90%”准确含义是方案6在
M4→M5诊断段的AUC差距恢复约90%，不是五段平均或所有模型均达到90%。C_i仍可能
含多个模型producer；不因名称对齐把真实混合缓存改成单一模型缓存。

`v6_refit64_quality128_01`与`v7_source_time64_quality128_01`分别完成，116.24/119.12s、
7313/7320MiB，相同2930条真实请求。方案6→方案7的M1/M3/M4恢复为
71.7→81.6% / 38.1→47.2% / 76.8→95.0%；M2 AUC同为.622349，仍低于Exact .630104
和Reuse .633982；M5从−22.1%改善到−9.7%，仍低于Reuse。这是固定前128人的结果，
不能替换此前512人的M5约90%或理解成同一人口的结果反转；全部原始请求保留。
当前source×time方向有一致改善信号，接下来顺序运行
`v6_refit64_quality512_01`和`v7_source_time64_quality512_01`，各自冻结64/16译者、
相同512人五段全部请求、编译CC/native128/release16。预计各350–700s、24GiB，
保守事件公式1773.82s，GPU0直接监控；逐桶编译数值/输出所有权检查和启动费用单列。
以这组完整开发比较判断方向，不用128人结果选取模型边或用户；确认仍未读。

用户要求取消统一超参数的僵化约束，采用对每次更新自适应的共同规则，并将测试扩大至
固定30,000人口的15%–20%。已冻结新split：development6000、confirmation6000、
calibration15512、legacy2488；历史所有拟合配置的UID并集为512，全部仍在calibration。
按n_theta0分层，两组评价各940/969/2026/2065人，覆盖原方案欠缺的短历史；选择只读
人口元数据，原512开发与2048确认成员保留，原split及输出证据不变。确认未读。

方案8先复用同64/16拟合的方案6、7，独立128名校准用户交错分成64强度拟合/64设计选择。
在真实发布前历史、原生查询时间基和连续源状态上，用各层响应相关性确定[0,1]强度，
再比较实际非线性修正forward到Current-Exact的logit MSE，在零修正/稳定时间项/
source×time中选一个。最初尝试把强度折入原decode/offset/time_weights；不混合预测分数，
不访问真实评价标签，不按M编号分支。两套候选拟合和选择teacher/回放费用均计入方法。
`v8_adaptive_canary_01`先用8校准用户、两目标验证折叠公式与真实流程，预计30–90s、
6GiB；通过后128用户五目标选择预计90–300s、18GiB，直接监控。选择完成才开放6000开发结果。

`run.py`新增冻结split、UID偏移及quality-only入口，复用原五段连续回放，避免每个分块
重复静态切换点诊断；各块保留原请求与ledger，最后重算全体指标。先检查原用户结果等价
及新增短历史实际读写，再按实测耗时启动6000人评价；长于30分钟使用tmux。

方案8第一次canary因FP32仿射项抵消的运算顺序变化未通过严格分解检查，最大rate差
4.99e−4，失败完整保留。改为每层输出后乘一个发布级标量，六个值进入target配置，
常量与时间项一致缩放，不改原权重。`v8_adaptive_canary_02`通过，9.84s/3543MiB；
原九项检查通过，现有分解检查补充非均匀强度覆盖。

`v8_adaptive128_01`完成，60.86s/13053MiB；M1/M4选择稳定设计，M2/M3选择source×time，
M5选择零修正。这是同一条独立校准规则的输出，尚不是质量结论。M5选择折上的teacher
logit MSE为Reuse .02732、稳定 .12435、source×time .11893，不能因旧512人的恢复率
较高而手动改选。当前零修正仍保留writer和view维护开销，不声称实现无成本No-op。
两候选拟合、独立校准teacher/历史/选择的必要准备合计89.05s（不含模型/数据IO），
比原单候选准备增加；尚不符合20%发布预算，后续需要共享候选计算或更简单的选择。

`quality_only_reference_canary_01`为17.69s/2544MiB，四人244条请求与原eager对照
元数据和三分支logits逐元素一致。新增短历史canary为166.73s/2320MiB，477请求/
5678写入，编译服务新增图数1/0/0/2/2，启动及服务重编译全部留账；短历史不能沿用
原长窗口无新增图的成本说法。扩大评价先统一使用eager reader，减少反复编译启动。
`cohort_merge_canary_01`用GPU2/3各两人、完整五段，19.87s，合并244条原始请求并
重算指标；基线与旧入口完全一致，M5零修正与Reuse逐元素一致，分块无UID交叉。

启动`v6_quality6000_01`与`v8_quality6000_01`：同一冻结6000 development、12块×500人、
五目标及14天尾段、native128/release16、eager读。分别GPU0/1和GPU2/3，每GPU顺序
六块，单用户全生命周期不跨任务重置；两组原始请求合并后重新计算AUC及全部指标。
依据436.15s/512完整旧对照、去除重复静态诊断、保留分块IO，预计每组双GPU墙钟
30–60分钟、每GPU约22GiB；保守3600s，因此均tmux detached，保留主/子日志和exit。
现有总授权覆盖此六层评价，确认6000人仍未读。新增选择已冻结，评价期间不改选择。

扩大样本的统计汇总改用固定概率并列组与UID重数计算同一ROC-AUC bootstrap，避免
1000次重复排序与无关PR-AUC计算。聚焦参考检查覆盖并列/单类别；既有方案7的128人
五目标1000次区间逐项完全一致，0.37s完成。当前机制/成本总表已更新57个完整入口，
未删除负结果；质量评价与自适应选择另有各自记录。分层双组比较已通过四人真实结果
canary，基线逐元素一致；6000主任务仍按冻结规则继续，不用中间分块结果改选择。

`v6_quality6000_01`和`v8_quality6000_01`均完成，双GPU墙钟1644.68/1651.95s，峰值
18280/18290MiB；各6000用户完整五段，4305人有反馈，共136610条请求。全部UID与
时间/标签/谱系元数据一致，两基线logits逐元素相同，24个分块exit均0，确认未读。
完整表见`results/design/analysis/v8_quality6000_01_vs_v6_quality6000_01.md`及同名JSON。

M1/M3/M4方案6→8的AUC差距恢复为87.34→87.28%、25.62→32.58%、55.02→57.19%；
M2 Exact .669870、Reuse .672662、固定 .665736、自适应 .666078，反号仍保留。
M5 Exact .678033、Reuse .658201、固定 .655195、自适应 .658201，零修正避免固定方法
的整体负收益，但没有恢复正差距；旧512人约90%不能外推到6000分层人口。
自适应−固定AUC的配对UID 95%区间仅M4在这轮排除0，[.000072,.000894]；
M1/M2/M3/M5区间均含0，不宣称这些小差异已经稳定成立。M5仍只是E14_partial诊断。

分层揭示问题：M5固定方案在初始1024–4095历史组恢复约61%，在256–1023及4096+
组均负收益；M4的两个短历史组Exact−Reuse反号。全局M5零修正也放弃了可恢复组，
发布级均值不足以表达用户状态差异。既有64/128校准UID全部初始历史≥1024，这是
与扩大人口的实际覆盖差异；下一步先做同预算的分层校准对照，再检验由摘要状态决定
可信度/修正强度的共同规则，不按M编号或已见评价分层人工指定动作。
当前五次必要准备26.35/89.05s，6000人初次summary backfill2.52/2.69s、发布2.10/2.20s；
Exact发布105.52/105.73s是eager且分块并行的实测，不与旧compiled分母混成达标比率。
两方案仍未达到阶段质量/成本目标；保存负结果并继续迭代，不读取6000独立确认。

2026-09-07，下一最小实验先只改变校准人群覆盖：从新split的calibration按相同初始
历史分层比例抽样，hash排序且轮流交错各层，使lifetime子集也覆盖长短历史；开发及确认
UID不进入拟合。增加短前缀的真实早期快照处理：不足五事件或所选时间点之前无历史时，
保留原切换点场景，不伪造空缓存或拆开同时事件。原长历史用法保持原样。
`v9_stratified_calibration_canary_01`先16 fitting/4 lifetime、8原development机制用户、
无质量轨迹，五目标、source×time8、rank32、三轮、native128/batch16，预计30–120s、
8GiB，GPU0直接监控。这是数值/资源canary，不能替代已经扩大的6000人质量评价。
通过后再同预算64人拟合对照，并检查独立校准中的状态误差；不因M5已见子组结果手写策略。

`v9_stratified_calibration_canary_01`通过，23.40s/4008MiB，真实六层identity/Exact差分/
query反馈与native追加检查通过；16人校准准备约5.83s，短历史已进入五段连续源状态。
仅八人机制读数不作为质量证据。下一步`v9_stratified_stable64_01`与
`v9_stratified_source64_01`使用同一分层64 fitting/16 lifetime、8旧开发机制用户、
无质量轨迹、五目标/三轮/rank32，区别仅source×time秩0或8；GPU0/1分别直接监控，
预计各60–150s、12GiB。拟合后再用独立分层校准检验状态误差，正式开发评价仍6000人。

分层64人拟合均完成：稳定/source×time为44.04/44.18s、7624/7723MiB，必要准备
22.20/22.58s。拟合误差不能代替质量，后续仍用固定6000人比较。另核对原请求manifest：
matrix_horizon实际从217日开始，因此五次发布均有前14天的真实反馈可用，不必使用
发布后的E14 AUC作为选择特征。`profile_release_quality.py`在排除上述64 fitting的独立
分层calibration上，测Parent/Current Full AUC、概率偏向和logit变化，所有标签早于发布，
每条前缀严格早于本请求；它只提供模型发布特征，不冒充persistent-Reuse质量或backbone
holdout。八人两目标canary9.09s/3378MiB、32请求，首batch原生标量对照及时间检查通过；
样本太小的AUC不作结论。下一步128人五目标预计30–120s、5GiB，GPU0直接监控。

`pre_release_auc128_01`完成，43.85s/3668MiB，128独立分层校准UID与当前64 fitting、
6000 development、6000 confirmation均无交叉。五个发布前窗口的真实请求数为
770/455/438/387/533，有反馈用户59/56/52/52/56；Current−Parent Full AUC为
.06136/.00245/−.04126/−.05525/.02329。全部标签严格早于对应发布，模型原生标量对照
通过。AUC可作为因果输入，但本小校准中用户/负例分布不均，不能直接按这些符号
重判已封存模型准入或声称它能预测下一窗口。Full profile的前缀/读取额外21s需计费。

先隔离校准覆盖因素，启动`v9_stratified_stable_quality6000_01`，冻结分层64人稳定设计，
与已保留`v6_quality6000_01`比较：只有拟合UID改变，其余rank32/三轮/时间项/逐层规则
一致。固定6000人仍完整五段，12块×500，GPU0/1/2/3各顺序三块，eager读/native128/
release16。上一同人口双GPU墙钟27.4分钟，各500人实际160–430s；此四GPU预计
15–25分钟、每GPU约19GiB，估计1500s，直接监控；每块原保守事件公式均小于30分钟。
这项是覆盖对照，不冒充新的状态自适应规则；在其运行时继续研究独立校准上的选择依据。

发布前AUC的UID bootstrap已保留：M3的Current−Parent区间为[−.08979,−.00905]，
M5为[.00243,.04432]，其余三个窗口含0；单用户请求占比17%–33%。这些是不同的
发布前分布，不能从一个AUC符号推出下一窗口的缓存兼容性或覆盖已封存准入。
后续将AUC与响应/偏向/状态信息一起检验，尚未把它设为硬门槛。当前6000人覆盖对照
与自动统计汇总继续运行，规则和原始请求不变。

`v9_stratified_stable_quality6000_01`完成，四GPU墙钟974.2s，6000用户/4305反馈用户/
136610请求；与原方案6的元数据及两基线logits完全相同。M1/M3/M4/M5恢复为
99.88%/44.45%/92.65%/20.82%；M2方法AUC .666307仍低于Exact .669870与Reuse .672662。
必要准备22.20s，对照26.35s。全部五目标AUC点估计高于原长历史校准，但配对UID 95%
区间均包含0，M4/M5的区间尤其宽，不能把点估计提升等同于稳定成立。
结果见`results/design/analysis/v9_stratified_stable_quality6000_01_vs_v6_quality6000_01.md`。

下一小检查`probe_release_feedback.py`复用已开放的发布前128人实际反馈，分别比较
Current使用Parent-Exact源缓存、稳定译者、source×time译者，另保留Current Full。
Parent-Exact是合法adjacent校准场景，不冒充持续混合缓存质量；所有前缀早于对应请求，
标签早于发布，UID与当前64人拟合不交叉。不用最终6000确认做选择。先8人两目标
`release_feedback_canary_01`，预计10–40s/5GiB，GPU0直接监控，检查已有Parent Full分数
匹配和时间关系；通过后128人五目标预计30–120s。查看真实质量与teacher MSE是否给出
同一选择，再决定发布/状态自适应规则，不从已见开发表手写每个M的设计。

`release_feedback_canary_01`通过，8.95s/2719MiB，11条已有独立发布前请求与Parent Full
分数、严格时间关系均吻合。`release_feedback128_01`完成，36.08s/3684MiB；五目标
同一2583条实际历史反馈上，稳定/source×time的AUC分别为
.55777/.55642、.47097/.47466、.51365/.51570、.41856/.41492、.52440/.52370。
M4的Current-on-Parent缓存Reuse AUC .46055高于两译者及Current Full .41204，
但其log-loss更差；这与发布后连续6000人M4约93%的恢复不同。不能把发布前、
Parent-Exact场景的AUC符号直接当成连续部署选择。这项揭示选择依据的分歧，未安装新门槛。

增加一个更直接的表示诊断：在原生连续校准状态与真实修正query上，把逐候选响应误差
分解为同时间候选差异和条件均值拟合误差。先检查当前时间修正能否表达主要误差，
再决定增加query条件还是改善摘要/可信度。`query_dispersion_canary_01`用8个拟合用户、
两目标验证正交MSE分解通过；随后`query_dispersion64_01`测原分层64用户、五目标、
同16 lifetime与全部发布前场景，预计30–120s、12GiB，GPU1直接监控。
这是校准表示诊断，不使用评价标签或确认，也不把响应MSE当质量结果。

`query_dispersion64_01`完成，37.62s/7617MiB。各层未表达的同时间候选方差占当前响应误差：
M1约2%–8%、M2约3%–13%、M3约3%–15%、M4约8%–27%、M5约12%–48%；
M5第三层47.5%，其余层条件均值误差仍占多数。仅为64拟合用户的表示诊断，不能称为
独立泛化结论。先完成已拟合source×time的6000人质量对照，再判断增添query条件的收益。

发布成本复测`v9_stratified_publication_01`使用冻结`v9_stratified_stable64_01`、同32名真实
混合状态用户，发布安装重复100次、初始摘要重复20次；编译Exact逐形状与eager数值对照，
单独记录首次编译和实际Python元数据分配。预计90–600s、约5GiB，GPU0独占测量并直接
监控。人口历史长度只用于计算外推，仍不是30000人IO/存储/服务完成证据。

随后`v9_stratified_source_quality6000_01`冻结`v9_stratified_source64_01`：与稳定设计同64/16
分层校准，唯一额外项是8维源状态参与时间修正。复用已通过的分层拟合与分块整链canary，
相同6000人、12块×500、完整五段、eager读、native128/release16，GPU0–3各顺序三块。
依据稳定设计974s，预计15–25分钟/每GPU约19GiB，估计1500s、直接监控；全部模型边保留，
不读确认、不依据中间分块改变方案。

`v9_stratified_publication_01`完成，56.87s/2704MiB，发布打包与原路径最大logit差7.15e−7，
编译Exact全部形状检查通过。五次30,000人计算外推：准备22.20s、发布6.90s、初始摘要
full-window代理4.81s，对应Exact204.30s，合计16.60%；这仅是resident-GPU计算估计，
人口IO/在线维护/等待覆盖尚未计量，不能写成20%预算已通过。32个实际Python初始状态
的保留分配约4.40MB，也只是局部测量，尚未包含GPU payload与人口RSS。

下一小变体`v10_source_confidence_canary_01`：在分层校准方案9上加入源状态可信度缩减。
每次拟合用当前模型校准源输入的标准化平方范数95分位，推理强度为
`min(1,sqrt(q95/norm²))`；同一强度乘常量与时间项，只随源状态变化，不看查询时间或
评价标签。所有模型使用同一规则，分位数先定为0.95，不做开发结果上的阈值网格。
首轮16 fitting/4 lifetime/8机制用户、五目标三轮、无质量评价，预计20–60s、5GiB，
GPU1直接监控。现有时间分解检查增加超出支持范围场景，验证两项一致缩放及时间独立。
通过后同64/16全拟合预计40–100s、9GiB；与6000人结果分开的独立校准先检验收益，
尚不把此启发式称为经过验证的不确定性估计。

`v10_source_confidence_canary_01`完成，23.56s/3978MiB，identity/Exact响应与梯度检查通过；
时间分解的超出支持范围检查通过。随后`v10_source_confidence64_01`完成同64/16拟合，
采用当前修正后的query重做三轮响应拟合。接下来`probe_source_confidence.py`复用发布前
独立128名分层校准UID；在真实连续、adjacent及lifetime场景比较Reuse、原稳定译者、
新译者及同新权重临时关闭缩减的对照，区分重新拟合和缩减本身的影响。仅计算teacher
logit距离，不选模型编号/阈值，不把它当作6000人质量评价。8人两目标canary预计10–40s，
通过后128人五目标预计30–120s、18GiB，GPU1直接监控。

`v10_source_confidence64_01`完成，45.87s/7624MiB，必要准备23.01s。
`source_confidence_probe128_01`完成，56.05s/10927MiB。源范数缩减仅影响各目标约
2.9%/5.0%/5.2%/3.3%/5.7%场景；UID等权teacher logit MSE稳定→缩减为
.003110→.003105、.054945→.054885、.041615→.041458、.070593→.070851、
.127446→.129281。M4/M5变差，因此不扩大方案10，不按负结果另调分位数。
同新权重关闭缩减的对照证明M5退化主要由缩减本身引入。大误差用户的源范数并不异常，
单纯输入幅度不能当作可靠的状态兼容性信号。原始场景表的writes_since_release是
进入新目标前scene.state的计数；实际Translator输入在pack_source(target)时按目标重置，
该记录列不能直接当作所有切换点的目标年龄。计算使用的是正确的pack_source输入。

当前`v9_stratified_source_quality6000_01`已按上文四GPU配置启动，评价期间冻结核心代码。
独立探索下一种发布自适应：利用相同拟合数据的按UID留出响应误差选择ridge正则，
不让同一用户的多个场景/时间复制进入两侧；先用微型显式删组拟合校验解析公式，再
评估额外计算成本。它只检验共同的发布规则，不能从开发AUC逐个指定M的参数。

`fit_adaptive_ridge.py`实现方案11的小校准入口，复用run.py全流程，仅在进程内替换拟合。
同一固定候选倍率.001/.01/.1/1/10，在每次发布、每轮实际query上，解析计算完整UID
删除后的响应误差，选最小者，再保留原rank32压缩。free intercept进入hat矩阵，
不同用户场景数用UID等权；所有时间副本归同UID。选择代理是未压缩ridge、固定本轮
源标准化和响应尺度下的交叉验证，不冒充最终低秩模型的独立质量。
聚焦公式对照直接删除用户、重拟合带截距ridge；初次参考代码的FP32单位阵使惩罚项
提前舍入，修为FP64单位阵后保持严格比较，不放松容差。随后16 fitting/4 lifetime、
五目标三轮`v11_adaptive_ridge_canary_01`预计30–120s、16GiB上界；GPU2与同机开发
评价短暂共用，合计低于46GiB，不将此canary并发计时当作独占成本基准。通过后64人
拟合等待空闲GPU，预计60–240s。阈值与候选倍率不会依据6000人中间结果更改。

`v9_stratified_source_quality6000_01`完成，四GPU902.04s/18282MiB，6000用户完整五段、
4305反馈用户、136610请求，与稳定对照元数据及两基线逐元素一致。M1/M3/M4/M5
AUC恢复约95.35%/45.75%/103.16%/20.89%；M2 AUC .666881仍低于Reuse .672662。
相对分层稳定设计，M4 AUC +.002511，配对UID区间[.000941,.004513]；其余四边
差值区间含0。M4超过100%只表示本次AUC点估计略高于Exact，非跨指标/模型/seed结论。
准备22.58s，原始全表与两组配对比较均保留。source×time保留为有局部收益的设计选择，
不宣布它解决M3/M5。核心代码在全部12个子任务中保持冻结。

方案11公式检查通过，`v11_adaptive_ridge_canary_01`25.75s/3983MiB，六层identity、
Exact响应和梯度检查通过。GPU2完成其6000人分块后，`v11_adaptive_ridge64_01`
独占该GPU完成，48.46s/7624MiB，必要准备24.70s。三轮倍率选择：M1 .001/.01/.01，
M2 .01/1/1，M3 .01/1/10，M4 .1/1/1，M5 1/1/1；由同一校准规则自动给出，
并非按目标编号设置。`adaptive_ridge_probe128_01`复用已通过的独立校准probe流程，
比较同一128名UID的真实连续/adjacent/lifetime场景，预计30–120s、18GiB，GPU2直接
监控；只改变冻结候选引用，不重复等价canary。随后才决定是否扩大6000人质量评价。

`adaptive_ridge_probe128_01`完成，52.66s/10927MiB。稳定→自动正则的UID等权teacher
logit MSE：M1 .003110→.003091、M2 .054945→.044966、M3 .041615→.085425、
M4 .070593→.050050、M5 .127446→.111047；M3约翻倍，且M2/M3/M5仍劣于No-op。
因此不扩大方案11，也不利用这个验证结果手写各M的正则值。它证明发布自适应规则
能够产生不同参数，但响应交叉验证代理不能保证闭环改善。全部失败权重/记录保留。

下一机制检查在`diagnose_query_dispersion.py`已有实际query分解上增加逐层历史head
修正/原生响应的范数比、teacher所需范数比、修正后的范数及方向余弦，另存每个UID场景，
不改变方法或对照。`--independent`复用独立128名校准用户；先8人两目标
`read_geometry_canary_01`，预计10–40s、5GiB，GPU0直接监控。通过后
`read_geometry128_01`五目标预计40–150s、14GiB，GPU0直接监控。这些范数只针对
history heads，未包含暂态self贡献；在判断读取放大或增加gate前先看证据。

读取几何canary通过，随后同时运行`read_geometry128_01`（独立128人）与
`read_geometry_fit64_01`（原64拟合用户），GPU0/1，各预计40–150s、14GiB以内。
后者提供拟合侧实际teacher修正分布，以免把独立校准上的异常阈值直接装进方法。
几何统计直接使用reader trace中的实际corrections，避免重新组合浮点运算；
整个方法权重、普通K/V和实际query不变。

读取几何两组完成。部分独立校准查询的history head响应为零或极小，当前常量修正仍非零，
导致相对范数出现极大值；这些比值的分母使用1e−20数值下界，因此不能把1e20级比值
当成可靠连续尺度或直接据此选阈值。接下来`probe_inactive_response.py`只做无阈值的
对照：同一冻结稳定译者、同一源状态，某head的完整原生历史响应向量恰为零时，
将该head修正置零；保留原native与暂态self项，并让后续层使用真实变化后的query。
它可能错失teacher因新K而打开原先静默head的合法变化，不能先验称为安全策略。
`inactive_response_canary_01`8人两目标预计10–40s/5GiB，GPU0直接监控；先验证关闭gate
时复制的六层reader与原reader一致，再测固定规则。通过后独立128人五目标预计40–150s/
14GiB。这是临时reader诊断，尚未安装到主执行器，不作人口质量/成本结论。

`inactive_response_canary_01`通过，9.65s/2849MiB；关闭gate时两目标的复制reader与原版
logits逐元素一致。8人小对照M1 teacher距离略降、M2基本不变，不据此作质量判断。
启动`inactive_response_probe128_01`，同128人五目标和原稳定权重、GPU0，预计40–150s；
记录实际零响应head查询数。当前无源状态或权重变更，独立确认仍未读。

`inactive_response_probe128_01`完成，52.84s/10927MiB，关闭gate的五目标reader对照均
完全一致。UID等权teacher距离稳定→gate：M1 .003110→.002984、M2 .054945→.053305、
M3 .041615→.047467、M4 .070593→.085275、M5 .127446→.144799。后3段变差，
不安装此规则；原生静默不能代表目标模型也应静默，追加屏蔽会删除合法的新响应。

转向摘要信息本身，方案12新增可增减的K²/V²统计。writer在实际cache写入/扣除时维护
FP32平方和，Translator按真实producer聚合均值和中心对角方差；输入由2类矩扩为
4类坐标，其余rank32、三轮实际query、时间项和native写入保持当前稳定设计。
不读取目标用户teacher作为方法输入，不改普通K/V；方差由平方和与均值推导，负舍入
残差截到0。该变体检验均值相近、内部离散程度不同的源缓存是否可区分，尚不预设收益。
逐事件/批量淘汰与原保留事件参考、producer方差与批量writer特征检查通过；10项适配
局部检查全部通过。`v12_variance_canary_01`用16 fitting/4 lifetime/8机制用户，五目标
三轮、无质量评价，预计25–90s、6GiB，GPU0直接监控。通过后同64/16拟合预计50–180s、
12GiB；新增摘要存储、增删和扩大输入的实际成本全部计入。

`v12_variance_canary_01`完成，24.04s/4006MiB，六层数值/梯度检查通过；主体仍是三轮
实际query响应拟合，8个机制用户不作质量结论。启动`v12_variance64_01`，与方案9稳定
设计完全相同的分层64/16和五目标，只增加摘要方差输入；预计50–180s/12GiB，GPU0。
独立校准probe已按各译者自己的输入维度提取特征，两个候选共享实际源cache；平方和
维护不能改变原稳定译者的均值输入，后续同时检查旧对照数值是否保持一致。

`v12_variance64_01`完成，47.16s/7721MiB，必要准备24.59s。两个初始方差运行的
静态source-view字段漏计一份9216-byte平方矩，原summary/源码保留，补正记录在
`analysis/v12_variance_storage_correction.json`；writer字节与实测GPU峰值已包含它，
后续执行器/汇总已修正，translated与source共享此份矩，不重复收费。
`variance_probe128_01`完成，54.70s/11144MiB，UID/场景及Reuse/原稳定误差与旧probe
逐元素一致。稳定→方差的UID等权teacher距离为.003110→.006348、.054945→.048585、
.041615→.034704、.070593→.091066、.127446→.198788；各目标改善用户比例
56%/57%/56%/63%/50%。这是分化的校准信息，并非质量成功。M2/M3有信号，而此前
source×time的teacher距离与6000人AUC并不总一致，因此用实际大样本回答新增摘要信息
是否有用，不根据少数校准用户误差替代质量结论。

先`v12_variance_quality_canary_01`，冻结同64译者，原开发前4名用户完整五段与真实
读写，预计15–45s/5GiB，GPU0；与已保留四人对照核对全部请求和两基线。通过后
`v12_variance_quality6000_01`固定同一6000人、12×500、GPU0–3、native128/release16、
eager读，预计15–25分钟/每GPU22GiB内（source×time实测902s），估计1500s并直接监控。
规则不依据分块结果更改，6000人确认集仍未读。

`v12_variance_quality_canary_01`通过，18.89s/2544MiB，四人244条实际请求；UID、请求、
时间、标签、旧状态数、目标年龄及Reuse/Exact logits与原四人对照逐元素一致。
`v12_variance_quality6000_01`已按上述配置启动，核心文件冻结直至12块全部完成。
汇总将重算全部请求AUC，并与同6000人稳定设计做UID配对比较，不依据中间结果改方案。

元数据覆盖复核保留在`analysis/calibration_length_coverage.json`：64 fitting含3名初始
历史≤16用户，实际场景最短6/11个token，并非完全缺失短缓存。不能把独立校准19-token
异常简单解释为“所有拟合都在长窗口”，因此未按看到的异常手写长度门槛。

另一可检验差异是校准query分布：原来每人8个最近历史item加随机item，时间取固定
1/60/3600秒及release gap；实际质量使用真实CC反馈。方案13`fit_observed_queries.py`
保持原均值摘要、四组时间×16查询及64/16预算，从该发布前14天、同拟合用户的真实CC
请求抽item，不加载label列；无请求用户共享该拟合组的真实请求池。时间为自己的历史
gap三个四分位及release gap，逐场景裁到[1,当前可见prefix的release gap]，不让query
落到源历史之前或发布之后。真实item是无标签探针，不当成负例或新的质量请求。
构造/请求IO费用单列`calibration_query_selection`，必要准备必须包含它。
`v13_observed_queries_canary_01`16/4/8机制用户、五目标，预计25–90s/12GiB上界，GPU1
短暂与开发评价共用，不将并发canary计时作最终成本。已有读写流程不变，另外检查
请求严格早于发布、gap严格为正及新query在合法范围；通过后64/16预计50–180s。

`v12_variance_quality6000_01`完成，909.51s/每GPU峰值18297MiB，6000人、4305反馈用户、
136610请求。M1/M3/M4/M5恢复89.75%/43.72%/94.49%/10.58%，M2仍为反向gap；相对
方案9稳定设计五段配对UID区间均含0。方差没有解决M3/M5短板，M5点估计从20.82%降至
10.58%，不能称显著下降或改进；完整负结果与开销保留，不扩展此变体。

`v13_observed_queries_canary_01`通过，24.00s/3978MiB；`v13_observed_queries64_01`
完成，45.63s/7624MiB，query选择0.444s计入必要准备。随后在同一独立128 calibration
UID上做两个预先声明的对照：原历史item/随机item探针，以及发布前真实CC item/gap探针；
每个对照的稳定与新译者必须共享同一组查询，不能跨任务比较误差绝对值或只报有利查询。
`observed_fit_original_probe128_01`和`observed_fit_observed_probe128_01`各预计30–120s/
18GiB，GPU0/1直接监控。它们检验校准分布迁移，均不是人口质量或独立确认。

随后以真实质量检验此查询分布改动：先`v13_observed_quality_canary_01`，原开发前4人、
固定五目标和native读写，预计15–45s/5GiB，GPU2；检查原始请求及两基线与保留四人
对照一致。通过后`v13_observed_quality6000_01`固定同6000人、12×500、GPU0–3、
native128/release16、eager读取，预计900–1500s/每GPU22GiB，直接监控；现有64人
拟合和query选择全部收费。这次同时保留两类独立校准误差及人口质量，不用代理误差
挑有利任务。执行期间不改核心文件或根据分块结果调整方案，确认集仍未读。

两个独立128人探针完成，54.80/54.26s、10927MiB。原探针稳定→方案13的UID等权
teacher logit MSE为.003110→.001302、.054945→.048472、.041615→.037540、
.070593→.074035、.127446→.128571；真实CC item/gap探针为.029184→.008183、
.064724→.061919、.059402→.049600、.123268→.096467、.251182→.232041。
真实探针五段平均误差下降，但大多数段仍明显差于Reuse到teacher的距离，不能称质量
已改善；误差被部分用户集中贡献，改善用户比例并非均超过半数。保留两个任务完整结果。
`v13_observed_quality_canary_01`通过，19.67s/2544MiB、244请求，全部非方法输出列
与原四人基线逐元素一致。按上述预算启动`v13_observed_quality6000_01`，冻结核心文件。

并行的小机制假设：固定方案9摘要/时间译者，在实际native历史响应上增加一个共享的
逐head斜率，校准目标是原修正后仍剩的teacher响应残差。每次发布用同一最小二乘公式
估计6层×6head共36个系数，UID等权、按实际事件数归一，不按M编号或开发标签设置。
读时使用已经算出的history响应，保留暂态self与原修正，并沿真实变化后的query继续；
沿用原逐层依赖消退mask。这检验读时可见信息能否补充只看source/time的估计，不是零点
ELU线性化，也不是屏蔽原生静默head。`probe_native_response.py`仅局部包装history读取，
不修改正在运行的6000人核心源码。`native_response_canary_01`原拟合前8人、两目标，
预计15–60s/5GiB，GPU0，关闭斜率包装必须与原reader逐元素一致。通过后64拟合与另一
128校准依次运行，各预计50–180s/14GiB，保存系数与完整对照；短暂共用GPU的计时不作
最终成本依据，两个集合始终与开发/确认分离。

`native_response_canary_01`关闭包装后两目标logits逐元素一致，拟合后M1/M2输出距离
略增，不作收益判断。最初canary复用了通用probe的“independent”文字，但此项是原拟合
前8人，configuration的independent=false与UID正确；保留原输出，此处纠正文案，后续
区分拟合/独立校准。按既定预算继续`native_response_fit64_01`与随后独立128人检查。

`native_response_fit64_01`完成，48.59s/7617MiB，五目标关闭包装均逐元素一致；拟合集
闭环logit误差均略升（M5 .028572→.028854），没有训练内输出收益。仍按先前声明用
`native_response_probe128_01`检查独立校准，固定上述36系数/目标，预计50–180s/14GiB，
GPU0直接监控；不依据单个head或模型的结果手写系数。

已完成6000人的请求集中度保存在`analysis/development_feedback_concentration.json`：
每段约2.7千反馈UID，但M3/M4最大单UID分别占31.2%/25.2%的负请求，M5最大单UID占
7.68%全部请求。该描述有助解释UID配对区间较宽，不改变冻结的请求级主指标、不删用户
或重加权争取更好结果；计数集中度不是AUC方差估计或训练重复数。

`native_response_probe128_01`完成，65.97s/10934MiB。五目标关闭包装与原reader逐元素
一致；稳定→额外native斜率的独立UID等权logit MSE为.003110→.003148、
.054945→.055240、.041615→.041679、.070593→.070895、.127446→.128271。
五段均略差，结合拟合集也无闭环收益，不把该后加斜率装入主方法或扩展6000人。
这否定了当前冻结译者上的36系数残差补丁，不等于所有联合query条件化方法都无效。
方案13必要准备核算为23.30s（含query选择.444s），校准发布前窗口各232/245/229/
248/223请求、29/30/27/29/29名自身有请求的拟合用户，其他人使用同拟合组请求池。

`v13_observed_quality6000_01`完成，939.09s/18280MiB，6000人、4305反馈用户、136610
请求；M1/M3/M4/M5恢复92.75%/43.99%/89.86%/16.61%，M2方法AUC .666442仍低于
Reuse .672662。真实请求分布校准没有解决M3/M5，原观察探针的误差改善不能代替任务
质量；必要准备23.30s，保留完整负结果。

下一方案14改变source→view映射，不继续调整这轮查询采样。用Gaussian source kernel
表达不同源状态之间的非线性关系；距离尺度取本次发布独立拟合源状态距离的中位数，
不读标签/开发/确认、不按M编号特调。仍以同一source统计、三轮实际query和rank32
生成常量/时间view，native写入与逐层依赖消退不变。时间保留producer条件化线性
Fourier块，并按维度归一；source kernel与时间块进入同一共享岭回归，自由截距需正确
居中。每次发布最多384个校准状态（同64/16实际191–258），共享support及刷新计算照计。
这是非线性译者假设，不预设更好泛化或20%成本成立。
新增公式检查用显式带自由截距的kernel方程验证未拟合source/time预测及发布后的时间
分解，连同已有适配检查11项通过（11.48s）。`v14_source_kernel_canary_01`16拟合/
4lifetime/8机制用户、五目标，预计25–120s/16GiB，GPU0直接监控；通过后同64/16
`v14_source_kernel64_01`预计60–240s/16GiB。主质量仍使用固定6000人，先看独立128
校准及实际读写检查，独立确认未读。

方案13与稳定设计的五段配对UID AUC区间均含0（M5[−.006485,.006979]），两基线
和完整请求元数据一致，不能称显著退化或改善。
`v14_source_kernel_canary_01`通过，28.61s/4268MiB，三轮更新后的响应/输出有限，
identity与Exact差分最大2.38e−7。启动`v14_source_kernel64_01`，同64/16、五目标、
三轮，预计60–240s/16GiB，GPU0直接监控；若进入人口质量，另用四人连续轨迹核对
冻结非线性view与原基线、写入和事件依赖，确认仍未读。

`v14_source_kernel64_01`完成，49.82s/8725MiB，必要准备27.86s（稳定22.20s）；新增
support与kernel求解/查找已进入实际耗时和显存。启动`source_kernel_probe128_01`，
与同64人线性译者在原独立128人、完全相同的查询/状态上比较，预计30–120s/18GiB，
GPU0。另`v14_source_kernel_quality_canary_01`固定原开发前4人、五次真实更新和244
请求，预计15–45s/5GiB，GPU1；核对全部非方法列与旧对照。均不读取6000确认。

`source_kernel_probe128_01`完成，54.57s/13169MiB，2099个场景及两基线/原稳定误差
与旧probe逐元素一致。稳定→kernel的UID等权teacher logit MSE为.003110→.005865、
.054945→.037192、.041615→.053102、.070593→.091556、.127446→.130338；改善
用户比例30%/39%/43%/38%/55%。只有M2均值改善，不能称广泛收益。由于译者表示变化，
仍按既定实际质量口径在同6000人完成对照，不用这些小代理数值替代任务结果。
`v14_source_kernel_quality_canary_01`通过，19.78s/2577MiB，244请求的全部非方法列
与旧四人对照逐元素一致。启动`v14_source_kernel_quality6000_01`，固定同6000人/
12×500、GPU0–3、native128/release16、eager读，预计900–1500s/每GPU22GiB，直接
监控并冻结核心文件；新support和kernel刷新计入实测，独立6000确认仍未读。

为定位剩余限制，`diagnose_time_view.py`区分共享Translator误差和时间view的表示误差。
仅对原拟合用户实际cache场景做两种teacher干预：每场景在原四组query直接拟合view；
以及在该prefix合法gap内64个对数间隔整数时间、同16个item拟合，再测原四组query。
每种都重算三轮实际修正query，保留原source/teacher、native写入和逐层消退mask。
前者看到了评价探针，后者也共享item与端点；它们不是独立质量、可部署方法或严格全局
上界。目的只是在更换共享估计器之前，检验现有32维Fourier时间view的局部表达能力。
`time_view_canary_01`8拟合用户、两目标，预计15–120s/16GiB，GPU1；检查时间边界、
有限值及双精度正规方程。若通过，再64用户五目标预计100–400s/16GiB，短暂共用GPU
的计时不作方法成本；核心文件仍为正在运行的方案14人口对照冻结。

`time_view_canary_01`通过，10.00s/3821MiB，正规方程相对残差≤3.2e−15。8人M1/M2
稳定logit MSE .000323/.004345；同探针teacher view降到约.0000197/.0000191，而
64时间拟合的同形式view为.000648/.013024。即使有逐场景teacher，跨时间拟合仍可能
损失效果；不把此小对照当作严格不可表达证明。下一步`time_view_fit64_01`固定原64
拟合、五目标，预计100–400s/16GiB，GPU1；新增记录最终真实query下拟合时间与探针
时间的逐层响应残差，以区分拟合/闭环与时间外推误差。校准群体、三个pass及所有
teacher介入都保留；函数局部绑定的lint修正没有改变已完成canary的计算。

`time_view_fit64_01`完成，89.25s/8119MiB，五目标时间与正规方程检查通过。同查询
teacher view的UID等权logit MSE为.0000138/.0000987/.000227/.000824/.001236；
64时间teacher view为.000499/.007107/.004270/.006114/.011754，均低于原拟合集
稳定设计，但距离同查询干预仍较远。它们使用逐场景teacher，不能代替共享方法质量。

一个结构线索：同查询view三轮后前3层平均响应残差已约1e−12，后3层仍有0.2%–2.3%
相对误差。固定source下，当前层query依赖更低层的修正，逐轮同时重拟合可能尚未把
变化传播到全部六层；全局rank32压缩的共享方法还会耦合各层，不能直接把oracle现象
推广为已证实原因。`time_view_six_pass64_01`只将此诊断的校准pass从3增到6，保持
同64用户/五目标/查询/teacher与读写，预计100–400s/16GiB，GPU1直接监控；复用已
通过canary的同一方程与时间检查，记录六层最终残差和输出。此处pass是内部校准次数，
不是M1–M5模型版本或方法方案编号。

同时用实际共享译者检验该线索：`convergence_six_pass64_01`复用方案9稳定的全部
64/16数据、查询、rank32与执行设置，只把校准pass从3改为6，不添加摘要、模型编号
分支或新推理字段。预计50–180s/12GiB，GPU0直接监控；现有方案9/14数值与读写
canary覆盖同一原拟合路径，这个配置对照无需重复同等检查。随后同独立128人比较，
用于判断收敛是否值得建立统一的发布自适应停止规则；不从已见五段质量手填pass数。

`time_view_six_pass64_01`完成，121.35s/8119MiB。同查询teacher view六层响应残差均
降到≤1.7e−11，支持逐层传播的诊断解释；但M5 logit MSE仅.001236→.001177，广时间
view .011754→.011219，未消除跨时间与候选差异。该局部收敛问题不是M5全部误差来源。
`convergence_six_pass64_01`仅完成资源计数，6.59s后因旧保守公式估2694.72s触发直接
运行限制，尚未模型执行/拟合；不是数值或质量失败，原probe保留。依据整阶段已有长
校准授权，改为`convergence_six_pass64_02`，同实验、实际预计50–180s/12GiB，保守
2694.72s，tmux `evokv_design_convergence_six` detached；前瞻配置引用通过的方案9
canary，`fit_passes.py`复用原执行器，保留log和exit，不放宽旧时长保护。

`v14_source_kernel_quality6000_01`完成，917.87s/18343MiB；同6000人/4305反馈UID/
136610请求，M1/M3/M4/M5恢复90.95%/41.78%/88.06%/8.17%，M2方法AUC .664990。
非线性source kernel没有解决短板，点估计五段均低于方案9稳定设计；完整区间/成本
继续汇总，不能只按点估计声称显著退化，保留此负结果。

下一表示诊断`diagnose_query_view.py`在同样64个合法时间×16item上拟合逐head的
仿射响应`b + Q A`，读取每层真实变化后的query，保留原生history与self；六轮使层间
传播有机会稳定。系数是逐场景teacher拟合，只用于与六轮广时间view比较，未安装
共享Translator或主读取器。它与先前零点ELU矩展开不同：系数由实际query区域拟合。
`query_view_canary_01`8拟合UID、两目标，预计15–120s/16GiB，GPU2；零view需与
原reader逐元素一致，校验时间和双精度正规方程。通过后同64/五目标预计100–400s/
16GiB，量化query形式能否显著缩小表示误差，再决定是否实施新的共享读取view。

`convergence_six_pass64_02`完成，56.49s/7624MiB，tmux exit=0。启动
`convergence_six_pass_probe128_01`，同独立128校准UID、冻结六次拟合权重，预计30–120s/
18GiB，GPU0。`query_view_canary_01`通过，15.44s/4176MiB，零view logits逐元素
一致，正规方程相对残差≤2.9e−15。8人M1/M2 query仿射teacher view误差约
7.87e−7/5.07e−7，明显低于广时间view的微型结果；它仍是逐场景teacher干预。
继续`query_view_fit64_01`，同原64/16、五目标、64时间×16item、六轮，预计100–400s/
16GiB，GPU2直接监控，先确定此表示线索能否跨五个目标成立。

`convergence_six_pass_probe128_01`完成，53.75s/10927MiB；六轮共享译者与三轮的
独立logit误差几乎不变（M5 .127446→.126832，其他段差≤.000043），必要准备从
22.20s增到30.54s。内部收敛的控制结果不足以支持扩大此固定六轮变体。
`query_view_fit64_01`完成，90.43s/8557MiB，零view与原reader一致、时间与方程检查
通过。五目标query仿射teacher view误差为8.61e−7/1.00e−6/1.50e−6/2.80e−5/
2.12e−5；同一64人六轮广时间teacher view为.000498/.007435/.003856/.005604/
.011219。此表示信号跨五目标成立，仍是逐场景teacher介入，不是方法恢复率。

据此实施方案15的共享query view：writer继续维护实际全历史producer均值/占比/年龄；
Translator从这些摘要预测每层每head的`b + Q A`。以拟合用户真实query的共享坐标均值/
尺度归一，并在发布时折回raw-Q的b/A，读时使用模型本层真实query。所有层的普通
K/V与self项不变，追加/淘汰使view失效，下一读刷新；后续版本继续读实际source。
校准按底层到顶层推进，拟合当前层前固定已经拟合的下层，用其真实修正query收集teacher
响应。每层一个rank32源映射（原方法为跨层共用rank32），逐场景仿射系数仅是校准监督，
开发时没有逐用户teacher。每个场景仍64个查询，改为同16items循环4次配64个合法
对数间隔时间；这是一组联动表示/校准改动，不能把未来差别归因于单一参数。
共享映射更大，query矩阵、校准与刷新全部收费；当前为eager原型，尚无编译服务声明。

新增坐标折叠、配对count权重、标量/批量发布和逐层消退检查后，计划
`v15_query_view_canary_01`：16拟合/4lifetime/8机制/4真实轨迹、五目标，预计45–240s/
18GiB，GPU0直接监控；用完整五次实际读写验证四组件。通过后64/16拟合预计90–400s/
18GiB，先独立128校准对照，再使用同6000人任务质量，不读取确认。

方案15数值检查12项通过；`v15_query_view_canary_01`完成，26.95s/4308MiB，
identity与Exact差≤2.39e−7，五个目标均拟合并发布。其继承的`tail_days=0`使真实服务
只覆盖M1–M4，共224请求；这224行的两基线、请求与状态计数和原参考逐元素一致。
因此不把这个运行称为完整五段服务canary；原记录保留，另以冻结权重执行
`v15_query_view_quality_canary_01`，同4用户、末段14天，预计15–60s/6GiB，GPU1。
同时启动`v15_query_view64_01`，同原64/16拟合、五目标，预计45–180s/12GiB，GPU0；
依据16人实测与已有64人回放成本。共享拟合和独立场景收益仍待验证，不由teacher诊断
推断方法有效。方案14完整配对区间现已汇总，五段均含0，不声称显著退化。

`v15_query_view_quality_canary_01`完成，19.21s/2562MiB，五目标共244条真实请求，
所有非方法列（含两条基线、覆盖与消退状态）和原参考逐元素相同；检查证据保存在
`analysis/v15_query_view_canary_checks.json`。`v15_query_view64_01`完成43.62s/8332MiB，
同原64/16用户，必要准备21.55s（教师、拟合、源构造和校准回放均计入）。
`v15_query_view_probe128_01`完成54.64s/12374MiB；独立用户等权logit MSE稳定→方案15：
M1 .003110→.001788，M2 .054945→.064499，M3 .041615→.025936，
M4 .070593→.076806，M5 .127446→.143733。M5约65%用户改善但均值变差，不能
据多数获益去删除坏例；这仍是教师距离，不是AUC。查询表示优势尚未被共享译者充分实现。
据M1/M3信号且主评价依赖真实请求，启动`v15_query_view_quality6000_01`，同固定6000人、
五次持续原生写入、冻结64拟合权重，4张GPU、每块500用户。预计900–1200s/每GPU22GiB，
依据方案13/14同任务实测939/918s及新完整canary19.21s；直接监控并保留log/分块exit。
其间不修改执行器或适配核心，确认输出保持未读。

用户要求先暂停并核对M2，已停止`v15_query_view_quality6000_01`（exit143），保留三个
完成分块及所有中间证据，`interruption.json`记录用户范围变更；不当作完整6000人结果。
核对`D14/reuse/E14/v1_to_v2/adjudication.json`：Parent .6387105824、Current .6399903384，
差.001279756，即.1279756个百分点；这是模型Full更新收益，不是方法恢复率，也不是
当前6000人的Exact−persistent-Reuse差。旧Full-only准入本身不被此次优化范围变更改写。
执行器新增显式`--no-op-targets 2`：M2不拟合/加载Translator、不发布修正view，读取走
实际native cache；模型M2和writer持续正常运行。M1/M3/M4/M5沿用共享方法，全部历史
记录留存，后续表单列M2边界但不把它作为本轮必须优化的目标。
先`v15_query_view_four_edge_canary_01`用原16人canary权重、同4用户/五次服务，预计15–60s/
6GiB，GPU0；检查M2逐元素等于Reuse，其他四段与原运行相同，并核对真实写入/producer
和无残留query修正。通过后同6000人四个适配目标预计900–1200s/四GPU，每GPU22GiB。

`v15_query_view_four_edge_canary_01`通过，18.11s/2561MiB；M2的63条请求逐元素等于
Reuse、没有活动修正；其他四目标181条请求的所有列与原canary逐元素相同，真实追加、
淘汰和发布计数一致。证据`analysis/v15_four_edge_canary_checks.json`，相关12项检查通过。
按用户新范围启动`v15_query_view_four_edge_quality6000_01`，原64人冻结权重，M2不加载，
`--no-op-targets 2`；四个适配目标、同6000人真实五阶段谱系，预计900–1200s/四GPU。
为保留边界，M2原生输出仍单列，主优化目标仅M1/M3/M4/M5。准备费用仍引用原64拟合
的完整21.55s（其中已发生M2拟合费用不删除），本轮不把它改写成节省后的估计成本。

## 2026-09-07：最终实验完成，按用户要求结束本轮探索

用户要求本轮完整实验结束后停止工作，分别总结论文设计与代码工程，完成后结束goal。
因此未再启动新的Translator拟合、表示诊断、确认或人口实验。

v15_query_view_four_edge_quality6000_01完成，893.09s、峰值18324MiB，父进程和12个分块
exit均为0。同6000选择UID、4305反馈UID、136610条五阶段请求；其中M1/M3/M4/M5
共4173反馈UID、111228请求。四个方法AUC为.6678707264/.6546076813/.6779190227/
.6655957821，恢复96.60%/28.95%/90.17%/37.29%。M2的25382条输出逐元素等于Reuse，
没有活动修正；所有请求保留，未将M2 No-op改善计入适配方法收益。

与方案9稳定/source×time对照的请求元数据完全相同，两基线最大logit差0。
相对稳定设计的M1/M3/M4/M5 AUC差区间为[-.001591,.001136]/
[-.003634,.001620]/[-.008346,.012082]/[-.002398,.012308]；相对source×time也均含0。
M5点估计改善、M3点估计下降，均不据此单独宣布显著差异。

方法五阶段服务账本952.31s，Exact920.19s，比率1.0349，另有原校准必要准备21.55s
和初次摘要3.09s；方法M2 No-op与Exact五阶段参考的范围差异、人口I/O和等待期缺口
均在工程总结中说明，不据此宣称四次迁移20%成本资格。最终质量、配对报告和保留运行
总表均已更新，原中断的五边适配运行仍标记为未完成，不改写历史不利结果。

两份总结已完成，逐项回答用户四个论文设计问题，并分别整理方案0–15的证据与工程经验。
没有修改或编译论文，确认输出保持未读，GPU模型作业与CPU汇总均已结束。
本轮goal按用户最新定义的收束点结束，不表示原质量/成本研究目标已经实现；以后是否
继续探索由用户重新决定。

## 后续记录格式（历史模板，不自动继续）

每轮一条，用以下字段写短记录即可，不为小变体新建整套文档：

```text
日期 / 方案版本 / run_id：
要回答的问题；为什么值得这次实验：
相对当前方案的改动；涉及哪些模块：
用户划分、模型边/轨迹、核心配置、代码与结果路径：
预计耗时/运行方式；实际耗时、峰值显存：
检查与主要结果（包括负结果和未覆盖的状态）：
解释：观察到什么，仍不确定什么；局部问题还是路线级反证：
决定：保留 / 修改 / 放弃此变体 / 暂停路线；下一步：
```

若放弃某种方案，说明原因与替代项，并更新本文件顶部和计划中的当前选择。
成熟设计才同步进论文正文；未经测量的建议继续标明 prospective。
