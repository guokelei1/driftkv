# Design 1 开发入口

研究目标与固定输入见[设计计划](../../docs/design/plan.md)。本目录使用冻结 Medium
V4→V5、一万用户/90,296 请求；完整 V0–V5 模型链留供后续连续状态研究。
原 Motivation Q/H 的源码、校准参数和结果继续保留。
当前目标为读时校正恢复≥80%、新增计算≤15%；已有均值/PCA原型是保留参照，
新方法可以更换输入和校正结构，不受其参数化限制。

第十八轮 **Item64＋Response256** 参照：固定面板AUC0.669820、缺口恢复83.65%、
新增FLOPs13.46%，但额外完整K/V为100%，不满足新增存储约束。[完整结果](../../results/design_one_2026_10/round18_item_response/evaluation_10k/summary.json)
与原始分数、校准权重、执行源码均已保留；第十一轮Item64校准为冻结依赖。
当前选定第二十轮：紧凑hidden执行＋紧凑Item校准，Item训练64轮、Response训练100轮。
恢复84.98%、总新增FLOPs10.46%、额外持久状态16.67%，[完整结果](../../results/design_one_2026_10/round20_compact_training/evaluation_10k/summary.json)
与两阶段权重已固定。第十九轮全部对照保留；以下历史入口不是自动重跑队列。

| 入口 | 用途 |
| --- | --- |
| `benchmark_data.py` | 从保留的全人口 Full/Reuse 评分准备固定请求面板，核对原3k参照及校准隔离 |
| `prepare_history.py` / `history.py` | typed UID semi-join 准备四份精确有界历史；后续直接按签名恢复，避免重复扫描 |
| `benchmark.py score/aggregate` | 同一因果滚动轨迹上的 Q/H 或 Design 1 评分、分片续跑与 pooled 指标汇总 |
| `calibrate.py` | 在256拟合/16验证用户上做按层共享读修正校准；小型前缀只用于 canary |
| `calibrate_nonlinear.py` | 两种新原型的同query Full响应蒸馏：`nonlinear_response`宽256、`kv_view`宽32；`--small`只做4拟合/2验证用户、两轮训练检查 |
| `release_config.py` | fresh Item/Response 校准的配置选择接口；当前原样返回固定参数，尚无自适应策略 |
| `rolling_calibration.py` | 四个发布前真实滚动场景，共享确切对齐的教师；每用户16个query分成4×4，保存不可变条目来源/出生占比 |
| `calibrate_affine.py` | 四场景逐条Full K/V监督的完整维度仿射读视图；FP64流式ridge，固定0.001，query仅用于拟合后诊断 |
| `scoring.py` | 增量维护实际缓存摘要，写后按需生成视图，候选读取复用该视图 |
| `launch.py` | 四卡各执行一个 UID 分片，记录日志/退出状态，完成后统一汇总；长任务在 tmux 内启动 |
| `compare.py` | 明确指定方法的配对用户bootstrap；描述固定面板的不确定性，不自动选优 |

模型原语在 `src/hstu_kvcache/design_one/`。第一版为 producer 分组 K/V 统计、
PCA 摘要和 `c=b+Aq+Tr`；不改原生写入，不包含后续 influence sketch 或重建策略。
唯一共享执行改动是为原有滚动快照增加可选状态观察回调，未启用时原方法路径不变。

新原型位于`src/hstu_kvcache/design_one/nonlinear.py`：响应网络直接从实际query与
原生响应率预测修正；历史网络逐条映射K/V，原生写入仍独立执行。映射视图只在初始化
及追加时计算，淘汰时同步删除；读取回调执行的第二次历史读取全额计费。
两种原型共用现有评分/聚合入口，按校准artifact的kind自动加载。
本轮配置与输出集中在`results/design_one_2026_10/round6_nonlinear/`，训练与质量结果
见[迭代记录](../../docs/design/iterations.md)，不覆盖首批结果。
`calibrate_nonlinear.py --method kv_view --initial-calibration <目录>`从既有KV32继续
联合拟合Full最终logit，费用包含继承校准。本轮输出隔离到`round7_logit_refinement/`。
KV策略列表可指定`producer_scope: parent_only`，仅变换初始Parent条目；
省略该字段时保持全部条目映射。作用范围随策略与请求签名保存，逐点独立计费。
第八轮使用`--method kv_view/kv_context --rolling-scenes`比较真实滚动校准与
增加条目来源/出生占比输入的KV32，均32轮；输出在`round8_rolling_context/`。
context方法限定全部条目映射，初始化和新增项各映射一次，单列上下文维护费用。
第九轮`calibrate_affine.py`复用同一场景和视图维护，拟合完整386→384残差映射，
输出在`round9_affine_kv/`；core是`src/hstu_kvcache/design_one/affine.py`。
第十轮沿用`calibrate_nonlinear.py --method kv_view`，新增仅适用于纯Parent fresh fit的
`--hidden-width 64 --queries-per-user 16/64`，分别检验容量与查询覆盖；
不修改冻结UID和配置文件，metadata单列实际校准预算。结果在`round10_capacity_queries/`。
第十一轮`--method kv_item`只增加当前物品embedding输入，保持宽64/16query/64轮。
回放使用真实初始化/追加的item ID，各条目只查表一次；旧observer未请求raw事件时保持原路径。
查表逻辑字节单列，网络与额外读取按FLOPs计费；结果在`round11_item_features/`。
第十二轮从该Item-KV64权重出发，使用`--method kv_item --initial-calibration <目录>`
及`--refinement-objective response/logit`分别做逐层读响应精炼、联合最终打分精炼。
两者均8轮、学习率0.0001、batch8，继承原输入/输出归一化；response还保留各层原RMS。
每点包含继承校准与新增精炼的完整费用，输出在`round12_item_refinement/`。
两种更新日程不同，结果只能比较两种精炼方案，不能只归因于监督目标。
第十三轮使用`--refinement-objective logit --refinement-epochs 12`，仍从第十一轮原始权重
开始；该轮数选项只适用于Item-KV64联合精炼，默认仍8轮，结果在`round13_item_duration/`。
第十四轮使用`--mixed-refinement`，同一原始权重、联合目标和8轮预算，奇数轮纯Parent，
偶数轮真实append768；每轮只训练一态、每人仍16query。两态终端item ID与教师逐条对齐，
额外状态采集、双态目标/诊断都计费；输出在`round14_mixed_refinement/`。
第十五轮`--producer-scale-refinement`冻结原Item64，只学六层Current残差缩放，8轮均用
真实a768场景，lr0.01、weight decay0；原归一化和pure RMS不变。输出在
`round15_producer_scale/`，自动生成独立`fixed_layer0_control/`作为不需新拟合的结构对照。
新原语`producer_scaled_item.py`保留原生缓存与Parent映射，新增缩放只作用于Current条目，
追加乘法记入`view_update_flops`，不存持久producer mask。
第十六轮`--method kv_item --recent-only-queries`保持fresh64轮，仅将原半随机query改为
最近16个不同known历史item，不足时固定seed补齐；输出在`round16_recent_queries/`。
第十七轮从第十一轮原权重出发，`--centered-logit-refinement`用实际batch内中心化误差
拟合教师相对logit，仍8轮；原未中心化MSE诊断保留，输出在`round17_relative_logits/`。
第十八轮`--method nonlinear_response --item-read-view <Round11目录>`冻结Item64读视图，
再拟合fresh Response256补偿Full−mapped响应。core在`composite.py`，拟合200轮，
校准历史每用户只映射一次；来源Item校准、新Response校准与服务分开计费。
输出在`round18_item_response/`，校准诊断基线明确标为Item而非Reuse。

第十九轮沿用同一artifact，在评价配置每条policy中设置`"execution_mode": "compact_hidden"`。
`src/hstu_kvcache/design_one/compact.py`只保存每层每条64维SiLU hidden，将线性输出
融合到query-key得分与加权value聚合；额外状态为原双K/V的16.667%，不保存展开视图。
默认`mapped_kv`仍复现历史路径。统计分别记录初始化/追加编码、额外读取、Response、
每worker一次投影初始化、持久state/native比及共享投影字节；临时query缓冲和峰值显存单列。
原权重等价测试与新结果在`round19_compact_hidden/`。
校准新增`--epochs`仅支持fresh Item32/48/64和Response100/200，默认旧轮数不变；
`--small`仍为两轮。新低预算点从零拟合，继承其真实Item校准费用。

第二十轮`--method kv_item --compact-training --epochs 64`保持同一网络参数、64轮
和loss，但训练读取使用可微的紧凑代数；每批live输出投影参与梯度并计费。
默认不启用时保持旧训练路径。随后用该Item权重执行
`--method nonlinear_response --item-read-view <目录> --epochs 100`。
结果在`round20_compact_training/`，服务继续使用`compact_hidden`；拟合费用按实际执行重新记录。

`release_config.py::select_release_config(release_id, stage, fixed)`在 fresh Item/Response
校准前接收已解析的宽度、训练轮数和紧凑训练开关，目前直接返回`fixed`。
后续可在此接入基于校准/验证信息的选择策略；现在不搜索参数、不读取评价结果。
上述64/100轮命令继续表示当前选定配置，历史CLI默认值和`--small`两轮限制不变。
新校准记录实际生效参数与接口源码哈希；已完成的结果保持原有来源记录。

结果根为 `results/design_one_2026_10/`。`benchmark_panel/binding.json` 记录固定
请求、参照与校准来源；各轮目录保留校准、逐请求评分、成本分项和汇总。
活动任务恢复要求用户、参数与源码签名一致；改方法另起轮次，保留先前结果。
首批完成目录已去掉重复分片，只保留最终分数及必要来源记录，不再原地续跑；
具体权重与原始载荷的保留范围见[结果索引](../../results/README.md#design-1-本地保留范围)。

评价同时报告 AUC、log loss、Full−Reuse 缺口恢复，以及包含校准、初始摘要扫描、
增量维护、视图生成和逐候选修正的新增 FLOPs。运行时间与显存单列。
小型 pilot 用于数值与资源检查，方法开发结论使用完整固定面板。

后续评价传入 `--history-packs results/design_one_2026_10/history_packs_fast`。
历史包经128用户与旧loader逐数组对齐、全一万人保存恢复一致性、全部请求时间因果核对。
初次构建的准备脚本从 `/tmp` 原样保存在本目录，SHA与包中的执行来源一致。

首批探索参照为 `calibration_rank8_c256/` → `rank8_10k/summary.json`：
纯Parent状态校准、`--rank 8`、`producer_mean`、个人化ridge倍率1，保留原生响应输入。
默认rank8与当前主参考一致；原rank32探索的执行源和结果保留。
各轮必须另用输出目录，不覆盖已完成的校准或评分。

保留校准选项 `--rolling-scenes`（真实原生追加和淘汰）、
`--summary-mode producer_mass`（分组贡献量）和
`--without-response`（独立重拟合移除Tr的输入对照）。
增长式mixed与个人化ridge倍率搜索已结束，其分支从活动入口删除，原结果与执行源保留。
当前生命周期使用`scoring.py`的`ViewObserver`，删除了未被评价调用的重复state包装器。
已完成的reference bootstrap专用脚本移到来源快照；常规比较继续使用`compare.py`。
评价 `--design-calibration-list configuration.json` 接收
`[{"tag": "方法名", "calibration_dir": "目录"}]`，多方法共享原生回放，各自独立计费。
原始结果、来源快照和全部迭代解释见[决策记录](../../docs/design/iterations.md)。
