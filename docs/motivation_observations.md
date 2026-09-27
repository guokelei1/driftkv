# 当前论文的 Motivation 与 Insights

测量记录：2026-09-20；导航更新：2026-09-22。

本文只记录当前论文采用的测量及其解释边界。保留实验的结果没有因目录清理而重算、筛选或改写。

## 当前统一 AUC：Motivation 1／2 与共享读取 Insight

[九预算统一真实反馈诊断](../results/insight/unified_auc_10k_teachers7144_20260920/README.md)已完成；
[原四预算记录](../results/insight/unified_auc_10k_20260920/README.md)完整保留。本轮六层
Medium／H192／seed17，四条已准入相邻边V0→V1至V3→V4，共用10000个开发用户、
每人1024-event发布快照和完整E14真实反馈，共212773条known请求。评价用户由3970个
原development和6030个未使用calibration-reserve用户按无标签资格及既有顺序选定，
与256拟合、全部512历史拟合、128pilot和两个保留确认组互斥。原split不改，没有读取final。

Parent／Reuse／Current Exact的四边AUC及全部分支见[完整表](../results/insight/unified_auc_10k_20260920/README.md)。
Current Exact相对Parent分别提高2.09／1.53／2.32／3.85个AUC百分点；Reuse损失占
对应更新收益的38.77%／81.71%／70.25%／109.70%。最后一边Reuse低于Parent，
其损失比例超过100%，保留原值。

共享读取两臂为`N*b`和`N*b+T*r`，其中N=1024，r为实际用户原生历史响应。
参数与归一化仅用对应32／64／128／256／512／1024／2048／4096／7144校准用户，
候选bank固定来自原256人的无标签历史，原256顺序及四边候选保持完全一致。
256教师预算下，前者四边AUC恢复为74.35%／-24.78%／34.62%／23.43%，
加入r后为105.80%／-14.95%／38.91%／80.59%，等权边均值从26.90%到52.59%。
加入r在这四边均优于同预算offset；V1→V2上两者仍不如Reuse。所有预算保留，
不以较优预算、版本或平均值遮盖负结果。

新增五预算在四条边形成20个同预算对照，加入用户响应的AUC均高于shared offset，
但V1→V2上两臂全部新增预算仍低于Reuse。256／1024／7144教师下两臂的等权边平均恢复
依次为26.90%／52.59%、27.09%／45.14%、27.27%／45.61%；恢复没有随教师数单调增加。
[全部九预算均值表及72行逐边CSV](../results/insight/unified_auc_10k_teachers7144_20260920/README.md)
保留32与64等较小预算的结果以及所有负值，最大7144预算的绝对AUC也完整列出。

7144教师由原256、未使用Medium成熟用户3329及已有Yambda共享历史中的外部用户3559
组成，即3585名Medium训练人口用户与3559名外部用户。按既有selector rank／UID顺序
扩充，全部day217前满1024历史，与原10k评价、pilot、确认组及其他历史拟合用户隔离。
四边模型和Medium item mapping／OOV规则不变，只扩大共享校准的人口。

横轴是含教师、拟合／profiling、状态处理和实际请求读取的额外兼容性运算，除以
10000份快照各一次Exact重建；普通读取为共同费用。256用户共享offset约占3.0256%，
增加用户响应为3.1050%–3.1082%。对比方法也有低成本有效点：V3→V4的一层DroidSpeak
在8.63%运算下恢复93.39%。
1024教师下两臂平均成本为12.1021%／12.2547%，7144时为84.4299%／85.1543%。
原五panel图及其生成器已清理；当时横轴为线性0–100%运算，纵轴恢复率不裁剪，
仅Exact−Reuse>1e-4时计算，本次四边均满足。[AUC表](../figures/tables/motivation_auc.tex)保留。
现有三规模[完整cost ledger](../figures/out/three_scale_auc_preview/cost_ledger.json)和
[图表索引](../figures/README.md)提供保留的数据与显示口径；当前图的截断和选边
不改写这里记录的原始恢复率。

48线程输入准备20.55秒；四GPU96人canary的25条路径与单GPU分数／AUC最大差均为0。
原四预算完整执行498.896秒、峰值16.431GiB，四边27条路径的raw AUC独立重算一致。
新增预算执行969.436秒、最高每卡3.19GiB，教师双缓存CPU125.58GiB；扩展保留原27路径，
新增10路径至每边37条，四边复算Reuse与原值最大差均为0，执行源hash未变。
秒数用于运行记录，不作为运算横轴。以下旧Motivation、概率保真度和
逐用户教师诊断均转为历史记录，数据与不利结果保留；
[5872人前序AUC pilot](../results/insight/auc_read_6l_20260920/README.md)只有两次失败，没有完成评价。

## 历史 Motivation：旧链完整 producer 三角

该组历史质量实验使用 Yambda-500M 固定 30,000 用户、六层 Medium、hidden 192、
六个 heads、1,024-event context、seed 17。V0 在前 217 天训练，
V1–V5 每次从直接父模型继续训练后续 14 天数据；六个版本都使用一 epoch。
每次更新使用既定的后续 E14 窗口，V5 对应 [287,301)。
十层 Large 的 V4/V5 两 epoch 设置不属于这组 Medium 数据。

图中左侧是模型更新的绝对 AUC 改善乘以 100，右侧是旧 cache 导致的 AUC 损失
除以该次更新的相邻模型改善，并用负号表示损失。二者不是同一个百分比口径。
相邻复用损失依次约为改善的 22%、73%、54%、87%、53%。
完整三角包含 15 个旧 producer 对比；更早 producer 的损失在本次运行内部与
Current 配对，再以相邻三路径测量的模型改善作为共同尺度，不能拼接成跨运行三路径结果。
版本年龄影响不是严格单调的。

证据入口：
[完整三角报告](../results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14/direct_long_age_reuse_v1/summary.md)。
普通 D14 相邻结果在同一模型树的 reuse/E14，V5 的既定E14 [287,301)结果保存在历史路径 `v5_extension_v1/reuse/E14_partial`。
这些汇总对应的 raw、seal、Full-only admission 和 checkpoint 均保留。
历史绘图生成器 `motivation1.py` 已清理；上述报告与原始证据仍保留。

成本表来自 [随机权重 GPU 测量](../results/release_cost_random_weight_v1/report.md)，
仅说明给定算子和计时边界下的 GPU 计算成本。当前表的五个配置、原始计时 JSON、
随机模型及 invalidation 记录保留。它不是模型质量或完整迁移延迟测量。

## 历史局部替换诊断（原 Insight 1）

实验在 3,000 固定用户、64 个无标签 probes 和五条版本边上执行全部 34 个冻结配置。
Exact-KV splice 是理想化诊断，横轴为替换的 K/V 覆盖率，不是真实重算 FLOPs。

旧论文四面板只展示 V0→V1、V1→V2、V2→V3 及三者平均值。
约 10%/20% token 覆盖对应论文平均恢复 23.5%/32.7%，不是五边平均的 30.0%/41.2%。
保留作者确认的 layer-only 减五个百分点显示口径，以及 0–100% 显示边界；
原始裁决数据不因此改动。

[完整报告](../results/yambda500m_medium_seed17/insight1_locality_v1/analysis/report.md)
和 formal_raw 保留全部五边及所有配置，不删除未画出的边或较差结果。
历史生成器 `insight1.py` 已清理；当时读取上述 analysis 中的
best_observed_by_edge.csv 并校验原 SHA-256，生成单栏四面板图。底层表与完整报告保留。

## 历史 2560 用户概率保真度诊断

同一 incoming query 下，Exact 与 Reuse 的历史读取差定义了所需响应修正。
这个代数关系只说明修正对象，不说明该差可被低成本预测。此前正文展示
[2026-09-20 共享诊断](../results/insight/query_read_6l_2560_20260920/README.md)：
当前 consolidated 六层 Medium 链的两条预选边 V1→V2、V2→V3，256 校准 UID、
128个独立pilot UID、2560个评价 UID，三组互斥，1024-event 静态历史，seed17。
评价组按既有development顺序，排除校准／pilot／保留组并筛 day217 前已满1024历史；
它是扩大开发评价，没有读取 final／confirmation 人口。

两种共享线性规则分别用校准用户的 16 个候选逐层拟合，归一化和参数也仅取自校准组。
两组为 `b+Aq` 与 `b+Aq+Tr`：`q` 是本次实际读取查询，`r` 是读取用户旧缓存所得的
原生历史响应。高层q已承接用户历史；加入r检验本层直接历史读取信息的贡献。
256人校准参数在128人pilot观测一次后原样固定，用于2560人相同32候选的比较，
不重拟合、不根据pilot调参。修正生成不用评价用户教师。

当时一张表同时给出 U（逐用户未裁剪恢复率的均值，冻结主指标）和 P（用户平均概率
误差的恢复率，辅助口径）。`b+Aq` 两边 U 为 -818.36%／-200.01%，P 为
-148.73%／-3.56%；`b+Aq+Tr` 的 U 为8.05%／17.90%，P 为65.54%／72.85%。
后者有2164／2154个用户优于Reuse，仍有396／406人不改善；全部负尾保留。
相对 `b+Aq`，加入r的平均概率误差下降86.15%／73.78%。共享收益由完整规则对Reuse
的比较说明，直接用户响应贡献由两组规则的比较说明，不主张 `b+Aq` 自身已取得正收益。
128人pilot的q-only U=-228.66%／-140.02%、q+r U=12.17%／34.71%及较差canary
均保留，不用较大的P替代主指标U，不把概率保真度恢复称为推荐AUC改善。

[运算估计](../results/insight/query_read_6l_2560_20260920/analysis/cost.md)计入每发布
3.9723 T 的q+r共享校准；每用户一次32候选的额外修正为16.961 M，约为普通读取的7.422%。
以cache-only重建4.1971 G/user为参照，本次2560用户各读一次且包含校准的新增兼容性
算术为重建的37.374%。这是形状推算，不是端到端时间比；复用pilot参数没有将校准成本记零。

此结果显示有限的跨用户共享与用户读取上下文信号，不验证具体摘要的必要性。
读端映射与部分 Value 映射可以代数等价；同预算 K/V translation 的质量／成本优势
仍需方法评价，不能由本表推出。

### 保留的历史诊断

历史 512-user、五边 discovery 的逐用户 shared offset／rank-1 平均恢复
95.34%／99.46%，[旧128用户诊断](../results/insight/user_information_6l_20260920/README.md)
已执行的每用户教师99.12%／99.50%，均保留于对应
原始结果和报告，但本次论文主表不展示它们，不用 oracle 结果证明共享可服务性。
历史有效口径仍为
[analysis_v2](../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/discovery_functional_boundary/analysis_v2/report.md)；
旧 analysis 将 gap-weighted 指标提升为主口径的 invalidation、原视图和 raw 均保留。

[旧持续性诊断](../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1/discovery/analysis/report.md)
继续保留全部五边／14 天轨迹、重新观察 93.39%、冻结修正失效及按旧历史占比缩放
33.85% 的原始结论。本次正文不再引用这些旧链数字，也不把它们算作当前 Design 的
持续维护证据。[前一摘要诊断](../results/insight/shared_read_6l_20260920/diagnostic/summary.json)
及旧表／源码快照同样保留；它不作为当前 Insight 的择优方案。

## 对当前设计的意义与边界

当前统一AUC结果显示复用旧状态会损失新模型的收益。
共享读取修正在固定256教师预算下三边优于Reuse，加入实际用户响应在四边均优于offset，
其中V1→V2的两种修正均低于Reuse。它为共享读取校准及用户历史信息提供了固定发布状态下的证据。
当前 Cross-Version Cache Adaptation 因此保留普通 K/V，并增加写时汇总、发布时翻译、
读时修正和连续状态维护。设计正文见 [paper/main.tex](../../paper/main.tex)。

当前四条更新和历史五条更新均来自一个训练 seed；训练seed是独立重复单位。
当前 Design 已完成设计，尚未实现或验证；历史六层原型的共享校准、实际追加／淘汰
及多次发布探索仍见[迭代记录](design/iterations.md)，不等于当前 Design 的实现。
后续按 [当前计划](design/plan.md)推进。用户已授权现有六层冻结模型上的本阶段探索，
长作业仍先有前瞻配置、资源估计和canary；此授权不包含backbone重训或未开放数据。

[结果索引](../results/README.md)列出保留范围和已知协议问题。
