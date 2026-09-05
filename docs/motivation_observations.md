# 当前论文的 Motivation 与 Insights

更新日期：2026-09-05

本文只记录当前论文采用的测量及其解释边界。旧 Small 观察长稿完整保存在
[结果清理包](../results/history/cleanup_results_2026-09-05.tar.gz)的同名路径下，
不再作为当前结论入口。清理没有重算、筛选或改写保留实验的结果。

## 实验对象与 Motivation

当前质量实验使用 Yambda-500M 固定 30,000 用户、六层 Medium、hidden 192、
六个 heads、1,024-event context、seed 17。V0 在前 217 天训练，
V1–V5 每次从直接父模型继续训练后续 14 天数据；六个版本都使用一 epoch。
每次更新观察后续 14 天，V5 包含已观察到的不完整 day-300 尾段。
十层 Large 的 V4/V5 两 epoch 设置不属于这组 Medium 数据。

图中左侧是模型更新的绝对 AUC 改善乘以 100，右侧是旧 cache 导致的 AUC 损失
除以该次更新的相邻模型改善，并用负号表示损失。二者不是同一个百分比口径。
相邻复用损失依次约为改善的 22%、73%、54%、87%、53%。
完整三角包含 15 个旧 producer 对比；更早 producer 的损失在本次运行内部与
Current 配对，再以相邻三路径测量的模型改善作为共同尺度，不能拼接成跨运行三路径结果。
版本年龄影响不是严格单调的。

证据入口：
[完整三角报告](../results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14/direct_long_age_reuse_v1/summary.md)。
普通 D14 相邻结果在同一模型树的 reuse/E14，V5 在 v5_extension_v1/reuse/E14_partial。
这些汇总对应的 raw、seal、Full-only admission 和 checkpoint 均保留。
绘图使用 [motivation1.py](../figures/src/motivation1.py)，不再使用四层数据。

成本表来自 [随机权重 GPU 测量](../results/release_cost_random_weight_v1/report.md)，
仅说明给定算子和计时边界下的 GPU 计算成本。当前表的五个配置、原始计时 JSON、
随机模型及 invalidation 记录保留。它不是模型质量或完整迁移延迟测量。

## Insight 1：低覆盖局部替换的恢复有限

实验在 3,000 固定用户、64 个无标签 probes 和五条版本边上执行全部 34 个冻结配置。
Exact-KV splice 是理想化诊断，横轴为替换的 K/V 覆盖率，不是真实重算 FLOPs。

论文四面板只展示 V0→V1、V1→V2、V2→V3 及三者平均值。
约 10%/20% token 覆盖对应论文平均恢复 23.5%/32.7%，不是五边平均的 30.0%/41.2%。
保留作者确认的 layer-only 减五个百分点显示口径，以及 0–100% 显示边界；
原始裁决数据不因此改动。

[完整报告](../results/yambda500m_medium_seed17/insight1_locality_v1/analysis/report.md)
和 formal_raw 保留全部五边及所有配置，不删除未画出的边或较差结果。
论文生成器是 [insight1.py](../figures/src/insight1.py)，直接读取上述 analysis 中的
best_observed_by_edge.csv 并校验原 SHA-256；该图为单栏、四面板，含红色 target area 和关键点标注。

## Insight 2：聚合处的响应可以紧凑修正，但不能冻结不变

论文响应修正表来自 512-user、五边、无标签 discovery。
历史聚合处的 shared offset 和 rank-1 correction 平均恢复为 95.34% 和 99.46%，
最差边分别为 94.46% 和 98.91%。
这些修正读取 Current-Exact anchor response，是 oracle 诊断，不是学习式迁移结果。

主口径是先逐用户计算不裁剪的 recovery，再用户等权、版本等权聚合。
论文只引用 [analysis_v2](../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/discovery_functional_boundary/analysis_v2/report.md)；
旧 analysis 曾把 gap-weighted 指标提升为主口径，已经 invalidated。
旧视图、INVALIDATED.md 与有效 raw 均保留，不因清理掩盖这次口径修正。

[持续性诊断](../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1/discovery/analysis/report.md)
保留全部五条更新及 14 天轨迹。同请求重新观察修正仍平均恢复 93.39%，
但冻结 cutover offset 在五条边都失效；按剩余旧历史占比缩放也只有 33.85%。
正结果与反例共同说明：紧凑响应结构不等于可跨请求保持不变的状态。

## 对当前设计的意义与边界

Motivation 提出低成本跨版本适配的需求；Insight 1 不支持依赖少量局部替换稳定恢复主要差距；
Insight 2 提示在历史聚合处表达版本变化，同时要求修正随 query 和历史演化。
当前 Cross-Version Cache Adaptation 因此保留普通 K/V，并增加写时汇总、发布时翻译、
读时修正和连续状态维护。设计正文见 [paper/main.tex](../../paper/main.tex)。

上述观察来自一个训练 seed；五条更新不是独立训练重复。
它们不证明所有 Transformer 都有相同效果，也不证明短状态翻译一定成功。
当前没有 translator calibration、完整在线更新或连续迁移的方法实验结果。
新实验仍需独立协议、资源估计、canary 和明确授权。

旧探索的结论与失败记录已压缩归档；废弃 raw 和四层权重不能从该包恢复。
[结果索引](../results/README.md)列出保留范围、归档清单和已知协议问题。
