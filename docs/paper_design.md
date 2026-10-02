# EvoKV 设计入口

论文叙述的唯一来源是 [paper/main.tex](../../paper/main.tex)。
当前 **Cross-Version Cache Adaptation** Design 已有论文叙述，尚未完整实现和验证。
它围绕个性化读修正、状态演化、质量回退与兼容性退出组织。
2026-10-01用户重新开放Design 1的方法选择：读时校正恢复至少80%、新增计算不超过15%。
具体摘要、输入和修正公式均可替换；最新探索决定以[开发计划](design/plan.md)为准，
2026-10-02 已将 Design 1 正文及对应结构图同步为紧凑历史编码、注意力内校正和
响应残差；共享编码参数，逐条编码作为可复用读取视图，不再生成旧的用户级 `(b,A)`。
引言/概述、当前校准和状态核算、E3 消融计划同步使用新参数化。旧线性原型的账本
明确保留为历史说明，不能套用到当前方法。
图源与导出文件见[图表入口](../figures/README.md)，论文不自动编译。

Design 2/3 保留为后续生命周期草案，已同步基本追加/淘汰接口。Influence 如何进入
新编码器/响应模型、非线性校准对应的支持度指标仍需后续设计与验证；
当前两阶段拟合不会产生旧闭式线性模型的 Gram 矩阵。正文与源码中的相关备注
标识这个边界，不把后续机制当作当前已完成的实验。

## 实现起点

当前 [Design 1 开发](../scripts/design_one/README.md)从固定 Medium V0–V5 模型链出发，
先在 V4→V5 一万用户上评价读时校正；摘要条件化、非线性响应与逐条K/V视图均已实现。
已实现原生状态维护并完成多轮固定面板评价，实验结论与最新进展见[迭代记录](design/iterations.md)。
当前Item64＋Response256只保存每层每条64维隐向量，将线性输出融合到attention读取，
并在训练阶段使用可微的同一代数。固定Medium V4→V5一万用户上恢复84.98%、
新增FLOPs10.46%、额外持久状态16.67%，达到质量/计算/存储三项开发目标；
旧摘要候选保留为59.45%/0.753%的参照，其摘要条件化的额外收益尚未稳定成立。
早期KV32的联合最终打分训练和仅映射旧producer对照均未超过当时主点，全部结果保留。
滚动校准、完整维度仿射映射及单纯加宽/加query未改善；加入Current物品信息后有明确收益，
联合精炼12轮、混合状态精炼与缩放未改善；固定Item读视图后的响应残差带来进一步收益。
当前保留Item64轮、Response100轮；通过校准读取的代数改写降低计算。
直接减少Item至32/48轮的四个点均约76%恢复，完整结果保留。
当前开发点与来源已固定，其他边、独立确认和完整连续适配尚待验证。
完整参数、所有负结果和来源见迭代记录。
学习式 influence sketch 与跨版本重建/退出仍属后续阶段。

[adaptation/](../src/hstu_kvcache/adaptation/) 的7个模块保留摘要、Translator、原生读取、
推理及混合producer状态接口；[Design脚本目录](../scripts/design/README.md)保留
10个文件的native服务与共享拟合依赖。这些是历史实现起点，不能视为当前论文的完整方法。
当前Q/H探针也复用其中的读取原语。

旧native C、均值/PCA条件修正和paired-summary原型保留其结果与解释范围，
不把旧超参数、同步执行方式或Translator结构作为当前设计承诺。
影响sketch、支持/重建策略和统一运行时属于后续问题，随验证后的Design 1继续设计。

## 工作入口

- [适配计划](design/plan.md)：保留接口、冻结六层范围和下一步。
- [实验设计](experimental_design.md)：资产、因果数据、评价与成本定义。
- [决策记录](design/iterations.md)：历史方案变化与证据。
- [当前读取修正](design/read_correction_probe_2026_09.md)：Q-v5与H-v4探针。
- [四种重算与保留基线原语](../src/hstu_kvcache/baselines/README.md)。
- [模型训练](unified_training_2026_09/README.md)：三规模选定模型。
- [结果索引](../results/README.md)：当前实验与论文依赖的历史证据。

## 历史证据

[质量报告](../results/design/analysis/native_base_quality4091_01_report/report.md)、
[FLOPs报告](../results/design/analysis/native_flops_01/report.md)和
[历史机制记录](design/expert_route_2026-09-07.md)继续提供原型报告入口；
原执行所用文档与哈希由各实验的源码快照保存。
旧10k用户AUC仍由论文附录描述；3.1使用的旧Medium三角与成本表证据也保留。
退休的是旧活动队列和重复出图，不改写这些结果或封存哈希。
