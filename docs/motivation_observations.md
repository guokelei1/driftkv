# Motivation 测量索引

导航更新：2026-10-01。当前正文探索使用三规模选定模型及其相邻版本边；
论文附录仍引用的旧实验单列在末尾。具体指标、配置和原始证据以各结果目录为准。

## 模型更新与缓存复用

[固定版本清单](unified_training_2026_09/model_versions.md)记录 Medium、Large、Max
各 V0–V5，共18个模型及训练数据链。三规模共15条相邻边的
[Full / Recompute / Reuse 评价](../results/unified_reuse_2026_09/README.md)已完成。
Full 与 Recompute 指当前模型按请求重算历史；Reuse 继承上一选定版本的缓存，
之后按时间追加与淘汰。每条边的两种读取使用同一批请求。

15条边均观察到 Reuse 的 AUC 低于 Current Full；原始分数和全部指标保留。
模型更新收益以所选上一版的同窗口 Full 为参照，不能用其他训练端点替代。

## 局部重算

[四种 baseline](../results/selective_recompute_2026_09/README.md)已完成全部15条边，
每边固定3,000名诊断用户，四种方法共享请求和 Full/Reuse 对照：

- **Layer**：在独立校准用户上选择层区间，重算该区间。
- **Tail**：选择最近的历史位置，经过当前模型各层重算。
- **Deviation**：按第一层当前投影与继承 K/V 的偏差选择历史位置。
- **Query**：按当前 query 对第一层继承 keys 的 attention 大小选择历史位置。

每种方法都有不同预算下的成本—恢复曲线。重算仅服务当前请求，随后丢弃临时状态。
用户面板根据已有 Full/Reuse 差异选取，属于条件化诊断；完整选择记录见
[用户面板](../results/selective_recompute_2026_09/panels/README.md)。

## 读取修正

当前采用[非线性 Q-v5 / H-v4 的90个实测点](../results/read_correction_2026_09/motivation_final/README.md)：

- **Q-v5**：`cross_phi_joint8`，C128/256/512；由 query 的非线性特征产生修正量，加到原始历史读取上。
- **H-v4**：`map_all`，C64/128/256；对继承 K/V 作临时非线性转换，再由当前 query 读取。

两者用少量独立用户拟合共享参数，并复用前述15条边的评价面板。
H C128及Q来自此前已完成结果，H C64/C256来自预算补齐；这些来源继续保留。
目标是观察低成本共享修正的恢复潜力，以及显式处理历史带来的恢复与额外成本，
不将两个不同结构的探针解释为严格单因素消融。

恢复率为 `(方法 AUC − Reuse AUC) / (Full AUC − Reuse AUC)`；
成本为方法相对 Reuse 的额外 FLOPs，除以 Full 相对 Reuse 的额外 FLOPs。
校准、选择和额外推理均计入对应方法账本。当前图表、逐点数据和结构图见
[图表索引](../figures/README.md)。

## 论文附录仍引用的历史证据

[原四预算10k用户AUC记录](../results/insight/unified_auc_10k_20260920/README.md)及
[扩展至7144教师的九预算记录](../results/insight/unified_auc_10k_teachers7144_20260920/README.md)
仍被论文附录引用，保留其输入、原始分数、全部预算和结论。它们使用旧的四边设置，
不作为当前三规模15边的执行入口，也不与当前Q/H指标混用。

其他历史记录及清理后的恢复边界见[结果索引](../results/README.md)。
当前 Cross-Version Cache Adaptation 尚未实现和验证；以上 Motivation 结果
不等于连续跨发布适配已成立。后续工作见[设计计划](design/plan.md)。
