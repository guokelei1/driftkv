# 图表入口

本目录保存当前实验结果图、论文结构示意图及对应生成器。生成器只读取已有结果。

## Design 1：紧凑历史编码与读时校正

[结构图 PDF](out/design_one/read_correction.pdf)、
[PNG 预览](out/design_one/read_correction.png)和
[可编辑 SVG](out/design_one/read_correction.svg)由
[生成器](src/design_one/read_correction.py)绘制。图为 7 × 2.75 英寸，
在论文中跨双栏放置，标签沿用 `fig:personalized-read-correction`。
PDF/PNG 同步到 `paper/pic/fig_personalized_read_correction.*`；
编辑和生成的唯一源仍在本仓库。

横向四步依次为：① 编码每条历史，② 保留并复用窄编码，③ 在注意力内校正读取，
④ 根据 query 和已修正读响应细化残差。前两步归属初始化/追加的历史准备，
后两步针对每次 query；键侧先投影 query、值侧先聚合编码再投影是第三步的重点。
正文四个编号段落与图标题同序；存储和追加/淘汰集中在第②步，共享校准作为末尾说明。
图中的物品输入、修正权重和响应模型分别标为 `e_i`、`α_i`、`g_ψ`，
与正文的 `Z`、`Δr`、修正后读取及最终读取符号对应。
橙色标识编码，蓝色标识查询/读取，灰色标识原生 K/V。编号、模块名和形状同时表达
含义，不依赖色阶区分。与 Motivation 逐请求编码完整历史的探针相比，本图强调
复用非线性编码和读取内的投影重排。少量用户的共享拟合仅作辅助说明。
图中的条目和权重为机制示意，不是实测数据；完整公式和归一化见论文。
该图对应当前紧凑 Item64＋Response256 的 Design 1，不包含待验证的 influence
传播或完整跨版本生命周期。生成无需模型、GPU 或重跑实验：

```bash
python figures/src/design_one/read_correction.py
```

## 论文 3.2 相邻双图

[架构图](out/motivation_branches_2026_09_30/v6/motivation_branches.png)置于上方，
[六联实验图](out/motivation_six_methods/quality_cost.png)置于下方；
[生成器](src/motivation_six_methods.py)只汇总四种重算方法和当前 Q-v5/H-v4 的已有结果。
论文 `/home/gkl/work/paper/main.tex` 中两图分别编号、分别配图注，
标签为 `fig:motivation-methods-architecture` 与 `fig:motivation-methods-quality-cost`。
同一个跨栏浮动体保证上下相邻，第一张图注后留 8 pt 间隔；
满正文宽放置时，含图注预计占正文页高的 40%–45%。当前是版式预览，尚未另选最终展示点。

曲线为 **Mean、Medium、Large、Max**，其中 Q correction 子图不显示 Max。
Mean 使用加粗深红线；Q correction 与 Q＋history 在空白处增加带坐标的局部放大框，
分别展示成本 1%–6% 与 59%–87% 的区域，主图保留统一成本尺度。
放大框采用白底，边框、刻度文字与指示箭头统一为蓝灰色，与黑色主坐标轴区分；
仅用一根细箭头指示来源区域，与主轴刻度留出间隔。
各规模等权平均五条相邻边，
Mean 等权平均全部十五条边；没有总体中位数线。
相同方法配置下的横、纵坐标分别取均值，不按请求数加权，也不插值到共同成本。
Layer 按原计划的层比例对齐，取 `ceil(层数 × 比例)`；Medium 前两个比例使用同一个已测层数点。
绘图纵轴从 0% 开始，负的汇总恢复率仅在展示时按 0% 绘制。
原始结果与汇总点表仍保留全部预算及负值，Mean 在裁剪前按全部十五条边计算，包含未显示的 Max。
Full/Reuse 只定义归一化端点，不作为方法曲线的补点。
[汇总点表](out/motivation_six_methods/aggregate_points.csv)和
[来源记录](out/motivation_six_methods/manifest.json)保存对应关系。

```bash
python figures/src/motivation_six_methods.py
```

## 3.2 Query／History 修正结果

- [当前 Q/H 总图](out/read_correction_2026_09/motivation_final/README.md)：Q-v5 非线性
  C128/256/512 与 H-v4 非线性 C64/128/256，保留全部 15 条边、90 个点及总体/分规模均值。
- [完整探索对照图](out/read_correction_2026_09/v5/overview_15panels.png)：保留 135 个点，
  包含当前 Q、H，以及原有 Q-v2、低成本 H-v1 对照。
- 生成器为 [当前 Q/H](src/read_correction_motivation_2026_09.py) 与
  [综合对照](src/read_correction_v5_2026_09.py)，共用
  [结果读取工具](src/read_correction_common.py)。均只读取已有结果。

当前两套输出分别承担选定方法展示与完整探索对照；原始来源见
[Q/H 结果索引](../results/read_correction_2026_09/README.md)。

## Motivation 两类方法结构图（2026-09-30）

当前紧凑左右版为 [PNG](out/motivation_branches_2026_09_30/v6/motivation_branches.png)、
[PDF](out/motivation_branches_2026_09_30/v6/motivation_branches.pdf) 和
[可编辑 SVG](out/motivation_branches_2026_09_30/v6/motivation_branches.svg)，
生成器为 [draw_v6.py](src/motivation_branches_2026_09_30/draw_v6.py)，
左侧模块为 [compact_recompute_v6.py](src/motivation_branches_2026_09_30/compact_recompute_v6.py)。
画布为 7 × 1.65 英寸；左侧四种选择策略排成一行，Layer/Tail 较窄，
Deviation/Query-guided 较宽；右侧并列展示 Q/H，共性内容放在各组上方。
两侧标题及公共重算流程分别居中。策略卡片上方说明选择依据，下方红字说明选择结果；
Deviation/Query 使用第一层 K/V 变化与注意力幅值，评分柱和被选位置对应。
Deviation 对照同一位置的新旧 K/V，经过差值比较得到各位置分数；
灰色/浅蓝色区分两组输入，只有输出中最高的分数使用红色。
Query-guided 明示 query 与继承 K 两个输入共同进入第一层 attention，再输出分数并选 top-k；
输入 K 不预先着色，只高亮输出中最高的分数，表示先评分再选择。
右侧以 `Correction model` 命名拟合结果，用文字标明少量用户，并突出旧读取加修正量的新读取路径。
四种选择策略、少量用户校准和 Q/H 读时路径均为机制示意，
掩码与注意力权重不表示实测数值。右侧将少量用户拟合、两种输入选择与统一的读更新串联。
Q 的修正量由 query 非线性特征预测；H 的修正量表示映射 K/V 后的读取相对原读取的变化。
共同加法是读出层的等价表达；H 实际直接读取映射后的 K/V，不增加残差计算过程或成本。
两者分别拟合且不改写持久缓存，具体图注见 [caption.txt](out/motivation_branches_2026_09_30/v6/caption.txt)。
该目录中的 v6 是当前选定结构图。

## 四种选择性重算 baseline（2026-09-26 已完成）

本轮生成器为 [selective_recompute_2026_09.py](src/selective_recompute_2026_09.py)，
读取[独立实验目录](../results/selective_recompute_2026_09/README.md)中的实际结果。
正式评测已输出[四种方法 × 三种规模的12张图](out/selective_recompute_2026_09/)，
每张包含五条相邻版本曲线，共 60 条曲线、295 个方法测量点；
Full/Reuse 另作为 120 个参考点，每个原始预算点均保留。

## 论文依赖的历史图表数据

旧五方法横向图、modified版本和三规模15子图的PNG/PDF/SVG及旧生成器已退休。
保留其数据和显示选择：

- `out/five_method_figures/`、`out/five_method_figures_modified/`：
  原selection、点表、说明及来源记录。
- `out/three_scale_auc_preview/cost_ledger.json` 与 `display_points.csv`：
  旧三规模AUC及计算量账本。

论文附录仍描述的旧10k/7144实验，其raw、summary与封存哈希保留于
[历史结果目录](../results/insight/unified_auc_10k_teachers7144_20260920/README.md)；
3.1旧Medium质量图和随机权重成本表的证据也保留，见[结果索引](../results/README.md)。
这些历史记录与本页当前四重算/Q/H输出分开。论文目录中的现用PDF未改动。
