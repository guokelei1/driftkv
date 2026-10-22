# E1：四配置质量—完整成本前沿（模拟，当前 Figure 6）

2026-10-21 按最新 `paper/main.tex` 更新：双栏 **1×4** 面板，依次为 Yambda-Large、Yambda-Max、RecFlow-Large、RecFlow-Max。画布为 **7.05×1.43 英寸**；裁边后 PDF 高度约为原十二面板版本的 50%，以四幅等宽面板和顶部双行公共图例缩短整体高度。轴刻度 6.8 pt、图例 6.9 pt、标题 7.5 pt，PDF 保留矢量文字。

## 图中信息

横轴为完整新增 FLOPs／同版 Rebuild 预算，纵轴为绝对反馈 AUC。各面板独立确定 AUC 范围，配置名称直接放在小标题中。四条方法曲线是 CacheTide、KV map（数据键 `Cache translation`：每次发布映射一次并存储，对应论文 3.3 节的 KV 映射），以及 deviation-guided 和 query-guided 两种选择性重算；query-only 等其余方法仍保留在数据文件中，不再绘制。曲线连接相同预算档的跨版本均值，小型浅色标记叠加 V3—V5 的逐版观测。Reuse 用空心圆、Rebuild 用实心方，各为跨版本均值的单个参考点；CacheTide 默认预算档加外圈。不把参考点连接成曲线。

## 数据与范围

- `frontier_data.json` 是本图的新模拟数据源，AUC 锚点对齐最新正文中 V3—V5 的模拟表；所有方法值均为人为构造的版式示例，不是实验结果，也没有捏造置信区间。
- `data.json` 是历史十二面板数据，保留不改，以免改变其他预览脚本的输入。当前 `plot.py` 不读取它。
- 正式数据应保留实际前沿，不能选择或调整真实实验去贴合模拟趋势。均值 AUC、完整成本和恢复率仍须按论文的最终报告规则分别计算。
- 图内无模拟水印，模拟标识保留在论文 caption、文件名、脚本及导出元数据。

```bash
python eval_figure/e1_quality_cost/plot.py
```

输出 `fig_6_1_simulated.png` 与矢量 PDF；论文对应 `pic/evaluation/{png,pdf}/fig_eval_quality_cost_simulated.*`，继续采用 `figure*` 和 `width=\textwidth`。本轮只重绘并导出该图，不重绘其他图，不运行实验。
