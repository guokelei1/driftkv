# 图表入口

所有 Python 绘图代码集中在 `figures/src/`，直接读取已有结果，输出到 `figures/pic/`。
实验执行与裁决仍在 `scripts/`；改图无需运行训练、评估或 discovery。
论文采用的图表以 [paper/main.tex](../../paper/main.tex) 为准。

| 内容 | 生成器或排版来源 | 数据 |
| --- | --- | --- |
| Motivation：更新改善与旧 cache 损失 | [src/motivation1.py](src/motivation1.py) | Medium D14 完整三角与匹配的相邻裁决 |
| Insight 1：论文四面板 locality 曲线 | [src/insight1.py](src/insight1.py) | [best_observed_by_edge.csv](../results/yambda500m_medium_seed17/insight1_locality_v1/analysis/best_observed_by_edge.csv) |
| Insight 2：响应修正表 | [paper/main.tex](../../paper/main.tex)，`tab:insight2-response` | [analysis_v2/frontier.csv](../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/discovery_functional_boundary/analysis_v2/frontier.csv) |
| Insight 2：持续性文字结果 | [paper/main.tex](../../paper/main.tex)，`sec:insight-two` | [持续性报告](../results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1/discovery/analysis/report.md) |
| 重算成本表 | [paper/main.tex](../../paper/main.tex) | [计时报告](../results/release_cost_random_weight_v1/report.md)；测量入口为 scripts/benchmark_release_cost.py |
| Design 架构图 | 直接定义于 [paper/main.tex](../../paper/main.tex) | 第 4 章五小节 |

在仓库根目录运行（需要 Matplotlib、NumPy）：

```bash
python figures/src/motivation1.py
python figures/src/insight1.py
```

Motivation 输出 `figures/pic/pdf/motivation1.pdf` 和 `figures/pic/jpg/motivation1.jpg`。
Insight 1 论文图输出 `figures/pic/pdf/insight1_locality.pdf`，可用 `--output` 指定 PDF，
或 `--preview /tmp/insight1-preview.png` 额外生成预览。
脚本的默认输入输出均相对仓库位置解析，可从其他工作目录调用。

Insight 1 论文图校验原 CSV 的 SHA-256，显示 V0→V1、V1→V2、V2→V3 及三者平均值，
保留四块 target area、关键点标注和 layer-only 减五个百分点的显示约定。
完整五条更新的数据及裁决保持原路径。旧 `paper/pic/plot_insight1.py` 已迁入本目录。
Locality 统一使用这一个生成器；原结果目录中的诊断图片保留为历史成品。
Insight 2 当前是 LaTeX 表格和文字，没有独立绘图脚本；只使用修正后的 analysis_v2。

论文仍从 `paper/pic/` 引用 PDF。确认新图后，从仓库根目录同步所需成品：

```bash
cp figures/pic/pdf/motivation1.pdf ../paper/pic/motivation1.pdf
cp figures/pic/pdf/insight1_locality.pdf ../paper/pic/insight1_locality.pdf
```

绘图脚本不会自动修改论文目录或编译论文。
Git 保留绘图源码、封存的小型输入和论文 PDF；JPG/PNG 预览与历史诊断图片保留在本地。
