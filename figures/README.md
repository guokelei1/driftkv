# 图表入口

论文采用的图表以 [paper/main.tex](../../paper/main.tex) 为准。Python 生成器位于 `figures/src/`，只读取保留结果，不运行模型训练或评估。

| 内容 | 生成器或排版来源 | 数据 |
| --- | --- | --- |
| Motivation：更新改善与旧 cache 损失 | [src/motivation1.py](src/motivation1.py) | Medium D14 完整三角与相邻裁决 |
| Insight 1：局部替换曲线 | [src/insight1.py](src/insight1.py) | `results/yambda500m_medium_seed17/insight1_locality_v1/analysis/` |
| Insight 2：响应修正表 | [paper/main.tex](../../paper/main.tex)，`tab:insight2-response` | `results/yambda500m_medium_seed17/insight2_functional_boundary_v1/` |
| 基础适配质量与费用图 | `figures/src/` 中保留的 Design 生成器 | `results/design/analysis/` |

旧 Design 2/3 的图表生成器、预览和输入结果已随其实验树删除；论文中的重构后 Design 2/3 由正文维护。
