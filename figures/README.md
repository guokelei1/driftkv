# 图表入口

保留两套已有图表结果，旧的 motivation／insight preview 输出已清理。

## 四种选择性重算 baseline（2026-09-26 已完成）

本轮生成器为 [selective_recompute_2026_09.py](src/selective_recompute_2026_09.py)，
读取[独立实验目录](../results/selective_recompute_2026_09/README.md)中的实际结果。
正式评测已输出[四种方法 × 三种规模的12张图](out/selective_recompute_2026_09/)，
每张包含五条相邻版本曲线，每个预算点均保留。下列已有图表不代表这四个新 baseline 的结果。

## 五方法横向总图

[five_method_figures](out/five_method_figures/) 包含最终五方法图：每张图选择 6L、10L、
16L 对应的版本线并计算平均线；横向总图为
[five_methods_horizontal.pdf](out/five_method_figures/five_methods_horizontal.pdf) 和
[five_methods_horizontal.png](out/five_method_figures/five_methods_horizontal.png)。
选择记录在 `selection.json`，整理后的绘图数据在 `selected_points.csv`，生成器为
[five_method_selected.py](src/five_method_selected.py)。

## 三规模 15 子图

[three_scale_auc_preview](out/three_scale_auc_preview/) 保留 6L、10L、16L 三个规模的
15 张 method 子图、每规模横排图和
[combined_15panels.pdf](out/three_scale_auc_preview/combined_15panels.pdf)。
完整理论 FLOPs 和 AUC ledger 在 `cost_ledger.json`，显示点在 `display_points.csv`，
这些文件是本次三规模评测的封存图表数据。

原始模型训练、评测和 motivation／insight 证据仍保存在 `results/`；本目录只保留上述
两套当前图表结果及其绘图所需数据。
