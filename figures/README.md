# 图表入口

Design 3当前跨路径实验的读取计时采用论文`tab:execution-read`，直接来自
`results/design3/cohort_01/summary.json`与`analysis.json`；历史绑定与初版对照保留在`binding_01`/`initial_01`。
系统图的已有生成器`../paper/scripts/plot_system_architecture.py`已同步实测执行组件，
输出`../paper/pic/system_architecture.pdf`；未编译论文。

Design2当前机制图由`src/design2/bounded.py`读取`results/design2/bounded_01/analysis`与旧扩展对照生成，
输出`results/design2/bounded_01/figures/mechanisms.pdf`及PNG，同步论文`pic/design2_bounded.pdf`。
图同时保留保持旧触发响应的负对照及完整费用，人口点只作同自然负载外推。

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
| Design 1机制图、流程图、理论规模图及质量表 | [src/design1.py](src/design1.py) | [基础定稿](../results/design/analysis/base_method_final_01/report.md)、[质量](../results/design/analysis/native_base_quality4091_01_report/report.md)、[FLOPs](../results/design/analysis/native_flops_01/report.md) |
| Design 2检测开发曲线（未进入论文） | [src/design2/detection.py](src/design2/detection.py) | results/design2/analysis/detection_01/summary.json |
| Design 2几何增量及总费用下界（未进入论文） | [src/design2/budget_audit.py](src/design2/budget_audit.py) | results/design2/analysis/budget_audit_01/summary.json |
| Design 2残余校准、续用风险及组覆盖（未进入论文） | [src/design2/scale_calibration.py](src/design2/scale_calibration.py) | results/design2/analysis/scale_calibration_01/summary.json |
| Design 2固定32方向分级占比与费用（未进入论文） | [src/design2/tiered32.py](src/design2/tiered32.py) | results/design2/analysis/tiered32_01/summary.json |
| Design 2精确225维状态收缩费用与候选数边界（未进入论文） | [src/design2/conditional225.py](src/design2/conditional225.py) | results/design2/analysis/conditional225_01/summary.json |
| Design 2真实重建闭环与两个策略备选（未进入论文） | [src/design2/lifecycle.py](src/design2/lifecycle.py) | results/design2/analysis/lifecycle_variants_01/summary.json |
| Design 2 Benchmark成本目标与服务费用（未进入论文） | [src/design2/benchmark.py](src/design2/benchmark.py) | results/design2/analysis/benchmark_01/summary.json |
| Design 2 2048UID、短压力、完整费用及正常M4退化（未进入论文） | [src/design2/scale_followup.py](src/design2/scale_followup.py) | results/design2/analysis/scale_followup_01/summary.json与benchmark.json |
| Design 2 计算机制分解、完整费用和预算推迟（未进入论文） | [src/design2/compute.py](src/design2/compute.py) | results/design2/analysis/compute_01/summary.json |
| Design 2 选择性更新：证据消融、全请求尾部及完整费用 | [src/design2/renewal.py](src/design2/renewal.py) | results/design2/evidence_01/analysis/evaluate.parquet；sparse_01/analysis/summary.json与full_reference.json |

在仓库根目录运行（需要 Matplotlib、NumPy）：

```bash
python figures/src/motivation1.py
python figures/src/insight1.py
python figures/src/design1.py
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
Design 2 的独立生成器只读取既有汇总，分别输出 figures/pic/design2/detection_01/
和 figures/pic/design2/budget_audit_01/；残余校准图输出 figures/pic/design2/scale_calibration_01/。
精确状态收缩图输出 figures/pic/design2/conditional225_01/，不重新执行模型或拟合。
真实闭环图输出 figures/pic/design2/lifecycle_01/，保留各边差异、配对区间及完整费用边界。
历史第二设计诊断图保持独立，不执行模型；当前论文选择性更新图的同步说明见下文。
Git 保留绘图源码、封存的小型输入和论文 PDF；JPG/PNG 预览与历史诊断图片保留在本地。

Design 1生成`design1_mechanism.pdf`、`design1_flow.pdf`、`design1_scale.pdf`及
`figures/tables/design1_quality.tex`，对应PNG预览在figures/pic/jpg/。
第六次专家路线已检查图示并同步三PDF到paper/pic、质量表到paper/tables；论文
按用户后续要求，main.tex现直接包含完整方法、实验与质量表正文，并引用三个PDF。
paper/tables保留生成表成品，但不再作为正文input；旧sections仅归档于draft/archive/。
机制图仅表示128已使用诊断UID的公共query聚合响应留出残余，不是AUC；质量表仅
4091成熟UID；规模图3万/10万/100万人口仅为固定校准及负载的计算外推。未编译论文。
# Design 2 Benchmark成本目标

`src/design2/benchmark.py`读取`results/design2/analysis/benchmark_01/summary.json`及旧闭环账本，
生成`pic/design2/benchmark_01/benchmark.png`/PDF。实测基线与“闭包重建”未验证费用假设分开标注；不运行模型。

`src/design2/renewal.py`生成`sparse_01/renewal.pdf`（快照证据/全请求尾部/费用三面板）及
`sparse_01/renewal_lifecycle.pdf`（按论文宽度绘制的生命周期尾部/费用双面板）。
后者同步到`../paper/pic/design2_renewal.pdf`，只读取既有结果，不重新拟合或执行模型。
“首次使用”线使用同0.5提交阈值的匹配时机消融；自然用户数实际2048，曲线人口仅为同负载外推。

最新`src/design2/expanded.py`读取`results/design2/scan_30k_01/analysis/summary.json`，
生成该结果目录`figures/expanded.pdf`与`special_auc.pdf`及PNG。前者同步到
`../paper/pic/design2_expanded.pdf`：左图富集2034UID分场景尾部、右图随机自然2048UID完整费用，
两者分母明确分开。后者保留短历史/旧谱系各边AUC与UID区间，不裁掉负结果。脚本不调用模型。
