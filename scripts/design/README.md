# 历史六层适配探索脚本

当前 Design 已完成设计，尚未实现和验证。本目录保留历史 native read-adaptation 数据生成链，供后续复用，不代表当前 Design 的实现或验证。下述校准和结果均指历史探索。

旧 runner 的 query-affine 校准已退休；仅保留通过 `--evaluation-from` 读取既有权重的评价路径，不再尝试导入已删除的拟合模块。

这些历史运行读取冻结的
Medium 模型、manifest 与已有事件；长训练仍由仓库根目录的 Medium/Large 入口负责。

`data.py`、`run.py`、`diagnose_query_holdout.py`、`diagnose_summary_objective.py` 和
`diagnose_decoder_closure.py`提供当前校准所需的场景、面板、坐标和共享执行原语。
`diagnose_native_input.py`、`diagnose_native_coverage.py`、`fit_native_ablation.py` 与
`native_service.py`生成 native C 和两个必要消融。

`evaluate_native_base.py`、`run_native_base.py`、`report_native_base.py`、
`report_native_coverage.py`、`report_native_flops.py`、`audit_native_residual.py` 与
`audit_native_response_scale.py`产生当前的质量、覆盖、残余和计算证据。
`diagnose_constant_affine.py` 与 `finalize_base_method.py`只读取冻结结果，生成论文
Design 1 的机制图表输入和定稿索引。

历史 v3--v15、PRO、宽源及已删除运行的入口不再保留。当前结果范围见
`results/design/analysis/`，论文叙述以 `/home/gkl/work/paper/main.tex` 为准。
