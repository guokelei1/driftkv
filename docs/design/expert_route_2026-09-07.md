# 六层机制证据

当前开发问题是：翻译后的历史状态能否在当前 HSTU 查询下恢复 Full 相对 Reuse 的质量缺口，并以远低于 Exact-All 的成本运行。

机制评价保留三项互补控制：

- [查询留出](../../results/design/analysis/mechanism_query_holdout192_report_01/report.md) 检查修正是否依赖当前查询；
- [native 输入](../../results/design/analysis/mechanism_native_input192_report_01/report.md) 检查实现路径是否真正使用 HSTU-native 输入；
- [解码闭环](../../results/design/analysis/mechanism_decoder_closure192_report_01/report.md) 检查读出与状态表示的闭环。

[聚合机制报告](../../results/design/analysis/mechanism_aggregate_factorial192_report_01/report.md) 记录当前可复查的联合证据。它们用于缩小接口选择，不替代 Full-only 准入、连续状态评价或规模质量结论。
