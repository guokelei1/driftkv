# 六层适配迭代记录

当前 Design 已完成设计，尚未实现和验证。本文件保留历史探索的发现和限制，供后续实现参考，不代表当前 Design 的验证结论。后续将重新训练模型并重做实验；旧结果不替代新验证。

## 已知结论

- HSTU-native 读出在冻结六层开发模型上有质量信号；证据见 [native 基础质量报告](../../results/design/analysis/native_base_quality4091_01_report/report.md)。
- 查询留出、native 输入和解码闭环控制仍是判断机制是否有效的必要诊断，分别见 [查询留出](../../results/design/analysis/mechanism_query_holdout192_report_01/report.md)、[native 输入](../../results/design/analysis/mechanism_native_input192_report_01/report.md) 和 [解码闭环](../../results/design/analysis/mechanism_decoder_closure192_report_01/report.md)。
- 开发点的计算组成已单独测量，见 [FLOPs 报告](../../results/design/analysis/native_flops_01/report.md)。高成本且恢复弱的变体没有进入后续范围。
- 兼容性与模型准入分开处理：候选版本不满足 H/S 时，服务父版本和缓存谱系保持不变。
- 只有实际跨发布、混合生产者、持续追加和淘汰的状态寿命才能支持连续适配主张。

## 当前约束

开发校准可以使用摘要或 K/V 派生监督，但必须与最终评价隔离，并报告教师访问及其成本。评价使用冻结协议和所有训练种子；不得以规模结果回调工作负载、版本边、指标或方法接口。

当前开发阶段已收束到 [计划](plan.md) 中的四个组件。任何新的规模训练先满足合同、canary、资源估计和明确启动要求。
