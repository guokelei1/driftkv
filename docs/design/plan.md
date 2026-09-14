# 六层跨版本缓存适配计划

状态：当前 Design 已完成设计，尚未实现和验证。已有代码与下列开发证据属于历史探索。后续计划重新训练所有模型并重做全部实验；新运行设置与启动另行确定，本文件的冻结六层范围描述现有开发资产。

当前方法是 Cross-Version Cache Adaptation：旧模型在写入时生成紧凑摘要；新版本发布时翻译摘要；读取旧缓存时修正 K/V；状态层维护跨版本缓存的来源、追加与淘汰。论文叙事以 [`/home/gkl/work/paper/main.tex`](/home/gkl/work/paper/main.tex) 为准，具体模型与评估协议见 [实验设计](../experimental_design.md)。

## 固定范围

- 开发使用冻结的六层 Medium 模型；训练种子是重复单位。
- 每条版本边先完成 Parent/Current 的 Full-only 准入，再解锁 Reuse 评价。
- 低 H/S 的候选版本保持 No-op，缓存谱系不变。
- 工作负载、发布窗口、历史、指标、动作集合和探针率在评价前冻结。

## 方法接口

1. **写入摘要**：从父版本的因果状态提取固定大小、可追加的摘要。
2. **发布翻译**：将旧摘要映射为当前版本可消费的表示；校准与最终评价分离。
3. **读取修正**：在 HSTU 聚合路径中结合当前 query、翻译表示和旧 K/V，生成兼容读出。
4. **状态维护**：跨发布保留混合生产者的缓存、后续追加、淘汰与继承误差。

首个原型必须同时包含这四个接口。一次重新初始化的单边实验不能代表连续适配。

## 当前证据与下一步

六层开发证据表明 native 读出路径有可测质量信号，见 [质量报告](../../results/design/analysis/native_base_quality4091_01_report/report.md)；其成本组成见 [FLOPs 报告](../../results/design/analysis/native_flops_01/report.md)。基础方法与机制控制分别记录在 [基础报告](../../results/design/analysis/base_method_final_01/report.md)、[覆盖率报告](../../results/design/analysis/native_coverage384_report_01/report.md) 和 [查询留出报告](../../results/design/analysis/mechanism_query_holdout192_report_01/report.md)。这些开发结果不构成规模结论。

下一次实验只在完整连续状态、冻结的版本准入和明确的 Exact-All 成本对照下检验缺口恢复。Medium 或 Large 的长训练仍需前瞻合同、资源估计、通过 canary 与用户明确启动。
