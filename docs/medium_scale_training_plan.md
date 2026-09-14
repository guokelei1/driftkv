# Medium 六层训练与评价

Medium 是当前方法开发的固定规模。保留六层模型、其依赖脚本、合同和必要 checkpoint；不以历史小模型资产补充当前流程。

训练、版本准入和 Full/Reuse 评价的可执行入口及依赖关系见 [scripts README](../scripts/README.md)。模型和数据协议以 [实验设计](experimental_design.md) 为准，适配接口以 [六层计划](design/plan.md) 为准。

每条版本边先封存 Parent/Current 的 Full-only 结果并决定是否准入；只有准入后才评价 Reuse。开发结果按训练种子完整报告，保留原始证据、哈希和裁决文件。任何新的 Medium 长训练都需要前瞻合同、资源估计、通过 canary 与用户明确启动。
