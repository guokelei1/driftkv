# 合同索引

本目录的 YAML 合同及其哈希全部保留、不可覆盖。这个 README 只负责导航。

- yambda500m_medium_*：六层模型训练、Full/Reuse、旧 producer 对比和两个 Insight。
- yambda500m_large_*：十层模型训练、Full-only、epoch 对照与 canonical 工作序列。
- yambda500m_small_*：四层历史实验边界，部分仍被基础工具或 Large 合同引用；
  旧实验权重与原始结果已删除，不能裸跑这些历史合同。
- scale population / streaming windows / unified scales：现行数据预处理边界。
- legacy_pointwise 和已退休的 Insight 合同：保留原算子事实、失败与否定结论，不作为新方法合同。

当前入口由 [scripts/README.md](../../scripts/README.md)指定。
下一阶段实验见 [experimental_design.md](../../docs/experimental_design.md)。
论文的新方法尚未运行 translator calibration；连续迁移也需要独立协议。
旧合同存在不等于授权启动，也不表示其历史结果文件仍在工作区。
结果保留边界见 [结果索引](../../results/README.md)。
