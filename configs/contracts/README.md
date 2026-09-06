# 合同索引

本目录的 YAML 合同及其哈希全部保留、不可覆盖。这个 README 只负责导航。

- yambda500m_medium_*：六层模型训练、Full/Reuse、旧 producer 对比和两个 Insight。
- yambda500m_large_*：十层模型训练、Full-only、epoch 对照与 canonical 工作序列。
- yambda500m_small_*：四层历史实验边界，部分仍被基础工具或 Large 合同引用；
  旧实验权重与原始结果已删除，不能裸跑这些历史合同。
- scale population / streaming windows / unified scales：现行数据预处理边界。
- legacy_pointwise 和已退休的 Insight 合同：保留原算子事实、失败与否定结论，不作为新方法合同。
- [evokv_design_v3_quality_development_v1.yaml](evokv_design_v3_quality_development_v1.yaml)：
  当前六层Design冻结译者的128用户开发质量评价，含资源预算、canary和原校准成本引用；不是confirmation。
- [evokv_design_v4_full_history_development_v1.yaml](evokv_design_v4_full_history_development_v1.yaml)：
  全部实际保留状态的响应校准与512用户五边开发轨迹，含历史回放成本和长作业预算；不是confirmation。
- [evokv_design_v4_cal512_v1.yaml](evokv_design_v4_cal512_v1.yaml)：
  同一v4扩至512校准用户，批量计算同一共享响应目标；固定开发用户检验，不读取确认。
- [evokv_design_v4_score512_v1.yaml](evokv_design_v4_score512_v1.yaml)：
  同一v4 source投影上增加共享decoder闭环输出蒸馏，全部五边开发校准；教师和拟合均计成本。
- [evokv_design_v5_temporal_development_v1.yaml](evokv_design_v5_temporal_development_v1.yaml)：
  原生时间基加权view与同一512用户五边轨迹；四组因果时间校准，批量发布与Exact采用相同batch。
- [evokv_design_v5_runtime_development_v1.yaml](evokv_design_v5_runtime_development_v1.yaml)：
  相同v5数学与数据的运行实现优化，重测完整教师/校准与真实生命周期成本。

当前入口由 [scripts/README.md](../../scripts/README.md)指定。
本阶段开发见 [design/plan.md](../../docs/design/plan.md)，技术定义见
[experimental_design.md](../../docs/experimental_design.md)。六层原型已运行共享Translator校准与开发评价，尚未达标或确认。
用户已撤销笼统 target-KV fitting 禁令，小规模共享校准与连续原型按计划记录配置；
正式人口评价和长训练再落实相应协议、资源、canary 与明确启动。旧禁止拟合条款只解释其原实验。
旧合同存在不等于授权启动，也不表示其历史结果文件仍在工作区。
结果保留边界见 [结果索引](../../results/README.md)。
