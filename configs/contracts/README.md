# 合同索引

本目录仅保留当前训练、论文数据生成和其冻结证据闭包所需的 YAML 合同。这个 README 只负责导航。

- yambda500m_medium_*：六层模型训练、Full/Reuse、旧 producer 对比和两个 Insight。
- yambda500m_large_*：十层模型训练、Full-only、epoch 对照与 canonical 工作序列。
- Small 合同及其 PRO 分支已退休。旧 Large v1 合同中的 Small PRO 字段只保留为
  已封存的历史元数据；当前 Full-only 执行器不读取它们。
- scale population / streaming windows / unified scales：现行数据预处理边界。
- 当前 Insight 合同只保留 locality、functional boundary、temporal persistence 及其
  temporal-coefficient 证据输入。

当前入口由 [scripts/README.md](../../scripts/README.md)指定。
本阶段开发见 [design/plan.md](../../docs/design/plan.md)，技术定义见
[experimental_design.md](../../docs/experimental_design.md)。六层原型已运行共享Translator校准与开发评价，尚未达标或确认。
用户已撤销笼统 target-KV fitting 禁令，小规模共享校准与连续原型按计划记录配置；
正式人口评价和长训练再落实相应协议、资源、canary 与明确启动。旧禁止拟合条款只解释其原实验。
旧合同存在不等于授权启动，也不表示其历史结果文件仍在工作区。
结果保留边界见 [结果索引](../../results/README.md)。
