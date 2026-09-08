# 文档地图

更新日期：2026-09-07

## 当前阅读顺序

| 要解决的问题 | 入口 |
| --- | --- |
| 论文现在如何定义方法 | [论文正文](../../paper/main.tex) |
| 方法结构和实现边界 | [paper_design.md](paper_design.md) |
| 最新专家路线：基础方法定稿与冻结 | [专家路线、完整结果与停止决定](design/expert_route_2026-09-07.md) |
| Design 1设计依据、统一机制对照与逐发布预算 | [基础方法定稿报告](../results/design/analysis/base_method_final_01/report.md) |
| 论文方法与基础实验的实际章节 | [完整英文正文](../../paper/main.tex)；[最新中文设计草稿](../../paper/draft/evokv_system_design_zh.md) |
| 理论计算、3万/10万/100万外推与Exact配对质量 | [增量FLOPs独立报告](../results/design/analysis/native_flops_01/report.md) |
| 最新真实质量与两个消融，供专家讨论 | [4091人基础真实质量报告](../results/design/analysis/native_base_quality4091_01_report/report.md) |
| 前轮native校准覆盖与连续状态边界 | [Native三臂校准报告](../results/design/analysis/native_coverage384_report_01/report.md) |
| 本阶段历史计划、执行边界与停止状态 | [design/plan.md](design/plan.md) |
| 当前方案、探索经过与失败记录 | [design/iterations.md](design/iterations.md) |
| 方案0–15的论文设计复盘、弱点与未来方向 | [研究总结](design/research_summary_2026-09-07.md) |
| 代码、实验执行、成本账本与停止点 | [工程总结](design/engineering_summary_2026-09-07.md) |
| 实验公式、资产与评价定义 | [experimental_design.md](experimental_design.md) |
| 已有测量及其证据范围 | [motivation_observations.md](motivation_observations.md) |
| 六层模型的已完成训练与评估 | [medium_scale_training_plan.md](medium_scale_training_plan.md) |
| 十层当前模型指针及历史对照 | [large_scale_training_and_qualification_plan.md](large_scale_training_and_qualification_plan.md) |

论文正文是设计文字的唯一来源；paper_design 是研究实现的简要映射，experimental_design
记录技术和执行约束。design/ 集中保存本阶段的落地计划和迭代记录，从六层完整 v0 开始，
再联动改进摘要、翻译、读取和连续维护，不另写平行论文。
旧的独立 Design 1 中文长稿、设计候选和专家讨论已退出活动目录，
不再维护第二套方法定义。motivation_observations 现只索引当前论文证据，
原来的长篇 Small 观察文档已完整归档；原始裁决不会因目录清理而改写。
两份训练记录中的早期预算、阶段和命令用于解释已有资产，不是当前待办。
用户已撤销笼统 target-KV fitting 禁令，明确连续优化为缓存跨多个版本的实际演化，
并要求初期也检验低计算、实质恢复的方向。决定与 review 见 design/plan.md 和迭代记录；
六层探索已覆盖方案0–15，最后四边适配的6000用户评价完成；按用户要求，在两份总结后
收束上一轮goal。用户随后提供的专家路线已完成查询留出、匹配二乘二与一次丰富源
预算诊断：表示证据强，共享方法仍未解决困难边，因此停止本次模型实验，没有进入
新的6000人评价；不恢复旧横向变体搜索。结果见专家路线末节，仍未通过质量与完整成本联合验证。
上述为早期探索历史。后续native C完成4091人成熟域真实质量与条件FLOPs核算，
第六次专家路线已完成基础方法定稿、统一常量/仿射机制对照及逐发布费用整理，Design 1冻结。
完整叙述现直接写入论文main.tex；用户提供的中文草稿保存于paper/draft/，
原分节文件仅归档于draft/archive/，不再被正文引用；不自动开启第二设计。
结果属于开发证据，未读取独立确认。实际冻结Medium为legacy ELU+1，读取严格复用其算子；
不能将其方法结果描述为SiLU-native模型验证。

开发与实验遵循 [AGENTS.md](../AGENTS.md) 的论文研究原则：优先用最小实现与小样本
验证想法，只做与当前问题相关的检查。已有计划中的阶段是研究问题和历史记录，
不要求每次探索先完成整套工程实现。配置和结论尽量更新现有入口，不为小改动另建
清理报告、审计文档或重复合同；测试选择见 [tests/README.md](../tests/README.md)。

## 最新论文的结构

Design 为 **EvoKV System Design**，按系统总览、缓存摘要、发布转换、评分与维护、
理论计算开销组织，正文约1400词。方法章、基础评价和质量表均直接维护在paper/main.tex，
配置、维度及求解目标集中在Implementation，PCA/回放及逐项算术放同文件附录；
不再使用sections/。
主文图表为常量/仿射公共query对照、方法流程、基础质量表及理论规模图。
不要再将当前方法命名为 Sketch-to-Sketch，也不要把所有组件都冠以 Sketch。

Motivation 使用六层 Medium 的五次更新与完整旧 producer 对比。
Insight 1 的论文图仅显示前三次更新及其平均值，保留原来的 layer-only
减五个百分点显示口径；底层五条更新的原始数据仍保留。
Insight 2 使用修正后的 analysis_v2 和 14 天持续性诊断。
具体生成器及路径见 [图表索引](../figures/README.md)。

## 资产与执行

模型以原有 manifest 和 checkpoint 路径为准，不复制、不重命名。
十层 Large 的 V4/V5 使用两 epoch 的当前工作序列，六层 Medium 各版本均为一 epoch。
Large 序列的 post-hoc 范围说明保留，不能混作独立 qualification。

当前入口见 [scripts/README.md](../scripts/README.md)，证据见
[results/README.md](../results/README.md)。
历史删除清单、归档位置及恢复限制统一维护在结果索引中。
历史合同及结果是审计记录，不是当前执行清单；不得据其存在启动旧实验。
