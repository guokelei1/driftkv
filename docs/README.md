# 文档地图

更新日期：2026-09-06

## 当前阅读顺序

| 要解决的问题 | 入口 |
| --- | --- |
| 论文现在如何定义方法 | [论文正文](../../paper/main.tex) |
| 方法结构和实现边界 | [paper_design.md](paper_design.md) |
| 下一阶段实验方案草稿（待讨论） | [experimental_design.md](experimental_design.md) |
| 已有测量及其证据范围 | [motivation_observations.md](motivation_observations.md) |
| 六层模型的已完成训练与评估 | [medium_scale_training_plan.md](medium_scale_training_plan.md) |
| 十层当前模型指针及历史对照 | [large_scale_training_and_qualification_plan.md](large_scale_training_and_qualification_plan.md) |

论文正文是设计文字的唯一来源；paper_design 是研究实现的简要映射，experimental_design
记录技术和执行约束。旧的独立 Design 1 中文长稿、设计候选和专家讨论已退出活动目录，
不再维护第二套方法定义。motivation_observations 现只索引当前论文证据，
原来的长篇 Small 观察文档已完整归档；原始裁决不会因目录清理而改写。
两份训练记录中的早期预算、阶段和命令用于解释已有资产，不是当前待办。
experimental_design 中的训练边界与首版实现约定仍待讨论，本轮状态整理不替代这些决定。

开发与实验遵循 [AGENTS.md](../AGENTS.md) 的论文研究原则：优先用最小实现与小样本
验证想法，只做与当前问题相关的检查。已有计划中的阶段是研究问题和历史记录，
不要求每次探索先完成整套工程实现。配置和结论尽量更新现有入口，不为小改动另建
清理报告、审计文档或重复合同；测试选择见 [tests/README.md](../tests/README.md)。

## 最新论文的结构

Design 为 **Cross-Version Cache Adaptation**，包括 System Overview、Cache Summarization、
Release-Time Translation、Read-Time Correction、State Maintenance Across Releases。
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
