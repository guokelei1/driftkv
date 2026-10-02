# EvoKV

EvoKV 研究 Transformer 推荐模型更新后，如何继续使用旧模型产生的持久化 K/V。
当前实验模型采用 HSTU，数据轨道为 Yambda 和独立的 RecFlow。

## 当前进展

- **模型与数据**：Yambda 的 Medium、Large、Max 各 V0–V5，共 18 个选定模型；
  RecFlow 的 4096 用户六层 A–F 链共 6 个选定模型，均已完成训练。
- **Motivation**：三规模 15 条相邻边的 Full/Recompute 与 Reuse、四种局部重算、
  非线性 Q-v5/H-v4 探索均已完成，保留完整结果及绘图代码。
- **Design**：论文已描述个性化读修正、写时影响摘要、状态维护与回退退出；
  Design 1 当前紧凑读修正在固定 Medium V4→V5 一万用户上恢复84.98%、新增FLOPs10.46%、额外持久状态16.67%，达到三项开发目标；
  校准占8.66%、服务新增计算占1.80%，仅保存每层每条64维历史编码。其他版本边和完整连续方法尚待验证。
  [开发计划](docs/design/plan.md)记录主线与边界，[执行入口](scripts/design_one/README.md)隔离新实验。

## 按任务查找

| 要找的内容 | 入口 |
| --- | --- |
| 数据处理与流式训练 | [训练链路](docs/unified_training_2026_09/README.md)、[执行配置](configs/unified_training_2026_09/README.md) |
| 模型版本、训练日期与 AUC 提升 | [18 模型固定清单](docs/unified_training_2026_09/model_versions.md) |
| Full/Recompute 与 Reuse 差距 | [15 条边的结果](results/unified_reuse_2026_09/README.md) |
| 四种局部重算与 Q/H | [Motivation 测量索引](docs/motivation_observations.md) |
| RecFlow 处理、训练与结果 | [开发协议](docs/recflow/plan.md)、[结果](results/recflow/README.md) |
| 论文当前设计与实现起点 | [论文](../paper/main.tex)、[实现边界](docs/paper_design.md) |
| 结果图、结构图和生成器 | [图表索引](figures/README.md) |

Full/Recompute 是当前模型按请求重算历史；Reuse 继承上一选定版本的缓存并随历史更新。
具体请求、时间语义和成本定义见[实验设计](docs/experimental_design.md)。

## 目录导航

| 目录 | 内容 |
| --- | --- |
| [docs/](docs/README.md) | 模型清单、实验定义、当前设置与设计记录 |
| [configs/](configs/README.md) | 执行配置和冻结合同 |
| [scripts/](scripts/README.md) | 数据准备、训练、评测与汇总入口 |
| src/hstu_kvcache/ | [模型](src/hstu_kvcache/models/README.md)、[数据](src/hstu_kvcache/data/README.md)、[重算方法](src/hstu_kvcache/baselines/README.md)及读取修正实现 |
| [results/](results/README.md) | 完整指标、原始证据来源与现存资产 |
| [figures/](figures/README.md) | 已有结果的绘图代码和当前图片 |
| [tests/](tests/README.md) | 数值、因果、聚合和相关回归检查 |

共享 HSTU attention 支持 Triton 与 PyTorch 对照；
后端选择、适用范围和实测见[后端记录](results/backend_acceleration/2026_09_19/README.md)。

## 维护约定

研究开发从最小可验证实验开始，检查聚焦实际数值、因果和证据风险；
长训练与正式评测的启动规则见 [AGENTS.md](AGENTS.md)。
文档按上面的入口就地维护，不另建重复进度或整理报告。

Git 保存源码、配置和紧凑证据；数据、权重、逐请求 raw 与运行输出在本地，
由 [.gitignore](.gitignore) 管理。复跑或从 raw 绘图需要相应本地资产。
当前方法与历史实验分开说明；完整结果包含失败和负值。
