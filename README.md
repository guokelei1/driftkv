# EvoKV

EvoKV 研究 Transformer recommender 在模型更新后如何继续使用持久化 K/V。
当前方法是 **Cross-Version Cache Adaptation**：保留逐事件缓存，在写入时维护短摘要，
发布时学习共享版本转换，请求时修正历史读取，并支持追加、淘汰和连续发布。

## 从这里开始

- [论文正文](../paper/main.tex)：当前论文及设计命名的唯一正文。
- [文档地图](docs/README.md)：设计、实验协议和模型资产。
- [脚本入口](scripts/README.md)：按数据、六层/十层/十六层训练、Motivation 和 Insights 查找。
- [论文图表](figures/README.md)：论文实际使用的生成器、数据来源和显示口径。
- [结果索引](results/README.md)：区分论文证据、保留模型与历史诊断。

共享 HSTU attention 默认在支持的 CUDA 输入上使用 Triton，保留原 PyTorch 算式和
已有权重格式。`EVOKV_ATTENTION_BACKEND=torch` 可选择原算式；适用范围、六层实测
和复查命令见 [后端升级记录](results/backend_acceleration/2026_09_19/README.md)。

当前 Design 已完成设计，尚未实现和验证。仓库保留的适配代码与结果属于历史探索，不代表当前 Design 已落地。
本轮模型训练与逐版评测已推进至 Max V5，进度见[统一训练索引](docs/unified_training_2026_09/README.md)；
三规模 AUC 诊断已保留，当前 Design 的适配实验仍待实现和验证。历史 Insight 的缺失依赖不在本轮清理中重建。
旧 PRO、KV-only replay 等方法不再作为当前 EvoKV 实现，但当前评估依赖的少量公共函数仍保留。

## 工作边界

这是论文研究仓库，开发和实验都以快速、可信地验证 idea 为主。
先做能回答当前问题的最小实现和小规模实验，再根据结果决定是否扩展；
序列化、并发、容错和通用框架按实际需要补充。
检查聚焦数值、数据因果、评价口径和证据，只运行与改动相关的必要测试，
不默认跑全量测试，也不追求完整覆盖。具体规则见 [AGENTS.md](AGENTS.md) 和
[测试入口](tests/README.md)。

六层 Medium、十层 Large、十六层 Max 的模型、训练记录和评估链保留。当前论文的原始结果、
seal、裁决、数据与冻结合同保留；废弃四层实验和旧探索原始结果已清理。
旧探索不再作为活动实验入口；历史归档的已知缺失及恢复限制见结果索引。
长训练和正式人口实验仍需前瞻协议、资源估计、focused canary 和用户明确启动。
已授权任务内的小规模开发探针使用轻量配置与记录，不为每次尝试另建审批流程。

历史删除清单、归档及恢复边界见 [结果索引](results/README.md)。

## Git 保存范围

Git 保存代码、论文文档、冻结合同、必要的小型结果表、汇总与 seals，以及论文 PDF。
数据、模型权重、逐请求 raw、运行日志、预览图和历史恢复包保留在本地，由
[.gitignore](.gitignore) 排除；小 CSV/JSON 不按扩展名一概忽略。
新增结果只提交复现图表和解释结论所需的紧凑记录，不用 `git add -f` 绕过规则提交大载荷。
仅克隆仓库可读取这些汇总和生成论文图；重新执行模型实验仍需本机的数据、权重和 raw。
