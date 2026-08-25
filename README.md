# EvoKV

EvoKV 研究模型发布后持久化用户 K/V 状态的预算化收敛：哪些状态可以继续复用，哪些需要依赖合法的部分重算，哪些必须 Exact Recompute。

当前论文主线聚焦 HSTU-native 推荐模型在版本发布后的 persistent K/V state compatibility：
新模型可以变好，但旧模型产生的状态可能阻碍这部分收益兑现。最新可复核结果记录在核心
motivation 文档中；具体实验设计和概念边界分别独立维护。

## 入口

- [论文总体设计](docs/paper_design.md)：概念层问题、场景、相关工作、比较对象和目标指标。
- [论文具体实验设计](docs/experimental_design.md)：架构、数据、版本训练、实验阶段和预期观察。
- [核心 Motivation 与 Observation](docs/motivation_observations.md)：当前已观察到的结果、数字和边界。

旧阶段文档、archive、legacy 和重复路线说明已清理。代码、合同、脚本和测试是实验设计
的执行材料，不再通过额外路线文档维护第二套叙述。

所有结果必须按 workload、release、lineage、seed、metric 和证据等级解释。不得跨协议拼接结果、筛选有利 seed/edge，或把 diagnostic K/V splice 当作可部署动作。
