# 博士研究方向分析：总索引

作者背景：TokaDB（面向推荐系统的数据管理优化，用户行为序列的组织/存储/访问）
目标：寻找支撑博士论文、具 CCF-A 潜力的研究方向
约束：4× NVIDIA A40 GPU；必须真实 workload、可实验验证、与已有积累关联

## 分析遵循的核心原则

1. 先确认已有研究边界，再寻找创新空间（不找新名字，找新未解决问题）
2. 不允许先方案后补价值（顺序：已有论文→已解决问题→限制→真实 bottleneck→新问题→方法）
3. 区分工程优化 / 方法改进 / 新研究问题
4. 涉及具体论文必须基于真实文献（本报告所有引用均经 arXiv API 核验）
5. 优先真实 workload 与 benchmark
6. 面向 CCF-A：寻找"新问题"而非"新实现"，先证明现有方法为何不能解决

## 文档导航

> ⭐ **下次工作起点：`08_core_insights_and_roadmap.md`** -- 包含方向演进、最终问题定义、核心数学 insight、分阶段工程建设路线（gated）、风险与未决问题。早期文档（`01-07`）部分结论已被 `08` 修正，以 `08` 为准。

| 文件 | 内容 |
|---|---|
| `08_core_insights_and_roadmap.md` | ⭐ **核心沉淀**：insight + gated 工程路线图 + 风险/未决问题（下次起点）|
| `01_executive_summary.md` | 执行摘要（早期，部分被 08 修正）|
| `02_landscape_boundaries.md` | 已有研究边界（附核验过的真实文献）|
| `03_new_problem.md` | 新问题定义（早期，场景已被 08 修正为"参数变化致失效"）|
| `04_thesis_direction.md` | 博士论文框架与子问题拆解 |
| `05_experiment_feasibility.md` | 真实 workload / benchmark / 4×A40 可验证性 |
| `06_alternatives_risks.md` | 候选方向对比与批判性自评 |
| `07_references.md` | 核验过的参考文献清单（含 arXiv ID）|

## 一句话结论

将 TokaDB 的"用户行为序列数据管理"延伸到 **LLM 化的推荐系统（LLM-based / Generative Recommendation）serving 阶段**：当推荐系统的个性化上下文从"embedding/特征"变为"LLM 的 KV cache / 自然语言 profile"这一**新型派生数据对象**后，在**流式用户行为更新**下出现了一个现有 LLM serving 系统（为 chat 设计、假设 KV 不可变）和现有推荐数据系统（为 embedding/特征存储设计）都**无法解决的一致性-新鲜度-延迟-精度**管理问题。这是一个新的问题定义、对应真实工业 workload、可在 4×A40 上用开源模型与公开数据集验证。

## 重要诚实声明

- 本报告所有被引用论文均通过 arXiv API 在本次会话中核验过标题/作者/摘要（见 `07_references.md` 的"核验状态"列）。
- 调研中发现一处记忆错误：arXiv:2111.05844 实为天体物理论文，并非"Persia"。凡未能在本次会话核验的论文（如 Persia、DeepSpeed/ZeRO、FlashAttention），一律标注"未核验，使用前须复核"，不作为立论基石。
- TokaDB 的具体内容基于用户自述（用户行为序列数据管理），未独立核验其原文；分析以其公开研究主题为准。
