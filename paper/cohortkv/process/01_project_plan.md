# CohortKV 论文项目计划

## 1. 目标与交付标准

目标不是把现有 design note 改写一遍，而是形成一篇可审计的系统论文工作稿：

1. 研究问题、motivation 现象、三个设计层、实验问题和结论一一对应；
2. 每一个数字都能回到当前协议和结果文件；
3. 已复现、单 seed 设计结果、接口正确性和未来实验明确分层；
4. 五篇目标论文只提供组织与写作技术，不替代本仓库的语义来源；
5. 不改动 `paper/` 之外的任何文件。

论文不会按一个 plan 一遍写完。流程要求先完成证据冻结和 Methods/Results 骨架，再做
**至少三轮实质性修订**。每轮都必须记录发现的问题、修改动作和仍未解决的缺口。

## 2. 硬边界

### 2.1 研究边界

当前对象是流式训练后、跨模型版本的 HSTU 前缀 K/V 迁移。系统输入是：

- 已物化的旧版本 `Norm(x)` capsule；
- 已发布的 source-to-target migration program；
- 固定的完整更新记录集合；
- 执行 GPU；
- 调用者显式指定的 HBM、DRAM、POSIX 文件或 remote-object destination。

系统输出是覆盖完整记录集合的 target-version K/V manifest。当前不解决训练、在线到达、
hotness、请求路由、训练/服务共置、自动 destination 选择和前台 SLO。

### 2.2 三个且仅三个贡献层

1. **Version-cohort migration compiler**：为 `(source version, target version)` 编译并验证共享
   迁移程序。
2. **Capsule-to-K/V operator**：用融合 affine 投影、bias、length mask 和 K/V split 直接
   生成目标 K/V。
3. **Destination-oriented update engine**：按显式 destination 执行、移动、分片并在完整
   覆盖后发布 manifest。

Coordinator 只作为 glue code，不写成第四个贡献。Version cohort 用于编译、批处理、
placement 和迁移执行，不用于预测“是否可以安全复用”。

### 2.3 证据边界

- 3×3 capacity 与 27-cell cohort-tiered 结果是跨训练 seed 的主证据。
- 4+12 long-context verified compiler 与 two-GPU runtime 是
  `adaptive_seed0_*` 设计证据。
- v4 destination backend 只有接口、readback、失败可见性和 manifest 语义。
- 不把 v2 的 64-record assigned mix 写成 organic mixed-version full-cohort workload。
- 不把 host-backed 结果写成 SSD、网络、remote GPU 或生产部署结果。
- 不把 exact recomputation 写成任务质量上界；只把它作为当前模型 K/V 语义参考。

## 3. 工作包

### WP0：材料盘点与来源优先级

- 完整阅读操作手册、写作准则和模板。
- 完整阅读五篇目标论文的 Introduction、Motivation、Design、Evaluation 和
  Discussion/Conclusion。
- 阅读 authoritative roadmap、evaluation protocol、当前 experiment records 和关键实现。
- 建立结果文件索引；不从已删除文档或旧 result path 恢复结论。

完成门：`03_claim_evidence_matrix.md` 中每个主张都有协议、结果和允许措辞。

### WP1：反向工程目标论文

对每篇论文记录：

- 章节骨架；
- Introduction 的问题推进方式；
- motivation 现象如何变成设计；
- 核心抽象如何扩展到 operator/system；
- evaluation 如何证明每个设计点；
- 本文禁止照搬的语义。

完成门：`02_reference_reverse_engineering.md` 明确“学什么、放在哪里、不学什么”。

### WP2：叙事与图表蓝图

- 一句话 thesis：模型更新使持久化 HSTU K/V 成为 versioned derived state；CohortKV 把
  source-to-target cohort 编译为可验证的状态变换，并用一遍式 operator 和
  destination job 发布目标版本。
- 建立 observation → requirement → mechanism → experiment 闭环。
- 先设计 Methods/Results，再写 Introduction；Abstract 和 title 最后定稿。
- 每张图先写出它回答的问题，再决定是否制作。

完成门：所有 Motivation 小节都能指向一个后续设计和一个 evaluation question。

### WP3：首稿 v0

先写：

1. Problem definition and scope
2. Design overview
3. Compiler
4. Operator
5. Destination engine
6. Evaluation
7. Limitations

再写：

8. Introduction
9. Related work
10. Conclusion
11. Abstract and title

首稿允许语言粗糙，但不允许证据占位符伪装成结果。

### WP4：修订一——事实与协议审计

逐句检查：

- 数字是否与 JSON/实验记录一致；
- metric 方向和分母是否准确；
- training seed 是否是统计单位；
- 不同 protocol family 是否被混合；
- fresh/stale/reuse/exact 的 serving 语义是否一致；
- padding、sequence length、catalog 和 t+1 预测语义是否写清；
- 所有 `up to`、`average`、CI、样本量和硬件边界是否有修饰语。

输出：v1 和 review log 中的逐项修改记录。

### WP5：修订二——叙事与系统闭环审计

按五篇目标论文的写作技术检查：

- HCache 式骨架是否突出“两个昂贵端点之间的中间状态机会”，同时清楚区分 same-model
  restoration 与 cross-version migration；
- Ekko 式应用重要性是否成立，但没有把论文写成模型发布系统；
- vLLM 式 core abstraction → operator → engine 是否连贯；
- DistServe 式每个 motivation 现象是否对应后续机制与实验；
- Orca 式 version cohort 是否成为稳定的系统执行单位，而不是 safety oracle。

输出：v2；允许重排章节、重写 Introduction 和删除不服务主线的结果。

### WP6：修订三——审稿人攻击面与语言审计

模拟系统审稿人的五类问题：

1. 新颖性是否被 DroidSpeak、HCache 或 recommender cache systems 覆盖？
2. 三个贡献是否都被实验证明，还是 architecture 多于 evidence？
3. 11.22× 是否来自不一致边界？
4. 单 seed compiler 是否被过度推广？
5. 为什么额外保存 50% capsule state 仍然合理，何时不合理？

修订时必须把不能用文字解决的问题移入 explicit limitation / open evaluation gate，而不是
用更强语气掩盖。

输出：v3、最终 Abstract/title，以及 `06_open_experiment_gaps.md`。

## 4. 质量门

论文工作稿完成需要同时满足：

- 全文只使用一套核心词汇：migration anchor、served K/V target、version cohort、
  compiled repair、exact recomputation、destination manifest；
- 三个贡献在 Introduction、Overview、Methods、Evaluation 和 Conclusion 中顺序一致；
- 每个主结果至少出现一次协议限定；
- 每个单 seed 结果在首次出现处明确标识；
- v4 不出现未测吞吐、SSD、network 或 full-cohort speedup；
- negative results 只用于界定设计，不重新成为项目 crux；
- 参考文献中的 closest work 有明确差异句；
- SVG 内文字、表格数字、正文数字和 claim matrix 一致；
- 所有仓库内链接有效；
- `git diff -- paper/` 之外没有由本任务引入的改动。

## 5. 本轮交付与后续迭代

本轮交付包含 v3 级工作稿和完整 review log，但这不意味着研究论文已经“写完即投稿”。
完成 full-cohort identical-boundary evaluation 后，需要再做一次 Results/Abstract/Introduction
联动修订；完成物理 SSD 或 remote backend 后，再决定 destination 部分是主贡献还是只保留为
architecture。任何新结果都应先更新 claim matrix，再改正文。
