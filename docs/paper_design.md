# EvoKV 论文总体设计（概念层）

更新日期：2026-08-25

本文只定义论文长期稳定的概念边界。它回答“在什么场景下解决什么问题、为什么这个问题重要、和哪些方向比较、用什么指标判断是否成功”，不记录某一次训练的超参数、版本窗口或具体结果。

## 1. 问题场景

推荐模型或其他序列模型持续发布新版本。模型升级之后，系统中已经存在大量由旧模型产生的用户持久化 K/V 状态：

~~~text
旧模型 θ_(t-1) + 用户历史 H_u
        -> persistent state C_u^(t-1)

新模型 θ_t + 旧 state C_u^(t-1)
        -> 当前请求预测
~~~

如果每次发布都把所有用户历史重新用新模型计算，状态语义最干净，但会产生巨大的后台计算、内存读写和迁移时间。如果直接复用旧状态，又无法保证旧状态与新模型的表示、注意力和 readout 语义兼容。真实系统因此需要在“立即全部重算”和“完全不处理”之间进行状态演进。

## 2. 要解决的大问题

EvoKV 研究的是：

> 当上游已经决定发布新模型时，如何在有限的后台计算和 I/O 预算下，使整个活跃用户状态集合从旧模型语义演进到新模型语义，并尽量接近新模型对完整历史执行得到的结果？

对每个发布时已物化的用户状态，系统可以选择：

- 继续复用；
- 对依赖闭合的局部状态做部分重算；
- 对必要部分做精确重算。

概念目标是：

~~~text
在总迁移预算 B 内，
最小化演进后状态与 Current Full 状态之间的偏差，
同时最小化对真实任务质量的损失。
~~~

这里的预算是发布期间的后台计算和 I/O 预算，不是请求时延 SLO。EvoKV 不负责决定模型是否发布，也不把模型训练本身作为贡献对象。

## 3. 核心对象与边界

论文必须区分三个对象：

1. **Model release**：上游训练与 Full-only validation 决定新模型是否值得发布；
2. **Persistent-state compatibility**：旧模型产生的状态能否被当前模型继续使用；
3. **State evolution**：在有限预算下选择复用、部分演进或精确重算。

Current Full 是当前模型完整执行语义的 reference，不是模型发布判定器。上游 release admission 与 EvoKV compatibility/migration 必须分离：被拒绝的候选不能成为 cache producer，只有已封存的 accepted release 才进入状态兼容性评测。

论文不依赖请求时 controller。发布时决策只能使用发布前可获得的状态、模型差异、历史统计和 target-free probe；未来请求标签只能用于事后质量验证。

## 4. 相关工作与比较对象

论文需要和以下几类工作明确区分：

### 4.1 持久化 KV、prefix cache 与 serving cache

已有工作关注在同一模型或相近模型之间复用 prefix/KV 以减少重复计算。EvoKV 的重点不是单次请求的 cache hit，而是模型版本改变后，已经持久化的用户状态是否仍然兼容，以及如何管理整个人口的跨版本状态。

### 4.2 Continual learning、模型更新与 release pipeline

持续训练、增量更新、checkpoint lineage 和模型发布流程解决的是模型参数如何产生与上线。它们不能自动解决旧模型状态如何迁移。EvoKV 接收一个已经确定发布的当前模型，把 release 后的 persistent-state convergence 作为独立系统问题。

### 4.3 推荐系统的用户状态、序列表示与特征缓存

推荐系统中的用户历史长、用户数多、item/embedding 持续变化，状态既有长期偏好又有近期行为。EvoKV 关注这些表示在模型版本切换时的跨版本语义兼容，而不是只比较一次请求的排序模型精度。

### 4.4 近似计算、部分重算与资源分配

近似推理、分层重算、预算分配和 learned scheduler 提供了降低计算成本的工具。EvoKV 的问题在于：哪些状态可以安全复用、哪些状态需要何种依赖闭合的演进，必须由跨版本兼容性和状态风险驱动，并在完整活跃人口上满足总预算约束。

### 4.5 系统迁移、缓存一致性与 version debt

传统缓存一致性通常围绕数据更新、失效和重新获取展开。EvoKV 面向的是神经网络内部持久状态的版本债务：每次模型发布都可能留下由旧 producer 产生的状态，连续 No-op 会使状态年龄和兼容性风险累积。

论文中的比较对象不是某一篇工作的实现复刻，而是这些方向共同覆盖的基线边界。

## 5. 论文要回答的研究问题

- **RQ1：存在性** 新模型已经带来模型发布收益时，直接复用父版本 persistent state 是否会损害这个收益？
- **RQ2：结构** 兼容性风险是否与用户、历史区域、序列组成、item/embedding 漂移、模型组件和版本年龄有关？
- **RQ3：演进** 依赖闭合的部分状态演进能否用低于 Exact 的成本恢复大部分质量或状态 fidelity？
- **RQ4：分配** 发布前可获得的 target-free 观测能否在相同预算下优于固定、随机和 metadata-only 分配？
- **RQ5：持续性** 在连续版本链和不同人口/模型规模上，状态债务和迁移收益是否仍然存在？

RQ1 是 motivation 的最低成立条件；RQ2 决定 recommendation-specific insight；RQ3–RQ5 属于在 motivation 成立之后的系统设计和资格验证。

## 6. 成功标准与主要比较

论文至少比较以下状态处理策略：

- No-op / direct Reuse；
- Exact-All / Current Full；
- 固定比例或固定区域的 partial；
- Random allocation；
- metadata-only allocation；
- target-free profiler/scheduler；
- offline oracle 作为上界。

主要指标分为三层：

### 任务质量

在同一当前模型、同一请求、同一 causal history 和同一 readout 下，报告：

- ROC-AUC、PR-AUC；
- event log-loss、Brier；
- user-equal 与 event-weighted 的配对差异；
- 若使用开放 catalog，再报告 Top-K、NDCG、MRR 等 ranking 指标。

Motivation 的直接质量量是：

~~~text
Reuse harm = Quality(Current Exact) - Quality(Reuse)
~~~

对 loss 指标则使用 Loss(Reuse) - Loss(Current Exact)，正值代表旧状态带来损害。

### 状态语义

报告 Bernoulli JS、normalized score RMS、probability shift、Top-K overlap、margin/pairwise disagreement，以及用户级尾部。状态 fidelity 是解释机制和指导迁移的 companion，不能替代任务空间质量。

### 资源与系统代价

报告 exact-equivalent compute、token-layer work、KV read/write bytes、history I/O、后台 worker-hours、迁移 makespan 和跨版本 state debt。计算和 I/O 必须分别报告，不能用任意权重混成一个没有解释的总分。

## 7. 论文贡献边界

论文的核心贡献应是一个跨版本 persistent neural state evolution 的系统抽象和可验证方法：

1. 明确模型 release、状态兼容性和状态演进的分离边界；
2. 用同一当前模型下的 Current Full/Reuse 对照建立版本化状态不兼容的任务质量证据；
3. 在有限后台预算下，将兼容复用、依赖闭合的部分演进和精确重算统一为人口级状态管理问题；
4. 用 target-free 风险观测、成本和质量验证评价状态分配；
5. 在连续版本、不同状态年龄和更大人口上验证方法的适用边界。

具体 migration granularity、predictor、scheduler、executor 和最终 action set 不应在动机证据之前被概念文档预先写死。

## 8. 不应过度声称的内容

- 一条有害边不能推出所有模型更新都必然有害；
- One-hop 直接复用不能单独证明 recursive debt 或最终迁移策略；
- 状态 fidelity 提升不能直接等同于线上排序质量提升；
- Yambda 单一 workload 不能代表所有推荐场景；
- 训练规模、checkpoint 大小或离线 GPU 数不能单独定义系统规模；
- 诊断性 KV splice 不能被写成可部署 action；
- 模型发布质量提升不能被归因于 EvoKV 的状态管理。

具体架构、数据、版本链和阶段性结果统一记录在
[具体实验设计](experimental_design.md) 与
[核心 Motivation 与 Observation](motivation_observations.md) 中。
