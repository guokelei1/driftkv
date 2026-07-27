# Target-paper reverse engineering

## 1. HCache：骨架与核心边界

### 学习

HCache 的有效骨架是：

1. 可复用状态很重要但 GPU 容量有限；
2. recomputation 与 offload/reload 是两个代价不同的端点；
3. intermediate activation 提供中间恢复点；
4. motivation 定量展示计算、I/O 和存储权衡；
5. pipeline bubble 与 storage layout 两个挑战分别变成 scheduler 和 storage manager；
6. evaluation 依次证明 overall、sensitivity 和 component value。

### 用在本文

- Introduction 先定义 stale reuse 与 exact recomputation 两个端点。
- Motivation 说明旧 `Norm(x)` capsule 允许在端点之间编译一个共享迁移程序。
- Design 按 compiler → operator → destination engine 展开。
- Evaluation 分开回答语义 fidelity、operator cost 和 endpoint completion。

### 不能照搬

HCache 在**同一模型**下从 token 或 intermediate activation 恢复状态；CohortKV 的输入状态由
旧模型版本产生，目标是当前模型版本的 HSTU K/V。本文不能把 cross-version 误写成
eviction restoration，也不能暗示 HCache 已经解决模型更新后的语义失效。

## 2. Ekko：应用背景，但不是模型发布论文

### 学习

Ekko 的 Introduction 很快建立三个事实：推荐系统需要持续学习；更新延迟会让最新行为和内容
不能及时反映；大规模更新链路本身是系统问题。它用生产场景把“更新”从 ML 训练问题提升为系统
问题。

### 用在本文

- 说明流式推荐模型持续产生新版本，持久化用户历史状态因此比普通请求内 KV 更容易跨版本。
- 强调更新后的模型与旧 derived state 共存，而不是只讨论一次性模型替换。

### 不能照搬

CohortKV 不传播模型参数，不设计 WAN dissemination，不绕过模型验证，不实现 rollback，也不
声称 model-update latency 或工业 SLO。训练只提供 checkpoint；论文处理的是 checkpoint 之后的
derived-state migration job。

## 3. vLLM：从 KV 特性到算子和系统

### 学习

vLLM 先识别 KV 的独特生命周期与布局问题，再提出 PagedAttention 这个核心抽象，最后围绕它
共同设计 memory manager、scheduler、distributed engine 和 kernels。核心机制不是孤立算法，
而是改变系统其余部分的接口。

### 用在本文

- HSTU 的 `K,V = W_{k,v} Norm(x)` 是可利用的结构，不只是一个回归 trick。
- 编译后的 affine program 成为 operator 输入。
- `(source,target)` program 与 capsule metadata 成为 batching、placement 和 publication 的
  system interface。
- operator 直接输出 destination-ready K/V，engine 维护 complete-version manifest。

### 不能照搬

当前 trace 上 page/jagged compaction 没有稳定正收益，因此不把 paging 或 compaction 作为论文
核心；它只保留为对短碎片 workload 的条件机制和 negative result。

## 4. DistServe：motivation 必须逐项映射设计

### 学习

DistServe 把 observation 写成设计义务：

- interference → disaggregation；
- resource coupling → independent allocation/parallelism；
- disaggregation communication → bandwidth-aware placement。

后续 evaluation 和 ablation 再逐项验证。

### 用在本文

| Motivation observation | Design requirement | Mechanism | Evaluation |
|---|---|---|---|
| stale K/V 留下可测 maintenance gap | 不能永久 reuse | unconditional compiled repair plus escalation | cross-dataset and 3×3 opportunity |
| age 与 task quality 不稳定、full endpoint 可近零/负 | 不能用 age/task gain 做 admission | version-pair program plus label-free semantic contract | age matrix and 6/9 strict gate |
| HSTU K/V 是 old normalized state 上的当前 projection 加共享 residual | 把 repair 变成一遍式 program | compiled affine projection | 27-cell fidelity/cost and verified compiler |
| kernel gain可能被 H2D/D2H、padding、publication 吞掉 | 端到端必须保持相同 endpoint | fused direct-write operator and destination engine | Stage-4 FP16 capsule path loses at 0/6; Stage-4.5 direct old K/V removes extra source state and passes scoped hot-HBM 1/2/4-GPU points |

## 5. Orca：version cohort 是系统执行单位

### 学习

Orca 从 autoregressive workload 的 iteration 语义推导新的 scheduling granularity，并让这个
单位贯穿 scheduler、execution engine、batching 和 distributed execution。论证的重点不是给旧
请求贴一个预测标签，而是选择一个能稳定协调系统组件的单位。

### 用在本文

Version cohort 定义为 `(migration anchor version, served K/V target version)`：

- compiler 对一个 cohort 拟合、验证和发布 program；
- operator 验证 capsule anchor 与 program source；
- engine 将同一 target 的多个 source programs 常驻每张 GPU；
- batching、LPT placement、extent metadata 和 manifest 都保留 cohort identity。

它不回答“这个版本是否安全 reuse”。每个 stale cohort 都先接受 compiled repair，再按照已发布
contract 进入更强同步或 exact fallback。

## 6. Closest-work boundary audit

### DroidSpeak

DroidSpeak 在相同架构的微调 LLM 之间选择性重算若干层并复用其余层，同时流水化重算和 cache
load。它证明“cross-model KV”本身不能作为 CohortKV 的新颖性声明。

CohortKV 的可辩护差异是：

- 连续流式训练产生的推荐模型版本，而非一组同时服务的 fine-tuned LLM variants；
- HSTU 旧 `Norm(x)` capsule 上编译的 affine source-to-target transform；
- 不使用 recommendation labels 的 cohort-level semantic certificate；
- 将固定完整记录集合转换并以 target-version manifest 原子发布的 update job。

正式投稿前仍需逐节阅读 DroidSpeak 全文并把 baseline/evaluation 差异写得更精确。

### MTServe

MTServe 处理 generative recommendation 中跨访问持久化的 per-user K/V，使用 GPU/host
hierarchical cache、Page–Chunk layout、异步传输和 locality-aware replacement。它与
CohortKV 的交集是 recommender K/V 的 storage/movement；其公开工作流把持久状态作为后续访问
可复用的 serving state，并未定义从 source model version 到 target model version 的语义变换。

因此本文不能把“推荐 KV 分层存储”作为新颖性，只能把重点放在 version invalidation 和
source-to-target transformation。

### HSTU

HSTU 是模型背景来源。本文只声称简化实现保留了研究所依赖的 pointwise unnormalized attention
和 first-class K/V；不把本仓库实验直接推广为生产级 trillion-parameter HSTU。

## 7. 名称审计

“StreamKV” 已被 2026 AAAI 论文 *StreamKV: Streaming Video Question-Answering with
Segment-based KV Cache Retrieval and Compression* 使用。论文工作名改为 **CohortKV**。
仓库协议和代码符号保持不变，以保留实验可追溯性。

## 8. 写作技术落点

| 论文位置 | 主要借鉴 |
|---|---|
| Introduction | Ekko 的应用紧迫性 + HCache 的两端点机会 |
| Motivation | DistServe 的 observation-to-design 映射 |
| Overview | vLLM 的 core abstraction → operator → system |
| Compiler/Engine | Orca 的 execution-unit 论证 |
| Evaluation | HCache 的 overall/sensitivity/component 分层 + DistServe 的逐设计验证 |
| Related Work | 明确 HCache、DroidSpeak、MTServe 三条最近边界 |
