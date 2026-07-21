# 04 博士论文框架与子问题拆解

## 论文主题（working）

*Data Management for LLM-based Recommendation Serving: Treating Derived Personalized KV as a First-Class Streaming Object*

## 核心论点

当推荐系统 LLM 化，"用户行为序列的派生 LLM KV 表示"成为一个新型数据对象。它既不是传统 embedding，也不是 chat 的静态前缀，而是**派生于行为流、需在 serving 时按新鲜度/一致性约束复用**的状态。管理这一对象的系统问题，是 TokaDB 工作（行为序列存储）的自然延伸，且现有 LLM serving 与推荐数据系统都不覆盖。

## 论文结构（4 个子工作，每个可独立成一篇 CCF-A）

### 工作 1（立论篇）：问题定义 + Freshness-Aware KV（最小可发表单元）
- **问题**：定义 SDKV-C（见 `03`），形式化新鲜度-延迟-精度 trilemma。
- **方法线索**（problem-first 之后才谈方法）：基于 causal LLM 的数学性质，区分"行为流变化下旧 KV 仍有效的操作"（如纯尾部追加）与"需局部重算的操作"（如语义重排、分块重摘要），给出**增量失效/维护机制**与新鲜度感知的替换策略。
- **实验**：在流式 LLM4Rec serving benchmark 上，对比 vLLM / SGLang / RcLLM（静态）基线，证明在等精度下降低重算、或等延迟下提升新鲜度。
- **目标会议**：SIGMOD / VLDB / ATC（偏数据管理与一致性）或 OSDI/SOSP（偏 serving 系统）。

### 工作 2（抽象篇）：派生 KV 的一致性模型
- **问题**：把 DB 的一致性级别（强/最终/有界滞后/读己之写）推广到"派生 KV over 行为流"，但要处理 LLM 的**有损性**（KV 不是无损投影，重算可能因非确定性略有差异）与**粒度**（段级而非记录级）。
- **贡献**：一致性级别谱系 + 不同级别下的成本/精度保证 + 选择策略。
- **目标会议**：SIGMOD / VLDB（一致性抽象是 DB 强项）。

### 工作 3（系统篇）：行为序列存储与派生 KV 的协同存储
- **问题**：TokaDB 类行为存储 与 派生 KV 存储的协同。分级（HBM/DRAM/SSD）、per-user 复制 vs item catalog 分片、与底层行为更新的失效传播。
- **贡献**：统一存储抽象 + 分级/复制/分片策略 + 失效传播协议。
- **与 TokaDB 的关系**：直接延伸--TokaDB 管原始序列，本工作管"原始序列 + 派生 KV"的联合。
- **目标会议**：SIGMOD / VLDB / ATC。

### 工作 4（serving 篇）：多候选 fan-out 的请求级调度
- **问题**：把"1 用户 × N 候选"作为一等调度单位。用户前缀 KV 在 N 间共享；catalog KV 跨用户共享；在新鲜度约束下做请求级 batching 与 locality 调度。
- **贡献**：fan-out 感知的调度器 + 与工作1/3 的失效/存储机制集成 + 端到端系统。
- **目标会议**：OSDI / SOSP / ATC / EuroSys。

## 论文叙事（thesis statement）

> "从 DLRM 到 LLM-based 推荐的迁移，不只是模型替换，而是引入了一个新的数据管理对象--派生于用户行为流的 LLM KV 状态。本论文证明该对象需要一套新的、流式一致性感知的数据管理方法，并提出相应的抽象、机制与系统。"

## 与已有积累（TokaDB）的关联路径

```
TokaDB: 用户行为序列的组织/存储/访问
   │
   ├── 对象扩展: 序列 → 序列 + 派生 LLM KV
   ├── 维度新增: 静态访问 → 流式一致性/新鲜度
   ├── 层次扩展: 单层存储 → 行为层 + KV 层协同(HBM/DRAM/SSD)
   └── 负载扩展: 单候选召回 → 1×N 生成式 fan-out serving
```

每一条都是已有工作的**纵深延伸**，而非换赛道，符合"与已有研究积累的关联性"要求。
