# 03 新问题定义（核心）

本节严格遵循"已有论文 -> 已解决问题 -> 限制 -> 真实 bottleneck -> 新研究问题"的顺序，并满足 principle #6："先证明为什么现有方法无法解决"。

---

## 3.1 真实 workload 的精确刻画

现代工业 LLM-based 推荐的部署范式（由多篇工业论文证实，见 `02`）是：

```
用户行为流（streaming）
     │ 持续到达：点击/播放/购买/停留...
     ▼
[行为序列存储]  ← TokaDB 类系统管理
     │ 周期性 / 触发式
     ▼
[个性化上下文编译]  ← nearline 预计算
   把行为序列 -> LLM 的 KV cache（或自然语言 profile）
     │
     ▼
[在线生成式 serving]  ← vLLM/SGLang/RcLLM 类系统
   请求 = system prompt + 用户个性化 KV + N 个候选 item
   输出 = 对 N 个候选的生成式排序/评分
```

**该 workload 的四个结构性特征（每一条都有工业证据）：**

| 特征 | 证据 |
|---|---|
| (W1) 个性化上下文**派生于行为流且持续演化** | Douyin-10K [WWW'26]；LIBER 流式分块；Netflix "cached serving makes next-token stale" |
| (W2) **nearline 预计算 + 在线 serving** 是主流部署 | LWGR "nearline precomputation + lightweight online serving"；LLM-persona at scale [RecSys'26 Industry] 用知识蒸馏+异步推断 |
| (W3) **1 用户 × N 候选** 的 fan-out 请求 | RcLLM 的 item catalog cache；Douyin Request-Level Batching；GR4AD LazyAR 多候选生成 |
| (W4) **item catalog 规模** 的 KV 共享 | RcLLM "massive item caches sharded"；AutoShard（embedding 时代即已分片）|

---

## 3.2 真实 bottleneck（不是臆测）

把上述 workload 跑在现有 serving 系统上，出现**三个被工业论文直接观察到、但未从系统层解决**的 bottleneck：

### B1. 新鲜度-重算困境（staleness-recompute dilemma）
- nearline 预计算的 KV 在两次刷新之间**陈旧**：Netflix 论文明确指出 staleness 使 "immediate next-token target stale"。
- 想要新鲜就要更频繁重算；但 LIBER 指出"each update to user sequences -> substantial computational overhead if LLMs recurrently called"。
- 现有 prefix cache 在行为流更新时**命中率为 0**（见 3.3），被迫 O(L) 全量重算。在 10K 长度序列（Douyin）下，单次全量重算代价不可接受。

### B2. 个性化 KV 的存储-访问成本
- 个性化 KV（per-user）体积远大于 embedding：8B 模型、10K 行为 token、bf16 -> 单用户 KV 约数十 MB；百万用户即数十 TB。
- 现有 serving 系统把 per-user KV 当普通 prefix block，无分级/复制/分片语义（RcLLM 开始做但仅静态）。

### B3. 多候选 fan-out 的调度空白
- 1 用户 × N 候选请求中，**用户前缀 KV 在 N 个候选间共享**，但 chat-oriented 调度器不把"用户级 fan-out"作为一等调度单位，导致重复 prefill 或次优 batching（Douyin 的 RLB 在模型侧缓解，serving 侧未系统化）。

---

## 3.3 为什么现有方法无法解决（principle #6 的核心证明）

逐一对照：

### (1) vLLM/SGLang prefix cache -> 失效
- 假设：`KV(text_prefix)` 一旦计算即不变，复用 = `text_prefix` 字符串相同。
- 推荐下：用户行为流更新 -> `text_prefix` 变化（即使只是尾部追加，若采用 ReLLa 式**语义检索重排**或 LIBER 式**分块重摘要**，则 prefix 内容整体变化）-> 命中率为 0。
- 即便纯尾部追加（causal LLM 下旧 KV 数学上仍有效），现有系统也**不识别"增量有效"**，按 cache miss 全量重算。
- 结论：**immutability 假设与行为流演化直接冲突。**

### (2) RcLLM -> 失效（部分）
- RcLLM 解决了空间复用（block 分解 + 分层存储），但 block 仍是**静态内容**。
- 行为流更新使 user-history block 内容变化时，RcLLM 无增量维护/失效机制，退化为全量重算。
- RcLLM 未定义/评测 staleness，无一致性模型。
- 结论：**RcLLM 覆盖空间维，未覆盖时间维。**

### (3) 模型侧工作（ReLLa/LIBER/HyMiRec）-> 不在系统层
- 它们改进的是"LLM 如何理解序列"（检索/摘要/codebook），把 serving 当黑盒。
- 它们承认 bottleneck（latency、feature fetching bandwidth）但不在系统层解决。
- 结论：**正交、互补，不构成对新问题的解决。**

### (4) 推荐数据系统（TokaDB/FIITED/AutoShard）-> 对象错配
- 管理 embedding/特征/行为序列，不管理 LLM 派生 KV。
- 派生 KV 有一致性约束（必须与底层行为流一致到某个新鲜度界）、访问模式（per-request 多 block 拼装）、生命周期（nearline 生成/在线消费）都与 embedding 不同。
- 结论：**对象不同，不能直接迁移。**

### (5) bicache [arXiv 2606.07571] -> 不适用
- 唯一挑战"KV 不可变"的近期工作，但针对 diffusion language model 的双向注意力，不针对推荐/流式行为。
- 结论：**证明"挑战 immutability"是前沿方向，但场景不同。**

> **综合：新问题在"派生个性化 KV 的时间演化一致性"这一精确位置，无现有工作覆盖。**

---

## 3.4 新研究问题的形式化定义

### 问题命名
**Streaming Derived-KV Consistency for Personalized LLM Serving（SDKV-C）**

### 形式化
给定：
- 行为流 `S_u(t)`：用户 u 在时刻 t 的行为序列（持续增长/变化）。
- 编译函数 `K(·)`：把序列段编译为 LLM KV 状态 `K_v = K(S_u)`（可为整段 KV、或摘要 profile、或 codebook）。
- 在线请求 `q(u, C)`：用户 u 对候选集 C 的生成式推荐请求，需拼装 `K_v` + 候选 item KV。
- 新鲜度要求 `F_u`：`K_v` 与 `S_u(t_now)` 的允许偏差（可定义为"未编译的新行为数"或"行为时间戳滞后"）。

求：一个数据管理系统，在满足
- **新鲜度约束** `Freshness(K_v, S_u(t_now)) ≤ F_u`
- **延迟 SLO** `Latency(q) ≤ L`
- **精度界** `|Accuracy(K_v) - Accuracy(K(S_u(t_now)))| ≤ ε`

下，**最小化重算/传输成本**（或最大化吞吐）。

### 为什么这是一个"新研究问题"而非工程/方法改进（principle #3）
| 判据 | 是否满足 |
|---|---|
| 产生新问题定义 | ✓ SDKV-C 形式化（派生 KV 的时间一致性） |
| 出现新 workload | ✓ 流式 LLM4Rec serving（W1-W4） |
| 需要新系统抽象 | ✓ "派生 KV 视图 over 行为流"作为一等管理对象，带一致性/新鲜度语义 |
| 存在新评价维度 | ✓ 新鲜度-延迟-精度 trilemma（现有 serving 评测无新鲜度轴） |

四项全中 -> 满足"新研究问题"标准，而非工程优化或模块拼接。

---

## 3.5 新问题衍生出的开放子问题（映射到博士论文，见 `04`）

1. **一致性模型**：派生 KV 与行为流之间的"KV 一致性级别"如何定义？（强一致 / 最终一致 / 有界滞后 / 读己之写等推广）--类比 DB 的一致性级别，但对象是派生 KV。
2. **增量维护**：行为流的小变化（追加/重排/重摘要）下，如何避免 O(L) 全量重算？哪些变化下旧 KV 数学上仍有效（causal 性质），哪些需要局部重算？
3. **存储与协同**：行为序列存储（TokaDB 类）与派生 KV 存储如何协同？分级（HBM/DRAM/SSD）、复制（per-user vs catalog）、分片策略？
4. **多候选 fan-out 调度**：把"1 用户 × N 候选"作为一等调度单位，用户前缀 KV 在 N 间共享的请求级 batching。
5. **评测体系**：构造流式 LLM4Rec serving benchmark，定义新鲜度-延迟-精度三维指标。
