# 02 已有研究边界分析

本节严格按"先确认已有研究边界"原则，梳理与候选方向相关的四条研究线、它们各自**已解决什么**、**覆盖到什么范围**、**留有什么限制**。所有引用见 `07_references.md`，均经 arXiv API 核验。

---

## 研究线 A：LLM serving 的 KV cache 管理

### A.1 已解决
- **vLLM / PagedAttention [SOSP'23, arXiv 2309.06180]**：把 KV cache 当作虚拟内存，按 block 管理，解决碎片化与跨请求共享。已验证。
- **SGLang / RadixAttention [arXiv 2312.07104]**：用 radix tree 索引共享前缀，加速结构化 LLM 程序。已验证。
- **FlexGen [ICML'23, arXiv 2303.06865]**：单 GPU 高吞吐离线推理，GPU/CPU/Disk 分层 offload + 4bit 压缩。已验证。

### A.2 近期进展（2026，arXiv 预印，多数尚未被广泛复现，仅作 landscape）
- 超越前缀的**段级/语义级 KV 复用**：SparseX [arXiv 2606.01751]、CacheTune [arXiv 2605.24022]、HYPIC [arXiv 2607.01299]。
- **语义感知驱逐**：SAECache [arXiv 2605.18825]（按 token 类型区分复用率）。
- **跨层存储**：Tutti [arXiv 2605.03375]（SSD-backed KV，GPU-centric I/O）、Prefill-as-a-Service [arXiv 2604.15039]（跨数据中心 KV 传输）、CALVO [arXiv 2603.21257]（KV 加载作为一等调度对象）。
- **流式驱逐**（注意：这里的"streaming"指**解码过程中的 token 流式驱逐**，不是行为流更新）：Nexus Sampling [arXiv 2606.23961]。
- **DLM 的 prefix cache**：bicache [arXiv 2606.07571]--指出双向注意力下 KV 不可不变，是少数挑战"KV 不可变"假设的工作，但针对 diffusion language model，不针对推荐/流式行为。
- **agent serving**：Pythia [arXiv 2604.25899]（利用 workflow 可预测性优化 multi-agent serving，指出 agent workload 的 prefix 命中率低等问题）。

### A.3 共同假设与限制（关键）
> **几乎所有上述工作都假设：一段 KV 一旦计算完成，其内容不变；"复用"= 同一段 KV 被多个请求共享。**

这一假设在 chat/RAG/agent workload 下成立（system prompt、文档、工具输出相对稳定）。但在**推荐系统**下不成立：个性化上下文（用户行为序列/profile）**随用户行为流持续演化**。这正是研究线 A 留下的核心缺口。

---

## 研究线 B：LLM-based / Generative Recommendation（模型侧）

### B.1 已解决（模型/算法层面）
- **长序列"不可理解"问题**：ReLLa [WWW'24, arXiv 2308.11131] 提出"lifelong sequential behavior incomprehension"--LLM 即使在 context 窗口内也无法有效利用长行为序列；用语义行为检索（SUBR）缓解。ReLLaX [arXiv 2501.13344] 进一步在数据/prompt/参数三层优化。均已验证。
- **流式/终身行为建模**：LIBER [arXiv 2411.14713] 用流式分块 + 级联 LLM 推断；HyMiRec [arXiv 2510.13738] 用残差 codebook 压缩/复用历史 embedding。
- **超长序列工业化**：Douyin-10K [WWW'26, arXiv 2511.06077] 端到端 10K 长度行为序列，STCA + Request-Level Batching + 长度外推训练。已验证。
- **benchmark**：ALPBench [arXiv 2602.03056] 面向长程个性化行为理解（属性级 ground truth）。
- **生成式推荐（GR）工业化**：Kuaishou GR4AD [arXiv 2602.22732]、OneMall [arXiv 2601.21770]、Taiji [arXiv 2606.03866]；Netflix GR scaling [arXiv 2605.23312]；LWGR [arXiv 2605.18771]；LLM-persona at scale [RecSys'26 Industry, arXiv 2606.12198]。

### B.2 限制（关键）
> **上述都是模型/算法论文。它们反复承认推理延迟与数据获取是部署瓶颈，但把 serving/数据栈当黑盒，不在系统层解决。**

直接证据：
- HyMiRec："due to inference latency and feature fetching bandwidth constraints, existing methods typically truncate user behavior sequences"。
- LWGR：部署方案是 "nearline precomputation with lightweight online serving"。
- Netflix GR："cached serving can make the immediate next-token target stale"。
- LIBER：流式更新下"substantial computational overhead if LLMs necessitate recurrent calls upon each update to the user sequences"。

这些"被承认但未在系统层解决"的陈述，正是新研究问题的入口。

---

## 研究线 C：推荐系统数据管理 / serving 系统（TokaDB 同域）

### C.1 已解决
- **Embedding 表的存储/压缩/分片**：FIITED [arXiv 2401.04408]（训练中细粒度维度剪枝 + virtually-hashed physically-indexed 哈希表）、Mixed-Precision Embeddings [arXiv 2409.20305]、AutoShard [KDD'22, arXiv 2208.06399]（embedding 表分片 RL）。均已验证。
- **特征/embedding serving**：工业界有 feature store 概念（Feast/Tecton 等，arXiv 上以"feature store"为题的论文极少，属 MLops 工业实践，不作为学术立论点）。
- **TokaDB（用户已有工作）**：用户行为序列的组织/存储/访问优化（按用户自述；未独立核验原文）。

### C.2 限制（关键）
> **现有推荐数据系统管理的对象是 embedding / 特征 / 行为序列本身，不包含其 LLM 派生 KV 表示。** 当推荐模型 backbone 从 DLRM 变为 LLM，"派生 KV"成为新的、体积更大、一致性约束不同的数据对象，现有栈不覆盖。

---

## 研究线 D：LLM4Rec 的 serving 系统（交叉地带，最相关）

### D.1 已解决
- **RcLLM [ICDCS'26, arXiv 2605.07443]**（最接近的直接竞品）：面向生成式推荐的超越前缀 KV 缓存。把 prompt 分解为可复用 block；**用户历史 cache 复制 + item cache 相似性感知分片**；affinity 调度 + 选择性注意力纠错。TTFT 降低 1.31×–9.51×。已验证。

### D.2 RcLLM 的边界与限制（关键，决定新问题的精确位置）
RcLLM 解决了 **静态、空间维度的 block 复用**，但明确**未解决**：
1. **时间维度**：假设 block 内容不变。行为流更新导致 user-history block 失效时，RcLLM 没有增量维护/失效机制，退化为全量重算。
2. **新鲜度-精度权衡**：没有定义/评测 staleness 对推荐精度的影响。
3. **多候选 fan-out 的请求级调度**：用了 affinity 调度，但未把"1 用户 × N item"作为一等 workload 抽象。
4. **与行为序列存储的协同**：RcLLM 是 serving 层的缓存，未与底层行为序列存储（TokaDB 类系统）协同。

> **RcLLM 的存在不是否定新问题，而是精确定位了新问题的剩余空间：时间演化维度（新鲜度/一致性）+ 行为存储协同。**

---

## 边界总结图（文字版）

```
                    LLM serving (chat/RAG/agent)        LLM4Rec serving
                    [研究线 A]                           [研究线 D]
KV 不可变假设 ✓     vLLM, SGLang, FlexGen, SparseX...   RcLLM (静态空间复用)
                                                          |
                                            缺口: 时间演化 / 新鲜度
                                                  + 行为存储协同
                                                          ↓
                                               ★ 新问题所在 ★

                    推荐 数据管理                        LLM4Rec 模型
                    [研究线 C]                           [研究线 B]
管 embedding/特征    TokaDB, FIITED, AutoShard...        ReLLa, LIBER, Douyin-10K, GR4AD...
不管 LLM 派生 KV    ←── 缺口 ──→                        承认 serving 瓶颈但当黑盒
```

新问题恰好落在 A/C（系统侧未覆盖 LLM 派生 KV 的时间演化）与 B/D（模型侧把 serving 当黑盒）的**交叉空白**。
