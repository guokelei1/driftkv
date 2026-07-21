# 01 执行摘要

## 推荐的博士研究方向

**数据管理 for LLM-based Recommendation Serving：以"派生个性化 KV 状态"为核心新型数据对象的流式一致性管理**

（Working title: *Streaming Personalized-Context Management for LLM-based Recommendation Serving*）

## 一句话论点（problem-first）

在 LLM 化的推荐系统中，用户的个性化上下文（用户行为序列/兴趣 profile）被**编译**成 LLM 的 KV cache（或等价的自然语言 profile）以支撑在线生成式推荐。该 KV 状态是一个**派生于用户行为流、随时间演化、需在 serving 时复用**的新型数据对象。**现有 LLM serving 系统假设 KV 不可变（为 chat 设计），现有推荐数据系统只管 embedding/特征（不管 LLM KV），二者都无法处理"流式行为更新 → 派生 KV 状态的新鲜度/一致性维护"这一新问题**，导致工业界只能退化为"nearline 预计算 + 在线轻量 serving"，并因此承受已知的 staleness 损失。

## 为什么是新问题（满足 CCF-A "新问题"标准）

| 维度 | 是否新 | 证据 |
|---|---|---|
| 新数据对象 | 是 | "派生于行为流、演化的个性化 KV/profile"——既非传统 embedding，也非 chat 的静态前缀 |
| 新 workload | 是 | 1 个用户 × N 个候选 item 的 fan-out；流式行为增量；catalog 级 item KV 共享 |
| 新一致性模型 | 是 | KV 新鲜度 vs. 重算延迟 vs. 推荐精度的 trilemma，现有 prefix-cache 无此维度 |
| 新评价维度 | 是 | 新鲜度-延迟-精度三维权衡（现有工作只评 TTFT/吞吐/精度） |

## 为什么现有方法不能解决（principle #6 的关键）

1. **vLLM/PagedAttention [SOSP'23]** 与 **SGLang [arXiv 2312.07104]** 的 prefix cache 假设 KV 一旦计算即不可变。当用户行为流更新导致个性化上下文变化时，prefix 命中率为 0，被迫 O(L) 全量重算——而真实变化往往只是少量新增/重排行为。
2. **RcLLM [ICDCS'26, arXiv 2605.07443]** 是最接近的工作（超越前缀的分块 KV + 分层分布式存储），但它处理的是**静态**分块复用，**没有时间演化/新鲜度维度**。
3. **模型侧工作**（ReLLa [WWW'24]、LIBER、HyMiRec、ReLLaX、Douyin-10K [WWW'26]）通过检索/摘要/codebook 缓解"长序列不可理解"，但都把数据移动/serving 当成黑盒，甚至明确承认"推理延迟与特征拉取带宽"是部署瓶颈而未解决。
4. **工业 GR 部署**（Netflix [arXiv 2605.23312]、Kuaishou GR4AD/OneMall/Taiji、LWGR）普遍采用"nearline 预计算 + 在线 serving"，**Netflix 论文明确指出 "cached serving can make the immediate next-token target stale"**——这是真实存在的、被工业界观察到但未从系统层面解决的问题。

## 与 TokaDB 的关联性

TokaDB 解决"用户行为序列的高效组织/存储/访问"。本方向把同一对象（用户行为序列）的**管理范围**从"原始序列 + embedding"扩展到"原始序列 + 其 LLM 派生 KV 表示"，并新增**流式一致性**维度。是已有积累的自然延伸而非跳跃。

## 4×A40 可验证性

- 模型：Llama-3-8B / Qwen2.5-7B/14B（4×A40-48G = 192GB，可 vLLM/SGLang serving，可做 LoRA 微调出 LLM4Rec 模型）
- 数据：Amazon Reviews / MovieLens / KuaiRec（公开）+ ALPBench [arXiv 2602.03056]（长程个性化行为 benchmark）
- workload：可构造流式 LLM4Rec serving trace（行为流到达 + 多候选生成请求）
- 指标：TTFT、吞吐、KV 命中率、**新鲜度 staleness**、推荐精度（Recall/NDCG/Hit）
- 详见 `05_experiment_feasibility.md`

## 目标会议

- 系统侧：OSDI / SOSP / ATC / EuroSys（LLM serving 系统）或 SIGMOD / VLDB（数据管理）
- 取决于切入点：若偏一致性/存储抽象 → SIGMOD/VLDB；若偏 serving 系统/调度 → OSDI/SOSP/ATC

## 最小可发表单元（第一篇论文候选）

**"Freshness-Aware KV Cache for Streaming LLM-based Recommendation"**
- 贡献：定义"派生个性化 KV 的新鲜度"问题 + 形式化一致性-延迟-精度 trilemma + 给出增量维护/失效机制 + 在流式 LLM4Rec serving benchmark 上证明相对 vLLM/SGLang/RcLLM 的优势。
- 这一篇即可验证整个博士方向的立论。
