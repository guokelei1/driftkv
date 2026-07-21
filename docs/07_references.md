# 07 参考文献（核验状态）

本清单仅列出在本会话中通过 arXiv API 核验过标题/作者/摘要的论文。"核验状态"列如实标注。**未在本会话核验的论文（如 Persia、DeepSpeed/ZeRO、FlashAttention）不列入立论引用**，仅在备注中提示使用前须复核。

格式：[编号] 标题 — 作者 — 发表/arXiv — 核验状态 — 用途

## A. LLM serving / KV cache（研究线 A）

- [A1] Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM) — Kwon et al. — SOSP 2023 / arXiv:2309.06180 — ✅ 核验 — 不可变前缀基线
- [A2] SGLang: Efficient Execution of Structured Language Model Programs — Zheng et al. — arXiv:2312.07104 — ✅ 核验 — RadixAttention 前缀复用基线
- [A3] FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU — Sheng et al. — ICML 2023 / arXiv:2303.06865 — ✅ 核验 — 单 GPU/分层 offload 背景
- [A4] SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving — Zhang et al. — arXiv:2606.01751 — ✅ 核验 — 段级静态复用（landscape）
- [A5] Adaptive KV Cache Reuse for Fast Long-Context LLM Serving (CacheTune) — Li et al. — arXiv:2605.24022 — ✅ 核验 — 非前缀复用（landscape）
- [A6] HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching — Liu et al. — arXiv:2607.01299 — ✅ 核验 — landscape
- [A7] Not All Tokens Are Worth Caching (SAECache) — Fang et al. — arXiv:2605.18825 — ✅ 核验 — 语义感知驱逐
- [A8] Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving — Qiu et al. — arXiv:2605.03375 — ✅ 核验 — SSD KV 分层
- [A9] Prefill-as-a-Service — Qin et al. — arXiv:2604.15039 — ✅ 核验 — 跨数据中心 KV
- [A10] CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands — Wang et al. — arXiv:2603.21257 — ✅ 核验 — KV 加载一等调度
- [A11] Forget Without Compromise: Nexus Sampling for Streaming KV-Cache Eviction — Duong et al. — arXiv:2606.23961 — ✅ 核验 — 流式 token 驱逐（区别于行为流）
- [A12] Enabling KV Caching of Shared Prefix for Diffusion Language Models (bicache) — Go et al. — arXiv:2606.07571 — ✅ 核验 — 挑战 KV 不可变（DLM 场景，非推荐）
- [A13] Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving — Yu et al. — arXiv:2604.25899 — ✅ 核验 — agent serving（D4 否决依据）
- [A14] Streaming Knowledge Compilation: Proactive Materiality-Scored Pinning for Time-Evolving LLM Wikis — Huerta — arXiv:2606.09877 — ✅ 核验 — 时变编译 KV 的相邻工作（非推荐）

## B. LLM-based / Generative Recommendation（研究线 B）

- [B1] ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation — Lin et al. — WWW 2024 / arXiv:2308.11131 — ✅ 核验 — 长序列不可理解问题；模型侧
- [B2] Full-Stack Optimized LLMs for Lifelong Sequential Behavior Comprehension (ReLLaX) — Shan et al. — arXiv:2501.13344 — ✅ 核验 — 模型侧
- [B3] LIBER: Lifelong User Behavior Modeling Based on Large Language Models — Zhu et al. — arXiv:2411.14713 — ✅ 核验 — 流式分块；承认重算开销
- [B4] HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation — Zhou et al. — arXiv:2510.13738 — ✅ 核验 — 承认推断/带宽瓶颈
- [B5] Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling (Douyin) — Guan et al. — WWW 2026 / arXiv:2511.06077 — ✅ 核验 — 10K 长序列工业实践
- [B6] ALPBench: A Benchmark for Attribution-level Long-term Personal Behavior Understanding — Ren et al. — arXiv:2602.03056 — ✅ 核验 — 评测 benchmark
- [B7] Towards Generalizable and Efficient Large-Scale Generative Recommenders (Netflix) — Xu et al. — arXiv:2605.23312 — ✅ 核验 — staleness 直接证据
- [B8] Generative Recommendation for Large-Scale Advertising (GR4AD, Kuaishou) — Xue et al. — arXiv:2602.22732 — ✅ 核验 — 工业 GR serving
- [B9] OneMall: End-to-End Generative Recommender Family at Kuaishou E-Commerce — Zhang et al. — arXiv:2601.21770 — ✅ 核验 — 工业 GR
- [B10] Taiji: Pareto Optimal Policy Optimization for Industrial LLM-Enhanced Recommendation — Li et al. — arXiv:2606.03866 — ✅ 核验 — 工业 LLM4Rec
- [B11] LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation — Mu et al. — arXiv:2605.18771 — ✅ 核验 — nearline 预计算+在线 serving 证据
- [B12] LLM-Based User Personas for Recommendations at Scale — Wang et al. — RecSys 2026 Industry / arXiv:2606.12198 — ✅ 核验 — 大规模在线 LLM 推断部署
- [B13] Recurrent Preference Memory for Efficient Long-Sequence Generative Recommendation (Rec2PM) — Chen et al. — arXiv:2602.11605 — ✅ 核验 — KV 存储代价证据

## C. 推荐数据管理 / embedding 存储（研究线 C）

- [C1] Fine-Grained Embedding Dimension Optimization During Training (FIITED) — Luo et al. — arXiv:2401.04408 — ✅ 核验 — embedding 存储/剪枝
- [C2] Mixed-Precision Embeddings for Large-Scale Recommendation Models — Li et al. — arXiv:2409.20305 — ✅ 核验 — embedding 压缩
- [C3] AutoShard: Automated Embedding Table Sharding for Recommender Systems — Zha et al. — KDD 2022 / arXiv:2208.06399 — ✅ 核验 — embedding 分片

## D. LLM4Rec serving 系统（研究线 D，最相关）

- [D1] RcLLM: Accelerating Generative Recommendation via Beyond-Prefix KV Caching — Zhao et al. — ICDCS 2026 / arXiv:2605.07443 — ✅ 核验 — 最接近竞品（静态空间维）

## 备注：未核验、不作为立论引用的常见论文

以下论文虽为领域常识，但在本会话未通过 API 核验，**不作为本报告立论依据**；正式撰写时须独立复核后再引用：
- Persia（分布式推荐训练系统）— ⚠️ 本次核验发现 arXiv:2111.05844 实为天体物理论文，记忆有误，须重新查证正确出处后再引用。
- DeepSpeed / ZeRO（Rajbhandari et al., SC 2020）— 未核验。
- FlashAttention（Dao）— 未核验。
- Feast / Tecton（feature store）— 属工业 MLops，arXiv 以"feature store"为题的论文极少，不作学术立论点。

> 原则#4 已落实：宁可少引，不引未核验。
