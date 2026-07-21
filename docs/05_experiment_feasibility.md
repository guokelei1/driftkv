# 05 实验可行性与评测方案（4×A40 约束）

## 5.1 硬件约束分析

- 4× NVIDIA A40，单卡 48GB，合计 192GB。
- 不适合：百卡级预训练基础设施优化（已排除）。
- 适合：7B–14B 开源 LLM 的 serving / LoRA 微调；流式 serving 系统原型；构造真实 workload trace。
- A40 相对 A100/H100 带宽较低，但这反而**有利于暴露 KV 搬运 bottleneck**（更真实地体现存储/一致性设计的价值），不削弱方法可验证性。

## 5.2 模型选择（均开源、可复现）

| 用途 | 模型 | 理由 |
|---|---|---|
| LLM4Rec backbone | Llama-3-8B / Qwen2.5-7B | 4×A40 可 vLLM/SGLang serving；社区有 LLM4Rec LoRA 配方 |
| 稍大规模验证 | Qwen2.5-14B | 4×A40 张量并行可跑，验证方法可扩展性 |
| 生成式推荐 backbone | 可选：在上述模型上 LoRA 微调出 seq-rec 模型（参考 ReLLa/HyMiRec 公开配方） | 避免依赖闭源工业模型 |

## 5.3 数据集与 benchmark（公开可构造）

| 数据集 | 用途 | 获取 |
|---|---|---|
| Amazon Reviews / MovieLens-25M | 行为序列 + 推荐评测 | 公开 |
| KuaiRec / KuaiRand | 短视频流式行为 | 公开 |
| ALPBench [arXiv 2602.03056] | 长程个性化行为理解 benchmark | 公开 |
| LongBench / Needle-in-Haystack | 长上下文 KV 评测（背景） | 公开 |

### 需自构造的 benchmark：Streaming LLM4Rec Serving Trace
**这是工作1/5 的关键产出之一，本身具发表价值**。构造方法：
1. 从上述数据集还原用户行为时间戳，生成**行为流**（按时间到达）。
2. 在线请求 = (用户, 候选集 C) 的生成式推荐，按泊松/真实到达模式生成。
3. 标注 ground-truth（用户下一交互/留存）用于精度评测。
4. 暴露可调参数：行为流速率、nearline 刷新周期、候选集大小 N、新鲜度阈值。

## 5.4 对比基线（均可在 4×A40 复现）

| 基线 | 说明 | 实现 |
|---|---|---|
| vLLM prefix cache [SOSP'23] | 不可变前缀基线 | 开源 |
| SGLang RadixAttention [arXiv 2312.07104] | radix 前缀复用 | 开源 |
| RcLLM [ICDCS'26] | 静态超越前缀 + 分层存储（最强调基线） | 论文公开则复现；否则按论文描述复刻核心机制作对比 |
| Oracle（全量重算，强一致） | 上界精度/最差延迟 | 自实现 |
| Nearline-only（不刷新） | 下界新鲜度 | 自实现 |

## 5.5 评测指标（三维，体现"新评价维度"）

| 维度 | 指标 |
|---|---|
| 延迟/吞吐 | TTFT、TPOT、QPS、P90/P99 延迟、SLO 达成率 |
| 新鲜度（新） | 平均 staleness（未编译新行为数 / 滞后时间）、新鲜度违约率 |
| 精度 | Recall@K、NDCG@K、Hit@K（推荐）；与 Oracle 精度差 ε |
| 系统 | KV 命中率、重算量、HBM/DRAM/SSD 流量、失效传播开销 |

**核心实验图**：新鲜度-延迟-精度 Pareto 前沿。证明所提方法在前沿上优于所有基线（尤其 RcLLM 在时间维上的空白被填补）。

## 5.6 可行性结论

- 全程无需超大规模集群；4×A40 足够跑通端到端原型与对比实验。
- 所有模型/数据集/基线开源或可构造。
- 自构造的 streaming serving benchmark 既服务自身实验，也是独立贡献。
- 风险点：RcLLM 若不开源需复刻；可通过作者联系/按论文复刻核心机制缓解。
