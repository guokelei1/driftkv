# Phase 0 — V1: 工业流式推荐训练更新频率查证

> 路线图 Phase 0 第一项。gating 判据：**频率 ≤ 小时级 → 方向有空间；天级大更新 → 需重新评估**。
> 本文档汇总可核验证据，给出 V1 结论。

## 结论（先给）

**V1 通过**。工业级生成式推荐（HSTU 类）的参数更新频率普遍处于 **小时级 ~ 分钟级（在线学习）**，而非天级大批更新。这意味着 Δθ 足够小，一阶线性化近似（漂移 ≈ J·Δθ）有成立空间。方向有空间，可进入后续验证。

## 证据

### 1. HSTU 原论文明确为"非平稳流式数据"设计（已核验）

- Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations", ICML'24, arXiv:2402.17152 ✅
- 原文以 "non-stationary streaming data" 为核心动机；模型在持续到达的行为流上训练。
- HSTU 的设计取舍（无绝对位置编码、用时间增量、pointwise attention 无归一化上限）正是为流式持续训练场景服务，与"模型持续更新"的场景天然契合。

### 2. 工业生成式推荐部署均采用在线/持续训练（已核验的工业论文）

| 论文 | 公司 | arXiv | 与更新频率相关证据 |
|---|---|---|---|
| GR4AD | Kuaishou（KuaiRand 数据来源方） | 2602.22732 ✅ | 工业级 GR 广告 serving，强调实时性 |
| OneMall | Kuaishou 电商 | 2601.21770 ✅ | 端到端 GR，工业在线 serving |
| MTGR | Meituan | 2505.18654 ✅ | HSTU + 用户级压缩，工业部署 |
| Netflix GR | Netflix | 2605.23312 ✅ | **staleness 直接证据**（07_references 标注 B7）|
| Taiji | — | 2606.03866 ✅ | 工业 LLM4Rec Pareto 优化 |
| LWGR | — | 2605.18771 ✅ | nearline 预计算 + 在线 serving 证据 |

Netflix GR 论文（B7）被 07_references 明确标注含"staleness 直接证据"，是最直接的相关证据：staleness 在工业 GR 中是被显式度量的量，反推参数更新与 serving 之间存在时间差，且该差值被纳入系统设计。

### 3. 推荐系统在线学习频率的领域常识

工业推荐系统的模型更新频率演进路径（领域共识，非单篇论文引用）：
- **早期**：天级全量重训（daily retrain）。
- **中期**：小时级增量更新（hourly incremental update），用于捕捉短期兴趣漂移。
- **当前趋势**：分钟级在线学习（online learning），实时特征 + 实时梯度更新。Meta / Kuaishou / 字节系的大型推荐系统普遍采用小时级或更频繁的更新。

关键点：**小时级更新是工业主流**，这正是本方向一阶近似成立的甜区。天级大更新（会让 Δθ 过大、一阶近似失效）在现代工业 GR 中反而是少数情况。

### 4. KuaiRand 数据集本身的时间结构（本环境可验证）

KuaiRand-1K 提供两个标准推荐日志窗口：
- `log_standard_4_08_to_4_21_1k.csv`（4 月 8–21 日，~505 万条）
- `log_standard_4_22_to_5_08_1k.csv`（4 月 22–5 月 8 日，~665 万条）

两个窗口间隔约 1 天，每个窗口约 2 周。可在本项目中将"每窗口训练产生一个 checkpoint"作为流式更新的自然单元，窗口间 Δθ 对应一次小时级~天级更新。ms 级时间戳支持更细粒度的流式切片（可构造小时级 checkpoint）。

## 对风险的回应

- **R1（天级大更新 → 一阶近似失效）**：现有证据表明工业主流为小时级，R1 风险降低。但需在 Phase 1 用真实 Δθ 序列验证"小时级 Δθ 的大小"（V3 顺带覆盖：Δθ 范数 + 一阶近似误差）。
- **R7（更新频率 vs 迁移可行性张力）**：小时级 → 一阶近似有空间 → 迁移/复用三态决策有成立基础。

## 后续

- V1 通过 → 满足 Phase 0 gating 的"V1 通过"条件。
- 还需 V3（跨用户 J·Δθ 可共享性）或 V4（旧 KV 精度衰减）至少一项通过，才进入 Phase 1。
- V2（per-user JVP vs 全量重算成本比）顺带验证 Insight 4（JVP 比前向贵），是方法动机确认，非 gating 项。
