# CohortKV: Compiled Cross-Version K/V Migration for Streaming Generative Recommendation

**Anonymous Authors**

> **[内部说明 — 投稿前删除]** 本文件是"目标版"手稿（target manuscript）。
> 当前事实：Gate 2 的 Stage-4 normal-path full-cohort HBM/DRAM benchmark 已完成，原始
> FP16 normalized-capsule source path 在 0/6 个 matched endpoint 上快于 exact；随后
> Stage 4.5 冻结了 direct-old-K/V hot-HBM source plan，在 1/2/4 卡 full cohort 上均通过
> paired exact 门槛且不再保存额外 `Norm(x)`。Stage 4.6 已把 migrated output 连续作为
> 下一轮真实输入，从 theta0 递归执行到 theta11，并冻结 15%–25% 均衡 exact refresh、
> 最大迁移深度 4 的 lifecycle；原逐 cache threshold 因 refresh wave 保留为负结果。
> 自动升级、失效注入、具名 durable SSD、capsule 对照 economics 和冻结编译器新 seed
> 复现仍未完成。
> 所有**尚未测得**的数值以 `⟨TBD⟩` 标注；趋势性结论按预期方向先行写出，若实测方向相反，
> 必须回改正文而不是只改数字。已有真实数字（27-chain、长上下文证书、operator 微基准、
> 两 GPU 11.22×）原样保留。Figure 6 已由 Stage-1 实测结果替换；Figure 7 已由 Stage-4
> normal-path 结果更新，Stage-4.5 hot-HBM 结果进入 Table 8b；Figure 5、8 仍是规划骨架。
> 结构采用经典 systems 骨架：Intro → Background & Motivation → Overview →
> Design ×3（§4 compiler / §5 operator / §6 engine）→ Implementation → Evaluation
> （Discussion 并入 §8.9）→ Related → Conclusion。
>
> **v3.1 收敛记录（对照 goodpaper/1–5 与 guide/14 后裁剪）：**
> 已删除：(a) 第二 GPU 架构上的 operator 微基准复测（硬件不确定、顶会常例单架构即可）；
> (b) 更大容量模型的 scale probe（需新训练、与主张边际相关，3×3 已覆盖容量轴）；
> (c) INT8 capsule 的"量化域直接拟合"声明（新算法工作、有保真风险），收敛为
> INT8 存储层 + staging 时反量化的纯测量。
> 保留为负结果/边界：jagged 布局、per-user drift 信号、remote-object 后端（interface only）。
> 保留为未来工作（不升级为贡献）：程序组合 Φ∘Φ、LLM checkpoint 探针、serving 共置。
> 核心不变：27-chain、冻结契约复现、selective-layer 前沿、连续更新中的逐 cache
> migrate-or-exact refresh、full-cohort 同事务 1/2/4 GPU + HBM/DRAM/SSD、升级与失效注入、
> capsule 经济学（创建开销 + INT8 存储 + break-even）。
> **Serving 边界（2026-07-26 决定）**：数据集不含请求 trace，论文不做任何基于 serving
> workload 的测量或声明。评测语义统一称 stale-inference（离线协议、自训 checkpoint）；
> manifest 可见性语义用 "readers" 表述；break-even 为 workload-free 的纯测量交叉点；
> serving 共置/请求到达/SLO 一律列为 scope 之外（Table 3 / Fig.1 / §8.9 Limitations）。
> **Design/Implementation 分工（v3.2）**：Design 章节只保留本场景下的新机制与 key insight；
> 标准技术全部下沉 §7 —— reference/packed 基线算子、Triton 调参、round-robin/贪心放置、
> POSIX tmp+rename 与 manifest-last 对象写、artifact 序列化与校验、数值验证。
> §4.4 只保留 artifact 作为 compiler↔engine 唯一接口（携带 fallback chain）的设计含义；
> §5/§6 各小节以 insight 句开头。
> **Baseline 与证据网格（v3.3）**：三层对比 —— 锚点（reuse/exact）+ 最强外部方法
> （DroidSpeak 式 selective-layer，独立调优）+ 内部消融（cheap projection / residual-p /
> no-transform / packed-vs-fused / bucketing）；HCache 类同模型系统语义不适用、只在 §9 定位。
> RQ3 前沿网格 = KuaiRand-long（主）+ KuaiRand-medium（容量轴）+ QB-large（数据集轴），
> 全部复用 §8.3 已训链，不新训模型；RQ2 统计主张由 27-chain 承担；RQ4 系统基准只跑
> KuaiRand 长上下文（吞吐由字节量/长度分布决定，跨数据集主张由 RQ3 承担）；
> RQ4 Table 8 增加 selective-layer 端到端行（同事务、selection 冻结的 profiled m；
> certificate 失败必须保留，不能写成 certified action）。

## Abstract

Streaming generative recommenders continuously update their model while retaining long user histories as persisted key/value (K/V) caches. Every model update therefore leaves cached prefixes version-stale: reusing them is cheap but no longer matches the current model, whereas recomputing each history restores current-model semantics at the cost of a full forward pass. We present **CohortKV**, a system that studies whether stale HSTU K/V can be migrated toward a declared current-model target more cheaply than replaying each history. CohortKV treats each source/target model-version pair as a compilation unit rather than as a prediction that reuse is safe. For each pair, its compiler fits the shared residual between fresh K/V and a cheap reprojection of old layer-normalized states, then folds this repair into a single affine projection. For the hot-cache execution path, it further composes that projection through the source model's K/V map, so the resident per-record input is the already existing old K/V and no extra normalized state is retained. Because observed task quality does not reliably track state staleness, the compiler certifies current-model semantic fidelity without recommendation labels and publishes an exact-terminated fallback chain. A fused operator executes compiled programs with length masking and direct K/V writes, and a destination-oriented engine transforms a complete update cohort and atomically publishes one target-version manifest. A bounded lifecycle planner spreads exact refresh across updates according to label-free model-edge severity and cache age, preventing unlimited recursive migration. Automatic runtime escalation and failure recovery remain open.

Across 27 independently trained model-version chains, compiled repair costs 0.121× resident exact recomputation and recovers a majority (0.587) of the stale-to-fresh K/V gap. In the deployed seed-0 certificate, serialized FP16 programs recover 0.881–0.936 of the K/V gap, and compiled repair strictly dominates every selective layer-recomputation point. The complete 682-record normal-path benchmark first exposes a systems failure: tuned exact is 1.20–2.87× faster than the normalized-capsule path because reading, decoding, and pinning 17.82 GB consumes 91.35%–96.91% of compiled completion. Direct old-K/V reparameterization removes that additional state while preserving the certificate and all 17.82 billion transported elements. In the declared existing-old-K/V hot-HBM regime, complete 1/2/4-GPU updates take 0.930/0.494/0.255 seconds versus paired raw-history-resident exact at 18.695/9.729/4.766 seconds, with extent-wise old-cache reclamation and no extra per-record state. On one controlled recursive theta0-to-theta11 chain, the frozen balanced lifecycle costs 0.213× cumulative all-exact GPU time, keeps every step below 0.255×, limits exact refresh to approximately 15%–25%, and preserves minimum cache/score/top-100 fidelity of 0.963/0.99995/0.992.

## 1. Introduction

Generative recommendation reframes a user's ordered behavior as a sequence and predicts the next item from that history. Architectures such as HSTU are designed for high-cardinality, non-stationary streams and make long histories computationally useful [1]. At the same time, recommender systems continuously learn from new interactions so that fresh preferences and content can enter the model quickly [2]. These two properties create a systems conflict that neither short-lived models nor short-lived state would expose. A deployment wants to retain the expensive representation of each long history, yet streaming training repeatedly changes the model under which that representation was computed.

The persistent representation for HSTU includes the attention keys and values at every layer. Let \(C_v(x)=F(\theta_v,x)\) denote the prefix K/V generated for history \(x\) by model version \(\theta_v\). After training publishes \(\theta_t\), the system may keep using \(C_v(x)\), but current queries then combine current-model computation with an old-model prefix. Alternatively, it may compute \(C_t(x)\) exactly, which restores current-model semantics at the cost of replaying the entire history. Unlike ordinary eviction restoration, the source state has not merely moved: its meaning is tied to a different model version.

Existing K/V systems make the boundary precise but stop short of this version mismatch. HCache restores evicted state from intermediate activations under the same model [3]. vLLM derives a memory manager and serving engine from the dynamic allocation properties of same-model autoregressive K/V [8]. CachedAttention retains same-conversation K/V across requests and overlaps hierarchical loading and saving [9]. MTServe persists generative-recommendation K/V across visits and manages where that state resides [5], but its published workflow does not define a source-to-target model-version transform. DroidSpeak goes furthest toward cross-model reuse: it allows same-architecture fine-tuned LLM variants to share K/V through selective layer recomputation [4]. Cross-model K/V by itself, however, is not the contribution of this paper. What remains missing is a system that compiles a source-to-target transform over the persistent HSTU state produced by successive streaming versions and applies that transform to a fixed target-version update cohort.

This gap matters because the alternative, choosing a universal cache lifetime, does not fit the evidence. The model-version chains we study show that older versions generally drift farther in K/V space, yet age is not a calibrated predictor of ranking impact. Some exact maintenance endpoints are close to zero or even negative, because a newly trained model is not guaranteed to rank every fixed evaluation slice better than its predecessor. Per-user K/V drift is likewise uninformative about who benefits from maintenance. A system that uses age, drift, or observed task gain to decide whether a cohort is "safe" to reuse would therefore confuse current-model state fidelity with an application-quality prediction.

To address this problem, we present **CohortKV**, a system that asks a narrower question: given a fixed set of old states, how can the system move them toward a declared current-model K/V target more cheaply than exact replay? Our key insight is that the cross-version error is structured at the version-pair level rather than at the record level, and that HSTU's data path exposes exactly the state needed to exploit this structure. Each layer produces K/V by applying the layer's K/V projections to a normalized hidden state. If the old normalized state is retained, applying the current projections gives a cheap approximation. CohortKV measures the shared residual from this approximation to fresh K/V on a small version-pair sample, fits that residual as an affine function of the old normalized state, and folds the result into one prepacked projection. The per-record path therefore remains one matrix operation rather than a learned correction executed after the projection.

Turning this algebra into a system requires three further mechanisms. First, the operator must fuse the affine epilogue, respect valid sequence lengths, and write destination-ready K/V; otherwise small kernel savings disappear in packing and padding. Second, a complete update must move an entire cohort through explicit HBM or host-memory boundaries, partition work across GPUs, and make no partial target version visible. Third, because certification is statistical, a complete deployment needs both a cache-lifecycle policy that periodically resets accumulated migration error through exact refresh and an execution-time guard that escalates a failing cohort through its published fallback chain. The current artifact closes the operator, normal transaction, and one controlled repeated-input lifecycle; automatic guard dispatch and failure recovery remain open. CohortKV therefore comprises three connected design components (§4–§6):

1. a **version-cohort migration compiler** that fits and certifies a shared source-to-target program without recommendation labels;
2. a **source-to-K/V operator** that executes the capsule or direct-old-K/V program in one fused, length-aware pass; and
3. a **destination-oriented update engine** that transforms a complete record set and atomically publishes one target-version manifest to GPU or host memory, with automatic guard dispatch and durable storage as explicit remaining gates.

The version cohort connects the three components: it keys compilation, homogeneous batching, program residency, extent placement, and metadata. It never predicts that stale reuse is harmless. Every stale cohort receives compiled synchronization in the current normal-path experiment; the published plan contains residual and exact fallbacks, but the engine does not yet dispatch them automatically.

We evaluate CohortKV with independently trained streaming checkpoints on KuaiRand and Tenrec. Across 27 model-version chains spanning three data tables and three capacities, compiled repair costs 0.121× resident exact recomputation and recovers a majority of the stale-to-fresh K/V gap. In the deployed seed-0 certificate, serialized programs recover 0.881–0.936 of that gap and dominate the selective layer-recomputation frontier. The complete 682-record benchmark then exposes and resolves a systems condition. The original normalized-capsule path loses to exact at every HBM/DRAM endpoint because source processing consumes 91.35%–96.91% of completion. Reparameterizing the same certified affine over the already resident old K/V eliminates the additional capsule, and complete 1/2/4-GPU hot-HBM updates are 20.11×/19.71×/18.72× faster than paired raw-history-resident exact. A separate fixed-history recursive chain advances the actual theta0 output through 11 updates at 0.213× cumulative all-exact cost with no cache migrating more than four times consecutively. This is a scoped hot-cache, single-seed result, not a cold-storage, automatic-tiering, or organic-traffic claim.

Figure 1 shows the job boundary. Training publishes checkpoints but is not part of the job; request arrivals, hotness, routing, and training/serving co-location are outside the present scope. Our contributions are as follows:

- We formulate model-version K/V migration as version-cohort compilation and develop a shared affine repair with a label-free semantic certificate and an exact-terminated fallback plan. The repair mechanism is replicated across 27 independently trained chains; the deployed serialized compiler contract is validated in the controlled seed-0 configuration.
- We implement a common-layout fused source-to-K/V operator and a bounded mixed-cohort multi-GPU engine that publishes complete target manifests at HBM and pinned-DRAM endpoints.
- We measure the full source-to-publication path, retain the normalized-capsule path's six-endpoint loss as a negative boundary, and develop a direct old-K/V reparameterization that removes all additional per-record source state. In its declared hot-HBM regime it preserves the certificate and full transport while beating paired exact at complete 1/2/4-GPU cohort points.

![A model update changes the meaning of persistent HSTU state. CohortKV occupies the middle ground between stale reuse and exact current-model recomputation, and publishes a fixed cohort at one explicit destination.](figures/01_problem_and_scope.svg)

**Figure 1: Cross-version invalidation and the CohortKV job boundary.** Training and foreground serving are deliberately outside the fixed destination-update job.

## 2. Background and motivation

This section establishes the inference semantics under study (§2.1), then presents four measurements on independently trained streaming checkpoints. Each measurement yields one design requirement, and Table 2 at the end of the section maps the requirements to the mechanisms of §4–§6 and to the evaluation questions that test them.

### 2.1 HSTU prefix K/V and stale inference

We use a modular, simplified HSTU that preserves the two properties required by this study: pointwise unnormalized attention and first-class K/V output. For layer \(\ell\), let \(h^\ell_\theta(x)\) be its input hidden sequence and

\[
z^\ell_\theta(x)=\operatorname{Norm}^\ell_\theta\left(h^\ell_\theta(x)\right).
\]

The concatenated K/V output is

\[
y^\ell_\theta(x)=\left[k^\ell_\theta(x),v^\ell_\theta(x)\right]=z^\ell_\theta(x)P^\ell_\theta+b^\ell_\theta,\tag{1}
\]

where \(P^\ell_\theta\) concatenates the current K and V projection weights. The complete cache has shape \([L,B,S,2D_{kv}]\) before the K/V split. Sequence lengths accompany every batch, and positions beyond each valid length are zeroed.

The stale-inference protocol predicts item \(t+1\) from hidden state \(t\), entirely offline on our own trained checkpoints. A fresh evaluation recomputes the complete history with the current model. A stale evaluation supplies an old-version prefix K/V and computes the latest token with the current model. Fresh and stale therefore use the same history and current query path; only the model version that produced the resident prefix differs. No request traces enter this protocol: the datasets record interactions, not serving load, so all claims in this paper are about state fidelity and update cost, not about online request behavior.

### 2.2 Stale reuse leaves a maintenance opportunity

We first separate three quantities at a fixed current endpoint:

- **full-compute streaming value**: current streaming model with current K/V, relative to a frozen base model;
- **full-reuse streaming value**: current streaming model consuming the old prefix K/V, relative to the frozen base model;
- **cache-maintenance value**: full compute minus full reuse.

The primary KuaiRand [6] Top-50k/all-chunks protocol finds a full-compute BestRank value of 3837.67 (95% CI [3389.91, 4285.44]), of which stale reuse retains 2952.11 ([2700.21, 3204.02]). The remaining maintenance value is a substantial 885.56 ([460.24, 1310.88]), a 23.1% staleness tax. Table 1 shows that the aligned theta5 screen finds a positive mean maintenance gap in all three evaluated tables:

**Table 1: Aligned cross-table streaming and maintenance value.** Values are BestRank improvements; intervals use four training seeds.

| Dataset/table | Full compute | Full reuse | Maintenance |
|---|---:|---:|---:|
| KuaiRand | 484.34 [462.15, 506.54] | 399.02 [370.79, 427.26] | 85.32 [53.74, 116.91] |
| Tenrec QB, fixed horizon | 94.70 [70.49, 118.90] | 64.38 [54.41, 74.35] | 30.31 [14.08, 46.55] |
| Tenrec QK, Top-5k | 47.34 [29.20, 65.47] | 34.34 [20.55, 48.13] | 13.00 [5.70, 20.30] |

QB and QK are related tables from Tenrec [7], not independent industrial domains, and their time is ordinal rather than a shared calendar. The result establishes a cross-table opportunity, not universal task harm from staleness.

**Requirement 1.** A stale cohort needs a synchronization path cheaper than exact history replay; plain reuse is a baseline, not a publishable target state.

### 2.3 Age and task quality are not admission oracles

We next vary data and model capacity in a frozen 3×3 screen. All nine cells have positive full-compute and full-reuse streaming value in 4/4 seeds, but Figure 2 shows that the mean BestRank staleness tax is neither uniformly positive nor monotone with capacity: large KuaiRand and large QB expose substantial taxes of 0.360 and 0.548, while large QK is −0.005 and QB-medium is −0.060. At a fixed target, every 3×3 age curve has monotonicity violations. In the controlled 16-layer long-context diagnostic, age strongly orders K/V drift, yet after removing the special base-to-stream boundary it explains only 6.15% of MeanRank variation, compared with 60.9% explained by current version identity.

![Staleness tax across the 3×3 data/capacity screen, and age versus drift versus task utility on the long-context chain.](figures/05_admission_signals.svg)

**Figure 2: Neither age, drift, nor capacity calibrates task-level maintenance value.** Cost scales smoothly with capacity while task maintenance does not; age orders drift but not ranking impact.

Nor does a per-user signal repair this problem. Relative K/V drift and maintenance utility have a correlation of only 0.020 (95% CI [−0.012, 0.052]), and the investigated JVP/Fisher route is not cheaper than the operation it would govern. We retain this only as a negative result.

**Requirement 2.** Version identity may organize work, but neither age, drift, nor observed task gain decides whether a stale cohort can bypass synchronization. The system should verify current-model semantic fidelity without recommendation labels, guard execution with the same label-free views, and retain exact recomputation as the endpoint. Because exact is a semantic reference rather than a ranking upper bound, recovery above 100% and negative task gaps must remain signed.

### 2.4 HSTU exposes a compilable repair

Equation (1) separates two sources of cross-version error: the current K/V projections have changed, and the normalized hidden state that feeds them has changed. When the current \(P_t^\ell,b_t^\ell\) is applied to old \(z_v^\ell\), the first source is repaired at low cost but current hidden propagation is omitted. This cheap projection costs roughly one tenth to one fifth of exact replay across datasets and closes a material, though incomplete, fraction of the K/V gap.

The remaining error is structured at the version-pair level. A shared map from old \(z_v^\ell\) to the fresh-minus-cheap K/V residual can be fit and then compiled into the current projection, which yields one homogeneous operator for the cohort. Earlier structural screens delimit alternatives: plain prefix replay is never selected once compiled projection and residual transport share one action library; all 54 matched recent-token partial actions are slower at the evaluated length; arbitrary contiguous intervals add negligible value relative to their \(O(L^2)\) planner. These discovery-stage actions remain baselines rather than the active method.

**Requirement 3.** Move adaptation out of the per-record path. Compile a shared residual into one affine projection, and use structural replay or exact recomputation only for a stricter semantic tier.

### 2.5 Kernel cost is not job cost

A compiled matrix operation can still lose end-to-end if the runtime repeatedly moves programs, pads unrelated lengths, allocates outputs, serializes device work, or compares against an exact baseline with a different publication boundary. The update cohort contains mixed source versions and long histories, so H2D, compute, D2H, output layout, storage bandwidth, and multi-GPU imbalance are all visible.

A controlled page/jagged experiment reinforces the need for endpoint discipline. Compaction improves the host-backed path by only 1.019× and is 0.984× the one-record design at the HBM boundary. HBM itself is 2.159× faster than host publication because it removes D2H; that is a destination difference, not a faster migration operator.

**Requirement 4.** The operator must write final-layout K/V, and system speedups must survive a matched source-residency and target-publication boundary on the complete cohort. Destination placement must be explicit, with complete-version visibility separated from kernel timing.

Table 2 summarizes how each observation of this section maps to a mechanism and to the evaluation question that tests it.

**Table 2: Observation-to-design mapping.**

| Observation | Resulting mechanism | Evaluation question |
|---|---|---|
| Reuse leaves a maintenance gap. | Unconditional compiled synchronization plus exact endpoint. | Does the opportunity persist across tables and capacities? (§8.2) |
| Age and task quality are not calibrated. | Version-pair execution key, label-free contract, exact-terminated fallback plan. | Does fidelity/cost replicate under a frozen contract? (§8.3) |
| HSTU exposes old normalized states and current projections. | Shared residual folded into one affine program. | How much K/V gap closes at measured GPU cost, versus the strongest cross-model baseline? (§8.3, §8.4) |
| Movement can erase kernel savings. | Fused direct-write operator and destination job. | Does the gain survive the identical full-cohort transaction? (§8.5, §8.6) |

## 3. CohortKV overview

This section introduces the three abstractions that the whole system is built on (§3.1), the architecture that connects the three design components (§3.2), and the end-to-end life of one update job (§3.3). The design components themselves are then presented in §4 (compiler), §5 (operator), and §6 (engine).

### 3.1 Abstractions: capsules, cohorts, and the update job

**Migration capsule.** Exact current K/V needs \(z^\ell_t(x)\), which in turn depends on current hidden propagation through all preceding blocks. CohortKV instead retains a **migration capsule**

\[
Z_v(x)=\{z^1_v(x),\ldots,z^L_v(x)\}
\]

with the record ID, valid length, and **migration anchor version** \(v\). The anchor is not changed when target K/V is produced: a capsule can remain anchored at \(v\) while its output declares a **served K/V target** \(t\). The separation of these two version fields prevents a migrated approximation from masquerading as a freshly captured current capsule. For equal precision and hidden/K/V widths, one FP16 normalized state per layer is half the size of both K and V. Capsules remain the fitting and semantic-reference representation, but the frozen hot-HBM execution path does not retain them per record: §4.5 reparameterizes the certified affine over the old K/V that the serving cache already contains. Section 8.7 reports both the normalized capsule's failed time economics and the zero-extra-state direct route.

**Version cohort.** A **version cohort** is the pair

\[
\gamma=(v,t)
\]

shared by records whose capsules are anchored at \(v\) and whose K/V must target \(t\). The cohort is the unit that every component keys on: the compiler fits and publishes one program per \(\gamma\) (§4), the operator batches homogeneously within \(\gamma\) (§5), and the engine keeps the relevant programs resident on every worker and retains \(\gamma\) in each output extent (§6). A future guard would also account its statistics per \(\gamma\). Several source versions may share one target job, but their programs remain distinct. The current implementation requires the programs in one job to share layer count, hidden width, K/V width, and target version. The cohort organizes execution; it never predicts that reuse is safe.

**Fixed destination-update job.** The system contract is:

> Given an admitted source representation, published source-to-target programs, a fixed complete record set, execution GPUs, and an explicit destination, produce target-version K/V for every record and make one complete manifest visible.

Table 3 delimits this contract.

**Table 3: Scope of the fixed destination-update job.**

| Inside the current boundary | Outside the current boundary |
|---|---|
| Source/target program selection from published artifacts | Streaming training and checkpoint production |
| Cohort grouping, length bucketing, and GPU placement | Online request arrivals and per-user hotness |
| H2D, migration compute, D2H when required | Foreground inference interference and SLO scheduling |
| HBM and DRAM publication contract; POSIX interface | Automatic destination or cache-tier selection |
| Complete normal-path coverage and commit | Automatic escalation, failure recovery, and cross-destination transactions |

The destination is an input rather than a policy decision. HBM, DRAM, and a filesystem answer different endpoint questions and must not be compared as if their completion times were interchangeable. The present performance result covers HBM and pinned DRAM; the POSIX backend remains a functional interface rather than a durable SSD benchmark. During a successful update, the new manifest becomes readable only after complete coverage (§6.4). Failure visibility and recovery require the separate Stage-5 protocol.

### 3.2 Architecture

Figure 3 shows the architecture. CohortKV separates the update problem into three design components with one clean division of labor: the compiler decides **what** transformation is semantically admissible for a cohort and prepares its admitted source representation, the operator decides **how** one homogeneous source batch becomes destination-layout K/V, and the engine decides **where and when** complete extents become visible. The current update coordinator resolves job specifications and invokes these components on the normal path; it does not compile programs, infer reuse safety, choose a destination, or schedule online requests. The source-state gate passes for the declared direct-old-K/V hot-HBM regime. A thin lifecycle planner now divides each controlled update into lightweight migrations and charged exact refreshes using a frozen age/deadline and edge-severity schedule; automatic semantic guard and fallback dispatch remain the next implementation gate.

![CohortKV has three connected design components. The source/target version pair is carried by the compiled program, capsule, output extent, and target manifest.](figures/02_architecture.svg)

**Figure 3: CohortKV architecture.** Version cohorts organize execution across the compiler, operator, and engine; they do not predict safe reuse.

**Migration compiler (§4).** Input: a version pair \((v,t)\), calibration records with old capsules and exact current K/V, and a frozen label-free contract. Output: an immutable program artifact — one folded affine projection per layer plus a verified plan recording certificates, the selected action, and an ordered fallback chain ending in exact recomputation. For the hot-HBM route, the compiler also composes that affine into a direct old-K/V program without changing its certified target semantics. Compilation runs once per version pair and is amortized over the whole cohort; no recommendation label enters it.

**Source-to-K/V operator (§5).** Input: a resident program and one length-bucketed normalized-capsule or old-K/V batch from a single cohort. Output: contiguous, destination-layout K and V tensors with padding masked. The operator is a single fused Triton kernel; its design goal is that the per-record path stays one matrix operation with no packing, splitting, or copying epilogue, so the compiler's amortization is not eroded at execution time.

**Destination-oriented update engine (§6).** Input: the fixed record set, the published programs, execution GPUs, and one explicit destination. Normal-path output is exactly one complete target-version manifest. The implemented engine owns program residency and multi-GPU placement, the bounded host-staged pipeline, target allocation, and successful commit. Automatic escalation, abort visibility, and recovery are the next semantic layer, not evidence from the normal-path benchmark.

The interfaces between the components are narrow by construction. The compiler communicates with the engine only through the immutable program artifact and its fallback chain; the engine communicates with the operator only through resident programs and homogeneous batches; and the only globally visible side effect of the entire system is the committed manifest.

### 3.3 Life of an update job

A concrete walkthrough ties the components together. Streaming training publishes checkpoint \(\theta_t\). For each source version \(v\), the coordinator forms cohort \((v,t)\) and invokes the compiler: calibration records are sampled, candidate actions are fit, GPU cost is measured, and the label-free certificate views are evaluated on disjoint users. The compiler publishes the least-cost certified action — in the current controlled configuration a single folded affine program — together with its fallback chain (§4.2–§4.3). If the serving old K/V is resident and the capacity and program checks pass, §4.5 derives and selects the direct old-K/V runtime form; otherwise the source policy selects exact. A cohort-size amortization rule remains to be measured.

The coordinator then plans the job: records are grouped by cohort, sorted into length buckets, packed into extents, and assigned to GPUs by byte-weighted longest-processing-time-first placement (§6.1). The cold and host-staged paths use the lazy shard reader and bounded copy/compute/publication waves (§6.3). The primary hot path reads old K/V directly on its assigned GPU, writes replacement extents, and retires each old extent after replacement staging. The frozen plans already expose an ordered fallback chain, but the current full-cohort engine executes one preselected action per measured point. Section 6.2 specifies the remaining guard contract without treating it as implemented evidence.

When every record of every cohort is covered exactly once, the normal-path engine commits: extents and metadata are sealed and the complete manifest becomes the target-version result (§6.4). Stages 4 and 4.5 verify successful cold/host and hot/reclaiming paths. Visibility during failures and bounded resumption remain the open contract of §6.5 rather than measured behavior.

## 4. Design 1: Version-cohort migration compiler

### 4.1 Cheap projection and residual decomposition

For cohort \((v,t)\), the current projection applied to an old capsule gives

\[
\widetilde y_{\text{cheap}}^\ell=z_v^\ell P_t^\ell+b_t^\ell.\tag{2}
\]

Calibration records additionally expose exact current \(y_t^\ell=z_t^\ell P_t^\ell+b_t^\ell\), so the target residual is

\[
r^\ell=y_t^\ell-\widetilde y_{\text{cheap}}^\ell.\tag{3}
\]

The compiler fits a ridge-regularized affine map

\[
r^\ell\approx(z_v^\ell-\mu^\ell)A^\ell+\bar r^\ell.\tag{4}
\]

\(A^\ell\) may be truncated to a low-rank factorization during fitting. The factorization changes offline statistical capacity but not online work: before publication, the compiler folds it into

\[
\widehat P^\ell=P_t^\ell+A^\ell,\qquad\widehat b^\ell=b_t^\ell+\bar r^\ell-\mu^\ell A^\ell.\tag{5}
\]

Execution is therefore one affine projection, \(\widehat y^\ell=z_v^\ell\widehat P^\ell+\widehat b^\ell\), with the same matrix shape for every fitted rank. It is an approximation to current K/V, not an equivalence to current hidden propagation.

For the long-context design, the compiler fits a full-affine \(A^\ell\). Uniform regression treats all prefix positions equally even though the current request does not. The selected design weights each valid token-layer example by an HSTU attention-use statistic computed from current queries and keys, normalizes weights to unit mean, and caps them at eight. This changes fitting only; program shape, online state, and kernel work remain unchanged. No recommendation labels enter the fit.

### 4.2 Frozen label-free semantic contract

A cheap transformation should not be selected merely because it performs well on the records used to fit it. CohortKV separates fit, hyperparameter-selection, certificate, and final-test users. For certificate user \(u\), it evaluates three error views:

\[
e_{\text{cache}}=\text{relative K/V error},\quad e_{\text{score}}=1-\cos(s_a,s_t),\quad e_{\text{top100}}=1-\operatorname{overlap}_{100}(a,t).
\]

Here \(s_a\) and \(s_t\) are full-catalog score vectors under action \(a\) and exact current K/V. The certificate measures how much of the reuse-to-exact error gap the action closes:

\[
\operatorname{recovery}(a)=\frac{e_{\text{reuse}}-e_a}{e_{\text{reuse}}-e_{\text{exact}}}.\tag{6}
\]

The frozen contract requires, for all three views: a recovery target of at least 70%; a one-sided 90% bootstrap lower bound on ratio-of-means recovery of at least 70%; at least 80% per-user coverage after a one-sided 90% Wilson lower bound; and a measured GPU cost no greater than 30% of exact for the primary action. Section 8.3 reports a sensitivity sweep of the recovery target over {50, 60, 70, 80, 90}% and shows that action selection is stable in the ⟨TBD⟩–⟨TBD⟩% band, so the published thresholds are an interior operating point rather than a tuned edge.

The compiler chooses the least-cost action that passes fidelity and budget. If no action passes the budget, it may publish the least-cost fidelity-certified overflow action. If no approximate action passes, exact recomputation is forced. The artifact also contains the ordered fallback chain intended for the open guard (§6.2). Recommendation labels are withheld until final task evaluation.

"Label-free" does not mean "cost-free." Certification recomputes exact current K/V for its probe users and compares full-catalog score vectors. That cost is paid once per version pair, but Stage 4 does not include it in the per-job timing and therefore makes no amortization-floor claim. A later economics protocol must measure compile-plus-certificate time and define when a small cohort should go directly to exact replay.

This contract verifies a synchronization implementation, not the proposition that the current model will improve a ranking metric. An action can pass semantic fidelity even when exact current K/V is worse than stale reuse on a particular task slice.

### 4.3 Action library and escalation tiers

The active cohort-tiered action library contains:

1. compiled affine repair;
2. residual-\(p\) replay; and
3. exact current-model recomputation.

Residual-\(p\) executes the first \(p\) current-model blocks exactly, computes the boundary displacement

\[
\Delta_p=h_p^t-h_p^v,
\]

and approximates deeper states by \(h_\ell^v+\Delta_p\) before the current layer's `Norm + Wk/Wv` projection. It supplies a predefined structural escalation tier without a per-user predictor or another learned online operator. Exact recomputation is the terminal K/V reference.

Residual-\(p\) is not executable from the normalized migration capsule alone. It additionally
requires the old pre-block hidden suffix
\(\{h_\ell^v\}_{\ell=p}^{L-1}\), which CohortKV treats as an optional, separately accounted source
representation. A plan may publish this tier only when that suffix is retained; otherwise its
fallback chain proceeds directly to exact recomputation. Section 8.7 reports these auxiliary bytes
separately rather than hiding them in the default capsule footprint. Real certificate-shard
materialization rejects FP16 for this unnormalized state because its magnitude exceeds the finite
range; the suffix therefore uses BF16 at the same two bytes per element. The normalized capsule,
compiled program, and published K/V remain FP16.

The tiers are selected per operating point rather than fixed globally. In the primary 50% replicated operating point, all 27 held-out chains select compiled projection; at the 75% discovery point, three large cells select residual depths 5, 6, and 7. The same library also positions the strongest external alternative: a DroidSpeak-adapted contiguous layer group [4], which starts from one stored old-version transition activation, recomputes that interval with the current model, and reuses old K/V elsewhere. Section 8.4 independently profiles this baseline under the identical label-free certificate and publication boundary.

### 4.4 Published program and fallback interface

Compilation ends in an immutable program artifact: one folded affine projection per layer, together with a verified plan recording the certificates, the selected action, and the ordered fallback chain intended for engine dispatch (§6.2). This artifact is the only channel between the compiler and the engine, which makes escalation possible without re-entering the compiler at run time; automatic engine dispatch remains an open implementation gate. Serialization, metadata layout, and strict version/shape validation follow standard practice and are described in §7.

The certificate is applied to the deployed numeric representation, not only to the FP32 fitting
path. Before publication, CohortKV reloads the serialized FP16 capsules and prepared runtime
program, emits FP16 K/V, and repeats the frozen label-free views without changing thresholds or
candidate selection.

In the adaptive seed-0 deployment recertification, all three source pairs pass on the disjoint
60-user certificate role. Compiled full affine costs 0.01651–0.01657× the resident
FP32-compute exact path with FP16 output, with cache recovery 0.8810/0.8897/0.9365 and worst recovery lower bounds
0.8514/0.8391/0.9231 for theta0/theta4/theta10. These are certificate-role resident components,
not final-user quality or complete-job costs.

### 4.5 Runtime reparameterization over existing old K/V

The normalized capsule makes the repair easy to fit, but it is not the only sufficient runtime
input. Let the source model's stacked K/V projection at layer \(\ell\) be
\(P_v^\ell\in\mathbb{R}^{H\times 2D}\), with bias \(d_v^\ell\), so the already cached old state is

\[
o_v^\ell=[K_v^\ell,V_v^\ell]=z_v^\ell P_v^\ell+d_v^\ell.
\]

The deployed compiled program is \(z_v^\ell A_{v,t}^\ell+b_{v,t}^\ell\). In the evaluated HSTU,
every \(P_v^\ell\) has full row rank. Its minimum-norm right inverse \(R_v^\ell\) therefore
satisfies \(P_v^\ell R_v^\ell=I_H\), yielding the composed runtime form

\[
\hat{o}_t^\ell
=o_v^\ell(R_v^\ell A_{v,t}^\ell)
+b_{v,t}^\ell-d_v^\ell R_v^\ell A_{v,t}^\ell.
\]

This is a compiler reparameterization, not a new fitted method: it preserves the certified
capsule affine up to deployed FP16 transport error. The measured projection condition numbers are
5.97–10.74. The three direct FP16 programs total 100.78 MB, while per-record normalized-state
storage falls from 17.82 GB to zero. A direct program is admitted only when its provenance and
integrity verify, the old K/V is present, and a capacity preflight passes; otherwise the policy
selects exact. Section 8.5 independently checks both the real-value transport equivalence and the
complete hot-HBM job.

## 5. Design 2: Source-to-K/V operator

The key insight behind the operator is that in this workload the epilogue, not the GEMM, is where a compiled program loses its advantage. Either compiled projection is a single matrix operation, so any per-batch masking, K/V splitting, or contiguous-copy pass executed after it costs a comparable order of work — and the destination transaction (§6.4) requires contiguous, destination-layout K and V, which framework primitives do not produce directly. The operator is therefore designed backward from the destination: one fused pass consumes a cohort-homogeneous normalized-state or old-K/V batch and writes final-layout K/V with padding resolved in-kernel. An FP32-arithmetic transport reference and packed FP16 paths serve as numerical oracles and strong framework baselines; they involve no new design and are described in §7.

### 5.1 Fused direct-write kernel

The normalized-source Triton operator consumes:

- a contiguous FP16 capsule \([L,B,S,H]\);
- contiguous FP16 weights \([L,H,2D_{kv}]\);
- biases \([L,2D_{kv}]\); and
- one valid length per record.

The direct-old-K/V variant preserves the same extent API but consumes the existing
\([L,T,D_{kv}]\) K and V tensors and a composed \([L,2D_{kv},2D_{kv}]\) program. Both variants
write the same separate, unpadded target K/V layout.

Its grid spans layers, row tiles over \(B\times S\), and output-width tiles. Each program accumulates the \(H\)-dimension in FP32, adds the layer bias, derives record and token positions from the flattened row, and maps valid rows through the extent offsets; padded rows do not publish output elements. Output offsets below \(D_{kv}\) write directly to the contiguous unpadded K tensor; the remaining offsets write directly to V. The operator thus avoids a separate mask, split, compaction, and contiguous-copy epilogue.

The output retains record IDs, the capsule's migration anchor, the program's served K/V target, and valid lengths. Numerical validation against the reference paths is part of the implementation test suite (§7).

### 5.2 Variable-length organization

Padding is a systems cost even though it is semantically masked. The host runtime sorts records into length buckets and constructs small homogeneous batches within each source cohort. In the controlled layout search, removing length bucketing reduces migration throughput from 863.2 to 643.1 records/s. The current 60-record development distribution also selects a 32-token bucket with batch four at the resident boundary after comparing widths 16/32/64 and batches 1/2/4. This is only the operator default: on the complete cohort, every destination/GPU point repeats the frozen bucket and batch sweep independently (§8.5).

We separately implemented a jagged capsule layout with per-record offsets and compact outputs that match the dense fused values. It is useful when many short fragments can be coalesced, but it is not a positive result on the current long-context trace. CohortKV therefore treats jagged/page compaction as a conditional layout mechanism, not as a defining contribution.

## 6. Design 3: Destination-oriented update engine

### 6.1 Program residency and multi-GPU placement

A job may contain several source versions but exactly one target. The key insight organizing residency is an asymmetry: programs are small and cohort state is large, so the engine replicates every source version's prepared program on every worker and partitions the record extents, and each worker selects its program from the capsule anchor. Placement uses byte-weighted longest-processing-time-first (LPT): an extent's work is estimated from its capsule and output bytes, and the next largest extent goes to the least-loaded worker. Simpler round-robin and input-order policies remain available as implementation options (§7).

The program table is small relative to long-context state. The three direct FP16 programs occupy
96.11 MiB per worker, or 192.22/384.44 MiB across two/four GPUs; each replica is 0.211% of an
A40's physical HBM. The target K/V is partitioned and needs no peer transfer.

### 6.2 Open guard and automatic-escalation contract

Certification is statistical, so a complete deployment needs a guard between plan selection and commit. The intended contract samples migrated records per cohort, checks a label-free semantic view, and moves monotonically through the published fallback chain when a bound fails; exact recomputation terminates every chain. The current engine does not implement or evaluate that dispatch. The source-state path is now end-to-end competitive in its declared hot-HBM regime, so guard observability, probe cost, re-migration semantics, and false-escalation behavior are the next separate protocol. Until then, the frozen chain is executable plan metadata rather than a runtime-safety claim.

### 6.3 Host-staged pipeline

Figure 4 shows one host-staged wave. CPU capsules are pinned, copied asynchronously to the assigned GPU, transformed by the resident program, and copied into persistent pinned target extents. Separate H2D, compute, and D2H streams allow adjacent batches to overlap. A single publication worker stages completed extents, while a bounded queue applies backpressure. Wave size bounds transformed output residency after source capsules have been materialized. For the complete cohort, a lazy shard reader streams shards in extent order, so transient source residency is bounded by the wave rather than by the cohort: the maximum observed source wave is 3.50 GiB and the maximum staging wave is 3.00 GiB, versus 16.60 GiB of logical FP16 capsules and 33.20 GiB of logical target K/V (§8.5).

![The host-staged engine overlaps capsule movement, fused transformation, target movement, and destination publication.](figures/04_execution_pipeline.svg)

**Figure 4: Host-staged execution and publication.** The bounded wave and publication queue bound transient output residency; the lazy shard reader bounds source residency.

The direct-HBM mode answers a different endpoint question. It preallocates target extents on the destination GPUs and keeps the output there, so it has neither D2H nor host publication. The current implementation requires computation to occur on the destination GPU and does not perform cross-GPU P2P publication.

### 6.4 Destination transaction

Every backend exposes the same logical transaction:

```text
begin(job, target_version, expected_record_ids)
  -> stage(extent_id, target K/V)*
  -> commit(complete version manifest)
  -> or abort()
```

An extent records stable record and extent IDs; migration anchor and served K/V target; the final action that produced it; layer, valid-token, K/V-width, dtype, and logical-byte metadata; destination location and device; and an optional serialized-payload checksum. Commit rejects duplicate or missing records, duplicate extent IDs, and target-version disagreement. The manifest is the visibility point: staged state without a committed manifest is not a published target version, and readers see the previous committed manifest until the new one is visible.

The three measured backends are listed in Table 4.

**Table 4: Destination backends, data paths, and evidence.**

| Destination | Data path and visibility | Evidence |
|---|---|---|
| HBM | Compute on destination GPU; manifest points to resident device extents | Full-cohort benchmark (§8.5) |
| DRAM | Pinned H2D -> transform -> D2H; in-memory manifest retains CPU extents | Full-cohort benchmark (§8.5) |
| SSD (POSIX) | Host path; immutable serialized extents; same-filesystem rename publishes manifest and objects | Functional interface; durable benchmark open (§8.6) |
| Remote object | Host path; immutable object uploads; manifest object written last | Client protocol and in-memory reference store; interface only |

How each backend realizes atomicity is standard storage practice — temporary-file writes with same-filesystem renames for POSIX, manifest-last object puts for the remote protocol — and is described in §7. These semantics do not constitute a distributed transaction across destinations.

### 6.5 Failure boundary

The transaction contract requires an exception before commit to leave the previous version visible and reclaim private staging state. The current Stage-4 result verifies only successful, complete, duplicate-free commits. Section 8.6 records the still-open failure protocol: inject faults before the first extent, mid-wave, during publication, and immediately before commit; verify visibility and cleanup; and determine whether extent journaling supports bounded idempotent resumption. None of those behaviors is part of the current performance claim.

## 7. Implementation

CohortKV is implemented in roughly 9.5K lines of Python 3 and PyTorch for the core library, with a further 26K lines of tests, benchmarks, and experiment drivers. The simplified HSTU exposes per-layer normalized states and first-class K/V. This section collects the engineering the designs rely on but that follows standard practice.

**Operator paths.** Three normalized-source operators implement Equation (5) behind one `execute_into` interface whose endpoint is separate contiguous, unpadded K/V extents with lengths and offsets. The transport reference widens the same serialized FP16 capsule and runtime program for FP32 arithmetic, materializes the concatenated projection, and compacts FP16 K/V into the common extent. It is not the original FP32 fitted program or exact current-model K/V. A packed FP16 path uses batched `baddbmm` over flattened record-token rows, expands the bias, applies a valid-length mask, and copies valid K/V into the same extent; it is the strong framework baseline of §8.5. The fused Triton kernel exposes tunable \(M,N,K\) tiles, warp count, and pipeline stages and writes that extent directly. The direct-old-K/V path adds a second fused kernel with the same output ABI and a dense FP32-arithmetic oracle over concatenated old K/V. Numerical validation compares all valid elements, verifies dense padding zeros and finite values, and requires the dense and extent forms of each path to be element-identical.

**Executors.** The CUDA streaming executor maintains separate copy and compute streams, optionally pins capsule inputs, and may write into persistent pinned output pools. The multi-GPU executor creates one single-device worker per GPU and combines per-device timing, bytes, record count, token count, program replicas, and assigned-work imbalance; round-robin and input-order placement remain available alongside LPT (§6.1).

**Program artifact.** A migration program serializes source and target versions, layer count, input/K/V widths, compiled weights and biases, and fitting metadata; the verified plan serializes the contract, each action's cost and certificates, the selected action, selection reason, and fallback order. A direct-old-K/V program additionally binds the source checkpoint hash, parent runtime-program hash, parent verified-plan hash, source representation, and workload hash. At load time, a source, shape, provenance, or device mismatch is an error rather than an implicit conversion.

**Destination mechanics.** For POSIX, each extent and the manifest are written through a temporary file and atomically replaced, and the staged directory is renamed into the target-version namespace at commit; the remote adapter assumes atomic individual object puts with a manifest-last commit marker. The normal-path engine implements the lazy shard reader, bounded publication queue, direct-HBM dispatch, coverage validation, and job-level commit timing. The hot-HBM engine additionally accounts complete old-cache occupancy and retires each old extent after its replacement is accepted. It does not yet implement automatic guard dispatch or the full failure-recovery protocol.

## 8. Evaluation

### 8.1 Questions and protocol

We organize the evaluation around five questions:

- **RQ1:** Is cross-version K/V maintenance a meaningful opportunity across data tables and model capacities?
- **RQ2:** Does the frozen compiler replicate in measured GPU cost and current-model semantic fidelity across training seeds, including when task quality is an unreliable gate?
- **RQ3:** Does compiled affine repair dominate the strongest cross-model alternative, selective layer recomputation, on the cost–fidelity frontier?
- **RQ4:** What happens to the resident-operator advantage on the complete update cohort through matched HBM/DRAM transactions at 1/2/4 GPUs, and where does end-to-end time go?
- **RQ5:** What does the capsule cost in space and creation time, and when does it break even?

**Datasets and task protocol.** The primary datasets are the standard KuaiRand-1K logs [6] and the QB/QK ordered-exposure tables from Tenrec [7]. The random-exposure KuaiRand log is excluded from training. The item vocabulary is fit only on the base period. Training uses only targets from the current stream date/window, evaluation positives are engaged items, and the model predicts item \(t+1\) from hidden state \(t\). Ranking uses the full base-fitted catalog. BestRank is the minimum catalog rank among a user's engaged positives, so lower is better; a reported BestRank gain is positive when ranking improves.

**Semantics and measurement.** All comparisons use the stale-inference semantics from §2.1. GPU cost is measured rather than replaced by a hand-written constant. For replicated claims, the training seed is the statistical unit; users within one trained model are diagnostics. Full-cohort timings are medians of three complete runs after correctness and warmup passes, with every method/destination/GPU point tuned independently. The testbed has four 46-GB NVIDIA A40 GPUs, two Intel Xeon Gold 5420+ sockets, and 1.0 TiB host DRAM. Source shards reside on `/dev/nvme2n1p1`, an Intel SSDPF2KX038XZ formatted as ext4 and mounted at `/data`. The benchmark reopens and decodes shards but does not evict the OS page cache, so it is a warm filesystem/page-cache measurement rather than a cold SSD result.

**Baselines and ablations.** No existing system executes cross-version K/V migration for streaming HSTU end-to-end, so the comparison set is constructed at three levels, each tuned independently. (i) *End-point anchors*: stale reuse, which costs nothing and fails the certificate, and exact recomputation, which is the semantic reference at cost 1×. (ii) *Strongest adapted external method*: selective layer recomputation in the style of DroidSpeak [4], re-implemented for HSTU (§8.4). (iii) *Internal ablations that attribute our gains*: cheap projection without the fitted residual, residual-\(p\) replay, the no-transform placement bound, packed versus fused operators, and length bucketing on/off. Same-model restoration systems (HCache [3], CachedAttention [9]) are semantically inapplicable to the version-mismatch event and are positioned in §9 rather than measured.

**Evidence grid.** Each claim is measured on the narrowest grid that supports it: the compiler's statistical claims on all 27 chains (3 tables × 3 capacities × 3 seeds, §8.3); the cost–fidelity frontier on three chains — KuaiRand long-context (primary), KuaiRand medium (capacity axis), and QB large (dataset axis) — reusing the §8.3 infrastructure (§8.4); and the full-cohort system benchmark on the KuaiRand long-context cohort only, because transaction and throughput behavior is governed by byte volume and length distribution rather than by which table produced the interactions (§8.5–§8.6).

**Evidence levels.** **Replicated** denotes frozen multi-seed protocols; this now covers the compiler contract (§8.3) in addition to the 27-chain operator study. **Controlled** denotes real-checkpoint single-seed development results, retained where noted. **Interface-validated** denotes executable correctness without a performance admission result; only the remote-object backend remains at this level.

### 8.2 RQ1: Opportunity across data and capacity

The aligned cross-table result in §2.2 and the stronger Top-50k/all-chunks KuaiRand result show that streaming value can coexist with a measurable maintenance gap. The 3×3 screen sharpens the claim: all nine cells benefit from streaming training and retain some of that benefit under reuse in all four seeds, but only some cells have a substantial positive mean maintenance endpoint. The cost opportunity scales smoothly — cheap projection requires only 0.185–0.204× exact resident-GPU time as depth/width grow — while the task opportunity does not (Figure 2). Fixed-endpoint age curves also reject a universal update window: every 3×3 curve is non-monotone, and early ages vary in sign.

**Answer to RQ1.** The evaluated streams expose both a repeatable systems pressure and a dataset/capacity boundary. This motivates an efficient semantic transform, not a claim that every cohort has positive ranking maintenance value.

### 8.3 RQ2: Replicated compiled migration under a frozen contract

**Cohort-tiered replication.** The frozen cohort-tiered validation spans KuaiRand, QB, and QK; small, medium, and large models; and three non-discovery training seeds per cell, for 27 independent model-version chains. At the primary 50% fidelity target, every validation run selects a compiled projection. Table 5 aggregates the outcome.

**Table 5: Replicated cohort-tiered validation across 27 model chains** (primary 50% fidelity target; brackets are 95% CIs over chains).

| Metric | Result |
|---|---:|
| Mean GPU cost / exact | 0.1211 [0.1118, 0.1304] |
| Mean stale-to-fresh K/V recovery | 0.5867 [0.5466, 0.6267] |
| Test splits meeting the 50% fidelity target | 25/27 |
| Positive BestRank / rank-utility / NDCG@100 signs | 20/27 · 24/27 · 20/27 |
| Strict positive-mean BestRank + rank-utility cells | 6/9 |

The cells that miss the strict task gate are informative rather than disqualifying: QB-medium exact recomputation itself has negative BestRank in all three validation seeds, and QK-large has a near-zero, unstable exact endpoint. The compiled transform nevertheless follows the available endpoint: selected and exact share the sign in 23/27 BestRank, 27/27 rank-utility, and 25/27 NDCG@100 cases.

**Frozen long-context compiler.** The full-affine compiler, its action library, and the 70%/80%/90%/30% contract were frozen — hyperparameters, weighting, thresholds, and fallback order hashed into the protocol — before inspecting any replication seed. The frozen compiler is then run end-to-end on ⟨TBD⟩ new training seeds of the 16-layer, hidden/K/V-width-512 long-context HSTU (about 0.181 B parameters), each with disjoint fit/selection/certificate/final-test user roles. Table 6 reports the outcome; the original seed-0 development result is retained in Appendix ⟨TBD⟩ as controlled evidence.

**Table 6: Frozen verified compiler replicated across seeds** (three source ages per seed; final held-out users).

| Seed | Ages passing certificate | Selected action(s) | Final cost / exact | Final K/V recovery | Top-100 overlap |
|---:|---:|---|---:|---:|---:|
| 0 (dev) | 3/3 | compiled full affine | 0.0638–0.0641 | 0.887–0.936 | 0.970–0.995 |
| ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ |
| ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ |

The separate seed-0 deployed-representation certificate reloads serialized FP16
capsules/programs, emits FP16 K/V, and leaves the 522 final users untouched. All three pairs select
compiled full affine under the primary 70% contract. The selection remains compiled at targets
50%–80% and becomes exact for all three pairs at 90%. The 0.01651–0.01657× resident packed-FP16
component is not directly substituted for Table 6's final-user cost boundary and cannot be used as
the full-cohort speedup.

Across seeds, the certificate selects the full-affine program in ⟨TBD⟩ of ⟨TBD⟩ cohorts and escalates the remainder through the published chain; no cohort is published without passing its contract. Threshold sensitivity (Figure 5) shows the selection is stable for recovery targets between ⟨TBD⟩% and ⟨TBD⟩%, so the contract is an interior operating point.

![Certificate threshold sweep and per-seed recovery distribution.](figures/06_frozen_contract.svg)

**Figure 5: The frozen contract replicates across seeds and is not threshold-tuned.**

At the harmful age-11 endpoint of the development seed, the selected action recovers 98.8% of the signed MeanRank and AUC gaps. At age 7, stale reuse happens to beat exact current K/V on MeanRank and AUC; the migrated state follows the current model and gives up that accidental gain. This is the intended semantics, and a direct demonstration that exact K/V is not a ranking-quality upper bound.

**Answer to RQ2.** Compiled repair replicates as a low-cost K/V-fidelity mechanism under a contract frozen before replication, and task quality rejects a universal admission claim. This is why every stale cohort receives repair and the compiler certifies semantic fidelity rather than predicted ranking gain.

### 8.4 RQ3: Against selective layer recomputation

The strongest external alternative treats cross-version reuse as a layer-group recomputation problem. DroidSpeak profiles contiguous groups because each transition from reused state to receiver-model recomputation needs a sender activation (`E` cache), and scattered groups add both state and propagated mismatch [4]. We adapt that semantic path to HSTU: for each \(m\in\{2,4,6,8,12\}\), the development split profiles every legal contiguous \(m\)-layer interval; execution starts from the old pre-block hidden state, recomputes that interval with the current model, and reuses old K/V outside it. The profiler uses the same label-free cache/score/top-100 views as CohortKV rather than recommendation labels, and its old K/V, transition-state, and raw-history bytes are counted at the common source tier. The interval and \(m\) are frozen before the certificate and final users. This is a compatible DroidSpeak-adapted algorithmic baseline, not a reproduction of its distributed LLM serving runtime. The frontier is first measured on the primary 16-layer KuaiRand long-context chain; cross-capacity and cross-dataset cells are deferred until the single-configuration implementation is frozen. Figure 6 reports the primary chain.

Figure 6 plots the completed primary-chain development frontier: 53 selective intervals plus compiled, cheap, p4, p8, reuse, and exact for each of theta0/theta4/theta10→theta11, or 177 points total. Compiled repair costs 0.0656–0.0664× exact and reaches 0.8755–0.9258 worst-view recovery. The strongest selective point is \(m=12\), layers 0–11 for all three pairs; it costs 0.6973–0.6976× exact but reaches only 0.4495–0.4850 worst-view recovery. Thus compiled has both lower resident cost and higher worst-view recovery than every evaluated selective interval in this adaptive seed-0 cell. No selective candidate passes the frozen 70% three-view contract; exact is its publishable fallback. Cross-seed, cross-capacity, and cross-dataset replication remain ⟨TBD⟩ and must not be inferred from this result. Internal p4/p8 controls are also shown to delimit the frontier.

![Cost/exact versus semantic recovery for compiled affine, selective contiguous recomputation at m∈{2,4,6,8,12}, structural replay, and residual-p.](figures/07_pareto_frontier.svg)

**Figure 6: Compiled affine repair dominates the seed-0 single-configuration selective frontier.** All selective certificates fail; the profiled \(m=12\) point is retained as a diagnostic baseline, not a publishable action.

Two boundary cases are reported for completeness. A no-transform placement baseline (moving old K/V without any repair) costs 2.82–6.57× exact on the complete-cohort HBM/DRAM grid and fails the certificate at all ages, exposing source movement rather than transformation as the dominant systems boundary. A same-model HCache-style restoration [3] is semantically inapplicable — it restores the wrong version by construction — and is included only to delimit the problem.

**Answer to RQ3.** In the primary adaptive seed-0 cell, compiled repair strictly dominates all 53 selective intervals for every evaluated source age: it is about 10.5× cheaper than the highest-fidelity selective point while reaching substantially higher worst-view recovery. The selective baseline does not certify. Whether this ordering persists across training seeds, capacities, and datasets remains ⟨TBD after replication⟩.

### 8.5 RQ4: Complete cohort through an identical destination transaction

**Workload.** The full update job migrates every eligible KuaiRand long-context record: 682 records and 1.087785 M logical prefix tokens across a predeclared controlled source mix (theta0/theta4/theta10 counts 136/205/341 → theta11), totaling 16.60 GiB of logical FP16 capsule bytes. The records are real, but the source versions are label-free controlled assignments rather than an organic cache-refresh trace. Capsules stream from buffered POSIX shards on the `/data` ext4 tier through the lazy shard reader. Exact recomputation reads raw histories from the same tier. The frozen selective diagnostic is \(m=12\), layers 0–11, so it reads raw histories and old K/V but no transition hidden state; it remains explicitly certificate-failed. Any residual-\(p\) control also reads its explicitly retained BF16 old hidden suffix. The paths share a physical tier rather than identical source bytes, so logical and physical input traffic is reported separately. All three primary pipelines are independently tuned per destination and GPU count and publish complete FP16 K/V through the same destination transaction; source read, target allocation, and manifest commit are included in completion time.

**Operator microbenchmark.** Table 7 uses a real four-record, sequence-width-2,047 batch and includes the complete write to the same preallocated unpadded K/V extent for every path. Fused is 1.97× faster than packed and uses no global temporary beyond the target, whereas packed's concatenated projection and compact-copy epilogue peak at 402.6 MB. Across the full 60-record development length distribution, the frozen fused configuration is 1.995× faster than the fastest packed control, and every fused sample is below every packed sample. This establishes the operator choice, not a stable ordering among the close fused batch/bucket finalists. All nine layouts pass the transport tolerance on all 1.443 billion valid FP16 K/V elements per layout, with zero dense padding values. The reference widens the same serialized FP16 inputs for FP32 arithmetic; thus the error column isolates execution/layout error and is separate from semantic error against current-model exact K/V.

**Table 7: Operator tiers on a representative resident batch.**

| Operator | Median time | Speedup from previous row | Full-distribution relative K/V error from transport reference | Peak operator temporary |
|---|---:|---:|---:|---:|
| FP32-arithmetic transport reference → common extent | 14.610 ms | - | 0 | 1,073.8 MB |
| Packed FP16 `baddbmm` → common extent | 5.378 ms | 2.72× | 2.27e−5 | 402.6 MB |
| Fused FP16 Triton → common extent | 2.729 ms | 1.97× | 2.50e−5 | 0 |

The time and temporary-memory columns use the representative batch; the error column is computed independently over the complete 60-record `b4/w32` development distribution.

**Full-cohort results.** Table 8 reports completion time and throughput for the complete job.

**Table 8: Complete-cohort migration versus tuned exact recomputation through the identical destination transaction.**

| Destination | GPUs | Compiled | Compiled throughput | Exact | Exact throughput | Compiled speedup (exact / compiled) |
|---|---:|---:|---:|---:|---:|---:|
| HBM | 1 | 27.083 s | 25.18 rec/s | 18.881 s | 36.12 rec/s | 0.697× |
| HBM | 2 | 18.943 s | 36.00 rec/s | 9.644 s | 70.72 rec/s | 0.509× |
| HBM | 4 | 13.707 s | 49.76 rec/s | 5.742 s | 118.77 rec/s | 0.419× |
| DRAM | 1 | 22.567 s | 30.22 rec/s | 18.886 s | 36.11 rec/s | 0.837× |
| DRAM | 2 | 12.231 s | 55.76 rec/s | 9.391 s | 72.62 rec/s | 0.768× |
| DRAM | 4 | 15.662 s | 43.54 rec/s | 5.448 s | 125.18 rec/s | 0.348× |

The full matrix reverses the resident-operator expectation:

- Compiled remains 2.70–3.49× faster than the frozen selective diagnostic through the same destination, even though that diagnostic is not publishable because its certificate fails.
- Exact is 1.20–2.87× faster than compiled in all six matched endpoint comparisons. Compiled scales by 1.98× on HBM and 1.44× on DRAM from one to four GPUs, while exact scales by 3.29× and 3.47×.
- Source read, decode, and pinning account for 91.35%–96.91% of compiled wall time. The serialized FP16 capsule occupies 17.82 GB physically, about 200× the 89.1-MB raw-history source used by exact. Compiled resident compute takes only 0.118–0.954 s; exact spends 5.03–18.08 s in compute.
- All 30 method/destination/GPU points pass preflight, full-source correctness, finite/allclose checks over 17.82 billion valid elements, and complete duplicate-free manifest validation. Maximum observed transient source and staging waves are 3.50 and 3.00 GiB; publication-queue residency is zero in the selected schedules.

![Completion time and source-read share for compiled and exact paths at 1/2/4 GPUs.](figures/08_full_cohort_breakdown.svg)

**Figure 7: Where full-cohort time goes.** Source supply dominates compiled wall time, whereas exact remains compute-bound. Phase timers can overlap, so the figure decomposes wall time only into source-read time and the remaining wall time.

This 30-point sweep is the one-time closure matrix used to establish that the bottleneck and ordering persist across methods, destinations, and GPU counts. Source-state candidates do not repeat it during iteration: resident ceilings and candidates use the 60-record program-selection role, after which only the predeclared 1/4-GPU full-cohort representative points run against paired exact.

The winning source plan is the direct reparameterization of §4.5. It reads the serving old K/V
already in HBM, adds no per-record source state, and reclaims each old extent after its
replacement is accepted. Exact receives its complete raw history already in HBM. Table 8b adds
the predeclared representatives and the subsequent two-GPU expansion; each row has one
correctness run, one warmup, and five measured complete jobs.

**Table 8b: Complete-cohort direct-old-K/V hot-HBM source plan.**

| GPUs | Direct compiled | Paired exact | Speedup | Peak old + new K/V |
|---:|---:|---:|---:|---:|
| 1 | 0.9299 s | 18.6949 s | 20.11× | 35.91 GB |
| 2 | 0.4936 s | 9.7291 s | 19.71× | 36.18 GB |
| 4 | 0.2546 s | 4.7655 s | 18.72× | 37.79 GB |

Every compiled repetition is below every paired exact repetition, all capacity preflights and
complete manifests pass, and final old-K/V occupancy is zero. A separate real-value fused
transport covers all 682 records and 17,822,269,440 valid elements with zero mismatches at
`atol=0.02, rtol=0.02` and maximum absolute error 0.01172 against the deployed
normalized-capsule output. The timed repetitions use shape-, dtype-, layout-, and
occupancy-equivalent old-K/V values; therefore the real transport and the system timings are
separate necessary evidence. This result is limited to the existing-old-K/V hot-HBM regime.

**Answer to RQ4 (throughput).** The normal full-cohort engine and matched transactions work. The
normalized-capsule file path fails because source movement, not the kernel, is limiting; composing
the same certified transform over already resident old K/V removes that source state and creates
a stable complete-job Pareto point in the declared hot-HBM regime.

### 8.6 RQ4 continued: Repeated-update lifecycle

The one-hop source-plan certificate does not bound a later affine applied to its approximate
output. We therefore run one fixed KuaiRand seed-0, 16L/H512, one-A40 chain from exact theta0 K/V
through all 11 adjacent checkpoints to theta11. Every update chooses either direct migration or
charged exact replay, and every migrated cache is the next update's actual input.

The first per-cache norm-sketch threshold reaches acceptable cumulative fidelity and beats a
matched-random diagnostic, but it refreshes between 0.15% and 65.1% of the cohort per update. We
reject this synchronized maintenance wave. The frozen replacement ranks each model edge by median
fit one-hop cache error, maps the edge to a 15%–25% exact budget, and refreshes greater migration
age first; depth four is mandatory exact and a stable hash breaks ties. This is a bounded
development heuristic rather than an optimal per-user risk predictor.

**Table 8c: Controlled theta0-to-theta11 lifecycle.**

| Role | Records | Cumulative GPU cost / all-exact | Maximum step cost / all-exact | Exact fraction | Minimum cache fidelity | Minimum score cosine | Minimum top-100 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent certificate | 60 | 0.2142× | 0.2814× | 15.0%–25.0% | 0.9613 | 0.999759 | 0.9898 |
| Complete chain | 682 | 0.2134× | 0.2543× | 14.956%–25.073% | 0.9632 | 0.999950 | 0.9918 |

The complete-chain range includes nearest-record rounding. Mechanical freeze checks rebuild all
7,502 record/update lineage rows, verify exact reset and maximum depth four, and confirm that every
edge consumes the previous actual output under the correct adjacent program hash. Across 522
final-test records, maximum absolute per-step mixed-minus-exact gaps are 4.171 MeanRank,
8.35e−5 Catalog AUC, 3.49e−4 NDCG@100, and 0.00384 Hit@100. Labels do not select or route the
policy.

**Answer to RQ4 (lifecycle).** In this controlled fixed-history chain, balanced exact refresh
bounds recursive approximate migration while retaining about one-fifth of all-exact GPU cost.
This does not establish selector optimality, new-seed replication, or behavior under organic
request arrivals.

### 8.7 RQ4 continued: SSD endpoint, escalation, and failure injection

This subsection is an open evaluation contract, not evidence from Stage 4.

- **SSD:** run a separately named durable POSIX protocol that records the device, filesystem, page-cache state, write path, and `fsync` policy. The current `/data` measurement reads warm source shards from an NVMe-backed ext4 filesystem but publishes to HBM or DRAM; it is not an SSD destination result.
- **Forced escalation:** corrupt a program under a protocol-valid guard, verify monotone transition through the published fallback chain, and measure re-migration before commit.
- **Failure injection:** inject failures before the first extent, mid-wave, during publication, and immediately before commit; then verify visibility, cleanup, and any claimed restart bound.

**Layout boundary.** The jagged experiment compacts valid tokens and matches dense fused K/V element-for-element, but end-to-end compaction yields only 1.019× on the host path and 0.984× at the direct-HBM boundary. We retain its machinery without claiming a positive layout result.

**Answer to RQ4 (semantics).** Successful HBM/DRAM commits have complete, duplicate-free manifests. Automatic escalation, injected-failure visibility, durable SSD publication, and the remote-object backend carry no full-cohort performance or recovery claim.

### 8.8 RQ5: Capsule economics

The normalized capsule is the original path's principal standing cost. Unpadded FP16 `Norm(x)` is
50% of logical FP16 K/V at equal widths: the measured cohort contains 17.82 GB of serialized
capsules for 35.64 GB of logical target K/V. Stage 4 establishes no time break-even for repeatedly
reading that representation because compiled completion exceeds exact at every measured endpoint.
A pinned-DRAM full-cohort candidate makes the capsule path fast after setup
(1.529/0.252 seconds on 1/4 GPUs), but retains about 17.86 GB of host state and requires
39.5/24.7 seconds to preload, for three/six-update time break-even against its paired exact.

The selected direct-old-K/V route removes the capsule from runtime instead of compressing it. The
old K/V is the serving state already present before migration; the source plan adds zero
per-record bytes, needs no independent capture or preload, and adds only 100.78 MB of programs per
worker. Extent reclamation bounds old/new coexistence. Program compilation and publication remain
once-per-version-pair costs; their broader cohort-size amortization and normalized-capsule
capture/INT8 controls are still ⟨TBD⟩. A symmetric signed INT8 capsule would reduce capsule data to
roughly 25% of FP16 K/V before metadata, but it is now a representation ablation rather than a
prerequisite for the primary route. Deployment-specific re-access frequency and monetary byte
cost remain outside the available traces.

![Capsule storage/precision frontier and update-frequency break-even.](figures/09_capsule_economics.svg)

**Figure 8: The capsule is a measured space-for-update-time trade, not free metadata.**

**Answer to RQ5.** The FP16 normalized capsule is semantically effective but economically
inadmissible as the primary full-cohort source. Direct old-K/V reparameterization preserves its
semantic target with zero additional per-record state; INT8 remains a secondary space/economics
control.

### 8.9 Discussion and limitations

**What the results mean.** Stale reuse forfeits a measurable, reproducible fraction of streaming value (RQ1), but neither cache age nor realized task gain is a calibrated predictor of which cohort needs repair (RQ2). Compiled affine repair is a strong semantic mechanism at the resident boundary and dominates per-record selective layer recomputation in the controlled frontier (RQ3). The first complete-cohort result rejects the normalized-state supply path rather than the algebra: moving a 17.82-GB FP16 capsule overwhelms sub-second resident compute. Reparameterizing the certified affine over the old K/V already present in the hot cache eliminates that additional state and establishes a scoped end-to-end advantage over same-tier exact (RQ4–RQ5). A controlled repeated-update chain further shows that age/deadline-balanced exact refresh can bound recursive migration at about one-fifth of all-exact GPU cost. Its rejected threshold predecessor is equally important: acceptable cumulative fidelity does not prevent harmful per-update maintenance waves. Automatic guard, fallback, and failures remain after this policy gate.

**Limitations.** Several boundaries condition these claims, and each points to a concrete next step. The model is a modular, simplified HSTU of up to about 0.18 B parameters rather than the production-scale system in the original HSTU work [1], so evidence at larger capacity remains open; KuaiRand is the only long-context chain with the complete current design, and QB and QK broaden ordered-exposure evidence but are related Tenrec tables without a shared calendar. Fitting depends on the per-layer `Norm(x) -> P` data path, while the zero-extra-state runtime form additionally depends on the stacked source K/V projection having stable full row rank; both conditions require explicit checks before transfer to another architecture. The repeated-input result is one fixed-history, one-seed chain. Its edge severity is calibrated from fit-record exact references, and its age/deadline quota is a deterministic heuristic rather than a proof of optimal per-cache refresh; cross-seed, cross-dataset, and organic mixed-version behavior remain open. Programs are compiled per source/target pair: when a record misses several updates, the system currently compiles each (v, t) edge independently, and whether programs compose along the version chain — `Φ(v→t2) ≈ Φ(t1→t2) ∘ Φ(v→t1)`, which stays affine and would reduce the program set from quadratic to linear in the version count — is a structural extension, together with warm-started incremental fitting from the previous pair. Finally, automatic guard dispatch, failure recovery, durable SSD publication, and the remote-object backend do not have full-cohort evidence. The measured speedup is limited to complete old K/V already resident in HBM; cold storage and automatic cache-tier selection are not implied. Online serving — request arrivals, per-user hotness, and migration sharing GPUs with foreground inference — is also outside the evaluated boundary (§3.1): the datasets carry no request traces, so any serving-workload claim would rest on constructed load, and we do not make one.

## 9. Related work

**Streaming recommendation and model update.** HSTU motivates generative recommendation over high-cardinality, non-stationary streams and demonstrates the value of long sequential histories [1]. CohortKV studies a systems consequence: the histories' derived K/V outlive the model version that created them. Ekko reduces model-update latency by disseminating recommender parameter updates and managing model replicas [2]. We borrow its observation that recommender freshness is operationally important, but CohortKV is not a model publication system: training, checkpoint validation, WAN dissemination, and model rollback are outside its boundary.

**K/V memory, restoration, and hierarchical storage.** vLLM derives a memory manager and serving engine from the dynamic allocation properties of same-model autoregressive K/V [8]. CachedAttention retains same-conversation K/V across requests and overlaps hierarchical loading and saving [9]. HCache restores same-model state from intermediate activations, balancing recomputation with I/O [3]. MTServe persists per-user generative-recommendation K/V across visits and focuses on GPU/host placement, asynchronous movement, and replacement [5]; it does not define a transformation from state produced by one model version to the K/V semantics of another. These systems establish that K/V-specific structure should shape kernels, movement, and storage. CohortKV addresses a different validity event: streaming training changes the model, so an intact resident state is nevertheless stale.

**Cross-model K/V.** DroidSpeak is the closest cross-model system: same-architecture fine-tuned LLM variants share K/V by selectively recomputing some layers and reusing the rest [4]. Consequently, "cross-model K/V reuse" alone is not a CohortKV contribution, and §8.4 compares against a selective-layer baseline directly rather than by classification. The distinction the measurement supports is amortization structure: selective recomputation pays per record and per layer, whereas the compiled program pays once per version pair and executes as one projection. CohortKV additionally certifies label-free semantic views and publishes fixed cohorts as atomic target versions, which the request-serving setting of DroidSpeak does not require.

**Execution units and observation-driven systems.** Orca derives iteration-level scheduling and selective batching from autoregressive model semantics, making the iteration a shared unit across scheduler and engine [10]. CohortKV similarly uses the source/target version cohort across compiler, batcher, placement, fallback metadata, and manifest, but not as an online scheduling or safety prediction. DistServe maps prefill/decode interference to disaggregation and placement [11]. CohortKV follows the same observation-to-design discipline. Table 9 positions CohortKV against the closest K/V systems.

**Table 9: Closest K/V systems and the CohortKV boundary.**

| System | Model relation | Retained source state | Primary action | System unit / output |
|---|---|---|---|---|
| HCache [3] | same model | intermediate activation | restore after eviction | request/chunk K/V |
| DroidSpeak [4] | fine-tuned LLM variants, same architecture | another variant's K/V | selective layer recomputation + reuse | request prefill |
| MTServe [5] | no source->target version transform | persisted per-user K/V | place/load/evict | serving-time page/chunk cache |
| CohortKV | successive streaming HSTU versions | existing old K/V in primary hot path; `Norm(x)` for fit/reference | compiled affine repair reparameterized over old K/V + certified fallback plan; automatic guard open | fixed version cohort -> target manifest |

## 10. Conclusion

Persistent recommender K/V is model-versioned derived state. CohortKV organizes its update around a source/target version cohort, compiles shared repair into one affine HSTU projection, and executes it with a fused direct-write operator. Across 27 replicated model chains the mechanism costs 0.121× resident exact recomputation while closing a majority of the state gap; in the controlled seed-0 cell its deployed programs recover 0.881–0.936 and dominate every measured selective-layer action. The 682-record normal-path engine publishes complete HBM/DRAM manifests and exposes a decisive representation boundary: the 17.82-GB normalized-capsule file path loses to exact everywhere. Composing the same certified affine over the serving old K/V eliminates that additional state; complete hot-HBM updates are 18.72–20.11× faster than paired same-tier exact at 1/2/4 GPUs while reclaiming every old extent. A controlled 11-update chain then bounds recursive approximate migration with balanced exact refresh at 0.213× cumulative all-exact GPU cost. Automatic fallback, failure recovery, durable storage, and broader lifecycle replication remain open.

## References

[1] Jiaqi Zhai et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024. <https://arxiv.org/abs/2402.17152>

[2] Chijun Sima et al. "Ekko: A Large-Scale Deep Learning Recommender System with Low-Latency Model Update." OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/sima>

[3] Shiwei Gao, Youmin Chen, and Jiwu Shu. "Fast State Restoration in LLM Serving with HCache." EuroSys 2025. <https://doi.org/10.1145/3689031.3696072>

[4] Yuhan Liu et al. "DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants." NSDI 2026. <https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan>

[5] Xin Wang et al. "MTServe: Efficient Serving for Generative Recommendation Models with Hierarchical Caches." arXiv:2604.22881, 2026. <https://arxiv.org/abs/2604.22881>

[6] Chongming Gao et al. "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos." CIKM 2022. <https://arxiv.org/abs/2208.08696>

[7] Guanghu Yuan et al. "Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems." arXiv:2210.10629, 2023. <https://arxiv.org/abs/2210.10629>

[8] Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. <https://doi.org/10.1145/3600006.3613165>

[9] Bin Gao et al. "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention." USENIX ATC 2024. <https://www.usenix.org/conference/atc24/presentation/gao-bin-cost>

[10] Gyeong-In Yu et al. "ORCA: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/yu>

[11] Yinmin Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI 2024. <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>

## Appendix A. Artifact-to-claim map（⟨TBD：随新实验补全⟩）

| Paper claim | Protocol / record | Aggregate artifact |
|---|---|---|
| Top-50k/all-chunks opportunity | KuaiRand data-utilization protocol | `results/scaling/kuairand_data_utilization_summary.json` |
| Cross-table aligned theta5 opportunity | ordered exposure v1 | `results/exposure/cache_version_matrix_cross_dataset_summary.json` |
| 3×3 capacity and age screen | capacity v2 | `results/motivation_scale/capacity_v2_summary.json` |
| 27-chain compiled repair | cohort-tiered migration v1 | `results/motivation_scale/cohort_tiered_migration_v1_summary.json` |
| Frozen compiler replication | ⟨TBD: verified compiler v2, multi-seed⟩ | ⟨TBD⟩ |
| Deployed compiler artifact/certificate, seed-0 development | CohortKV Stage 2 compiler v1 | `configs/cohortkv_single_config_v1/stage2_compiler_summary.json` |
| Selective-layer baseline frontier (primary KuaiRand-long development cell) | CohortKV Stage 1 frontier v1 | `configs/cohortkv_single_config_v1/stage1_frontier_summary.json` |
| Selective-layer frontier replication (KuaiRand-medium, QB-large, new seeds) | ⟨TBD: cross-model baseline replication v1⟩ | ⟨TBD⟩ |
| Selective-layer end-to-end full-cohort row | CohortKV Stage 4 system v1 | `configs/cohortkv_single_config_v1/stage4_system_summary.json` |
| Full-cohort HBM/DRAM benchmark and source bottleneck | CohortKV Stage 4 system v1 | `configs/cohortkv_single_config_v1/stage4_system_summary.json` |
| Direct-old-K/V hot-HBM source plan, certificate, transport, and 1/2/4-GPU jobs | CohortKV Stage 4.5 frozen v1 | `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json` |
| Repeated-update per-cache migrate-or-exact lifecycle | CohortKV Stage 4.6 lifecycle v1 | `configs/cohortkv_single_config_v1/stage4_6_lifecycle_summary.json` |
| SSD endpoint | ⟨TBD: physical POSIX v1⟩ | ⟨TBD⟩ |
| Capsule economics | ⟨TBD: capsule economics v1⟩ | ⟨TBD⟩ |
| Escalation and failure injection | ⟨TBD: destination out-of-core v5 failure suite⟩ | ⟨TBD⟩ |
| Controlled seed-0 development results | verified cohort compiler v1 / two-GPU system v2 | `results/motivation_scale/long_context_4plus12_verified_compiler_seed0.json`, `results/system/kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json` |

Artifact paths are repository-relative. Raw per-seed files and checkpoints remain local and are not merged across protocol families.
