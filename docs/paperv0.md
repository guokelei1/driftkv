# EvoKV: A Release-Time Runtime for Versioned Recommendation State

## Abstract

Long-history recommender models increasingly materialize per-user intermediate state to avoid repeatedly encoding thousands of historical interactions. Once retained across requests, this state becomes a long-lived system object: it has a producer model, a materialization time, an append and eviction lineage, and consumers that may evolve independently from it. A model release can therefore leave the serving system with a population of structurally readable yet semantically outdated states. Reusing all states is inexpensive but may deviate from the current model’s execution semantics, whereas rebuilding all states requires replaying the complete retained history for the entire population.

We present **EvoKV**, a release-time runtime for the versioned evolution of persistent recommendation state. EvoKV treats cached key–value tensors as **versioned materialized neural state** rather than ordinary cache entries. Its control plane analyzes release compatibility, profiles a sparse sample of states without future requests or labels, and allocates a bounded migration budget across the complete cutover population. Its data plane executes dependency-closed transition plans—including No-op, local projection, causal tail replay, and exact reconstruction—while preserving each state’s rolling append and eviction lineage.

On a Yambda-50M development platform, EvoKV shows that state compatibility depends jointly on workload semantics, release semantics, and individual state history. Output-only releases preserve old state to numerical precision, while cache-producing updates create measurable staleness and, under stronger encoder refreshes, degrade recommendation quality. At equal cost, EvoKV’s state-level planner exceeds the strongest fixed or metadata-only policy by 24.9–43.0 percentage points in the low-budget regime. Its grouped GPU executor reduces runtime by 35.8%–80.4% relative to Exact-All, and the frozen policy remains effective when state accumulates through a true two-release recursive lineage. These results establish the runtime abstraction and its end-to-end mechanisms on the development release chain; blind temporal qualification is deliberately kept separate.

## 1. Introduction

Modern recommender systems increasingly model user behavior as a sequence rather than as a fixed collection of aggregate features. SASRec demonstrated that self-attention can select relevant events from a user’s interaction history, and subsequent industrial models extended sequential recommendation toward substantially longer behavioral histories and candidate-conditioned interest computation [1, 2]. HSTU further formulates recommendation as sequential transduction over high-cardinality, non-stationary streams [3]. These architectures make long histories more expressive, while also making their repeated execution increasingly expensive.

A common response is to retain history-side intermediate results, such as the key and value tensors produced by the sequential encoder. Candidate queries can then reuse the materialized history state rather than replay the complete sequence. This optimization changes the role of the tensors. They are no longer ephemeral values owned by one forward pass; they become persistent, per-user state that may survive many requests and multiple model releases.

Recommendation models themselves also evolve continuously. Systems such as Monolith support online training, Ekko accelerates dissemination of sparse model changes, and QuickUpdate reduces the bandwidth and latency required to publish large recommendation models [4–6]. These systems address an essential part of freshness: moving new parameters to the serving fleet. They do not, however, determine what should happen to the population of model-generated states that was materialized by earlier parameters.

This separation creates a distinct systems problem. Suppose a model transitions from (\theta_{t-1}) to (\theta_t), while the serving system already stores state for millions of users. The old tensors remain shape-compatible with the new model and can usually be read without an exception. Nevertheless, the current encoder may assign different semantics to the same history. A serving path that combines a current query with old history state therefore executes neither the complete old model nor the complete new model.

The conventional cache choices are too coarse. **Reuse-All** avoids migration work but may retain substantial semantic error. **Exact-All** reconstructs the entire population under the current model but requires raw-history reads, full encoder execution, and state writes for every materialized user. A single fixed partial refresh can reduce this cost, yet it cannot exploit the fact that some releases preserve the state producer, some users accumulate almost no error, and some state regions are far more recoverable than others.

EvoKV approaches this problem as the lifecycle management of **versioned materialized neural state**. This perspective is related to the way partially stateful dataflow systems treat intermediate results as materialized, reconstructable system objects, rather than as undifferentiated cache entries [7]. The defining difference is that EvoKV’s materialization function is a learned model whose definition changes at publication time. Consequently, state compatibility depends on the release, the query semantics, the state lineage, and the dependency structure of the model graph.

EvoKV is organized as a release-time runtime with a control plane and a data plane. The control plane maintains version and lineage metadata, determines whether the cache-producing subgraph changed, uses sparse target-free probes to estimate action benefit, and produces one transition plan for every state in the cutover population. The data plane validates that each plan satisfies model dependencies, executes a mixed population of No-op, partial, and exact actions, and preserves the rolling append and eviction semantics of persistent state.

This paper makes four contributions. First, it introduces **versioned recommendation state** as a first-class systems abstraction and separates model admission from state compatibility. Second, it defines a dependency-closed transition-plan interface that distinguishes executable state evolution from diagnostic tensor replacement. Third, it presents a target-free, full-population planner that charges profiling and migration against the same release budget. Fourth, it realizes these abstractions in a lineage-preserving GPU executor and evaluates the resulting cost–fidelity–quality frontier across multiple workloads, release semantics, model conditions, training seeds, and a recursive two-release state lineage.

## 2. Background and Motivation

### 2.1 From Transient KV to Persistent Model State

For user (u), let (h_u) denote the retained interaction history and let

[
Z_u^{\theta}
============

# \operatorname{Encode}_{\theta}(h_u)

\left{
K_{u,\ell}^{\theta},
V_{u,\ell}^{\theta}
\right}_{\ell=0}^{L-1}
]

denote the multi-layer state generated by encoder version (\theta). Given a candidate-conditioned query (q), the current model produces

[
y_u^{\mathrm{full}}
===================

\operatorname{Score}_{\theta_t}
\left(q,Z_u^{\theta_t}\right).
]

If the serving system retains the parent state, it instead produces

[
y_u^{\mathrm{reuse}}
====================

\operatorname{Score}*{\theta_t}
\left(q,Z_u^{\theta*{t-1}}\right).
]

The two executions share the same user, history, query, candidates, timestamp, current output layers, and evaluation protocol. Their only intended difference is the version and lineage of the persistent history state. EvoKV uses Current Full as the reference execution semantics of (\theta_t). It does not assume that Current Full is a theoretical upper bound on every future ranking metric.

Persistent state is useful only when the workload genuinely depends on the retained history. A next-item evaluation can otherwise be dominated by repeated items, short-term recency, catalog popularity, or candidate-generation rank. EvoKV therefore evaluates three distinct workload roles. Natural Next-Listen represents a short-horizon negative control. Return-to-Familiar evaluates ranking over a familiar-item universe. Explicit Like/Dislike evaluates candidate-conditioned long-term preference. A frozen base model absorbs count, popularity, recency, and proposal-rank signals, while the sequential model contributes a residual over those compact statistics.

Under this protocol, long history beyond the most recent 32 events contributes measurable incremental value to Explicit Like/Dislike. It does not pass the same qualification for every workload. This distinction is fundamental to the runtime: short-state workloads define legitimate No-op or compact-state regions rather than failed experiments.

### 2.2 Model Releases and State Compatibility

EvoKV separates two publication decisions. The **model-admission gate** determines whether the new model itself is suitable for release. The **state-compatibility gate** begins after that decision and determines whether state generated by previous versions can continue to represent the current model.

This separation avoids conflating model regression with stale-state error. If (\theta_t) is worse than (\theta_{t-1}) on a particular slice, rebuilding every state under (\theta_t) reproduces the current model’s semantics but cannot repair the model itself. EvoKV is responsible for state convergence, not model selection.

We study three release semantics. **R0**, an output-only release, modifies a path that does not produce persistent state. **R1**, a routine continual update, modifies the complete model through warm-start incremental training. **R2**, a periodic encoder refresh, retrains the cache-producing encoder under the same architecture and prediction objective. These release families create a compatibility spectrum rather than assuming that every publication invalidates every state.

### 2.3 Why Ordinary Cache Invalidation Is Insufficient

Traditional cache invalidation asks whether an entry can be reused or must be rebuilt. Persistent neural state introduces three additional dimensions.

First, compatibility is not purely binary. A state may preserve most current-model behavior while containing localized error. This creates useful intermediate actions between No-op and Exact-All.

Second, tensor replacement must respect the computational dependencies of the model. An upper-layer key or value depends on the current hidden state produced by preceding layers. Replacing an arbitrary upper-layer region with a current-model tensor may reveal where error is located, yet the resulting state cannot necessarily be produced from the old state and available raw history. Such a replacement is a diagnostic intervention, not a deployable transition.

Third, state evolves after publication. A user state may be materialized under (\theta_0), receive new events under (\theta_1), evict old events at the context limit, and later be consumed by (\theta_2). Its version is therefore not a single scalar attached to an immutable tensor. It is a lineage over materialization, append, eviction, and partial reconstruction.

These properties make state evolution a runtime problem. The system must represent state provenance, reason about release compatibility, validate transition legality, allocate finite resources across a heterogeneous population, and execute the selected plans without replacing the true rolling lineage with a request-local approximation.

## 3. EvoKV Runtime

### 3.1 System Model

At release (t), let (\mathcal{S}_t) denote the complete set of states materialized at cutover. This population is determined before observing which users will issue future requests. For each state (u), the runtime selects an action (a_u) from a legal action set (\mathcal{A}_t(u)):

[
\min_{{a_u}}
\sum_{u\in\mathcal{S}_t}
L_u(a_u)
]

subject to

[
C_{\mathrm{probe}}
+
\sum_{u\in\mathcal{S}_t}C_u(a_u)
\le B_t.
]

Here, (L_u(a_u)) is the remaining divergence from Current Full after the action, (C_u(a_u)) includes exact-equivalent token-layer work and the associated state and history movement, (C_{\mathrm{probe}}) is the complete cost of sparse profiling, and (B_t) is the background release budget.

The objective is deliberately target-free. At cutover, the runtime does not know which users will return, which candidates they will receive, or what labels those interactions will produce. Recommendation quality is connected only after the assignments have been sealed and is used to evaluate whether restoring current-model fidelity also restores task behavior.

### 3.2 Runtime Architecture

EvoKV separates policy from execution.

The **control plane** consists of a versioned state catalog, a release compatibility analyzer, a sparse profiler, and a population planner. It executes once per model release and emits an immutable transition assignment for the full state population.

The **data plane** consists of a transition-plan validator and a lineage-preserving rolling executor. It groups states by action and effective shape, performs the required model computation, and materializes the resulting current-version state.

This separation allows the planner to reason in logical work units while allowing the executor to reorganize equivalent operations for GPU efficiency. An executor optimization may change batching, buffer allocation, or kernel count; it may not change the action assigned to any state.

### 3.3 Versioned State Catalog

EvoKV’s runtime interface associates each materialized state with a descriptor

[
D_u =
\langle
u,
v_{\mathrm{producer}},
g_{\mathrm{producer}},
\tau_{\mathrm{materialized}},
I_{\mathrm{history}},
n_{\mathrm{effective}},
\lambda_u,
d_u
\rangle .
]

The descriptor records the producer model, the signature of the cache-producing subgraph, the materialization time, the covered history interval, the effective retained length, the append/eviction lineage (\lambda_u), and the accumulated version debt (d_u).

These fields turn several experimental bookkeeping rules into system semantics. The cutover population is defined by states that actually exist at release, not by future served users. A state is materialized once and subsequently changes through append and capacity eviction. Consecutive No-op decisions preserve the mixed lineage rather than retrospectively regenerating the complete prefix under an older model.

In the current prototype, the necessary catalog fields are represented by frozen cutover manifests, release metadata, prefix-length and activity features, and rolling-cache records. At the first two development edges, the cutover populations contain 8,229 and 8,488 states, while only about 36% of those users later appear in the explicit-feedback request stream. Planning only over future served users would therefore use information unavailable to a real release-time system.

### 3.4 Release Compatibility Analyzer

The release pipeline provides EvoKV with a descriptor of the old and new model versions, including the identity or signature of the subgraph that produces persistent state. The compatibility analyzer first determines whether that producer changed.

If the producer signature is unchanged, the analyzer emits a **No-op compatibility certificate** and bypasses profiling and migration for the complete population. R0 is the concrete instance of this path. Its output layers change, while the persistent-state producer remains identical. Across all R0 experiments, Current Full and Reuse differ by at most (4.44\times10^{-15}) JS divergence. The metadata gate therefore selects No-op for every state with zero probe and migration cost.
If the producer changed, the analyzer admits a set of dependency-closed candidate plans and invokes the profiler. A producer change does not itself prove that Exact-All is required; it establishes that compatibility must be measured and allocated.

### 3.5 Dependency-Closed Transition Plans

EvoKV represents a migration action as a transition plan

[
P_a =
\langle
\operatorname{precondition},
\operatorname{readset},
\operatorname{compute},
\operatorname{writeset},
\operatorname{cost}
\rangle .
]

A plan is legal only when every tensor it writes can be derived from the persisted state, available raw history, and current model through a dependency-closed execution path. This contract distinguishes a system action from an oracle splice.

For example, a layer-0 key or value depends directly on the input embedding, normalization, and current layer-0 projection. EvoKV can therefore reproject a selected layer-0 region without constructing an unavailable current hidden state from a lower layer. In contrast, an arbitrary layer-2 replacement normally requires current-model layer-1 hidden states for the same positions. Copying exact layer-2 tensors from an offline Current Full execution is diagnostically useful, but it is not an executable transition plan.

The current EvoKV prototype instantiates the transition-plan abstraction with a frozen action library:

| Action           | Transition semantics                                                                      | Exact-equivalent token-layer work |
| ---------------- | ----------------------------------------------------------------------------------------- | --------------------------------: |
| No-op            | Retain the existing rolling state                                                         |                                0% |
| Layer0-Recent128 | Reproject layer-0 K/V for the most recent 128 positions                                   |                             6.48% |
| Layer0-Middle    | Reproject a fixed middle region at layer 0                                                |                            12.50% |
| Layer0-Full      | Reproject layer-0 K/V over the retained prefix                                            |                            25.00% |
| Hybrid-Tail128   | Retain the old prefix and replay the latest 128 events through the complete current stack |                            25.92% |
| Exact-All        | Replay the complete retained history under the current model                              |                              100% |

Layer0 actions exploit a dependency boundary in the model graph. Hybrid-Tail128 exploits a positional boundary: its tail is reconstructed causally through every layer, while its earlier prefix remains old. Exact-All is the complete reconstruction plan. All upper-layer arbitrary splices remain diagnostic-only.
This interface is more important than any single action. It gives the runtime a uniform object on which to enforce legality, estimate resource consumption, compare fidelity recovery, and schedule mixed execution.

### 3.6 Sparse Profiler

When migration may be required, EvoKV profiles a deterministic 1% sample of the cutover population. The profiler uses only information available at release: effective prefix length, state age, activity over the preceding 1, 7, and 30 days, recent unique-item count, organic-interaction ratio, and repeat ratio.

For each sampled state, the runtime executes target-free probes over the legal action set and measures the marginal fidelity recovery relative to No-op. A fixed `StandardScaler + Ridge(alpha=1.0)` model predicts the benefit of each action for the remaining population. The low-capacity model is intentional: the profiler is a release-time systems mechanism rather than a quality model trained on future labels.

Probe work is charged against the same release budget as migration. A larger sample therefore provides more observations while directly reducing the work available for state transitions. On the development population, 1% offers a stronger low-budget cost–recovery tradeoff than the fully reported 2% companion configuration.

### 3.7 Population Planner

The planner receives, for every state, an estimated benefit and cost for each legal action. It then selects exactly one action per state under the release budget. No-op remains a normal choice: the planner can spend its resources on a small high-risk region while leaving compatible or low-benefit states unchanged.

This formulation differs from a request-triggered cache policy. The decision is made for the complete materialized population, not only for states that happen to receive a request during evaluation. It is also different from a fixed release-level action. The same release can assign Exact-All to a small set of high-benefit states, dependency-closed partial reconstruction to a larger middle region, and No-op to the remainder.

Assignments are sealed before held-out recommendation labels are connected. The evaluation therefore measures whether a target-free systems objective transfers to task quality; labels do not participate in action selection.

### 3.8 Lineage-Preserving Rolling Executor

A correct executor must preserve persistent-state lineage. EvoKV materializes a state once at release cutover. During the following serving interval, each new event is appended under the active model semantics, and the oldest event is evicted when the 512-position capacity is reached.

This detail materially affects the experiment. Approximately 86.5%–87.0% of evaluated requests experience at least one rolling eviction before their query. Recomputing the surviving parent-model prefix independently for every request would create a request-conditioned diagnostic state, not the state that a deployed runtime would retain.

The rolling executor therefore replays the actual append and eviction sequence and applies each transition to the resulting state object. Exact reconstruction is checked against Current Full. R0 remains numerically exact. Recursive experiments additionally verify that a state materialized under (\theta_0), appended and evicted under (\theta_1), and consumed under (\theta_2) is not replaced by an artificial prefix recomputed entirely with (\theta_0).
To execute a mixed assignment efficiently, EvoKV groups operations by transition type and effective prefix length, preallocates output buffers, and removes avoidable cache cloning. These changes preserve each user’s frozen action. A numerical canary bounds the maximum per-user K/V difference between the original and grouped paths at (2.62\times10^{-6}), below the predefined (10^{-5}) tolerance.

### 3.9 Runtime Invariants

The design is governed by five invariants. **Compatibility** requires No-op to reproduce Current Full when the state producer is unchanged. **Dependency closure** requires every partial write to have a realizable current-model execution path. **Population integrity** requires the budget and assignment to cover the complete cutover population, with probe cost included. **Lineage fidelity** requires one-time materialization followed by real append and eviction, rather than request-local state synthesis. **Policy sealing** requires transition assignments to be fixed before recommendation labels are observed.

Together, these invariants define EvoKV as a state-evolution runtime rather than a collection of tensor-editing heuristics.

## 4. Evaluation

### 4.1 Experimental Setup

We evaluate EvoKV on a Yambda-50M platform derived from the Yambda music recommendation data. Yambda provides temporally ordered listening interactions, explicit preference events, and an `is_organic` marker that distinguishes organic behavior from recommendation-driven events [8].

The platform contains 46,467,212 listening events from 9,238 users, covering 877,168 observed items and 144,441 artists over approximately 301 days. User histories are long and highly skewed: the P50, P90, and P99 lengths are approximately 3,024, 13,044, and 23,765 interactions.

The development model contains four HSTU layers, hidden dimension 128, and a retained context of 512. We evaluate a single-task explicit-feedback model, M0-F, and a multi-task model, M1, that shares a persistent encoder across the N, R, and F workloads. Every formal condition retains training seeds 17, 37, and 71. No seed is selected according to long-state utility, staleness, quality, or scheduler performance.

Our evaluation asks five questions. Does the workload use persistent long-history state? Do natural model releases make that state observably stale? Can dependency-closed plans recover the error below Exact-All cost? Does the release-time planner outperform simpler policies at equal work? Do the mechanisms remain effective under measured GPU execution and recursive version debt?

### 4.2 Long-State Utility and Release-Dependent Compatibility

The Explicit Like/Dislike workload establishes the required long-state utility. Relative to Recent-32, Full-512 improves aggregate log loss by 0.00204 for M0-F and 0.00279 for M1-F. M1-F is positive in all three seeds. Natural Next-Listen and Return-to-Familiar do not pass the same long-state gate, providing short-state controls rather than being forced into the migration narrative.

State compatibility then separates by release semantics. R0 produces a maximum Current-Full-versus-Reuse JS divergence of (4.44\times10^{-15}), demonstrating that a release outside the state producer can safely retain every state. Both R1 continual updates and R2 encoder refreshes create nonzero target-free divergence.

The strongest task-quality evidence occurs in M1-R2. Current Full improves over stale parent state by 0.003274 in log loss, 0.01331 in ROC-AUC, 0.01773 in dislike PR-AUC, and 0.000999 in Brier score. R1 produces stable fidelity differences, while its task-quality effects are smaller and more release dependent. The system implication is not that every release requires migration; it is that release semantics must enter the compatibility decision.

### 4.3 Local Structure and Executable Recovery

Diagnostic tomography shows that stale error is structured across model layers, history positions, and users. The strongest diagnostic regions recover approximately 78%–99% of the error. Recent-128 is positive across every non-R0 condition and seed, whereas recent-1 recovers only about 1%–2%. The effect is therefore not a single-token shortcut.

In M0-F R2, the diagnostic `layer0 × middle` region recovers 92.06% of JS divergence in aggregate and is positive in all three seeds. It simultaneously improves log loss, ROC-AUC, dislike PR-AUC, and Brier score. User-level error is also concentrated: depending on the condition, the highest-risk 10% of states account for 18.4%–70.5% of total No-op risk.
The dependency-closed actions preserve this recovery opportunity under the true rolling executor. For R2, Hybrid-Tail128 recovers 85.5% of aggregate fidelity loss for M0-F and 94.2% for M1 at approximately 27.2% of Exact-All logical work. Some release-specific Layer0-Full conditions recover 98%–99.6% at approximately 25% work, while the full action library retains variation across releases and seeds.

On held-out M1-R2 requests, Hybrid-Tail128 removes approximately 83.3% of equal-seed aggregate log-loss harm and approximately 94% of AUC harm. These results establish that the legal plans recover task-relevant behavior rather than only reducing an internal tensor distance.

### 4.4 State-Level Planning at Equal Cost

With a deterministic 1% probe, EvoKV obtains the following equal-seed target-free recovery:

| Release and model | 5% budget | 10% budget | 25% budget |
| ----------------- | --------: | ---------: | ---------: |
| R1 edge 1 / M0-F  |     38.0% |      54.1% |      72.9% |
| R1 edge 1 / M1    |     41.6% |      60.8% |      87.9% |
| R1 edge 2 / M0-F  |     39.7% |      54.9% |      78.5% |
| R1 edge 2 / M1    |     41.3% |      67.5% |      95.4% |
| R2 / M0-F         |     54.1% |      79.2% |      95.8% |
| R2 / M1           |     38.1% |      58.2% |      83.4% |

Every non-R0 seed produces positive recovery. R0 is removed from the learned path by its compatibility certificate and assigns No-op with zero probe and migration work.

The relevant baseline is not merely No-op. EvoKV is compared at the same total budget with the best release-level uniform action, feasible uniform partial policies, random Exact selection, and zero-probe rankings based on prefix length, state age, activity, or recent unique-item count.

At the 5% budget, the Ridge planner exceeds the strongest non-learning baseline in each of the six non-R0 release–model conditions by 24.9–43.0 percentage points on average. Every condition is positive in all three seeds, and paired-user bootstrap intervals remain positive. This result establishes that the planner exploits action-specific state heterogeneity rather than benefiting only from the existence of a strong fixed partial action. Its largest relative value appears in the low-budget regime, where the runtime must choose which states and which transitions deserve scarce work.

The assignments are sealed before quality labels are attached. In M1-R2, the 25% policy improves aggregate log loss in all three seeds. The equal-seed average improvement is 0.001509, accompanied by an average ROC-AUC improvement of approximately 0.0117 in the initial held-out evaluation. Lower budgets retain positive average improvements with greater seed variability.

### 4.5 Measured GPU Execution

Logical token-layer work does not directly determine wall-clock runtime. A mixed policy creates multiple operation types, small batches, ragged prefixes, state copies, and additional kernel launches. EvoKV therefore reports both logical work and measured execution time.

Before executor grouping, mixed-policy rollout consumes 36.4%–72.2% of Exact-All runtime. The semantics-preserving executor optimizations improve all ten preregistered runtime conditions by (1.12\times)–(2.56\times), with a geometric mean of (1.60\times). Operation-batch count falls by 21%–61%. The final executor saves 35.8%–80.4% of runtime relative to Exact-All.

These measurements also define the correct systems claim. A 5% logical token-layer budget does not imply a 95% wall-time saving. EvoKV’s frontier explicitly retains both quantities, allowing the planner’s logical allocation and the executor’s hardware efficiency to be evaluated separately.

### 4.6 Recursive Version Debt

A one-hop parent state, an artificial age-2 state, and a true recursive state are not equivalent. The true lineage materializes (\theta_0) state at the first edge, appends and evicts events while (\theta_1) is active, and later presents the mixed state to (\theta_2).

A canary shows that recomputing the complete edge-2 prefix under (\theta_0) overstates actual debt. Relative to a one-hop (\theta_1) state, the true recursive state increases mean JS by approximately 3.5%, whereas the artificial direct-(\theta_0) state increases it by approximately 24%. EvoKV therefore evaluates version debt through the persistent rolling lineage rather than through a convenient age label.

Across the complete recursive population, M1 exhibits stable action structure. Hybrid-Tail128 recovers 55.62% of No-op divergence, Layer0-Middle recovers 77.47%, and Layer0-Full recovers 97.45% on average across seeds. User-level debt remains concentrated, with the highest-risk 10% contributing 30.8%–54.6% of total recursive JS.

EvoKV applies the same frozen profiling and allocation algorithm to the recursive population. It does not reuse assignments from the previous release; it recalibrates with the same 1% deterministic probe, feature set, Ridge model, and allocator contract. At a 25% budget, the M1 planner recovers 93.3%–95.4% of fidelity across the three seeds and exceeds the strongest deterministic baseline by 17.3 percentage points on average. At the 5%, 10%, and 25% budgets, both M0-F and M1 outperform the strongest deterministic baseline in every seed, with positive paired-user bootstrap intervals.

On recursive held-out quality, M1 aggregate log loss improves in all three seeds at the 10% and 25% budgets. ROC-AUC also improves in all three seeds, with mean absolute gains of approximately 0.000483 and 0.001641, respectively. These results show that EvoKV’s runtime mechanisms continue to operate when state debt is produced by actual multi-version evolution rather than by a single isolated model edge.

### 4.7 Scope of the Current Evidence

The evaluation establishes the state abstraction, release-dependent compatibility, dependency-closed transition plans, state-level planning, rolling execution, and recursive version debt on the (\theta_0)–(\theta_2) development chain. The current implementation is a PyTorch release-time prototype using a 4-layer, 128-dimensional encoder and context length 512 on Yambda-50M.

Rare dislike-only calibration is not monotonic across every seed. Exact-All frequently moves in the same direction as the policy, indicating that part of the effect belongs to the current model’s complete semantics rather than to mixed-state allocation alone. EvoKV therefore claims budgeted recovery of Current-Full fidelity and its principal aggregate quality effects; it does not equate aggregate fidelity with monotonic improvement in every downstream slice.

## 5. Discussion

EvoKV changes the unit of model publication. A release is not complete when new parameters reach the serving process. It is complete when the model and the persistent state population have entered an explicitly managed compatibility regime.

This regime does not require every state to become physically identical to a complete current-model reconstruction. A compatible output-only release may converge through No-op. A moderate release may converge through a mixture of retained and partially reconstructed states. A high-risk refresh may allocate exact reconstruction to a small subset while using cheaper dependency-closed plans elsewhere. The runtime therefore defines convergence as a controlled population transition under a measurable fidelity and resource contract.

The versioned-state abstraction also explains why multiple apparently separate observations belong to one system. Release semantics determine whether compatibility analysis can terminate at No-op. Dependency structure determines which partial actions are legal. State features and sparse probes determine where those actions are valuable. The executor determines whether their logical advantages survive real batching and rolling lineage. Recursive debt determines whether a local decision remains meaningful over multiple publications.

Under this abstraction, EvoKV is not a more flexible cache policy and not merely a stronger partial-recomputation point. It is a runtime for managing the lifecycle of model-generated state:

[
\text{materialize}
\rightarrow
\text{append and evict}
\rightarrow
\text{analyze compatibility}
\rightarrow
\text{plan transition}
\rightarrow
\text{execute}
\rightarrow
\text{advance lineage}.
]

The development evidence closes this lifecycle end to end. Long state is useful under a qualified workload; its compatibility changes with release semantics; its error has recoverable structure; legal plans expose a cost frontier; a target-free planner improves that frontier at equal cost; and the resulting assignments execute efficiently over both one-hop and recursive rolling state.

## References

[1] W.-C. Kang and J. McAuley. “Self-Attentive Sequential Recommendation.” *IEEE International Conference on Data Mining*, 2018.

[2] Q. Pi et al. “Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction.” *ACM International Conference on Information and Knowledge Management*, 2020.

[3] J. Zhai et al. “Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.” *International Conference on Machine Learning*, 2024.

[4] Z. Liu et al. “Monolith: Real Time Recommendation System with Collisionless Embedding Table.” 2022.

[5] C. Sima et al. “Ekko: A Large-Scale Deep Learning Recommender System with Low-Latency Model Update.” *USENIX Symposium on Operating Systems Design and Implementation*, 2022.

[6] K. K. Matam et al. “QuickUpdate: A Real-Time Personalization System for Large-Scale Recommendation Models.” *USENIX Symposium on Networked Systems Design and Implementation*, 2024.

[7] J. Gjengset et al. “Noria: Dynamic, Partially-Stateful Data-Flow for High-Performance Web Applications.” *USENIX Symposium on Operating Systems Design and Implementation*, 2018.

[8] A. Ploshkin et al. “Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking and Retrieval.” 2025.
