# CohortKV: Compiled K/V Translation Across Streaming Recommender Versions

**Anonymous Authors**

> **Evidence status — remove before submission.** This manuscript reflects the frozen
> Stage 0–6 single-configuration evidence as of 2026-07-28. Every reported number is bound to a
> completed artifact. New-seed and cross-configuration experiments belong to Stage 7 and are
> described only as open evidence. The paper deliberately separates one-hop hot-HBM data-plane
> speed, repeated-update GPU work, one-time compiler cost, host-device movement, and
> failure-safe publication. It is now the frozen **D1-only baseline draft**. Under the current
> paper structure, Sections 4–6 are connected components of D1; the deadline-based lifecycle in
> Section 5 is not the current Design 2. Current D2 is fixed-action, wave-compiled segmented
> execution over row-sharded embeddings and remains under formal evaluation. Its authoritative
> definition and claim boundary are in `docs/08_core_insights_and_roadmap.md`,
> `docs/eval_protocol.md`, and `docs/future_design/DESIGN2_FINAL_PLAN.md`; they must be integrated
> in a later manuscript revision rather than filled into this frozen evidence snapshot.

## Abstract

Streaming generative recommenders update model parameters while retaining long user histories as
persistent key/value (K/V) caches. Each model publication therefore turns an intact cache into
model-version-stale derived state: reuse is inexpensive but no longer matches the current model,
whereas exact replay restores current-model K/V at the cost of forwarding the complete history.
We present **CohortKV**, a system for translating stale HSTU K/V across streaming model versions.

CohortKV has three connected mechanisms. First, a version-pair compiler fits the shared residual
between fresh K/V and a cheap reprojection of old normalized states and folds that residual into
one affine program. For execution, CohortKV reparameterizes the program directly over the already
resident old K/V, eliminating all additional per-record normalized-state storage. Second, a
label-free staggered-renewal lifecycle divides each model edge between lightweight migration and
exact refresh, assigns each continuously resident cache a scheduled renewal deadline, and
recursively advances the actual mixed cache through progressively growing canonical-date
histories. Third, a destination runtime verifies program, version, source-state, empirical
semantic, and capacity conditions before updating a complete cohort. A separate copy-on-write
mode provides atomic manifest publication.

In an earlier 50%-fidelity cohort-tiered mechanism protocol over 27 independently trained model
chains, compiled repair costs `0.121×` resident exact recomputation while recovering `0.587` of
the stale-to-fresh K/V gap. Separately, in the controlled seed-0 configuration, direct old-K/V
translation preserves `0.881–0.936` cache-gap recovery. An occupancy-equivalent hot-HBM timing
job completes 682-record updates in `0.930/0.494/0.255` seconds on one/two/four A40 GPUs, versus
`18.695/9.729/4.766` seconds for paired exact replay; complete real-value transport is validated
independently. Over an 11-edge growing-history chain, the
deadline-based lifecycle uses `0.100×` the paired retained-prefix exact GPU time and preserves `99.746%`
of weighted NDCG@100. A separate two-GPU integration commits all 682 records on normal and
exact-fallback paths, while injected mid-job and pre-commit failures expose no partial target and
preserve readable old state.

These measurements have distinct boundaries. The hot-HBM timing result assumes prepublished
programs and uses shape-, dtype-, layout-, and occupancy-equivalent old-K/V tensors; a separate
complete real-value run validates transport tolerance. The lifecycle ratio excludes a common
target-model append and reports `662.87 GB` of host-device movement separately. Fitting,
preparation, and empirical semantic validation take `308.90 seconds` for the three evaluated
non-adjacent version pairs; the 11 adjacent lifecycle programs receive fit/provenance validation
but not that same 60-user gate. The failure-safe experiment supplies correctness rather than
throughput. Current system and lifecycle results are single-configuration development evidence;
cold storage, online serving, and cross-seed system replication remain outside the present claim.

## 1. Introduction

Generative recommendation predicts a next item from a user's ordered behavior history.
Architectures such as HSTU make long histories computationally useful and are designed for
high-cardinality, non-stationary streams [1]. At the same time, production recommenders update
their parameters continually so that recent interactions and content can affect predictions [2].
These two properties create a state-maintenance problem: the system wants to retain the expensive
representation of a long history, but the model under which that representation was computed
keeps changing.

For history \(x\), let

\[
C_v(x)=F(\theta_v,x)
\]

denote the prefix K/V generated under model version \(\theta_v\). Once training publishes
\(\theta_t\), \(C_v(x)\) is still physically intact but is no longer current-model state. Reusing
it is cheap; recomputing \(C_t(x)\) is semantically exact but replays the entire history. This is
not ordinary eviction restoration. The bytes have not moved or disappeared—the model-relative
meaning of those bytes has changed.

Existing K/V systems clarify nearby but different boundaries. vLLM manages same-model
autoregressive K/V allocation [8]. CachedAttention retains same-conversation state across
requests [9]. HCache restores evicted state from intermediate activations under one model [3].
MTServe persists generative-recommendation K/V and manages placement [5]. DroidSpeak addresses
cross-model reuse among fine-tuned LLM variants through selective layer recomputation [4].
Consequently, cross-model K/V reuse alone is not our contribution. We study successive streaming
recommender versions and ask whether a source-to-target transform can be compiled once per model
pair, applied to a complete cache cohort, and controlled across repeated updates.

Three difficulties make that question more than a kernel problem.

First, application quality cannot safely decide whether state is current. Across our model-version
chains, age orders K/V drift more consistently than it orders ranking impact, and exact
current-model K/V is not guaranteed to improve every fixed evaluation slice. A system that predicts
"safe reuse" from age, per-user drift, or observed task gain conflates semantic maintenance with a
ranking-quality oracle that the data does not support.

Second, a cheap one-hop transform is not a recursive guarantee. Once an approximate cache becomes
the source for another model edge, the next program consumes state outside its one-hop empirical
validation distribution. Exact refresh therefore remains necessary, but synchronized or
unbounded refresh can erase the benefit or create maintenance waves.

Third, resident arithmetic can be irrelevant to complete-job cost. Our first compiled path stored
one FP16 normalized state per layer and record. Its kernel was cheap, but reading and preparing the
`17.82-GB` source consumed `91.35%–96.91%` of completion, making compiled execution slower than
exact replay at all six HBM/DRAM endpoints. A system result must include source representation,
destination placement, capacity, and publication semantics rather than substituting kernel time
for job time.

We present **CohortKV**, which connects three mechanisms around a source/target model-version
cohort:

1. a **compiled direct-old-K/V translator** that learns a shared affine repair from old normalized
   states, empirically validates the deployed representation without recommendation labels, and
   reparameterizes the program over already resident old K/V;
2. a **deadline-based growing-history lifecycle** that chooses migration or exact refresh before
   execution, assigns stable deadlines to continuously resident caches, and advances the actual
   mixed state through consecutive model edges; and
3. a **destination runtime and transactional closure** that uses a common final-layout K/V
   operator, fixed preflight, exact fallback, and complete-manifest publication.

The current evidence supports these contributions with deliberately separated measurements.
In an earlier 50%-fidelity cohort-tiered protocol, 27 model chains spanning KuaiRand and two
Tenrec tables at three capacities average `0.121×` resident exact cost and `0.587` K/V-gap
recovery. This is mechanism replication, not replication of the final system. In the larger
seed-0 configuration, three non-adjacent deployed FP16 programs close `0.881–0.936` of that gap
and dominate all 53 evaluated DroidSpeak-style contiguous intervals. Composing those programs
through the source model's K/V projection removes the extra normalized capsule. In a separate
shape-, dtype-, layout-, and occupancy-equivalent HBM timing experiment, the success-path data
plane is `18.72–20.11×` faster than paired raw-history-resident exact at one, two, and four GPUs;
a complete real-value transport establishes numerical tolerance independently.

For repeated updates, CohortKV first migrates or exactly refreshes the retained prefix under the
target model, stops the maintenance timer, and then appends newly observed behavior under that
target model. The frozen H12 renewal policy costs `0.100017×` paired retained-prefix exact over 11
edges and reaches weighted AUC/NDCG@100/Hit@100 ratios of
`1.000039/0.997463/1.000000`. It is chosen over a lower-cost token-debt endpoint because the
latter has no per-cache renewal deadline. These adjacent programs are evaluated by the complete
rollout but do not carry the separate three-view gate used for the non-adjacent one-hop programs.

Finally, a two-A40 copy-on-write integration binds the H12 action partition for
\(\theta_0\rightarrow\theta_1\). A normal job publishes all 682 records; a shape-preserving program
perturbation is detected before target execution and publishes through exact fallback; mid-job and
pre-commit failures reveal no partial target and preserve checksum-valid readback of all old
records. This is correctness evidence, not a throughput result. The faster extent-reclaiming path
is measured separately and is not abort-safe.

Our contributions are:

- **An implemented affine translator with zero additional per-record runtime state.** The
  compiler moves shared adaptation out of the record path, and its direct-old-K/V
  reparameterization turns the serving cache itself into the runtime source. Missing direct-ridge
  controls still prevent attribution of the measured gain specifically to residual centering.
- **A deterministic migrate-or-exact lifecycle under growing canonical-date histories.** It separates
  maintenance from foreground append, recursively consumes previous actual output, and supplies a
  scheduled deadline for continuously resident caches without using recommendation labels.
- **An implementation and measurement study that exposes the system boundary.** It demonstrates
  why the normalized source fails, validates a same-boundary hot-HBM data plane, and separates
  fast reclamation from failure-safe copy-on-write publication.

The paper does not claim a universal task benefit from maintenance, a formal semantic certificate,
an end-to-end growing-history state-movement speedup, cold-storage performance, online serving
latency, or replication of the final 16-layer lifecycle beyond seed 0.

## 2. Background and motivation

### 2.1 HSTU prefix K/V and stale inference

We use a modular simplified HSTU that preserves the two properties needed here: pointwise
unnormalized attention and first-class per-layer K/V output. For layer \(\ell\), let

\[
z_\theta^\ell(x)=\operatorname{Norm}_\theta^\ell
\left(h_\theta^\ell(x)\right)
\]

be the normalized input state. Concatenated keys and values are

\[
y_\theta^\ell(x)=
\left[k_\theta^\ell(x),v_\theta^\ell(x)\right]
=z_\theta^\ell(x)P_\theta^\ell+b_\theta^\ell, \tag{1}
\]

where \(P_\theta^\ell\) concatenates the K and V projections. A batch carries valid sequence
lengths, and positions beyond each length are zeroed.

The stale-inference protocol predicts item \(t+1\) from hidden state \(t\). A fresh evaluation
forwards the complete history with the target model. A stale evaluation supplies an old-version
prefix cache and evaluates the latest token with the target model. Both branches use the same
history, target checkpoint, latest token, candidate catalog, and engaged positives; only the model
version that produced the resident prefix differs. Exact K/V is the semantic reference but not a
guaranteed upper bound on ranking quality.

The datasets contain interactions rather than request traces. Training, request arrival, user
hotness, foreground interference, and service-level objectives are outside the evaluated job.

### 2.2 Maintenance opportunity and its boundary

We separate:

- **full-compute streaming value**: a current streaming model with current K/V, relative to a
  frozen base model;
- **full-reuse streaming value**: the same current model consuming an old prefix cache; and
- **cache-maintenance value**: full compute minus full reuse.

The primary KuaiRand Top-50k/all-chunks protocol reports a BestRank full-compute value of
`3837.67` (95% CI `[3389.91, 4285.44]`), of which stale reuse retains `2952.11`
(`[2700.21, 3204.02]`). The remaining maintenance value is `885.56`
(`[460.24, 1310.88]`), a `23.1%` staleness tax.

**Table 1: Aligned cross-table streaming and maintenance value.** Values are BestRank
improvements; intervals use four training seeds.

| Dataset/table | Full compute | Full reuse | Maintenance |
|---|---:|---:|---:|
| KuaiRand | 484.34 [462.15, 506.54] | 399.02 [370.79, 427.26] | 85.32 [53.74, 116.91] |
| Tenrec QB, fixed horizon | 94.70 [70.49, 118.90] | 64.38 [54.41, 74.35] | 30.31 [14.08, 46.55] |
| Tenrec QK, Top-5k | 47.34 [29.20, 65.47] | 34.34 [20.55, 48.13] | 13.00 [5.70, 20.30] |

QB and QK are related tables from one collection and use ordinal rather than shared calendar
time. The result establishes a cross-table opportunity, not three independent deployment domains.

A frozen 3×3 data/model-capacity screen sharpens the boundary. All nine cells have positive
full-compute and full-reuse streaming value in all four seeds, but the mean BestRank staleness tax
is not uniformly positive or monotone with capacity. Large KuaiRand and large QB expose taxes of
`0.360` and `0.548`, while large QK is `-0.005` and QB-medium is `-0.060`. Scale increases
maintenance cost and can widen the quality opportunity; it does not mechanically create one.

### 2.3 Why age, drift, and task gain are not routers

At a fixed target, every 3×3 age curve has monotonicity violations. In the controlled 16-layer
diagnostic, age strongly orders K/V drift, yet after the base-to-stream boundary it explains only
`6.15%` of MeanRank variation, compared with `60.9%` explained by current version identity.
Per-user relative K/V drift and maintenance utility correlate at only `0.020`
(95% CI `[-0.012, 0.052]`). The explored JVP/Fisher route is also not cheaper than the operation it
would govern.

These measurements lead to two rules. Version identity may organize work, but no observed
recommendation label decides whether one cache is migrated or refreshed. Empirical semantic
validation compares approximate state with current-model state; final recommendation metrics
evaluate the resulting policy but do not route it.

### 2.4 Two negative results that shape the final design

Equation (1) exposes a cheap current-projection baseline, but the representation used to execute
that baseline matters. The original design retained every layer's old \(z_v^\ell\) in FP16.
Although the compiled resident operator was cheap, the complete 682-record capsule occupied
`17.82 GB` and made the path slower than exact at every matched HBM/DRAM endpoint. Source
read/decode/pinning consumed `91.35%–96.91%` of wall time. This negative result motivates the
direct-old-K/V translator in Section 4.

Repeated maintenance exposed a separate failure. A per-cache norm-sketch threshold produced
acceptable cumulative fidelity but refreshed between `0.15%` and `65.1%` of records from one
edge to the next. Such synchronized work is operationally undesirable even when average quality is
acceptable. A later growing-history control also found that current-edge norm shift has mean
Spearman correlation `0.0341` with realized candidate error. These results motivate a deterministic
renewal schedule rather than a learned or thresholded risk oracle.

## 3. CohortKV D1 overview

### 3.1 Job boundary and state objects

CohortKV treats a model update as translation and publication over persistent derived state. The
job accepts:

- a fixed set of committed old caches and raw histories;
- a declared target model version;
- immutable source-to-target programs;
- a frozen migrate-or-exact action partition;
- execution devices; and
- one explicit destination.

In copy-on-write mode, it produces one complete target-version manifest or aborts without changing
the visible version. The faster reclaiming mode is evaluated only for successful completion and
does not promise this abort behavior. Destination selection, training, requests, and foreground
scheduling are outside the job.

Four objects connect the system:

1. A **version cohort** \(\gamma=(v,t)\) groups records with source version \(v\) and target
   version \(t\). It keys compilation, program lookup, homogeneous execution, and lineage.
2. A **translation program** is an immutable, hash-bound affine and empirical semantic record for
   one cohort.
3. A **cache lineage** records the action, source/target versions, last exact version, and
   migration depth of every committed cache.
4. A **destination transaction** makes only a complete target manifest visible.

A version cohort is not a prediction that reuse is safe. Every reusable prefix receives either
compiled translation or exact refresh, and natural cache misses are exact.

### 3.2 Logical decomposition of one model edge

For edge \(\theta_v\rightarrow\theta_{v+1}\), the logical sequence is:

```text
previous committed post-append K/V
  -> derive the retained prefix required by the next history
  -> verify artifact, version, source, semantic canary, and capacity
  -> choose migrate or exact from the frozen lifecycle
  -> update the retained prefix under theta_(v+1)
  -> stop the maintenance timer
  -> append newly observed behavior under theta_(v+1)
  -> stage complete per-record target extents
  -> commit one complete manifest, or abort
```

The append is common foreground inference and is measured separately from migration. Only the
post-append full cache may commit or become the next edge's source; a retained prefix is private
intermediate state. The diagram is a logical decomposition, not one integrated benchmark: the
recursive evaluator, hot-HBM reclaiming path, and copy-on-write publication are three separately
scoped experiments.

### 3.3 Claim firewall

The current evidence consists of separate experiments whose numeric gains are not composed.

**Table 2: Measurement boundaries used throughout the paper.**

| Property | Measured boundary | Explicit exclusions |
|---|---|---|
| `18.72–20.11×` one-hop speed | Prepublished program; shape/dtype/layout/occupancy-equivalent old-K/V source in HBM; normal extent-reclaiming path; real-value tolerance checked separately | Compile/validation, recursive lifecycle, abort safety, cold storage, foreground interference |
| `U/E=0.100017×` lifecycle work | One A40; corrected growing history; GPU work for the matched retained prefix | Target append, CPU scheduling, catalog scoring, `662.87 GB` H2D/D2H movement, end-to-end latency |
| `308.90 s` setup | Fit, runtime preparation, and empirical semantic validation for the three non-adjacent \(\theta_0/\theta_4/\theta_{10}\rightarrow\theta_{11}\) pairs | Not H12 adjacent-program setup; not hidden in data-plane timing; no measured overall break-even claim |
| Copy-on-write closure | Two A40s; one \(\theta_0\rightarrow\theta_1\) normal/fallback/fault integration | Throughput, process/GPU crash, journal, resume, durability |
| `0.121× / 0.587` replication | Earlier 50% cohort-tiered protocol over 27 trained chains | Replication of the final 70% full-affine/direct-old-K/V/H12 system |

No single experiment simultaneously establishes the one-hop hot-HBM speedup, the repeated-update
`U/E` ratio, and abort-safe publication.

## 4. Design 1: Compiled direct-old-K/V translation

### 4.1 Shared residual compilation

For cohort \((v,t)\), the target projection applied to old normalized state gives

\[
\widetilde y_{\mathrm{cheap}}^\ell
=z_v^\ell P_t^\ell+b_t^\ell. \tag{2}
\]

Calibration records also expose exact target K/V
\(y_t^\ell=z_t^\ell P_t^\ell+b_t^\ell\), so the residual is

\[
r^\ell=y_t^\ell-\widetilde y_{\mathrm{cheap}}^\ell. \tag{3}
\]

The compiler fits a ridge-regularized affine

\[
r^\ell\approx (z_v^\ell-\mu^\ell)A^\ell+\bar r^\ell. \tag{4}
\]

Before publication it folds the residual into

\[
\widehat P^\ell=P_t^\ell+A^\ell,\qquad
\widehat b^\ell=b_t^\ell+\bar r^\ell-\mu^\ell A^\ell. \tag{5}
\]

The record path remains one affine:

\[
\widehat y_t^\ell=z_v^\ell\widehat P^\ell+\widehat b^\ell.
\]

This is a projection-centered ridge map, not an equivalence to target hidden propagation.
Attention-use weighting affects fitting only; it does not change program shape or online work.
Recommendation labels are not used.

### 4.2 Empirical semantic validation

The following frozen contract is implemented for the three non-adjacent source pairs used in the
one-hop frontier and hot-HBM source study. The 11 adjacent programs used by the growing-history
experiment have a different evidence boundary described in Sections 7.2 and 8.5.

Fit, hyperparameter-selection, empirical-validation, and final-test users are disjoint. For each
validation user, the deployed serialized representation is compared with exact target K/V using:

\[
e_{\mathrm{cache}}=\text{relative K/V error},\quad
e_{\mathrm{score}}=1-\cos(s_a,s_t),\quad
e_{\mathrm{top100}}=1-\operatorname{overlap}_{100}(a,t). \tag{6}
\]

Recovery is the fraction of the reuse-to-exact error gap closed by action \(a\):

\[
\operatorname{recovery}(a)=
\frac{e_{\mathrm{reuse}}-e_a}
     {e_{\mathrm{reuse}}-e_{\mathrm{exact}}}. \tag{7}
\]

The frozen seed-0 contract requires at least 70% recovery in all three views, a one-sided 90%
bootstrap lower bound of at least 70% for ratio-of-means recovery, at least 80% qualifying-user
coverage after a one-sided 90% Wilson lower bound, and measured resident GPU cost no greater than
30% of exact for the primary action. Sensitivity targets from 50% through 80% select compiled
affine for all three source pairs; 90% selects exact.

We call this an **empirical semantic gate**, not a formal certificate. Passing supports the
evaluated version pair and held-out role; it does not guarantee unseen distributions or recursive
use of an approximate output.

### 4.3 Reparameterization over existing old K/V

Normalized state is convenient for fitting but unnecessary at runtime. Let the source model's
stacked projection be \(P_v^\ell\), with bias \(d_v^\ell\), so existing old K/V is

\[
o_v^\ell=[K_v^\ell,V_v^\ell]
=z_v^\ell P_v^\ell+d_v^\ell. \tag{8}
\]

When \(P_v^\ell\) has full row rank, its minimum-norm right inverse \(R_v^\ell\) satisfies

\[
P_v^\ell R_v^\ell=I.
\]

Substituting
\(z_v^\ell=(o_v^\ell-d_v^\ell)R_v^\ell\)
into the deployed affine yields

\[
\widehat o_t^\ell
=o_v^\ell(R_v^\ell\widehat P^\ell)
+\widehat b^\ell
-d_v^\ell R_v^\ell\widehat P^\ell. \tag{9}
\]

Equation (9) is a reparameterization of the deployed program, not a second fitted model. In the
evaluated source checkpoints, stacked projections have full row rank and condition numbers
`5.97–10.74`. Three direct FP16 programs occupy `100.78 MB` per worker; the extra per-record runtime state
falls from a `17.82-GB` normalized capsule over the measured cohort to zero.

Admission requires verified source checkpoint and parent-program provenance, compatible shapes,
finite prepared tensors, expected old K/V, and capacity. The equivalence assumes exact
source-version old K/V. Once an approximate cache is recursively migrated, Equation (9) remains
the executed affine but its one-hop empirical validation no longer applies; Section 5 supplies
exact renewal.

### 4.4 Actions and immutable artifacts

The primary action library contains:

1. compiled affine translation;
2. residual-\(p\) structural replay when its auxiliary state exists; and
3. exact target-model recomputation.

Residual-\(p\) executes the first \(p\) target-model blocks exactly and transports a boundary
displacement to deeper old states before target `Norm + Wk/Wv`. It needs raw history and every old
pre-block hidden state from \(p\) through the final layer. Real shards overflow FP16 for this
unnormalized suffix, so it is stored in BF16 and accounted separately. When the suffix is absent,
the valid fallback skips residual-\(p\) and terminates at exact.

Compilation publishes immutable program tensors, source and target identities, shapes, fitting
metadata, empirical-gate evidence, selected action, and an exact-terminated fallback order. The
artifact is the only compiler-to-runtime channel; runtime never silently converts a mismatched
program.

The present baseline suite includes reuse, exact, cheap current projection, residual-\(p\), and a
DroidSpeak-style contiguous selective-layer implementation. It does not yet isolate the
projection-centered residual formulation from an unconstrained direct K/V ridge map or a plain
\(z_v\rightarrow y_t\) ridge map. Those are required algorithmic ablations before a final novelty
claim.

The frozen growing-history H12 integration does not retain the BF16 suffix required by
residual-\(p\), so its executable action set is compiled translation or exact. Residual-\(p\)
remains a quality-tier action in plans whose declared auxiliary state is present; the runtime may
not pretend that state exists or estimate its cost as zero.

## 5. Design 1 continued: Deadline-based migration under growing histories

### 5.1 Correct canonical-date and timing boundary

Let \(R_v\) be the retained suffix of the previously admitted history after cropping enough old
tokens to make room for newly observed window \(\Delta_{v+1}\). The paired update is:

```text
mixed: previous actual K/V(R_v)
       -> migrate or exact under theta_(v+1)
       -> stop mixed timer
       -> append Delta_(v+1) under theta_(v+1), outside timer

exact: raw R_v
       -> exact under theta_(v+1)
       -> stop exact timer
       -> append Delta_(v+1) under theta_(v+1), outside timer
```

This separates two invariants. First, target-model computation for newly observed behavior is
foreground inference and enters neither side of the primary migration ratio. Second, the append
occurs after the retained prefix is updated and uses the target model. Encoding the new window with
the source model before migration changes the synchronized prefix and is not the primary protocol.

The exact post-append branch is checked against a one-shot target-model forward on
\(R_v\Vert\Delta_{v+1}\) for K/V, hidden state, scores, and Top-100 output. One-shot exact is the
authority if they disagree.

### 5.2 Recursive state and action classes

Every committed mixed post-append cache is the next edge's actual source. Each record carries
source and target versions, last exact version, migration depth, final action, and fallback reason.
There are three action classes:

- **natural exact** for a cold, re-entered, zero-overlap, or expected-but-missing retained prefix;
- **scheduled exact** when the lifecycle deadline is due; and
- **migration** otherwise.

Exact actions reset lineage. Migration increments depth and consumes the previous actual state.
Recommendation labels and next-window positives are unavailable to the scheduler.

### 5.3 Work-balanced staggered renewal

The selected policy assigns every reusable record a phase in a renewal horizon \(H\). New records
are ordered by decreasing retained-prefix tokens and greedily assigned to the phase with the
smallest current token load. A record is exactly refreshed when its due version arrives, after
which its deadline advances by \(H\). Natural exact also resets its next deadline. Otherwise it
migrates.

This construction is deterministic, label-free, and balances estimated exact work by prefix
tokens. Unlike a global budget, it gives each continuously resident cache a renewal deadline.
For a continuously resident record, it is a deadline-bounded systems heuristic, not an optimal
scheduling theorem or a per-cache failure predictor.

### 5.4 Candidate selection

Development evaluated four preregistered label-free families with four parameters each:
work-balanced renewal, total token debt, AoI MaxWeight, and model-time renewal. These screens used
an earlier append order and therefore serve only to choose candidates, not as formal primary
numbers under the corrected protocol.

Two candidates were rerun sequentially on the same A40 with new paired exact denominators:

- `token_debt_total10` minimizes measured retained-prefix work but controls only aggregate debt;
- `staggered_renewal_h12` costs more but provides a per-cache renewal horizon.

The frozen deployment candidate is H12. This choice uses no recommendation label. The measured
trajectory has 11 updates, one shorter than its 12-edge horizon, so the experiment does not observe
a complete renewal cycle. The candidate pair and the rule preferring H12's per-cache deadline over
the cheaper aggregate-debt endpoint were frozen before either corrected same-device run was
examined.

## 6. Design 1 system closure: Destination runtime and transaction

### 6.1 A common destination extent

The translator does not publish framework tensors. Its output is a destination-ready extent with
separate contiguous K and V, record identifiers, valid lengths, source and target versions,
actions, and lineage. Padding is excluded from the logical extent, and invalid dense positions are
zero when a dense view is required. The normalized-source and direct-old-K/V operators implement
the same extent ABI, allowing placement, validation, and publication to remain independent of the
chosen source representation.

The implementation provides an FP32-arithmetic transport reference, a packed FP16 framework path,
and fused FP16 kernels. The fused direct-old-K/V path consumes
\([K_v,V_v]\), applies Equation (9), masks by valid length, and writes final K/V without a global
concatenated-output or compaction temporary. This kernel is enabling implementation rather than a
separate algorithmic contribution.

CohortKV has two execution modes with different guarantees:

- The **extent-reclaiming mode** retires an old extent after its replacement is staged. It supplies
  the one-hop hot-HBM performance result and bounds normal-path occupancy, but it is not
  abort-safe.
- The **copy-on-write mode** retains the complete old manifest and old extents until one complete
  target manifest commits. It supplies failure-visibility evidence, but the present work makes no
  throughput claim for this mode.

The numerical gains of these modes are never combined.

### 6.2 Fixed preflight and exact fallback

Preflight resolves conditions before any target extent is produced. The runtime verifies artifact
and model-version identity, program identity and shape, finite prepared tensors, old-cache
presence, and physical capacity. It also executes a fixed, label-free semantic canary on four
selection-role records and compares K/V relative L2 with a frozen threshold of 0.2.

Failures have two classes:

1. An artifact/target-version mismatch or insufficient copy-on-write capacity is a fatal admission
   error. Exact recomputation cannot make the requested target or destination capacity safe, so no
   transaction begins.
2. A program identity or shape failure, missing required old K/V, or semantic-canary failure
   changes the affected migration cohort to exact before target execution. Scheduled and natural
   exact records are unchanged.

This is job-level pre-execution fallback. CohortKV does not currently sample runtime waves, detect
natural distribution drift online, invalidate already written extents, or resume a failed job.
The canary establishes that the implemented fallback can detect the evaluated constructed
perturbation; it is not a general drift detector.

### 6.3 Multi-GPU placement and capacity

Prepared programs are replicated across workers, whereas record extents and target state are
partitioned. Placement weights a record by its valid tokens and greedily assigns extents to the
least-loaded device. The action partition is frozen before placement, so device assignment cannot
change which records are exact.

Copy-on-write admission accounts for, on every device,

\[
B_{\mathrm{req}} =
B_{\mathrm{model+program}}
+ B_{\mathrm{old}}
+ B_{\mathrm{complete\ new}}
+ B_{\mathrm{transient}}
+ B_{\mathrm{margin}}. \tag{10}
\]

The complete-new term is essential: capacity for one wave does not imply capacity for an atomic
target version. In the faster reclaiming mode, old and new bytes may instead overlap only by the
in-flight extent wave, but that lower peak comes with the weaker failure boundary stated above.

The evaluated destinations are HBM and pinned host DRAM. A POSIX path is a correctness interface
only. Although the host filesystem and device are recorded, the experiment does not isolate
physical-device I/O from the warm page cache. We therefore make no SSD, GDS, remote object-store,
or cold-start performance claim.

### 6.4 Atomic visibility

Each job follows

```text
preflight -> begin -> stage private extents -> validate complete manifest -> commit
                         \-> on error: abort and reclaim private staging
```

The manifest is the only visibility point. Commit requires the declared target version, the exact
expected record set, no duplicates, complete metadata and lineage, finite tensors, and valid
extent checksums. A retained prefix is never publishable because it omits the common target-model
append. Only the complete post-append cache may enter the target manifest or serve as the next
edge's source.

During copy-on-write, reads continue through the old manifest. On abort, private target extents are
discarded and the old manifest remains authoritative. This provides atomic version visibility
within the process and allocator used by the experiment; it does not imply durable logging or
recovery from process, host, or GPU loss.

## 7. Implementation

### 7.1 Model and streaming pipeline

The complete vertical slice uses a modular simplified HSTU with 16 layers, hidden width 512,
concatenated K/V width 1,024, and maximum history length 2,048. Pointwise unnormalized attention,
per-layer valid-length handling, and first-class K/V output are preserved because they define the
state being migrated. Training and evaluation predict item \(t+1\) from hidden state \(t\), fit
the item vocabulary on the base period only, train on targets from the current stream date, and
use engaged items as evaluation positives.

The primary KuaiRand split contains four base windows followed by 12 streaming endpoints
\(\theta_0,\ldots,\theta_{11}\). The growing-history evaluator admits each canonical-date window
only after evaluating the preceding model edge. It crops histories at 2,048 tokens identically for
mixed and exact branches. The fixed full-cohort workload has 682 records and 1,087,785 prefix
tokens. Its non-adjacent source mixture contains 136/205/341 records from
\(\theta_0/\theta_4/\theta_{10}\), all targeting \(\theta_{11}\); the lifecycle uses adjacent
edges from \(\theta_0\) through \(\theta_{11}\).

### 7.2 Compiler and immutable programs

The frozen user-role manifest allocates 40 users to fitting, 60 to program selection, 60 to the
empirical semantic gate, and 522 to final testing. For the three non-adjacent deployment pairs,
the compiler samples up to 8,192 valid tokens per layer on fit users, searches its ridge and
structural choices on the selection role, and serializes FP16 weights and biases. The empirical
gate reloads the serialized capsule, program, and output rather than validating an unmaterialized
FP32 fit. An optional residual hidden suffix uses BF16 because real values overflow FP16; it is not
required by the primary direct-old-K/V path.

The growing-history experiment separately fits 11 adjacent direct programs on the 40-user fit
role and validates serialization, provenance, shapes, and loading. That artifact takes
`86.606 s` in aggregate but does not execute the 60-user, three-view empirical gate above.
Consequently, lifecycle task metrics validate the complete measured rollout, while the paper does
not claim that every adjacent program is independently admitted by the non-adjacent gate.

The three non-adjacent deployed artifacts record source and target checkpoint hashes, tensor
shapes and dtypes, role-manifest provenance, hyperparameters, validation summaries, and the
exact-terminated fallback plan. Their direct-old-K/V composition additionally records the source
projection, rank and condition checks, parent-program identity, and composed-program hash.
Adjacent lifecycle descriptors record their fit, checkpoint/program hashes, shapes, provenance,
and load checks, but not a three-view gate or structural fallback plan.

### 7.3 Operators and state stores

Reference, packed, and fused operators share one preallocated unpadded-extent API. The packed
baseline uses batched framework matrix multiplication followed by valid-token compaction. The
fused kernels combine projection, bias, masking, K/V separation, and final-layout write. Dense and
extent forms are cross-checked over every valid element, and dense padding must remain zero.

The growing-history evaluator uses a CPU FP16 recursive store and groupwise H2D/D2H staging to fit
the 11-edge experiment in the declared device boundary. It records retained-prefix CUDA work,
foreground append, and logical movement in separate ledgers. The formal transaction integration
instead uses a two-device HBM extent store with private copy-on-write staging and manifest
readback.

### 7.4 Reproducible result boundaries

The implementation emits protocol strings, hashes, record counts, timings, lineage rows, action
counts, capacity ledgers, and claim-boundary flags into JSON artifacts. A deterministic final
assembler verifies 18 frozen development inputs, the aggregate schema, candidate binding,
cross-field semantics, and artifact-to-claim mappings. It performs no GPU experiment and must not
be counted as a replication.

Experiments run on NVIDIA A40 GPUs. CUDA-event repetitions estimate timing variability within a
trained model; they are not independent samples. Training seed is the replication unit.

## 8. Evaluation

We ask five questions:

- **RQ1:** Is model-version-stale K/V meaningful across the evaluated data and capacity regimes?
- **RQ2:** Does compiled translation improve the resident cost/fidelity frontier?
- **RQ3:** Does the translated action remain useful at a complete-job source and destination
  boundary?
- **RQ4:** Does deadline-based migrate-or-exact maintenance survive recursive growing histories?
- **RQ5:** Does the implemented destination close fallback and failure visibility correctly?

### 8.1 Methodology

**Datasets.** KuaiRand-1K [6] supplies the primary calendar-time streaming trace. Tenrec QB and QK
[7] supply two positive ordered-exposure extensions; they are related tables from one collection,
not independent datasets, and their time is ordinal rather than a shared global calendar. The
random-exposure KuaiRand log is excluded from training. Only items in the base-period vocabulary
are eligible targets.

**Evidence tiers.** Motivation uses four training seeds. Mechanism replication uses three
non-discovery seeds in each of three data tables and three model capacities, for 27 independently
trained version chains. The complete compiler, operator, source-path, lifecycle, and transaction
experiments use the frozen 16-layer KuaiRand seed-0 configuration. We do not pool numbers across
these protocols.

**Metrics.** BestRank and MeanRank are lower-is-better; AUC, NDCG@100, and Hit@100 are
higher-is-better. Cache recovery is the fraction of the stale-to-fresh relative K/V-error gap
closed. Score and Top-100 recovery apply the same construction to catalog scores and recommendation
sets. Exact has recovery one by construction for all three semantic views—cache, score, and
Top-100—but is not a task-quality upper bound. Task ratios above one are therefore accompanied by
the paired absolute difference.

**Cost.** Each cost ratio uses measured CUDA work against exact target-model recomputation on the
same records, batch, device, and retained prefix. Resident, source-to-manifest, lifecycle, and
copy-on-write timings are separate result families. CPU work, transfer, append, scoring, and
publication are included only when the named boundary says so.

### 8.2 RQ1: Opportunity exists, but it is not universal

Table 1 establishes positive maintenance value across the aligned KuaiRand, QB, and QK tasks.
The 3×3 data/model-capacity screen provides a more important qualification. Full current-model
compute and stale reuse are positive in all four seeds of all nine cells, but the within-seed
BestRank staleness tax ranges from `-6.0%` for QB-medium to `54.8%` for QB-large. KuaiRand-large
is `36.0%`; QK-large is `-0.5%`. Thus more data or a larger model can make exact maintenance more
expensive without creating a larger task-quality opportunity.

A separate seed-0 long-context Tenrec stress screen reaches the same negative boundary. Exact
prefix replay costs `4.504 ms/batch` on QB and `3.578 ms/batch` on QK, but the oldest-cache
BestRank tax is only `1.1%–5.6%` and `3.6%–6.0%`; NDCG maintenance is non-positive in five of
six cells. We therefore evaluate semantic K/V repair independently from whether a particular
task slice rewards freshness.

**Answer to RQ1.** Stale K/V removes a replicated and sometimes large fraction of streaming-model
value, but neither model capacity nor recomputation cost guarantees such an opportunity. CohortKV
is a state-maintenance mechanism, not a policy for deciding whether a product should value a model
update.

### 8.3 RQ2: Compiled translation improves the measured resident frontier

**Cross-table and capacity replication.** Table 3 summarizes the earlier 50%-fidelity
cohort-tiered protocol. Its 27 chains are independent at the trained-model level. The interval
describes these prespecified observations; it is not a superpopulation interval over datasets,
because the nine data/capacity cells are heterogeneous and QB/QK are related.

**Table 3: Compiled mechanism over 27 trained version chains.** Intervals summarize the 27
chain-level values within this prespecified matrix.

| Quantity | Result |
|---|---:|
| Resident GPU cost / exact | 0.1211 [0.1118, 0.1304] |
| Stale-to-fresh K/V-gap recovery | 0.5867 [0.5466, 0.6267] |
| Chains meeting the frozen 50% target | 25 / 27 |
| Positive BestRank / rank-utility / NDCG@100 signs | 20 / 27, 24 / 27, 20 / 27 |
| 3×3 cells with positive mean BestRank and rank utility | 6 / 9 |

The semantic mechanism transfers across all three tables and capacities, but task endpoints are
not uniformly positive. This is precisely why task quality is evaluated after maintenance rather
than used as an admission oracle. The 50% protocol in Table 3 is not replication of the final
70% seed-0 program or the H12 lifecycle.

**Structural and recent-token controls.** An earlier unified-library screen rejects several
apparently natural partial-replay choices. All 54 matched recent-token partial actions are slower
than their complete-span counterparts because splitting, concatenation, and small-kernel overhead
dominate. Arbitrary contiguous intervals add material value in only one discovery cell, fixed deep
suffix replay does not transfer to scaled QB/QK, and plain prefix is never selected. These
negative results motivate the compact library used here rather than an \(O(L^2)\) interval
planner.

**Closest selective-recomputation baseline.** On the frozen selection role, we evaluate every
contiguous interval in a DroidSpeak-style selective implementation: 53 intervals for each of
\(\theta_0,\theta_4,\theta_{10}\rightarrow\theta_{11}\), or 177 complete frontier points after
adding CohortKV and anchors.

**Table 4: Resident FP32 algorithmic frontier on the seed-0 selection role.**

| Source | Compiled cost / exact | Compiled worst-view recovery | Best selective cost / exact | Best selective worst-view recovery |
|---|---:|---:|---:|---:|
| \(\theta_0\) | 0.0656 | 0.8787 | 0.6973 | 0.4530 |
| \(\theta_4\) | 0.0663 | 0.8755 | 0.6976 | 0.4850 |
| \(\theta_{10}\) | 0.0664 | 0.9258 | 0.6976 | 0.4495 |

Compiled translation strictly dominates every evaluated selective interval for all three source
ages. The strongest selective action recomputes layers 0–11 of 16, yet no selective point passes
the 70% three-view gate. This is a controlled single-seed result, not a replicated claim about all
fine-tuned-model K/V methods.

**Deployed serialized program.** Table 5 evaluates reloaded FP16 programs on the disjoint
empirical-validation role. The runtime cost here is resident program execution; source reads and
destination publication enter RQ3 instead.

**Table 5: Deployed FP16 empirical semantic gate.**

| Source | Cost / exact | Cache recovery | Score recovery | Top-100 recovery | Worst recovery lower bound | Worst coverage lower bound |
|---|---:|---:|---:|---:|---:|---:|
| \(\theta_0\) | 0.01657 | 0.8810 | 0.9845 | 0.9479 | 0.8514 | 0.9224 |
| \(\theta_4\) | 0.01652 | 0.8897 | 0.9201 | 0.9046 | 0.8391 | 0.9005 |
| \(\theta_{10}\) | 0.01651 | 0.9365 | 0.9717 | 0.9470 | 0.9231 | 0.9459 |

All three programs pass. Recovery targets 50%–80% select compiled translation, while 90% selects
exact. The result validates held-out state, scores, and recommendation-set overlap; it does not
assert that an arbitrary recursive input remains in distribution.

The `0.0656–0.0664×` values in Table 4 and `0.01651–0.01657×` values in Table 5 are not
interchangeable. Table 4 is a resident FP32 algorithmic frontier on the selection role; Table 5
uses a different role, serialized FP16 program/output, deployed operator, and freshly measured
exact denominator. Their semantic conclusions align, but their cost ratios remain within their
own protocols.

**Setup economics.** Historical fitting takes `31.243 s`, FP16 runtime preparation `4.316 s`,
and empirical validation `273.343 s`, for `308.901 s` over the three pairs. Composing the three
direct-old-K/V programs later takes another `1.546 s`. At 682 records, this setup dominates the
one-GPU `18.695-s` exact job in RQ3. The resident-only per-pair amortization floors are
2,865–2,936 records, but they exclude source and destination work and are not end-to-end
break-even points.

**Baseline completeness.** The current suite includes reuse, exact, cheap target projection,
residual-\(p\), and selective recomputation. It does not yet include two especially important
same-capacity learned controls: direct ridge from old K/V to target K/V, and unconstrained ridge
from old normalized state to target K/V. Until those controls are run with the same roles,
serialization, semantic gate, and operator boundary, Table 4 supports the measured frontier but
not a claim that projection-centered residual fitting is uniquely responsible for it.

**Answer to RQ2.** Compiled-program execution is cheap and semantically effective at the resident
boundary, replicates as a mechanism over 27 chains, and decisively beats the implemented selective
baseline in the controlled large configuration. The final formulation still needs the two direct
learned ablations for a complete algorithmic novelty claim.

### 8.4 RQ3: Source representation decides the complete-job outcome

**Resident operator.** On a representative four-record, length-2,047 batch, the three paths write
the same preallocated unpadded extent.

**Table 6: Resident transport operator.**

| Path | Median latency | Peak temporary beyond target |
|---|---:|---:|
| FP32-arithmetic transport reference | 14.610 ms | 1,073.8 MB |
| Packed FP16 | 5.378 ms | 402.6 MB |
| Fused FP16 | 2.729 ms | 0 B |

Across the complete 60-record selection distribution, fused execution has
`30.142/31.070/31.154 ms` samples versus a `61.970-ms` packed median, a `1.995×` advantage.
All nine layouts pass over 1.443 billion valid elements. This chooses an implementation; it is not
a source-to-manifest result.

**Normalized capsule failure.** The first complete implementation reads the `17.8235-GB` physical
FP16 normalized capsule and writes all 682 target records. Table 7 contains the six matched
compiled/exact endpoints from a 30-point matrix covering five methods.

**Table 7: Complete normalized-source job, seconds.**

| Destination | GPUs | Compiled | Exact | Compiled source-processing share |
|---|---:|---:|---:|---:|
| HBM | 1 | 27.083 | 18.881 | 91.35% |
| HBM | 2 | 18.943 | 9.644 | 92.95% |
| HBM | 4 | 13.707 | 5.742 | 96.91% |
| Pinned DRAM | 1 | 22.567 | 18.886 | 96.66% |
| Pinned DRAM | 2 | 12.231 | 9.391 | 94.83% |
| Pinned DRAM | 4 | 15.662 | 5.448 | 95.19% |

Compiled translation loses to exact at all six endpoints, although it remains 2.70–3.49× faster
than the empirical-gate-failed selective diagnostic. In HBM, compiled arithmetic itself takes only
`0.954/0.244/0.118 s`; source read, decode, and preparation erase that advantage. Calling this a
kernel speedup would therefore be misleading.

**Direct old-K/V path.** Equation (9) removes the additional capsule from execution. In a complete
real-value transport over 17,822,269,440 valid elements, all elements satisfy
`atol=rtol=0.02`; the maximum absolute difference from the deployed capsule program is `0.01172`.
Timing is a separate shape-, dtype-, layout-, and occupancy-equivalent experiment.

**Table 8: Hot-HBM prepublished-program, extent-reclaiming normal path.** Peak K/V is aggregated
across all participating devices.

| GPUs | Direct compiled | Paired exact | Speedup | Aggregate peak old + new K/V | Abort-safe? |
|---:|---:|---:|---:|---:|---|
| 1 | 0.9299 s | 18.6949 s | 20.11× | 35.91 GB | No |
| 2 | 0.4936 s | 9.7291 s | 19.71× | 36.18 GB | No |
| 4 | 0.2546 s | 4.7655 s | 18.72× | 37.79 GB | No |

Every compiled repetition is below every paired exact repetition. Both timing sources begin in
HBM: compiled receives tensors with the shape, dtype, layout, and occupancy of the existing
source-version serving K/V, whereas exact receives raw history. Real source values are exercised
in the separate tolerance run above. These are not equal resource footprints. Existing old K/V
occupies `35.645 GB`; raw history occupies about `89.1 MB`. The direct source adds no per-record
state. The three-program set occupies `100.78 MB` per worker and is replicated when more workers
participate.

**Answer to RQ3.** The same compiled affine changes from a 0/6 loss to an 18.72–20.11× one-hop
normal-path speedup when it consumes the state that a hot-cache deployment already owns. This is a
prepublished-program hot-HBM data-plane result. It excludes the `308.901-s` setup, lifecycle
movement, copy-on-write, cold storage, and serving interference.

### 8.5 RQ4: Scheduled renewal preserves task quality over the measured chain

The corrected evaluator runs H12 and token debt sequentially on the same A40 with one warmup and
three timed repeats per edge. Both recursively consume their previous actual post-append cache and
receive a fresh paired exact denominator on all 11 edges.

The 11 adjacent programs come from a separate 40-user fit and serialized
provenance/shape/load-validation artifact with `86.606 s` aggregate wall time. They do not receive
the 60-user three-view gate in Table 5. That setup is excluded from \(U/E\), just as the
non-adjacent `308.901-s` setup is excluded from the one-hop data plane; the two setup values are
not interchangeable.

**Table 9: Corrected growing-history lifecycle.**

| Candidate | \(\sum U/\sum E\) | Scheduled exact / reusable record-edges | Weighted AUC ratio | NDCG@100 ratio | Hit@100 ratio | Per-cache deadline |
|---|---:|---:|---:|---:|---:|---|
| Token debt, total 10 | 0.071319 | 221 / 6,711 | 1.000030 | 0.996890 | 0.999060 | No |
| Staggered renewal, H12 | 0.100017 | 462 / 6,711 | 1.000039 | 0.997463 | 1.000000 | Yes |

For H12, per-edge \(U/E\) stays between `0.0963` and `0.1053`, and scheduled exact refresh stays
between 35 and 50 records. The aggregate task ratios correspond to mixed/exact absolute values of
`0.7657145/0.7656843` AUC, `0.0154512/0.0154905` NDCG@100, and
`0.2435897/0.2435897` Hit@100. The AUC ratio above one is a paired difference of only
`+0.0000302`, not superiority over exact. The worst per-edge H12 ratios are `0.999765` AUC,
`0.943696` NDCG@100, and `0.957746` Hit@100, so the aggregate does not imply identical behavior
on every edge.

H12 is selected because it guarantees a scheduled renewal deadline for every continuously
resident cache. Token debt is cheaper but may leave one cache approximate indefinitely. The
observed maximum H12 migration depth is 11 because the trace ends before the first 12-edge
deadline completes.

The main ratio covers only retained-prefix GPU work. H12 separately records `662.870 GB` of
cumulative logical H2D+D2H movement and `116.602 s` of movement GPU-event time. Mixed and exact
foreground append ledgers are `190.181 s` and `190.129 s`; append is deliberately outside both
sides of \(U/E\). The evaluator is groupwise host-staged rather than a full-cohort HBM-resident
lifecycle. A separate q90 cache-fidelity diagnostic—defined as
\(\max(0,1-\text{q90 exact-relative K/V error})\)—is `0.0792`, so this result supports the
reported recommendation outcomes under the measured rollout, not semantic equivalence of every
recursively migrated cache.

**Answer to RQ4.** Over 11 canonical-date updates, deterministic H12 renewal reduces
retained-prefix GPU work to 10.00% of exact while keeping aggregate weighted NDCG@100 at 99.75% of
exact. It does not yet establish a full H12 renewal cycle, an end-to-end movement speedup, or
replication beyond the seed-0 chain.

### 8.6 RQ5: The destination closes the evaluated failure cases

The formal transaction test binds the H12 partition for the first edge:
553 migrations, 50 scheduled exact refreshes, and 79 natural exact records. The complete target
contains 866,210 tokens and 28.384 GB of K/V, partitioned across two A40s.

This first-edge target is a different workload from Table 8's non-adjacent
\(\theta_0/\theta_4/\theta_{10}\rightarrow\theta_{11}\) cohort, which explains why its
28.384-GB target is smaller than the 35.645-GB old-K/V footprint reported there.

**Table 10: Two-GPU copy-on-write correctness integration.** Elapsed wall times validate case
completion and are not throughput comparisons.

| Case | Outcome | Validation wall time |
|---|---|---:|
| Normal 682-record job | Complete \(\theta_1\) commit; 682/682 target readback | 90.93 s |
| Shape-preserving program perturbation | Canary fails; all affected migrations become exact; complete corrected commit | 105.27 s |
| Mid-job fault | Abort after 1/280 extents; no target visible; 682/682 old readback | 30.16 s |
| Pre-commit fault | Abort after 280/280 private extents; no target visible; 682/682 old readback | 85.64 s |

The nominal canary relative L2 is `0.06678`, below the 0.2 threshold. The perturbed real program
reaches `1228.08`; the fallback job makes zero migration-operator calls. On the two devices,
copy-on-write requires `32.933/32.905 GB` against `47.700 GB` capacity, with
`45.803 GB` free before the job. Both abort paths reclaim private staging and preserve version,
shape, finite-value, and checksum-valid old state.

**Answer to RQ5.** The implementation closes complete-manifest visibility for the normal path, one
pre-execution semantic fallback, and two representative injected exceptions. It does not measure
the performance of this copy-on-write mode or establish protection against process/GPU failure,
durable restart, runtime drift, or online rework.

## 9. Discussion and limitations

### 9.1 What the three results jointly establish

CohortKV's contribution is not an isolated affine fit, pseudoinverse, renewal heuristic, fused
kernel, or copy-on-write store. It is the decomposition of model-version-stale recommender state
into three explicit responsibilities: compile a shared source-to-target approximation, bound how
long approximate state may persist, and publish only a complete target version. Each responsibility
has executable evidence. The measurements nevertheless remain separate: no run simultaneously
establishes the 20× one-hop data plane, the 0.100× repeated-update ratio, and abort-safe
publication.

This separation changes how the result should be used. A deployment may choose the reclaiming path
when failed maintenance can be retried from another authoritative source, or copy-on-write when
the old cache must remain continuously readable. The current paper establishes speed for the first
and visibility semantics for the second; it does not show that copy-on-write retains the first
path's speed.

### 9.2 Total economics and standing resources

Direct translation succeeds by consuming an existing hot cache, not by reducing its standing
size. The measured old K/V occupies `35.645 GB`, while exact's raw-history source is about
`89.1 MB`. CohortKV removes the additional `17.82-GB` normalized capsule. The three non-adjacent
programs in the one-hop study occupy `100.78 MB` per worker; this is not the full lifecycle
library. The 11 adjacent serialized program files total `369.52 MB`, although a sequential
edge-by-edge executor need retain only the active edge's program. Persistent K/V is still more
expensive than storing histories. A system should therefore compare avoided future prefill with
HBM opportunity cost, eviction rate, and user hotness. The available interaction datasets do not
provide the request trace needed for that calculation.

Compiler economics are also incomplete. Fit, preparation, and empirical validation take
`308.901 s` for three version pairs, versus `18.695 s` for one 682-record one-GPU exact job.
Programs may amortize over a larger cohort, multiple cache replicas, or multiple executions, but
the current artifacts do not measure a complete time break-even. We report setup and data-plane
cost separately rather than calling compilation free or offline.

### 9.3 Approximation, renewal, and validation

Equation (9) is equivalent to the deployed normalized-source affine only for exact source-version
old K/V and a verified full-row-rank source projection. It does not make an approximate cache
exact, nor does it prove that programs compose across model versions. H12 supplies periodic exact
renewal because recursive translation is empirically useful but not algebraically guaranteed.

The measured trace contains 11 edges and therefore stops before completing H12's first 12-edge
cycle. Aggregate task metrics remain near exact, but some per-edge task ratios are lower and the
q90 cache-fidelity diagnostic
\(\max(0,1-\text{q90 exact-relative K/V error})\) is only `0.0792`. Longer traces must verify the
renewal boundary rather than infer it. Direct composition of adjacent affine programs,
warm-started fitting, and reducing the potentially quadratic source/target program set remain open
design questions.

The empirical semantic gate is also narrower than a safety proof. It uses disjoint held-out users
and the serialized deployed representation, which protects against several implementation errors,
but it guarantees neither unseen distributions nor recursive inputs. The transaction canary
catches one intentionally large, shape-preserving perturbation. Natural drift detection,
false-fallback rates, and mid-execution invalidation are unevaluated.

### 9.4 Evaluation and generality

The final compiler, direct source path, lifecycle, and transaction use one KuaiRand training seed
and a simplified 16-layer, width-512 HSTU. The 27-chain experiment supports the earlier compiled
mechanism across three tables and capacities; it does not replicate the final 70% gate,
direct-old-K/V execution, H12, or copy-on-write integration. Training seeds, not 682 users or
timing repetitions, are the independent units needed for the next replication phase.

The 682-record workload is a controlled cohort rather than an organic cache trace. Its
non-adjacent source mixture and adjacent lifecycle are constructed from available histories.
KuaiRand windows obey the prepared canonical-date admission order used by training and evaluation,
but the source log is not strictly ordered at every raw timestamp boundary: 3,521 of 11,797,055
history tokens cross the next partition's raw-time minimum, with a maximum lead of 6,128.6
seconds. We therefore claim canonical-date streaming semantics, not strict event-time causality.

QB and QK broaden ordered-exposure evidence but are related Tenrec tables. Taobao UserBehavior is
rejected for the target task because it lacks true unclicked impressions, and ZhihuRec exposes a
negative maintenance boundary rather than a compatible replication. Generalization to another
architecture requires an exposed projection path, source-rank checks, and new empirical
validation.

The baseline comparison is not yet complete. The strongest implemented external control is
selective layer recomputation, but direct old-K/V ridge and unconstrained normalized-state ridge
are needed to isolate the value of projection-centered residual fitting. Classical mean-shift or
orthogonal alignment, low-rank deltas, and a small nonlinear map would further locate the
cost/fidelity frontier if measured with identical roles and runtime boundaries.

### 9.5 Deployment boundary

We do not evaluate online request arrivals, latency SLOs, foreground interference, training and
maintenance co-location, admission under changing hotness, or state eviction. The recursive
evaluator reports host-device movement but does not optimize it; the transaction integration
reports copy-on-write correctness but no throughput. POSIX and remote adapters are interfaces
rather than physical-storage results.
Physical SSD, GPUDirect Storage, cold restart, quantized full-cohort execution, process/GPU crash,
journaling, and resume are outside the current claim.

These limitations are not hidden implementation details. They define the next evidence needed to
turn the single-configuration vertical slice into a submission claim: complete the missing learned
baselines, replicate the final design across training seeds and predeclared dataset/capacity
cells, observe at least one full H12 cycle, and measure one integrated lifecycle path with state
movement and the selected publication mode.

## 10. Related work

### 10.1 Streaming recommendation and model publication

HSTU motivates generative recommendation over high-cardinality, non-stationary streams and shows
the value of long sequential histories [1]. CohortKV studies a systems consequence: the histories'
derived K/V can outlive the model version that created them. Ekko reduces recommender model-update
latency by disseminating parameters and managing replicas [2]. CohortKV begins after checkpoint
publication; training, checkpoint validation, parameter dissemination, and model rollback are
outside its job boundary.

### 10.2 Same-model K/V memory and restoration

vLLM derives allocation and serving mechanisms from the dynamic memory behavior of same-model
autoregressive K/V [8]. CachedAttention retains same-conversation state across requests and
overlaps hierarchical loading and saving [9]. HCache restores evicted same-model state from
intermediate activations, balancing recomputation and I/O [3]. MTServe persists
generative-recommendation K/V and manages placement, asynchronous movement, and replacement [5].
These systems establish that K/V structure should inform kernels and storage. CohortKV addresses a
different validity event: the state remains intact, but a new model version changes its semantic
target.

### 10.3 Cross-model K/V reuse

DroidSpeak shares K/V across same-architecture fine-tuned LLM variants by selectively recomputing
layers [4]. Cross-model K/V reuse is therefore not itself a CohortKV contribution. Our adapted
selective baseline tests its central per-layer alternative under HSTU semantics, but is not a
reproduction of DroidSpeak's distributed serving system. CohortKV instead amortizes a shared
affine over a model-version cohort, reparameterizes it over resident old K/V, controls recursive
use through exact renewal, and publishes a complete target-version manifest.

### 10.4 Scheduling and execution boundaries

Orca derives iteration-level scheduling and selective batching from autoregressive execution
semantics [10]. DistServe maps prefill/decode interference to disaggregated placement [11].
CohortKV similarly makes one semantic unit—the source/target version cohort—visible to the
compiler, batcher, lineage, fallback, and destination. The cohort is neither an online request
batch nor a prediction that reuse is safe.

**Table 11: Closest system boundaries.**

| System | Model relation | Retained source | Primary operation | Published unit |
|---|---|---|---|---|
| HCache [3] | Same model | Intermediate activation | Restore after eviction | Request/chunk K/V |
| DroidSpeak [4] | Fine-tuned LLM variants | Another variant's K/V | Selective layer recomputation and reuse | Request prefill |
| MTServe [5] | No source-to-target transform | Persistent per-user K/V | Place, load, and evict | Serving page/chunk |
| CohortKV | Successive streaming HSTU versions | Existing old K/V; normalized state only for fit/reference | Compiled translation plus scheduled exact renewal | Complete target-version cohort |

## 11. Conclusion

Persistent recommender K/V is model-versioned derived state. CohortKV compiles a shared affine for
each source/target model pair, reparameterizes it over the old K/V already present in a hot cache,
and schedules deterministic exact renewal to limit recursive use. A destination runtime gives
migration, scheduled exact, natural exact, and fallback one complete publication boundary.

In the earlier 50%-fidelity protocol across 27 trained version chains, the compiled mechanism costs
`0.121×` resident exact and closes `0.587` of the stale-to-fresh K/V gap. In the controlled
full-cohort configuration, the normalized source loses to exact at all six endpoints, while
occupancy-equivalent direct old-K/V timing is `18.72–20.11×` faster than paired hot-HBM exact on
one, two, and four GPUs; real-value numerical transport is validated separately. Over 11
corrected growing-history edges, H12 uses `0.100017×` paired retained-prefix exact GPU work and
preserves `99.746%` of weighted NDCG@100. A separate two-GPU copy-on-write integration commits
complete normal and exact-fallback targets and exposes no partial target under mid-job and
pre-commit faults.

These results establish three separately scoped executable properties in one controlled
configuration: translation, deadline-based recursive maintenance, and failure-visible
publication. They do not establish cold-storage speed, total compiler amortization, an end-to-end
lifecycle movement advantage, online serving performance, or cross-seed replication of the final
system.

## References

[1] Jiaqi Zhai et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers
for Generative Recommendations." ICML 2024. <https://arxiv.org/abs/2402.17152>

[2] Chijun Sima et al. "Ekko: A Large-Scale Deep Learning Recommender System with Low-Latency
Model Update." OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/sima>

[3] Shiwei Gao, Youmin Chen, and Jiwu Shu. "Fast State Restoration in LLM Serving with HCache."
EuroSys 2025. <https://doi.org/10.1145/3689031.3696072>

[4] Yuhan Liu et al. "DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants." NSDI 2026.
<https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan>

[5] Xin Wang et al. "MTServe: Efficient Serving for Generative Recommendation Models with
Hierarchical Caches." arXiv:2604.22881, 2026. <https://arxiv.org/abs/2604.22881>

[6] Chongming Gao et al. "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly
Exposed Videos." CIKM 2022. <https://arxiv.org/abs/2208.08696>

[7] Guanghu Yuan et al. "Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender
Systems." arXiv:2210.10629, 2023. <https://arxiv.org/abs/2210.10629>

[8] Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with
PagedAttention." SOSP 2023. <https://doi.org/10.1145/3600006.3613165>

[9] Bin Gao et al. "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with
CachedAttention." USENIX ATC 2024.
<https://www.usenix.org/conference/atc24/presentation/gao-bin-cost>

[10] Gyeong-In Yu et al. "ORCA: A Distributed Serving System for Transformer-Based Generative
Models." OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/yu>

[11] Yinmin Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
Large Language Model Serving." OSDI 2024.
<https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>

## Appendix A. Artifact-to-claim map

Paths are repository-relative. A path supports only the boundary named in the final column.

| Paper result | Primary artifact | Protocol boundary |
|---|---|---|
| KuaiRand Top-50k maintenance opportunity | `results/scaling/kuairand_data_utilization_summary.json` | Four-seed all-chunks motivation |
| Aligned KuaiRand/QB/QK opportunity | `results/exposure/cache_version_matrix_cross_dataset_summary.json` | Four-seed table-specific tasks; QB/QK related |
| 3×3 data/model-capacity screen | `results/motivation_scale/capacity_v2_summary.json` | Motivation only; does not tune migration |
| 27-chain compiled mechanism | `results/motivation_scale/cohort_tiered_migration_v1_summary.json` | Earlier 50%-fidelity operator protocol |
| Selective frontier | `configs/cohortkv_single_config_v1/stage1_frontier_summary.json` | Seed-0 resident FP32 selection role |
| Serialized compiler and empirical gate | `configs/cohortkv_single_config_v1/stage2_compiler_summary.json` | Seed-0 resident FP16; 40/60/60/522 roles |
| Common-extent operator | `configs/cohortkv_single_config_v1/stage3_operator_summary.json` | Resident operator only |
| Normalized-source complete matrix | `configs/cohortkv_single_config_v1/stage4_system_summary.json` | Warm source path; HBM/DRAM are destinations |
| Direct old-K/V source plan | `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json` | One-hop prepublished-program hot-HBM normal path |
| Corrected lifecycle | `results/system/cohortkv_single_config_full_chain_v1/stage4_9_same_device_confirmation_seed0.json` | One-A40 host-staged 11-edge retained-prefix work |
| H12 detailed result | `results/system/cohortkv_single_config_full_chain_v1/stage4_9_staggered_renewal_h12_seed0.json` | Selected candidate; movement reported separately |
| Copy-on-write closure | `results/system/cohortkv_single_config_full_chain_v1/stage5_full_cow_theta0_theta1_seed0.json` | Two-A40 one-edge correctness, not throughput |
| Source-state and setup ledger | `results/system/cohortkv_single_config_full_chain_v1/stage5_source_state_accounting_seed0.json` | Derived accounting; no new representation run |
| Frozen aggregate | `results/system/cohortkv_single_config_full_chain_v1/final_summary_seed0.json` | CPU-only Stage-6 assembly; no new GPU evidence |

Raw per-seed artifacts and checkpoints remain local and are not pooled across protocol families.
