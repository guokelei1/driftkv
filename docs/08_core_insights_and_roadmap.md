# Core insights and roadmap

> Status: authoritative as of 2026-07-23. This file replaces all earlier problem statements and
> phase plans.

## 1. Current thesis

Streaming recommendation training creates a sequence of model versions

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_t.
$$

For a fixed user-history prefix $x$, a cache produced under version $v$ is

$$
C_v(x)=F(\theta_v,x).
$$

After the model advances to $\theta_t$, consuming $C_v$ with the current model is cheaper than
recomputing the history but no longer version-consistent. The research question is:

> Can the internal structure of HSTU be used to migrate a version-stale prefix K/V cache toward
> the current model at materially lower cost than a complete history forward, while recovering a
> controllable fraction of fresh recommendation quality?

The active abstraction is **structure-aware cache migration**. It is not a per-user prediction
problem and it is not ordinary tail-token cache append.

## 2. What the corrected evidence supports

### 2.1 Cache staleness has a useful time scale

Under `validity_v1_incremental_prefix_cache`, fresh recomputation improves Best Rank over stale
reuse by `4.15` on average for one-step cache staleness and by `63.39` for a theta-0 cache carried
across multiple versions. The cumulative gain has a seed-level 95% t interval of
`[41.57, 85.20]`; cumulative parameter distance and Best Rank gain have mean Spearman correlation
`0.975` across five cache ages.

The defensible conclusion is limited but useful: one small update often does not justify immediate
global recomputation, while indefinite reuse accumulates visible loss. Cache age therefore creates
an operating region between the two endpoints.

### 2.2 Raw user-level drift is a discarded decision signal

Across 20 seed-window cells, the correlation between an individual sample's relative K/V norm
change and its realized rank-utility gain is `0.020`, with 95% interval `[-0.012, 0.052]`.
Moreover, a JVP-based estimate is not cheaper than the forward it was intended to avoid.

This is retained only as a negative result. The project must not return to “estimate drift for
every user, then choose reuse/migrate/recompute” unless new evidence changes both the quality-signal
and cost arguments.

### 2.3 HSTU exposes a migration decomposition

At layer $l$,

$$
K_l=W^K_l\operatorname{Norm}_l(x_l), \qquad
V_l=W^V_l\operatorname{Norm}_l(x_l).
$$

Version change affects both the direct projection parameters and the hidden state propagated from
earlier layers. The current operator separates them:

- **cheap refresh** reuses cached old $\operatorname{Norm}(x_l)$ and applies current $W^K_l/W^V_l$;
- **full propagation** starts from a cached split hidden and executes current blocks through a
  continuous region;
- executing all layers from current embeddings is exactly full current-model K/V recomputation.

This decomposition produces a real computation-quality curve rather than only a cache-error
estimate.

### 2.4 The first six-layer curve passes the continuation gate

On the current six-layer model and four training seeds:

| Configuration | Measured GPU time / full | Rank recovery at theta-3 | Rank recovery at theta-5 |
|---|---:|---:|---:|
| cheap all | 0.187 | 50.8% | 72.9% |
| cheap + suffix-2 | 0.372 | 65.3% | 84.6% |
| cheap + suffix-4 | 0.635 | 87.5% | 101.3% |
| cheap + suffix-5 | 0.767 | 88.8% | 103.0% |
| full recompute | 1.000 | 100% | 100% |

In this small-gap table, paired differences around 100% contain zero and do not support superiority
over fresh. Suffix-5 is not proven equivalent to full; four seeds only fail to distinguish their
current Best Rank and NDCG@100.

Two structural observations matter more than the exact numbers:

1. Relative stale K/V error rises with depth in the current model, so a deep propagation region is
   a reasonable first heuristic.
2. The final block output is not consumed by another prefix layer. Removing its
   attention/gate/residual gives exactly equal K/V and reduces suffix-1 to `0.229x` versus cheap at
   `0.187x`; the quality gain remains negligible.

### 2.5 The first scale pass is positive on KuaiRand and unresolved across datasets

The optimized suffix was frozen and tested without another layer search. At batch 32, increasing
synthetic resident sequence length from 128 to 512 raises optimized full latency from `2.08 ms` to
`14.41 ms`; cheap/full falls from `0.189` to `0.058` and suffix-2/full from `0.377` to `0.248`.
The lightweight paths therefore gain a larger relative advantage once full-prefix attention leaves
the under-utilized regime. Suffix-5 stays near `0.8x`, so the high-quality endpoint remains costly.

The quality-cost curve also survives depth 3, 6, and 9 over four seeds at fixed hidden/KV width.
An approximately two-thirds suffix recovers `94.6%`, `101.3%`, and `95.3%` of the cross-seed mean
Best Rank gain at costs `0.554`, `0.636`, and `0.668` respectively. In these depth cells, paired
comparisons do not distinguish values around 100% from full. This supports the structural
decomposition across the first depth range; it does not prove suffix optimality.

Along controlled interpolation from theta-0 to theta-5, stale K/V error grows monotonically from
`0.206` to `0.656`. Recommendation utility does not grow monotonically: full maintenance Best Rank
gain peaks at alpha `0.75` before falling at the trained endpoint. Update norm and cache error are
therefore severity indicators rather than calibrated utility predictors.

The MovieLens-1M chronological-holdout transfer check is a boundary result. After two updates,
full maintenance improves Best Rank by only `1.48` with seed interval `[-6.37, 9.33]`; NDCG and the
provided 20-candidate metrics are also unresolved. The operator itself remains exact, but the
KuaiRand problem strength has not generalized under this short version chain. Exact scale results
and protocol limitations are in `experiments/scaling/SCALING_V1.md`.

### 2.6 More complete KuaiRand use strengthens the problem and preserves the operator

The earlier experiments did not use all local KuaiRand data. The top-5k vocabulary retained only
3.67% of 11.71M standard-log rows, and the base iterator used only each user's latest truncated
sequence. A fixed 2x2 stress test first increased the catalog/context bundle to top-20k/length-256
and the model to 12 layers/hidden-192. The maintenance gap remained positive on Best Rank and
NDCG@100 in all four cells. In the combined cell, optimized full latency was 9.1x the original
baseline while cheap/full fell to 0.099. Exact results are in
`experiments/scaling/KUAIRAND_FACTORIAL_V1.md`.

A second protocol then used top-50k, length 512, and overlapping chronological base chunks. This
raises eligible base targets per epoch from 230,945 to 620,958 without changing the 130,239 stream
targets. Across four seeds, theta-5 full maintenance Best Rank grows from 82.68
`[40.23, 125.13]` under latest-only training to 885.56 `[460.24, 1310.88]`; NDCG@100 grows from an
unresolved 0.00109 to 0.00250 `[0.00169, 0.00330]`. Full compute over frozen is 3837.67 Best Rank,
so the stronger gap is not caused by streaming training becoming useless. Cumulative parameter
distance is slightly smaller rather than larger.

At this stronger operating point, cheap costs 0.058x full and recovers 54.6% of the mean Best Rank
gap; suffix-2, suffix-4, and suffix-5 cost 0.248x, 0.613x, and 0.796x and recover 58.9%, 76.2%, and
84.1%. The curve is more conservative than the small-gap result but remains useful. Details are in
`experiments/scaling/KUAIRAND_DATA_UTILIZATION_V1.md`.

A single-seed bridge then combined top-50k chunked training with 12 layers and hidden size 192.
Full maintenance gains 659.04 Best Rank; cheap, proportional one-third, and two-thirds suffixes
cost 0.054x, 0.312x, and 0.653x full and recover 61.1%, 63.4%, and 82.9%. This descriptive gate
shows no interaction failure, but it is not a substitute for cross-seed inference.

Full recomputation must now be treated as the version-consistency and cache-fidelity reference,
not an unconditional upper bound on realized ranking quality. In the top-20k/12-layer cell,
one-third suffix beats full Best Rank by 25.84 with paired seed interval `[7.69, 44.00]`, while its
paired NDCG difference contains zero. Future tables must report paired differences and multiple
quality views rather than dismissing every recovery value above 100% as noise.

## 3. Current contribution hypothesis

A complete paper could make four contributions if the remaining gates pass:

1. Define model-version invalidation of generative-recommendation prefix K/V as distinct from
   behavior append and fixed-model cache management.
2. Characterize the time and layer structure of the resulting quality loss under leak-free
   streaming evaluation.
3. Introduce a migration operator that decomposes projection refresh from hidden-state
   propagation and allocates computation across layer regions.
4. Build a version-aware executor that applies an age/budget-appropriate suffix configuration by
   cache-version cohort and demonstrates end-to-end savings.

The reusable idea is the decomposition and budgeted allocation of propagation work. A dynamic
arbitrary-layer planner is optional rather than assumed: it must first show reproducible benefit
over the optimized deepest suffix.

## 4. Next research gates

### Gate A: remove structurally wasted compute — passed

For every propagation region, execute full current blocks only while their output will affect a
later layer's K/V. At the terminal layer, compute current `Norm + Wk/Wv` without attention, gate,
output projection, or residual.

Completed checks:

- unit-test K/V equality against the existing suffix operator for every suffix depth;
- remeasure GPU time with the same resident batch and CUDA-event protocol;
- confirm that the optimized operator changes cost only, not ranking output.

All suffix depths have maximum absolute K/V difference `0.0` from the legacy operator. The new
operator takes 62.4% of legacy suffix-1 time, 75.1% of suffix-2, 86.4% of suffix-5, and 89.1% of
legacy full-recompute time. Costs normalized to optimized full are stable across four seeds:
cheap `0.187`, suffix-2 `0.372`, suffix-3 `0.504`, suffix-4 `0.635`, suffix-5 `0.767`.

### Gate B: test whether deepest suffix is actually the right region — passed with a negative result

Use the six-layer model as a small oracle space before increasing scale. Represent a candidate
interval `[s, e]` as full propagation through layers `s..e-1` followed by terminal projection at
layer `e`; layers outside the interval use cheap cached states. There are only 21 contiguous
intervals in a six-layer model.

Seed 0 searched all intervals at theta-3/theta-5; seeds 1-3 evaluated only selected candidates.
Deep suffixes recover more Best Rank than same-cost middle intervals on every held-out seed. Small
middle-interval NDCG advantages at theta-5 occur on only two of three seeds, reverse at theta-3,
and their paired intervals include zero. Early intervals are also consistently worse on Best Rank.

The current decision is to retain the optimized deepest suffix. Arbitrary or disjoint interval
selection is deferred unless a larger model or second dataset produces new evidence. Exact results
are in `experiments/validity/INTERVAL_ORACLE.md`.

### Gate C: establish the full streaming-training value chain — passed

Under the same validity protocol and future evaluation window, compare:

1. `frozen`: theta-0 model with its consistent theta-0 cache;
2. `full reuse`: current theta-t model consuming a theta-0 prefix cache;
3. `full compute`: current theta-t model with a fresh theta-t prefix cache.

This separates the benefit of stream training from the cache inconsistency it creates. Under the
six-layer validity protocol and four seeds, full compute over frozen improves Best Rank by `111.43`,
`237.97`, and `484.34` at theta-1/3/5. Full reuse retains most of this value, while cache maintenance
adds `5.66`, `41.68`, and `85.32` Best Rank respectively. At theta-5, maintenance accounts for
17.6% of total Best Rank value and 30.8% of NDCG@100 value; both seed-level intervals exclude zero.

Thus streaming training is necessary, stale reuse remains useful, and the recoverable gap grows
with cache age. MRR does not uniformly favor maintenance, so the claim is restricted to the primary
Best Rank and NDCG views. Exact results are in
`experiments/validity/STREAMING_VALUE_CONTROL.md`.

### Gate D: dynamic selection without returning to per-user estimation — deferred

If the oracle region changes across model updates or cache ages, select one configuration per
`(old_model_version, current_model_version, compute_budget)` cohort. A small held-out probe cache
set may evaluate candidate regions after each update; all caches in that version cohort then use
the selected GPU-friendly operator.

Only after this oracle planner is useful should cheaper update-level features be studied, such as
cache age, layerwise parameter-update norms, or aggregate probe errors. Training and evaluation
probe sets must be separated. A user-specific JVP is outside the active design.

The current held-out interval result does not satisfy this prerequisite. Do not build the planner
until scale or cross-dataset experiments show reproducible region changes.

### Gate E: expand scale and generality after the operator is fixed — KuaiRand passed, cross-dataset open

Completed without changing the operator:

- resident sequence lengths 16-512 and batch sizes 1-128;
- active KuaiRand lengths 32/64/128 over four seeds;
- depth 3/6/9 at fixed hidden and K/V width over four seeds;
- controlled theta-0 to theta-5 update magnitude over four seeds;
- a four-seed MovieLens-1M two-update chronological-holdout check.
- a top-5k/top-20k by 6-layer/12-layer four-seed factorial stress test;
- a top-50k, length-512 four-seed comparison of latest-only and complete base-chunk training.

The KuaiRand scale gate passes: the curve survives length, batching, depth, model width, larger
catalog/context, controlled update magnitude, and materially greater training-data utilization.
The generality gate remains open because MovieLens shows only a tiny and inconsistent maintenance
gap. Do not tune the short MovieLens chain until it becomes positive.

The Taobao UserBehavior audit fails the target-semantics gate before model training: it records user
actions but not true unclicked impressions, so using it would change the prediction problem or
require synthetic negatives. It remains a documented data boundary rather than the selected
second stream.

Tenrec QK/QB and ZhihuRec pass the exposure-semantics and capacity audit. QK is the closest
multi-feedback video match, QB is a compact pipeline check, and ZhihuRec supplies explicit
timestamps and a different domain. Under a base-only top-50k audit, they retain 70.05%, 92.29%, and
95.42% of raw rows respectively. Exact distributions, limitations, and the frozen ordered replay
are in `docs/dataset_expansion_audit.md`. These are data results only; the generality gate remains
open.

Proceed in this order:

1. Implement one length-128 ordered-exposure loader with a 64-impression base, six 8-impression
   windows, base-only top-50k vocabulary, and base-only cohort selection.
2. Use QB for data parity and one seed of `frozen / full reuse / full compute`. Do not evaluate
   migration configurations before establishing a measurable maintenance gap.
3. If the motivation gate passes, repeat the frozen control on base-only 5k-user QK and ZhihuRec
   cohorts. Preserve the distinction between Tenrec ordinal replay and ZhihuRec calendar time.
4. Reproduce successful cells over four seeds and evaluate only the frozen endpoints and
   proportional suffix budgets: full reuse, cheap, roughly one-third/two-thirds suffixes,
   all-but-first suffix, and full recompute. Do not reopen arbitrary interval search.
5. Report a negative boundary if the gap remains unidentifiable; do not change target or window
   semantics to obtain a positive result. Then add organically mixed cache versions to the
   surviving KuaiRand and second-dataset settings.

### Gate F: turn kernel savings into a system result — parallel systems task

The current cost is GPU-resident kernel time. A paper-grade system evaluation must include:

- extra normalized/split-hidden state capacity;
- HBM reads and writes, host-device transfer, allocation, and cache admission;
- batched throughput and tail latency across version cohorts;
- fused or grouped `Wk/Wv` projection only after profiling identifies it as a bottleneck;
- end-to-end comparison with periodic full recomputation under equal quality or equal cost.

## 5. Evaluation rules for the next phase

- Keep full reuse, cheap refresh, every selected propagation configuration, periodic recomputation,
  and full recompute as explicit baselines.
- Report absolute rank/NDCG gains together with normalized recovery. Do not report a recovery ratio
  when the fresh-over-reuse denominator is too small to be identifiable.
- Treat full recomputation as the fidelity reference and report each method's paired task-quality
  difference from full; do not assume full is the ranking-quality ceiling.
- Treat training seed as the replication unit and use paired intervals for methods evaluated on
  the same run.
- Separate configuration search from final evaluation to avoid selecting and reporting on the same
  cells.
- Version any material protocol change and never merge it into existing validity-v1 summaries.
- For a new dataset, publish the temporal audit and freeze target semantics before method results.
- Report operator state and data-movement cost alongside arithmetic time.

## 6. Claims that are not yet supported

- deepest suffix is optimal;
- the current method is statistically equivalent to full recomputation;
- relative kernel-time savings transfer to industrial end-to-end latency;
- the effect generalizes beyond the current simplified HSTU and KuaiRand setup;
- the problem is novel relative to all current literature;
- dynamic layer selection can be predicted cheaply without probe evaluation.

## 7. Stop or pivot conditions

The route should be reconsidered if any of the following persists after the scoped checks:

1. the optimized suffix cannot produce a meaningful Pareto curve against cheap-all, periodic full
   recomputation, and full recompute after end-to-end data movement is included;
2. extra state movement removes the measured compute saving in end-to-end execution;
3. the streaming-training and maintenance gains fail to generalize beyond the current control;
4. the chosen region does not generalize across seeds, cache ages, or a second dataset;
5. a related-work audit shows that model-version cache migration and the same structural operator
   have already been established.

## 8. Recommended execution order

1. [x] Implement terminal projection optimization and its equivalence tests.
2. [x] Run the one-seed interval oracle and validate selected candidates on held-out seeds.
3. [x] Decide whether update-level arbitrary-interval selection is currently necessary: no.
4. [x] Run the corrected frozen/full-reuse/full-compute control across seeds.
5. [x] Freeze the optimized suffix and complete the first length, batch, depth, update-magnitude,
   and MovieLens transfer pass.
6. [x] Complete the KuaiRand data/model factorial and top-50k chunked-data utilization pass.
7. [x] Audit Taobao and reject it as the primary second stream because it lacks true unclicked
   impressions.
8. [x] Audit Tenrec QK/QB and ZhihuRec under a base-only ordered-exposure protocol.
9. [ ] Implement the shared loader and run the one-seed QB motivation control, followed by frozen
   QK and ZhihuRec controls if the gap is identifiable.
10. [ ] Reproduce successful second-dataset cells across seeds with the frozen suffix endpoints.
11. [ ] Evaluate organically mixed cache versions on the settings with an identifiable gap.
12. [ ] Measure end-to-end state movement, throughput, tail latency, and periodic recomputation.
13. [ ] In parallel, complete a primary-source related-work audit before making novelty claims.
