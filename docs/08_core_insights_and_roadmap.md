# Core insights and roadmap

> Status: authoritative as of 2026-07-25. This file replaces all earlier problem statements and
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
an operating region between the two endpoints. The fixed-endpoint matrix in Section 2.8 further
shows that age is a coarse ordering variable, not a calibrated quality trigger.

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

A cross-dataset follow-up adds a shared version-level calibration without returning to per-user
estimation. On a small cache sample, it fits a low-rank affine map from cached old
`Norm(x)` to the residual between cheap and fresh K/V. Because this residual map is linear in the
cached state, it is folded into a prepacked K/V projection once per model-version pair. Online
migration remains one layer-batched projection rather than executing the low-rank factors for
every cache. Rank is chosen on a disjoint probe as the smallest value closing 50% of the measured
cache-fidelity gap.

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

### 2.5 The first scale pass is positive on KuaiRand; MovieLens is unresolved

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

### 2.7 The pre-design motivation now aligns on KuaiRand, QB, and QK

The shared loader retains true zero-feedback exposures and real positive targets. Tenrec uses
official within-user ordinal order, so it is an ordered replay rather than calendar-time drift.
After the original top-50k transfer exposed two different measurement problems, one data axis was
changed at a time before seed expansion:

- QK moved to a top-5k vocabulary, matching the dense KuaiRand operating point and reducing the
  5k-user model from 5.09M to 0.77M parameters.
- QB retained top-50k but uses a fixed-horizon cohort whose users have at least the complete
  64+6x8 raw exposure horizon. This restriction uses activity availability, not future labels or
  observed model outcomes, and removes the old 4,049-to-2,552 user-composition shift.

Under four seeds, the cumulative BestRank maintenance gap has a positive cache-age relationship
on all three datasets: seed-level Spearman is `0.925 [0.845, 1.000]` on KuaiRand,
`0.600 [0.005, 1.000]` on QB, and `0.700 [0.475, 0.925]` on QK. At theta-5:

| Dataset | Full compute over frozen | Full reuse over frozen | Cache maintenance |
|---|---:|---:|---:|
| KuaiRand | `484.34 [462.15, 506.54]` | `399.02 [370.79, 427.26]` | `85.32 [53.74, 116.91]` |
| QB fixed horizon | `94.70 [70.49, 118.90]` | `64.38 [54.41, 74.35]` | `30.31 [14.08, 46.55]` |
| QK top-5k | `47.34 [29.20, 65.47]` | `34.34 [20.55, 48.13]` | `13.00 [5.70, 20.30]` |

Thus the pre-design logic is consistent: streaming training becomes valuable, reuse preserves
substantial value rather than failing immediately, and old caches leave an additional recoverable
gap. Absolute BestRank changes are not compared across catalog sizes. QB and QK are related Tenrec
tables rather than two independent sources, QB conditions on a complete activity horizon, and
Tenrec lacks global timestamps; these limitations must accompany the claim.

The original aligned method gate remains a useful negative design result. QK cheap refresh costs
`0.194 [0.187, 0.200]x` full and gains `9.17 [6.71, 11.63]` BestRank, while the frozen suffixes
lose seed-0 BestRank relative to cheap. Aligned QB cheap recovers only 5.0% of its seed-0
BestRank gap and every tested suffix has a secondary-metric conflict. Thus a fixed suffix is not
the transferable method.

The new `compiled_low_rank_migration_v1` screen changes the operator rather than retuning the
suffix. It separates adapter fit, rank-selection probe, and held-out users; seed 0 discovers the
50% cache-fidelity target and seeds 1-3 replicate that frozen rule. Selected ranks vary as
`2/8/4/2` on KuaiRand, `8/4/2/16` on QB, and `16/16/8/8` on QK. Across four seeds, the compiled
operator costs `0.123 [0.122, 0.124]x`, `0.115 [0.113, 0.117]x`, and
`0.109 [0.106, 0.112]x` full and closes `52.1%`, `51.5%`, and `53.6%` of the held-out cache gap.
BestRank, rank utility, and NDCG@100 improve over reuse in 4/4 seeds on every dataset. Seed-level
rank-utility intervals are positive on all three; QB/QK BestRank and QK NDCG remain imprecise with
four seeds. This is the first method-level cross-dataset pass, although QB and QK are related
Tenrec tables and the result is still kernel-level. Exact design, intervals, and amortization
limits are in `experiments/migration/COMPILED_LOW_RANK_V1.md`.

### 2.8 Fixed-endpoint evidence weakens an age-only recomputation policy

The raw theta-5 full-compute gains `484.34/94.70/47.34` and maintenance gaps
`85.32/30.31/13.00` are not cross-dataset effect sizes. Catalog and task difficulty differ. A
fixed-endpoint cache-version matrix now holds the current model, final evaluation histories,
targets, and users fixed while varying only the model version that produces the same prefix K/V.
Its primary dimensionless effect is

$$
\text{staleness tax} =
\frac{\text{full compute}-\text{reuse}}
     {\text{full compute}-\text{frozen}},
$$

with metric direction corrected and the ratio computed within each training seed. It measures the
fraction of useful streaming-training value forfeited by stale reuse.

At the coarse theta-5 endpoint, BestRank tax is `0.176 [0.115, 0.237]` on KuaiRand,
`0.315 [0.225, 0.404]` on QB, and `0.276 [0.168, 0.385]` on QK. The largest value is only 1.79x
the smallest, rather than the 10.2x span in raw full-compute gains. MeanRank, rank-utility, and
NDCG@100 tax spans are 1.60x, 2.40x, and 2.47x respectively, although QK's NDCG interval includes
zero.

A separate four-seed fine matrix uses one-day KuaiRand updates and four-exposure Tenrec updates.
BestRank endpoint tax remains aligned at `0.177 [0.118, 0.236]`, `0.350 [0.263, 0.437]`, and
`0.141 [0.056, 0.227]`, a 2.48x span. It also exposes local deterioration:

- KuaiRand's common age 13→14 step adds `0.063 [0.028, 0.099]` tax, positive in 4/4 seeds.
- QB's common age 6→7 step adds `0.094 [0.052, 0.137]`, positive in 4/4 seeds.
- QK has a positive maximum local step in every seed, but its location moves across
  8→9, 9→10, 4→5, and 7→8; no single fine-age transition has a stable interval.

The first age at which BestRank tax reaches 10% is 14 in every KuaiRand seed, but
`2/1/5/4` on QB and `1/10/5/8` on QK. Every fine curve is also non-monotone. Thus a universal
fixed reuse window is not a calibrated quality policy: it can recompute during a benign plateau
or miss an earlier harmful update. This does not prove that every tuned periodic policy fails;
KuaiRand is a counterexample at the current operating point and periodic recomputation remains a
required baseline. The supported design implication is an update-aware action chosen once per
old/current model-version cohort, not a return to per-user JVP estimation.

The local maximum-step statistic was identified in exploratory analysis and must be frozen before
new confirmatory data. Fine QK MeanRank, rank-utility, and NDCG intervals include zero, so the
cross-resolution claim remains restricted to full-catalog BestRank. Exact definitions and
per-seed transitions are in `experiments/exposure/CACHE_VERSION_MATRIX_V1.md`.

### 2.9 A wide joint opportunity exists on KuaiRand, not automatically on every long cohort

The strongest current joint quality-cost regime is the four-seed KuaiRand top-50k/all-chunks cell.
Full compute improves BestRank over frozen by 3,837.67, stale reuse retains 2,952.11, and cache
maintenance recovers the remaining 885.56. Thus the staleness tax is 23.1% and reuse retains
76.9% of useful streaming-training value. At length 512, cheap refresh costs 0.058x full. This is
the desired interior regime: neither permanent reuse nor full recomputation dominates, and there
is substantial room between them.

A separately frozen seed-0 stress screen asked whether longer Tenrec histories create the same
joint regime. QB top-50k retains a median 244 events and QK top-20k a median 206. At those lengths,
full resident-GPU recomputation costs 4.504/3.578 ms per batch versus 0.512/0.439 ms for cheap
refresh, so the systems-side opportunity passes. The quality-side opportunity does not: across
stream learning rates `1e-4`, `2e-4`, and `4e-4`, oldest-cache BestRank tax is only 1.1%-5.6% on
QB and 3.6%-6.0% on QK, below the pre-frozen 15% lower bound. NDCG maintenance is non-positive in
five of six cells. No migration quality or extra seeds were run after this failure.

This negative result rules out a convenient but false shortcut: longer context makes full
recomputation more expensive, but does not by itself make model-version incompatibility consume a
larger fraction of streaming value. The paper should use KuaiRand top-50k/all-chunks as the primary
joint opportunity result, the aligned coarse QB/QK settings as cross-dataset problem evidence, and
the long-context Tenrec screen as a boundary. Exact protocol and compact evidence are in
`experiments/exposure/OPPORTUNITY_REGIME_V1.md` and
`results/exposure/long_context_opportunity_summary.json`.

### 2.10 Joint data/model-capacity motivation passes with a dataset-dependent boundary

A new pre-design matrix couples nested training data with model capacity instead of feeding the
same data volume to three models. Each dataset keeps one fixed prediction task and base-fitted
vocabulary. Small, medium, and large models are 3L/H64, 6L/H96, and 9L/H128; KuaiRand uses
250/500/980 users and QB/QK use 1k/3k/5k users. Effective base and stream targets strictly increase
at every tier. Four independent training seeds cover all nine cells, producing 36 trained
model-version chains and 108 matched core/control/cache-age artifacts.

Two motivation endpoints are unusually stable. At theta-11, full compute over frozen and full
reuse over frozen improve BestRank in 4/4 seeds in every cell, and all 18 seed-level intervals
exclude zero. Thus scaling the data and model does not remove the need for streaming training, nor
does a theta-0 cache immediately destroy its value.

The recoverable maintenance gap is not monotonic in capacity. Large KuaiRand has full/reuse/cache
BestRank gains of 4415.56/2819.20/1596.37 and a 36.0% staleness tax
`[25.3%, 46.7%]`. Large QB has 141.87/66.70/75.17 and a 54.8% tax
`[38.3%, 71.3%]`. Both gaps are positive in 4/4 seeds and corroborated by rank utility. Medium QK
has a positive BestRank maintenance sign in 4/4 seeds but a wide interval; large QK falls to 3/4
signs, a -0.5% mean within-seed tax with a wide interval, and no NDCG support. QB medium also has a
BestRank direction conflict despite positive rank utility.

The paired large-minus-small BestRank-tax change is +23.2 percentage points on KuaiRand
`[-1.0, 47.4]`, +53.4 points on QB `[36.9, 69.8]`, and -13.1 points on QK
`[-81.4, 55.3]`. Therefore the supported scale statement is existential rather than universal:
larger data/model operating points can widen the migration opportunity substantially, but capacity
alone does not guarantee stronger cache invalidation.

The systems pressure grows cleanly. At batch 32 and length 128, full resident-GPU prefix cost is
about 1.05/2.09/3.36 ms from small to large, while cheap projection refresh remains
18.5%-20.4% of full. Only large KuaiRand's seed-0-selected local jump replicates at the same fixed
age: age 10→11 adds 16.8 tax points `[6.0, 27.5]` over seeds 1-3. Large QB and QK do not share a
replicated jump location. This reinforces an update-aware policy motivation, but does not justify
a universal sudden-failure age.

This matrix completes the requested capacity expansion of the motivation. It does not select or
tune a migration method. Large KuaiRand and large QB are the strongest joint quality-cost
development regimes; the earlier accepted 6L/H96 QK result and the medium-QK signs supply the QK
problem evidence, while large QK must be reported as a scale boundary. Exact settings and all nine
cells are in `experiments/motivation/CAPACITY_V2.md`; the reproducible aggregate is
`results/motivation_scale/capacity_v2_summary.json`.

### 2.11 Cohort-tiered migration scales as an operator; task quality is not an admission oracle

The capacity design screen no longer treats a fixed suffix or prefix depth as the method. For each
old/current model-version cohort it calibrates a shared residual map, folds that map into one
prepacked K/V projection, and retains residual-delta prefix replay plus exact recomputation as
higher-fidelity tiers. The controller selects by measured GPU cost and probe K/V fidelity, without
task labels or a per-user predictor.

On the nine motivation-selected discovery checkpoints, the inherited 50% fidelity rule chooses
compiled ranks 2-32. Mean cost is `0.122x` full, mean test K/V recovery is 0.564, and BestRank plus
rank utility improve over reuse in 9/9 cells. NDCG@100 improves in 8/9; QB-medium is the exception,
and its exact full endpoint is also negative. At a 75% target the same controller remains on a
compiled projection in six cells and selects residual depths 5/6/7 in large KuaiRand/QB/QK, giving
a genuine version-dependent action curve rather than a fixed layer count. Plain prefix replay is
never selected in the unified library.

The frozen 27-seed replication separates operator behavior from maintenance opportunity. Every
primary action is a compiled projection, with mean cost `0.121 [0.112, 0.130]x` and mean K/V
recovery `0.587 [0.547, 0.627]`. The strict positive-mean BestRank/rank-utility gate passes 6/9
cells, not 9/9. QB-small has essentially zero rank-utility effect, QB-medium full recomputation is
negative in BestRank in all three replication seeds, and QK-large is near zero and unstable.

The migrated result nevertheless tracks the full endpoint: selected/full signs agree in 23/27
BestRank, 27/27 rank-utility, and 25/27 NDCG cases. When full is positive, the selected action is
also positive in 18/20, 24/24, and 19/20 cases, with median gain recovery
0.907/0.972/0.966. These are descriptive cross-cell diagnostics. The supported conclusion is that
the compiled operator scales in cost and cache fidelity and usually preserves an available
quality gain. The unsupported conclusion is that task labels can identify which model-version
cohort should reuse or migrate. Full recomputation is the semantic reference but is not a ranking
upper bound, and neither cache age nor label-free drift predicts realized task gain reliably.
Consequently the active system does not make a version-level reuse admission decision. Every stale
cohort follows the same monotone semantic-synchronization ladder; version cohorts exist to compile,
batch, place, and schedule work.

The frozen architecture and complete result are in
`experiments/migration/COHORT_TIERED_MIGRATION_V1.md`; the reproducible aggregate is
`results/motivation_scale/cohort_tiered_migration_v1_summary.json`.

### 2.12 The larger KuaiRand 4+12 run rejects a fixed reuse window

The superseded 12+4 seed-0 exploration trained theta0-theta3. Theta0-cache MeanRank loss across
its three moving endpoints was 1,327/1,597/2,284 and catalog-AUC loss was
2.65/3.19/4.57 percentage points. This was enough to freeze MeanRank as the next protocol's
primary metric and AUC as a robust secondary, but three changing dates could not establish a
causal cache-age jump. Its prepared data, checkpoints, and raw results were removed after the
metric decision and cannot be pooled with the replacement protocol.

The replacement uses eight base dates and eight daily updates. It keeps all 5,820,867 exposures
for 965 base-eligible users, fits a base-only top-50k prediction catalog plus 262,144 context-only
hash buckets, and trains all chronological chunks. Histories use an eight-day window capped at
2,048 tokens. The 16L/H512 model remains 181,082,112 parameters (0.181B). Four DDP workers use
logical per-device batch four, micro-batch one, and four-way gradient accumulation, preserving
effective global batch 16 while bounding the length-2,048 backward graph.

The data-only dry run covers 2,914,284 base tokens and 422,097 eligible base targets per epoch.
The fixed D16 theta7 endpoint has 746 eligible users; median history length is 2,048, and 392
users are token-truncated after applying the eight-day window. Their logical unpadded FP32 prefix
K/V totals 77.58 GB decimal. Cached normalized state adds 38.79 GB, making the compiled-method
state 116.37 GB. These are capacity facts, not allocated-GPU peaks, latency, or quality results.

Seed-0 theta0-theta8 training completed in 831 seconds. The first evaluator stored a partial matrix.
The replacement v3 evaluator then completed the full strict lower triangle: for every current
theta-i from theta1 through theta7, it evaluated cache theta0 through theta-(i-1), giving 28
distinct pairs.

The complete matrix does not show repeated late cache-age cliffs. The large discontinuity is the
theta0-to-theta1 base-to-stream boundary. Excluding theta0, MeanRank, AUC, standard top-k metrics,
K/V drift, hidden cosine, score cosine, and top-10 changes evolve smoothly enough that no
non-theta0 quality step survives the post-hoc multiple-comparison screen. The supported conclusion
is that stale reuse has a measurable endpoint cost and theta0 is special; the unsupported
conclusion is that the 8+8 run establishes a generic “safe for two versions, then collapse”
pattern.

The new exploratory route changes only the temporal split, not model scale or serving semantics.
It uses D1-D4 as base and D5-D16 as 12 updates, producing theta0-theta12 and 11 leak-free
next-day current versions. The complete matrix has 66 pairs; theta0 has 11 longitudinal points and
theta2 has nine. The prepared cohort has 945 users and 5,780,499 selected rows. At D16 its logical
FP32 prefix K/V is 71.29 GB across 682 eligible users. Seed-0 four-GPU training completed in
860.7 seconds and produced theta0-theta12; the 66-pair motivation matrix is complete.

At the fixed theta11/D16 endpoint, cache age is strongly ordered with K/V drift but not with task
quality. Across all pairs, age-to-MeanRank Spearman is approximately zero while age-to-K/V-drift
Spearman is 0.817. After removing the special theta0 base boundary, age alone explains 6.15% of
MeanRank variation, whereas current-version identity explains 60.9%. Even adjacent one-update
cohorts have opposite effects: theta0→theta1 loses 538 MeanRank on average, while theta1→theta2
improves it by 137. The complete matrix therefore supports a non-linear, update-dependent
staleness effect and rejects cache age as a sufficient statistic for a fixed reuse window. It does
not support a universal delayed cliff or a predictor that decides which version is safe to reuse.

The first large-model progressive-sync diagnostic fixes theta11/D16 and evaluates source theta0,
theta4, and theta10. On 16 held-out diagnostic users, rank-32 compiled repair costs approximately
0.061x exact recomputation and recovers 0.610/0.589/0.636 of the K/V fidelity gap. At age 11,
residual p8 costs 0.550x and recovers 0.846; p12 costs 0.775x and recovers 0.934. These values
validate the action ordering and implementation only. They use one training seed, four fit users,
and altered evaluation settings, so they are not formal paper evidence. The frozen formal
protocol uses 40 fit, 60 label-free probe, and all remaining test users. Evaluation may use two or
four workers because worker count changes only inference sharding and wall time.

The formal two-worker evaluation is now complete on 582 test users. Rank-32 costs
`0.0640/0.0642/0.0641x` exact and recovers `0.579/0.673/0.609` of the K/V gap at ages
11/7/1. P8 costs approximately `0.549x`; it is not a reliable quality tier, because its K/V
recovery is only `0.840/0.784/0.798` and its age-7 task deviations are worse than rank-32.

A separate three-round design search keeps that endpoint and user split fixed. It exposes an
important compilation fact: ranks 16 through 512 all fold into the same
`[16,512,1024]` affine projection, so low rank is offline regularization rather than an online
cost knob. The label-free probe selects full rank 512. An attention-use-weighted version of the
same full-affine regression is the final exploratory candidate. It costs
`0.0640/0.0642/0.0642x` exact and recovers `0.886/0.891/0.936` of the K/V gap. Relative to
rank-32, its three-age mean absolute deviation from fresh falls by 44.5% for MeanRank, 44.5% for
AUC, and 47.3% for NDCG@100 without changing the online matrix, program bytes, or measured cost.
At age 11 it recovers 99.2% of the signed MeanRank and AUC gap, 94.4% of the NDCG@100 gap, and
90.9% of the Hit@100 gap.

The intermediate ridge search is a negative result: ridge `1e-2` wins a tiny probe
score-cosine difference but loses test top-100 overlap and MeanRank fidelity to `1e-3`. Attention
weighting adds only a further 1.0%-1.6% average task-fidelity improvement over uniform full rank.
Because later rounds were designed after inspecting earlier test results, the selected program is
exploratory and must be frozen on new seeds or datasets before becoming confirmatory evidence.

A verified compiler now turns that candidate into an enforceable cohort contract. It preserves the
40 fit and 60 earlier program-selection users, withholds another 60 users for label-free
certification, and leaves 522 users for final recommendation evaluation. For cache error, fresh
score cosine, and top-100 overlap, the contract requires at least 70% recovery, a one-sided 90%
bootstrap recovery lower bound of at least 70%, at least 80% user coverage after a one-sided 90%
Wilson bound, and primary cost no greater than 0.30x exact. The candidate library contains current
projection, compiled full affine, structural p4/p8, and exact.

At ages 11/7/1, the compiler selects compiled full affine at
`0.0627/0.0631/0.0631x` certificate cost. The worst recovery lower bounds are
`0.853/0.837/0.923`, and the worst coverage lower bounds are `0.922/0.900/0.946`. P8 is retained
as a budget-overflow fallback only at ages 11 and 1; at age 7 it fails the frozen contract, so that
plan falls directly to exact. On the 522 final users, the selected action costs
`0.0638/0.0640/0.0641x`, recovers `0.886/0.891/0.936` of the K/V gap, and remains above every
frozen certificate bound. At the harmful age-11 endpoint it recovers 98.8% of the signed MeanRank
and AUC gap, 90.3% of NDCG@100, and 88.9% of Hit@100. No recommendation labels enter compilation
or certification. This is still adaptive seed-0 exploration because the preceding program was
developed after inspecting this seed; confirmation requires a frozen new seed or dataset.

Cache age remains checkpoint-update distance over the identical resident prefix, not literal
residence of one physical snapshot. Physical rolling eviction and organically mixed per-token
versions remain separate system gates. The 8+8 protocol is in
`experiments/motivation/LONG_CONTEXT_8PLUS8_V2.md`; the split exploration is in
`experiments/motivation/LONG_CONTEXT_SPLIT_EXPLORATION_V1.md`; the design loop is in
`experiments/migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md`; the verified compiler is in
`experiments/migration/VERIFIED_COHORT_COMPILER_V1.md`.

## 3. Current contribution hypothesis

A complete paper could make four contributions if the remaining gates pass:

1. Define model-version invalidation of generative-recommendation prefix K/V as distinct from
   behavior append and fixed-model cache management.
2. Characterize the time and layer structure of the resulting quality loss under leak-free
   streaming evaluation.
3. Introduce a verified cohort migration compiler that generates migration actions, certifies
   label-free current-model semantic fidelity and user coverage, then publishes the cheapest
   qualifying program and an ordered fallback chain. Its fast path compiles an
   attention-use-weighted full-affine residual into one K/V projection.
4. Build a version-cohort executor that applies an unconditional cheap synchronization, advances
   cache extents through progressively stronger fidelity tiers under a resource budget, and
   demonstrates end-to-end savings against periodic recomputation.

The reusable idea is to move adaptation work out of the per-cache path: measure one version-pair
cohort, learn a shared correction, compile it into a GPU-friendly operator, and batch one action
per cohort. A dynamic arbitrary-layer planner is unnecessary: the all-interval and recent-token
screens do not justify their search or fragmentation cost.

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

### Gate D: shared version-level calibration without per-user estimation — kernel gate passed

The first update-aware operator is now implemented. For each theta-0/theta-5 cohort, a small
fit set supplies fresh K/V residuals, a disjoint probe chooses the smallest rank closing at least
50% of the cache gap, and held-out users evaluate the compiled projection. No labels are used to
fit or select the rank. The same frozen selection rule produces positive BestRank, rank utility,
and NDCG@100 signs in all 12 dataset-seed cells.

Compilation is essential: executing the low-rank factors online costs about `0.4x` full in the
seed-0 screen, whereas folding them into one prepacked projection lowers held-out kernel cost to
`0.106-0.124x` full. The current unoptimized calibration takes roughly `0.3-0.4 s`; descriptive
kernel-only break-even cohorts are about 4.8k, 6.3k, and 7.4k caches on KuaiRand, QB, and QK.
These amortization numbers exclude state movement and cannot yet support an end-to-end claim.

The remaining policy question is how quickly each stale extent advances through compiled repair,
residual replay, and exact recomputation under organic mixed ages and a finite update budget.
This is scheduling of repair strength, not prediction of whether a version is safe to reuse.
Fixed-period and age-only schedules remain mandatory baselines. This gate does not reopen
arbitrary-layer search, and a user-specific JVP remains outside the active design.

The capacity-tiered follow-up preserves the compiled fast path across 3L/H64, 6L/H96, and 9L/H128.
Its 27 replication runs take `0.121 [0.112, 0.130]x` full and recover
`0.587 [0.547, 0.627]` of the K/V gap. The operator gate passes; the unconditional task-quality
gate does not, because three cells have near-zero or negative full maintenance endpoints. This
prevents using ranking gain as a synchronization contract; it does not invalidate semantic
fidelity as the contract or justify an unobservable task-quality admission predictor.

### Gate E: expand scale and generality after the operator is fixed — motivation passes with scope limits

Completed without changing the operator:

- resident sequence lengths 16-512 and batch sizes 1-128;
- active KuaiRand lengths 32/64/128 over four seeds;
- depth 3/6/9 at fixed hidden and K/V width over four seeds;
- controlled theta-0 to theta-5 update magnitude over four seeds;
- a four-seed MovieLens-1M two-update chronological-holdout check.
- a top-5k/top-20k by 6-layer/12-layer four-seed factorial stress test;
- a top-50k, length-512 four-seed comparison of latest-only and complete base-chunk training.
- an ordered-exposure QB/QK/ZhihuRec transfer with frozen targets and migration methods;
- a one-axis motivation alignment follow-up using fixed-horizon QB and dense-catalog QK.
- coarse and fine fixed-endpoint cache-version matrices over four seeds on KuaiRand/QB/QK.
- a pre-frozen long-context QB/QK opportunity screen with one seed and a bounded learning-rate
  rescue.
- an aligned QB/QK method gate, expanding only the QK cheap endpoint to four seeds.
- a fit/probe/test-separated compiled low-rank migration screen on aligned KuaiRand/QB/QK over
  four seeds, with a frozen 50% cache-fidelity target after seed 0.
- a fixed-task 3x3 joint data/model-capacity motivation matrix over KuaiRand/QB/QK and four seeds.
- a motivation-selected nine-checkpoint structural design screen and a frozen 27-seed
  cohort-tiered migration replication.

The KuaiRand scale gate passes: the suffix curve survives length, batching, depth, model width, larger
catalog/context, controlled update magnitude, and materially greater training-data utilization.
The original MovieLens gate remains negative. The exposure-compatible motivation now reproduces
the same three-part logic on KuaiRand, fixed-horizon QB, and top-5k QK over four seeds: positive
long-horizon streaming value, substantial value retained under stale reuse, and a positive
age-dependent maintenance gap. ZhihuRec remains a negative boundary. Cheap projection refresh now
has positive four-seed BestRank evidence on aligned QK, but the strong suffix quality-recovery
curve still does not generalize. This closes the motivation gate and rejects the fixed suffix as
the transferable method. Fixed-endpoint BestRank staleness tax is also on one scale across the
three positive datasets, while its local jumps show that age alone is not a calibrated
recomputation trigger.

The compiled adapter now passes the aligned method-level transfer gate at kernel scope: its frozen
selection rule improves all three reported task-quality views in every individual seed on
KuaiRand, QB, and QK, while the rank-utility interval is positive for all three datasets. This
supersedes QB's earlier seed-0 suffix failure as the current method status; that failure remains
evidence against transferring the fixed suffix. It does not establish end-to-end savings or an
independent third organization/domain, because QB and QK are related Tenrec tables.

The long-context Tenrec screen is negative and must not replace the aligned motivation protocols.
It confirms a large kernel-cost separation at median histories 244/206, but its staleness tax
never reaches the frozen opportunity range. This strengthens the case for keeping data-regime
selection independent of method quality and makes the replicated KuaiRand top-50k/all-chunks cell
the current primary joint quality-cost regime.

The completed capacity matrix sharpens this boundary. Streaming value and value retained under
reuse reproduce in all nine cells, while the maintenance opportunity becomes strong at large
KuaiRand and large QB but not large QK. Full prefix cost still rises approximately 3.2x from the
small to large model. The paper can therefore claim that scale increases systems pressure and can
widen the quality opportunity; it cannot claim that a larger model or cohort mechanically creates
a larger staleness tax.

The Taobao UserBehavior audit fails the target-semantics gate before model training: it records user
actions but not true unclicked impressions, so using it would change the prediction problem or
require synthetic negatives. It remains a documented data boundary rather than the selected
second stream.

Tenrec QK/QB and ZhihuRec pass the exposure-semantics and capacity audit. QK is the closest
multi-feedback video match; QB and QK now supply two related ordered-exposure reproductions, while
ZhihuRec supplies explicit timestamps and a different-domain negative boundary. Exact
distributions, cohort conditioning, and limitations are in `docs/dataset_expansion_audit.md`;
model results are in `experiments/exposure/ORDERED_EXPOSURE_V1.md` and
`experiments/exposure/CACHE_VERSION_MATRIX_V1.md`.

Proceed in this order:

1. Freeze the verified compiler, including its action library and 70%/80%/90%/30% contract, and
   replicate it on new training seeds or accepted cross-dataset checkpoints without another
   search on the current seed's users.
2. Add organically mixed cache versions and version-cohort batching; compare unconditional compiled
   sync plus progressive refinement against reuse, periodic full recomputation, and age-only
   fixed-window policies at equal cost and equal fidelity.
3. Measure end-to-end state reads/writes, calibration amortization, throughput, and tail latency
   before optimizing another arithmetic kernel.
4. Keep ZhihuRec as a reported boundary unless a task-independent protocol, rather than
   result-driven per-dataset tuning, creates a reason to revisit it.

### Gate F: turn kernel savings into a system result — parallel systems task

The current cost is GPU-resident kernel time. A paper-grade system evaluation must include:

- extra normalized/split-hidden state capacity;
- one-time adapter fitting, compilation, and cohort-size amortization;
- HBM reads and writes, host-device transfer, allocation, and cache placement;
- batched throughput and tail latency across version cohorts;
- profile the prepacked projection and normalized-state reads before another kernel optimization;
- end-to-end comparison with periodic full recomputation under equal quality or equal cost.

## 5. Evaluation rules for the next phase

- Keep full reuse, cheap refresh, every selected propagation configuration, periodic recomputation,
  and full recompute as explicit baselines.
- Report absolute rank/NDCG gains together with normalized recovery. Do not report a recovery ratio
  when the fresh-over-reuse denominator is too small to be identifiable.
- For cross-dataset motivation, report within-seed staleness tax in addition to raw maintenance;
  omit it when full-compute streaming value is not positive.
- Treat full recomputation as the fidelity reference and report each method's paired task-quality
  difference from full; do not assume full is the ranking-quality ceiling.
- Treat training seed as the replication unit and use paired intervals for methods evaluated on
  the same run.
- Separate configuration search from final evaluation to avoid selecting and reporting on the same
  cells.
- For calibrated migration, keep adapter-fit users, rank-selection probes, and final test users
  disjoint; record the frozen fidelity target and one-time calibration cost.
- Version any material protocol change and never merge it into existing validity-v1 summaries.
- For a new dataset, publish the temporal audit and freeze target semantics before method results.
- Report operator state and data-movement cost alongside arithmetic time.

## 6. Claims that are not yet supported

- deepest suffix is optimal;
- the current method is statistically equivalent to full recomputation;
- relative kernel-time savings transfer to industrial end-to-end latency;
- the full KuaiRand suffix quality-recovery curve generalizes across datasets or architectures;
- the problem is novel relative to all current literature;
- dynamic layer selection can be predicted cheaply without probe evaluation;
- the compiled adapter's kernel-time savings survive normalized-state movement and end-to-end
  serving;
- the current 50% fidelity target is optimal, or transfers beyond the tested 3L/6L/9L
  theta-0/theta-11 capacity cells;
- every tuned fixed-window policy fails; current evidence only rejects age as a universal,
  calibrated quality state;
- longer context by itself increases the relative cache-maintenance quality gap; the Tenrec stress
  screen shows cost can grow while staleness tax shrinks.

## 7. Stop or pivot conditions

The route should be reconsidered if any of the following persists after the scoped checks:

1. compiled migration cannot produce a meaningful Pareto point against cheap-all, periodic full
   recomputation, and full recompute after calibration and end-to-end data movement are included;
2. extra state movement removes the measured compute saving in end-to-end execution;
3. the streaming-training and maintenance gains fail to generalize beyond the current control;
4. the compiled adapter or fixed progressive ladder does not generalize across seeds, cache ages, or
   a second dataset;
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
9. [x] Implement the shared loader and run frozen QB, QK, and ZhihuRec motivation controls.
10. [x] Expand QB to eight seeds and QK to four seeds with frozen suffix endpoints; retain
    ZhihuRec and the 12L/192 checks as negative gates.
11. [x] Diagnose the QB changing-cohort confound and QK sparse-catalog regime; validate
    fixed-horizon QB and top-5k QK over four seeds.
12. [x] Fix the current model/evaluation endpoint, normalize cross-dataset effect size, and map
    coarse/fine cache-version curves over four seeds.
13. [x] Freeze and run the long-context QB/QK opportunity screen; retain its failed quality gate
    as a boundary and do not run migration quality on those cells.
14. [x] Re-evaluate frozen migration methods on aligned QB/QK; expand QK cheap to four seeds,
    stop QB and all aligned suffix branches at seed 0.
15. [x] Screen renormalization, embedding-delta, and low-rank residual candidates; compile the
    only cross-dataset candidate into a fused projection.
16. [x] Freeze the 50% cache-fidelity rank rule after seed 0 and validate it on KuaiRand/QB/QK
    seeds 1-3 with separate fit, probe, and test users.
17. [x] Complete the fixed-task 3x3 joint data/model-capacity motivation matrix over four seeds.
18. [x] Transfer the frozen compiled operator across the capacity matrix, including
    top-50k/complete-chunk KuaiRand, and run the frozen 27-seed tiered replication.
19. [x] Complete the large KuaiRand scale bridge: 4+12 seed-0 training and all 66 motivation
    pairs are done; age is not a sufficient task-quality state and no universal delayed cliff is
    claimed.
20. [x] Complete the full-user theta11/D16 progressive-sync evaluation, bounded compiled-program
    search, and verified compiler. The compiler uses disjoint 40/60/60/522
    fit/selection/certificate/final roles and publishes a measured fallback plan; the current seed
    remains adaptive exploration and cannot supply confirmatory evidence.
21. [ ] Evaluate organically mixed cache versions with unconditional cheap synchronization,
    progressive refinement, periodic full, and age-only fixed-window baselines.
22. [ ] Measure end-to-end state movement, calibration amortization, throughput, tail latency, and
    periodic recomputation.
23. [ ] In parallel, complete a primary-source related-work audit before making novelty claims.
