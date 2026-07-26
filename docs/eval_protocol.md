# Current evaluation protocol

> This file defines the valid artifact boundary. Any material change to targets, data split,
> serving semantics, model family, or timing semantics requires a new protocol name and separate
> result files.

The active implementation sequence is
[`09_single_configuration_full_chain_plan.md`](09_single_configuration_full_chain_plan.md).
That document is a development plan, not a result protocol: it creates no comparable evidence by
itself. Before an integrated run is promoted from development evidence, its final configuration,
metrics, timing boundary, baselines, and artifact schema must be frozen here under a new protocol
name. Existing protocol strings and result families must not be silently reused for that run.

## 1. Protocol families

### `validity_v1_incremental_prefix_cache`

Used by `scripts/motivation_validity.py` for the repaired staleness motivation and for producing
the model-version checkpoints consumed by method evaluation.

### `layerwise_validity_v1`

Used by `scripts/layerwise_validity.py` for cheap projection plus continuous deep-suffix migration.
It must point to a matching validity-v1 run and checkpoint directory.

### `interval_oracle_v1_terminal_projection`

Seed-0 discovery over all 21 one-based inclusive contiguous intervals. It uses terminal
`Norm + Wk/Wv` instead of executing a full terminal block and retains legacy suffix timing only as
an equivalence/cost comparison.

### Held-out interval validation

`interval_oracle.py` records `study_stage=heldout_validation` for seeds 1-3 and evaluates only
candidates selected on seed 0. These files must not be used to search additional configurations.

### `compiled_low_rank_migration_v1`

Uses matching 6L/H96 theta-0/theta-5 checkpoints from KuaiRand validity, fixed-horizon QB, and
top-5k QK. A user split separates adapter fitting, rank-selection probe, and final evaluation.
The adapter fits fresh-minus-cheap K/V residuals from cached old `Norm(x)`, then folds the
low-rank affine correction into one prepacked K/V projection for online execution.

Candidate ranks are `2/4/8/16/32/64/96`. The frozen selector chooses the smallest rank closing at
least 50% of the stale-to-fresh relative K/V error gap on the probe. Seed 0 is
`seed0_discovery`; seeds 1-3 are `frozen_rule_replication`. Task labels are not used for adapter
fitting or rank selection. Raw files from the earlier uncompiled candidate screen have a different
protocol string and cannot be pooled with this family.

### `progressive_prefix_replay_v1`

Uses the nine motivation-selected `motivation_capacity_v2` checkpoints for method discovery and
the remaining 27 training seeds for frozen-rule replication. Its \(L+1\) action ladder ranges from
cheap current projection over cached old `Norm(x)`, through exact current-model prefix replay, to
full recomputation. A 60-user label-independent probe selects the lowest measured GPU-cost action
closing at least 20% of the stale-to-fresh K/V error gap. Discovery and
`frozen_rule_replication` records remain distinct. This family is a replicated structural
baseline, not the current primary operator.

### `cohort_tiered_migration_v1`

Uses the same capacity-v2 checkpoint boundary with a disjoint 40-user adapter-fit set, 60-user
planning set, and remaining held-out users. The production library contains a calibrated residual
compiled into one affine K/V projection, prefix replay with residual-delta transport as a
high-fidelity tier, and exact recomputation. Plain prefix replay is measured only as a discovery
baseline and is excluded from production selection.

The primary selector chooses the minimum-cost action reaching 50% probe K/V recovery; 75% and 90%
are frozen secondary curve points. Compiled ranks have one online kernel shape, so their median
measured projection cost is used and the smallest eligible rank is selected. A structural partial
action must save at least 1% relative to full on the probe. Task labels do not fit the adapter or
select an action. The nine motivation-selected checkpoint files use
`cohort_tiered_migration_discovery_v1`; only the other 27 files use
`cohort_tiered_migration_v1` with `study_stage=frozen_rule_replication`.

### `kuairand_long_context_8plus8_*`

This new, separate bring-up family has five protocol records:

- `kuairand_long_context_8plus8_data_v2` is the immutable prepared-data boundary;
- `kuairand_long_context_8plus8_training_v2` trains theta0 through theta8;
- `kuairand_long_context_8plus8_motivation_v2` is the completed partial matrix containing the
  moving theta0 curve and fixed-D16 theta7 row;
- `kuairand_long_context_8plus8_motivation_all_pairs_v3` evaluates all 28 strictly
  older-cache/current-model pairs through theta7;
- `kuairand_long_context_8plus8_compiled_migration_v2` evaluates a fixed rank-16 compiled
  residual at the theta0-to-theta7 boundary.

The family uses eight base dates and eight online dates and must not be merged with the 14-day-base
validity, scaling, exposure, capacity, or superseded 12+4 families. Seed-0 training, the v2 partial
motivation evaluation, and the active v3 all-pairs evaluation are complete. Only the method
evaluation remains pending. Exact settings and commands are in
`experiments/motivation/LONG_CONTEXT_8PLUS8_V2.md`.

### `motivation_joint_scale_v1_*`

This rejected seed-0 family crossed KuaiRand, QB, and QK with three matched data/model operating
points.
Small, medium, and large jointly increase the retained catalog/user data and model capacity; it is
not a model-only factorial. Every cell trains theta-0 through theta-11, runs the
frozen/full-reuse/full-compute value control, fixes theta-11 while varying the stale cache version,
and measures resident-GPU prefix cost. Its simultaneous catalog/task change and underfed 12L/H192
Tenrec cells failed the gate. Do not run more seeds or use it for the primary capacity claim.

The target, base-only vocabulary rule, full-catalog ranking, and stale-serving semantics remain
shared. KuaiRand retains calendar order and complete chunked training. QB/QK share a 64-exposure
base plus 12 four-exposure ordinal windows and an activity-only complete retained-horizon cohort.
Calendar age and ordinal age are not numerically interchangeable. Exact frozen settings are in
`experiments/motivation/JOINT_SCALE_V1.md`.

### `motivation_capacity_v2_*`

This corrected pre-design family holds each dataset's accepted task and vocabulary fixed while
jointly increasing nested training users and model capacity. KuaiRand/QB retain top-50k and QK
retains top-5k. Models scale as 3L/H64, 6L/H96, and 9L/H128; dataset-specific user counts increase
at every tier. Training, target, serving, endpoint, and timing semantics otherwise match the v1
screen. Exact settings and the seed-expansion rule are frozen in
`experiments/motivation/CAPACITY_V2.md`.

The family is complete over seeds 0-3: 36 independently trained model-version chains, 36 streaming
controls, and 36 fixed-endpoint cache-age matrices. GPU-resident operator cost is recorded at seed
0 for all nine cells. `results/motivation_scale/capacity_v2_summary.json` is the only compact
cross-cell summary; raw files remain separate by dataset, tier, stage, and seed. Training seed is
the statistical unit, and all ratios are formed within seed before aggregation.

The frozen gate supports positive full-compute streaming value and positive stale-reuse value in
all nine cells. It does not support a uniform monotonic capacity claim for cache maintenance:
large KuaiRand and large QB have replicated BestRank maintenance gaps, medium QK has four positive
signs but an imprecise interval, and large QK is unstable across seeds. This boundary cannot be
removed by pooling tiers or selecting a favorable seed.

### `streaming_value_control_v1_incremental_prefix_cache`

Six-layer seeds 0-3 compare theta-0 frozen serving, current-model full reuse of a theta-0 prefix,
and current-model full compute on the same history and future positives. The three contrasts are
total streaming-training value, value retained under stale reuse, and cache-maintenance value.

### Fixed-endpoint cache-version matrices

`cache_version_matrix_v1` and `cache_version_matrix_full_eval_v1` hold the current theta-5 model,
final evaluation histories, targets, and users fixed while changing only the old model version
used to produce the prefix K/V. The KuaiRand matrix retains its 300-user evaluation; the
`full_eval` QB/QK matrices evaluate every eligible final-window user up to 5,000. These files must
not be pooled with moving-window cumulative-age records.

`cache_version_matrix_fine_full_eval_v1` is a separate temporal-resolution family. KuaiRand uses
one-day updates through theta-15. QB/QK use four ordinal exposures per update through theta-10.
The current model and final evaluation window remain fixed within every matrix. KuaiRand fine
runs retain the existing latest-sequence training mode and are diagnostic rather than a
replacement for the all-chunks primary operating point.

### `scaling_v1_fixed_optimized_suffix`

Freezes the optimized deepest suffix and changes one KuaiRand axis at a time. Active sequence
length, model depth, and controlled parameter interpolation use the same full-catalog target
semantics as validity-v1. Depth 3/6/9 and every reported quality point use seeds 0-3. No interval
is selected from these result cells.

### `operator_cost_scaling_v1_resident_cuda_events`

Uses synthetic full-length tensors resident on one A40 to isolate shape scaling. Sequence length
and batch size are separate axes. The record includes absolute CUDA-event latency and ratios to
optimized full; it excludes transfers, allocation, admission, and scheduling.

### `movielens_chronological_holdout_scaling_v1`

Uses the same MovieLens-1M users at consecutive train/dev/test chronological targets. Theta-0 is
trained on train histories, theta-1 ingests the train target and evaluates the dev target, and
theta-2 ingests the dev target and evaluates the test target. It is a two-update transfer check,
not a substitute for a long wall-clock stream. Equal effective lengths are batched together and
evaluation uses highest float32 matmul precision so full/incremental parity is not obscured by
shape-dependent TF32 approximation.

### `kuairand_factorial_v1_training` and `kuairand_factorial_v1_fixed_suffix`

Four-seed 2x2 KuaiRand stress test. The data bundle changes top-5k/length-128 to
top-20k/length-256; the model factor changes 6-layer/hidden-96/four-head to
12-layer/hidden-192/eight-head. Base/stream dates, optimization, full-catalog evaluation, and
fixed proportional suffix budgets remain unchanged. The data bundle is not a one-variable causal
axis and must be labeled as such.

### `kuairand_data_utilization_v1_latest` and `kuairand_data_utilization_v1_all_chunks`

Top-50k, length-512, six-layer comparison of base training coverage. `latest` preserves the old
one-sequence-per-user iterator. `all_chunks` uses complete chronological base histories split into
length-512 chunks with stride 511. Streaming chunks execute only when they contain an engaged
target from the current date. Both modes must record context tokens and eligible targets; they
have identical stream targets in the current artifact.

Method files use `kuairand_data_utilization_v1_fixed_suffix`. Frozen/full-reuse/full-compute files
use `kuairand_data_utilization_v1_streaming_control`. These are a new protocol family and must not
be pooled with latest-only scaling-v1 cells.

`kuairand_data_utilization_v1_large_model_*_gate` is a seed-0-only top-50k/all-chunks bridge using
12 layers and hidden size 192. It is descriptive and must not be pooled with the four-seed
six-layer data-utilization summary.

### `ordered_exposure_data_audit_v1`

Data-only audit for Tenrec QK-video, Tenrec QB-video, and ZhihuRec. The first 64 raw impressions
per user fit the item vocabulary and user cohort; six later windows contain eight impressions per
user. Catalog and cohort decisions cannot use later windows. Negative rows are observed
impressions, not sampled negatives. Tenrec uses the official within-user file order and is an
ordinal replay, while ZhihuRec uses impression timestamps. This protocol contains no model
training and cannot support a maintenance-gap or generality claim.

The families may be linked through their explicit `source_run`, but their records must not be
merged as if they were the same experiment.

## 2. Shared validity invariants

- Target: hidden state at position `t` predicts item at position `t+1`.
- Training labels: only engaged targets from the currently ingested stream date contribute.
- Vocabulary: item IDs are fitted only on the protocol's frozen base period. Legacy validity
  families use 14 dates; long-context families use the base side of their recorded split.
- Padding: lengths determine valid hidden and K/V positions; padding cannot become the last state.
- Temporal order: evaluate a day before ingesting it into history or training on it.
- Evaluation positives: engaged items on the next unseen day.
- Ranking: full fitted catalog, not sampled negatives.
- Fresh serving: current model over the complete available history.
- Stale serving: prefix K/V produced by an older model, followed by the latest behavior token under
  the current model.
- Parity: current-model fresh prefix plus latest-token incremental execution must match the complete
  fresh forward within numerical tolerance.

Old full-history stale-K/V forwards, same-position reconstruction targets, padded `h[:, -1]`, and
future-fitted item vocabularies are invalid and must not be reintroduced.

## 3. Current motivation setup

- Dataset: KuaiRand-1K standard logs.
- Time split: 14 base dates followed by 17 stream dates.
- Stream update: three dates per window, two epochs per date, at most five windows.
- Model: three-layer simplified HSTU, hidden 128, four heads, head dimension 32, ReLU pointwise
  attention, sequence length 128, 5,000-item fitted catalog, 908,160 parameters.
- Base training: six epochs, AdamW learning rate `3e-4`.
- Stream training: AdamW learning rate `1e-4`.
- Evaluation: at most 300 eligible users per seed, batch size 32.
- Replication: training seeds 0, 1, 2, and 3.

Staleness modes:

- `one_step`: cache from the version immediately before the latest three-date update;
- `cumulative_theta0`: fixed-input prefix cache from the base model, consumed by later current
  versions. This is a controlled cache-age stress test, not an organic mixed-version rollout.

## 4. Current method setup

The original compact paper-facing method result uses the six-layer run:

- hidden 96, six layers, four heads, head dimension 24;
- sequence length 128, 5,000 items, 770,496 parameters;
- the same temporal split, training settings, 300-user limit, and seeds 0-3;
- cumulative theta-0 pairs at current versions 1, 3, and 5.

The original configurations are `reuse`, `cheap_all`, `cheap_plus_topN_full`, and `recompute`.
The optimized configurations use one-based names such as `interval_l5_l6`: full current blocks
run through all interval layers except the terminal layer, which runs only current
`Norm + Wk/Wv`. `[L1,L6]` must match current-model full prefix K/V recomputation. Arbitrary
intervals are an oracle ablation; the retained method remains the deepest suffix.

The three-layer `layerwise_seed*` family is a correct sanity run, not the main method table.
The strongest current KuaiRand scale table is the separate top-50k/all-chunks six-layer protocol
in Section 5; it does not retroactively replace the original run's protocol or artifacts.

The current cross-dataset method result is the compiled low-rank family. It retains the same old
normalized-state requirement as cheap refresh, learns one shared adapter per old/current
model-version pair, and precompiles it before cache migration. It does not execute a separate
adapter model per user.

## 5. Scaling-v1 setup

KuaiRand shape and quality axes:

- six-layer active sequence lengths 32, 64, and 128 on the same theta-0/theta-5 samples;
- synthetic resident cost at lengths 16, 32, 64, 128, 256, and 512 and batches 1, 8, 32, 64, 128;
- depth 3, 6, and 9 with hidden size 96, four heads, head dimension 24, and otherwise matching
  validity-v1 training;
- controlled states `theta(alpha) = theta_0 + alpha * (theta_5 - theta_0)` for alpha 0.25, 0.5,
  0.75, and 1.0. Intermediate states are mechanistic probes rather than trained checkpoints.

MovieLens transfer setup:

- `movielens_1m_hard_v5/pilot20`, 3,883 items and 5,923 shared users;
- six layers, hidden size 96, sequence length 128, four base epochs, and two epochs per update;
- 1,000 deterministic seed-specific evaluation users, full-catalog metrics plus the provided
  20-candidate view;
- four training seeds;
- recovery ratios are omitted when the full-over-reuse gap is too small to identify.

Exact tables are in `experiments/scaling/SCALING_V1.md`. Scaling files are a separate protocol
family and must not be pooled with validity-v1 user records.

KuaiRand factorial and data-utilization setup:

- the factorial uses top-5k/128 and top-20k/256 crossed with 6L/H96 and 12L/H192, four seeds;
- data-utilization-v1 uses top-50k, length 512, 6L/H96, four seeds, and compares `latest` with
  `all_chunks` while holding stream targets fixed;
- both retain the 14 base dates, five three-day updates, two epochs per stream date, 300 users,
  full-catalog ranking, and theta-0-to-theta-5 method comparison;
- neither protocol is “full KuaiRand”: top-20k and top-50k retain 8.18% and 13.35% of standard-log
  rows respectively; the random log and KuaiRand-27K are excluded.

Exact tables are in `experiments/scaling/KUAIRAND_FACTORIAL_V1.md` and
`experiments/scaling/KUAIRAND_DATA_UTILIZATION_V1.md`.

### 5.1 Ordered-exposure reproduction

The original method-transfer family uses `ordered_exposure_reproduction_v1_all_active`:

- the vocabulary is the top 50k items fitted only on each user's first 64 raw exposures;
- the 5,000-user cohort is selected only from retained base activity, with no future-window
  filtering;
- base contains 64 raw exposures and the stream contains six consecutive 8-exposure windows;
- every exposure enters context, while only observed positive feedback is a target;
- Tenrec uses official within-user ordinal order and ZhihuRec uses impression time;
- the primary model is 6L/H96, length 128, with 6 base epochs and 2 epochs per update;
- every positive-active user is evaluated against the full retained 50k catalog;
- QB uses eight seeds, QK four, and ZhihuRec one descriptive seed;
- only reuse, cheap-all, deepest suffix-2/4/5, and full recompute are evaluated at theta-5.

The motivation-alignment follow-up retains the same exposure target, model, optimizer, base length,
window length, and seed-level inference, but uses two explicitly versioned data regimes:

- `ordered_exposure_fixed_horizon_v3` for QB: top-50k and 5,000 users with at least 112 raw
  exposures. This conditions only on activity availability, never on future feedback labels or
  model outcomes. It is a controlled fixed-cohort mechanism test, not a population estimate.
- `ordered_exposure_kuai_matched_top5k_v2` for QK: top-5k fitted from the base prefix and the
  original base-activity cohort. The catalog axis was frozen before four-seed expansion.

Both evaluate every positive-active cohort user at theta-1/3/5 and also record five cumulative
cache-age points. A valid motivation claim requires all three conditions at theta-5 to improve
BestRank over their relevant baseline and a positive seed-level relationship between cumulative
cache age and maintenance gain. Absolute BestRank changes cannot be compared across catalogs.

The fixed-endpoint matrix uses the same aligned cohorts and checkpoints but reconstructs one final
evaluation set for every stale cache version. For an oriented quality gain $G$, cross-dataset
effect size is the per-seed staleness tax

$$
\tau_G =
\frac{G(\mathrm{full\ compute},\mathrm{reuse})}
     {G(\mathrm{full\ compute},\mathrm{frozen})}.
$$

It is reported only when the within-seed streaming-value denominator is positive. The ratio is
computed before cross-seed aggregation. Raw BestRank differences and numeric cache ages are not
cross-dataset effect sizes: catalog sizes differ, and Tenrec age denotes ordinal exposure blocks
rather than KuaiRand calendar days.

These protocols are not pooled with KuaiRand user records; only seed-level signs, value partitions,
and age structure are compared. The original method results are not relabeled as measurements on
the aligned cohorts. A separate 12L/H192 single-seed gate keeps the old data settings fixed and is
a negative scale diagnostic. Exact tables and limitations are in
`experiments/exposure/ORDERED_EXPOSURE_V1.md` and
`experiments/exposure/CACHE_VERSION_MATRIX_V1.md`.

### 5.2 Long-context opportunity screen

`long_context_opportunity_v1` is a separate seed-0 screening family. It was frozen before
migration-quality evaluation:

- QB uses top-50k, 1,000 complete-horizon users, 64 base plus 12x16 raw exposures, and length 256;
- QK uses top-20k with the same raw horizon and model settings;
- both use 6L/H96, six base epochs at `3e-4`, two epochs per update, and oldest-cache theta-0 versus
  current theta-11;
- the primary acceptance interval is BestRank staleness tax 0.15-0.45 with positive streaming and
  maintenance value, at least 55% reuse retention, and one agreeing secondary metric;
- only stream learning rates `1e-4`, `2e-4`, and `4e-4` are allowed, in that order, with all other
  optimizer settings fixed.

Both datasets failed the quality gate at every allowed learning rate and therefore have no valid
migration-quality result in this family. Their resident-GPU cost measurements are valid systems
diagnostics but must not be merged with the aligned four-seed QB/QK motivation or KuaiRand method
records. Exact negative results are in `experiments/exposure/OPPORTUNITY_REGIME_V1.md`.

### 5.3 Aligned ordered-exposure method gate

`aligned_ordered_exposure_method_gate_v1` uses the accepted Section 5.1 motivation cohorts and
their unchanged checkpoints:

- fixed-horizon top-50k QB and top-5k QK at theta-0 to theta-5;
- seed 0 evaluates cheap-all, frozen deepest suffix-2/4/5, and optimized full recompute;
- a partial configuration expands only if it creates a material quality-cost point on BestRank
  without a conflicting secondary-metric failure;
- QB stops at seed 0; QK expands cheap-all, and only cheap-all, to seeds 1-3;
- seed-level method-minus-reuse and method-minus-full differences remain the inferential units.

The QK four-seed result is valid evidence for cheap projection refresh, not for the suffix family.
The unexpanded QK suffix and QB results are seed-0 gate diagnostics. They must not be pooled with
the original top-50k method-transfer family. Compact results are in
`results/exposure/aligned_method_gate_summary.json` and
`results/exposure/qk_top5k_aligned_method_summary.json`.

### 5.4 Compiled low-rank migration

`compiled_low_rank_migration_v1` uses the unchanged aligned checkpoints but creates a new,
non-overlapping user split:

- KuaiRand: 40 adapter-fit, 60 rank-probe, and 200 held-out test users;
- QB/QK: 80 adapter-fit, 120 rank-probe, and 400 held-out test users;
- split seed is `9151 + training_seed` and is independent of labels and outcomes;
- the adapter target is current-model fresh prefix K/V, while its input is the old-version cached
  normalized state used by cheap refresh;
- only valid prefix positions enter the fit; sequence lengths mask padding during migration;
- seed 0 fixes the 50% fidelity target, and seeds 1-3 may not change the rank ladder or target.

Online time is measured with CUDA events after adapter compilation. One-time target collection,
low-rank fitting, compilation, shared parameter bytes, and descriptive break-even cohort size are
reported separately. Kernel-time ratios must not include calibration in the numerator and then be
described as end-to-end cost; both components must be shown.

The primary selector signal is cache-fidelity recovery

\[
\rho_C =
\frac{E(C_{\mathrm{reuse}},C_{\mathrm{fresh}})
      -E(C_{\mathrm{method}},C_{\mathrm{fresh}})}
     {E(C_{\mathrm{reuse}},C_{\mathrm{fresh}})},
\]

where \(E\) is mean per-sample relative K/V error. The selector chooses the first action in
`cheap_prepacked -> increasing adapter rank -> recompute` whose probe \(\rho_C\) reaches 0.5.
Held-out ranking metrics evaluate the selected action; they do not select it.

Exact design and four-seed results are in
`experiments/migration/COMPILED_LOW_RANK_V1.md`.

### 5.5 Capacity-tiered migration

`progressive_prefix_replay_v1` and `cohort_tiered_migration_v1` consume the fixed-task 3x3
capacity checkpoints without retraining or changing their streaming semantics. One checkpoint per
cell is selected by the motivation-only rule in
`results/motivation_scale/design_discovery_seeds.json`; migration outcomes do not enter this
selection. These nine checkpoints are discovery units. The other three training seeds in every
cell are the 27 frozen-rule replication units.

For cohort-tiered migration:

- adapter fit, probe planning, and held-out evaluation users are disjoint;
- all rank candidates are compiled before resident-GPU timing;
- full recomputation is the cache-fidelity reference, not a guaranteed ranking upper bound;
- primary method evidence is the 50% target fixed before held-out replication;
- the 75% and 90% targets show the same frozen action library under stricter quality SLAs;
- rankings from the fit or probe users cannot be pooled into final quality results.

At the 50% target, all 27 replication runs select a compiled projection. Mean kernel cost is
`0.121 [0.112, 0.130]x` full and mean K/V recovery is `0.587 [0.547, 0.627]`. The strict
positive-mean BestRank/rank-utility gate passes 6/9 cells. Endpoint tracking is a required
diagnostic because QB-medium full recomputation is itself negative in BestRank and QK-large is
near zero. This result supports scalable operator fidelity and cost; it does not establish that
every version cohort should be admitted for migration.

Exact architecture, frozen gates, cell results, and limitations are in
`experiments/migration/COHORT_TIERED_MIGRATION_V1.md`. The compact reproducible summaries are
`results/motivation_scale/progressive_prefix_replay_v1_summary.json` and
`results/motivation_scale/cohort_tiered_migration_v1_summary.json`; bounded architecture
comparisons are in `results/motivation_scale/structural_design_discovery_summary.json`.

### 5.6 KuaiRand 8+8 long-context bring-up

The prepared cohort contains 965 users with at least five D1-D8 exposures; reaching exactly 1,000
users is not an objective. All 5,820,867 selected standard-log exposures remain in the context
stream. The prediction task uses a D1-D8-fitted top-50k catalog. Long-tail items map to 262,144
deterministic context-only buckets and cannot become positives.

The frozen HSTU is 16L/H512 with eight 64-dimensional heads, ReLU pointwise attention, an
eight-day history window, and length 2,048. It has 181,082,112 parameters. Base training and every
daily update use all chronological chunks. D9-D16 produce theta1-theta8. Theta1-theta7 are
evaluated on D10-D16 before ingestion; theta8 has no next-day endpoint inside the frozen horizon.
Four FP32 DDP workers each use logical batch four, micro-batch one, and four-way gradient
accumulation, preserving effective global batch 16. Dynamic length-bucketed dense execution is
part of the protocol.

The active motivation record is a 28-cell strict lower triangle. For each current theta-i,
`i=1..7`, it holds that model's next unseen evaluation date, histories, positives, and users fixed
while evaluating every cache theta-j with `0 <= j < i`. The theta0 column is the seven-point
moving curve; the theta7 row is the fixed-D16 endpoint. Every cell contains its own complete
current-model fresh reference, so diagonal self-pairs are omitted. MeanRank is primary, AUC is the
robust secondary, and NDCG@100/Hit@100 are standard secondaries; all other rank and top-k metrics
remain in the raw result. Adjacent cache ages are paired within each current-version row, but a
user bootstrap within one training run is diagnostic rather than independent replication.

Cache age in this controlled family is checkpoint-update distance. Every version encodes the same
deterministically tail-cropped resident prefix; it is not a claim that one physical D9 cache
snapshot survives unchanged through D16. Literal snapshot residence, rolling eviction, and
organically mixed per-token versions belong to a separate lifecycle protocol. At D16, 746 users
are eligible, median length is 2,048, and 392 users are token-truncated after applying the
eight-day window. This truncation is recorded in every result and cannot be hidden by referring
only to the model maximum.

The preliminary method uses 40 label-independent D16 fit users and fixed compiled rank 16; all
remaining users are held out. It reports CUDA-event kernel time and cache fidelity for reuse,
compiled migration, and exact recomputation. It does not include a probe-selected rank, quality
tier, admission rule, physical mixed-version rollout, transfers, or end-to-end scheduling.

The prepared artifact SHA256 is
`2db3f76992ab490802fc586d47cbc2e1b4e38e1adc45a221c123dfe159633b36`. Training JSON records this
hash, and both evaluators reject a mismatch or incomplete training record. The data-only dry run
is valid protocol evidence about counts and state size; it is not a model-quality result.

The seed-0 28-cell v3 evaluation is complete. Its only large discontinuity is the theta0
base-to-stream boundary; it does not establish a repeated non-theta0 cache-age cliff. This result
is valid evidence for the 8+8 protocol but is not pooled with alternate temporal splits.

### 5.7 KuaiRand temporal-split cliff exploration

The exploratory family supports 4+12 and 6+10 splits over the same fixed 16 dates. Each split
refits its user cohort and base-only prediction catalog, receives a distinct prepared-data,
training, and motivation protocol string, and must be reported separately from 8+8 and from one
another. Model shape, maximum sequence length, history window, all-chunks training, optimizer
hyperparameters, and four-worker execution remain unchanged.

The primary split is 4+12. D1-D4 train theta0; D5-D16 produce theta1-theta12. Theta1-theta11 are
evaluated on D6-D16 before ingestion, while theta12 has no unseen endpoint. The motivation matrix
contains all 66 strictly older-cache/current-model pairs. Besides the fixed-current rows, the
result must expose fixed-cache longitudinal trajectories: theta0 under theta1-theta11, theta1
under theta2-theta11, and so on. Every point retains the complete fresh current-model path on the
same endpoint as its reference.

The prepared 4+12 artifact contains 945 users and 5,780,499 rows and has SHA256
`e03f3e80dacf9deccd5783d26a184d8ced7b339275bf13fa3b90de42a4b028b8`. The data audit and tiny
four-GPU smokes, seed-0 theta0-theta12 training, and the 66-pair matrix are complete. The evaluator reports all raw
metrics, successive loss increments along each fixed-cache trajectory, and an exploratory late
jump after at least two earlier transitions. Adjacent longitudinal endpoints use different
next-day tasks, so this jump ratio is descriptive; the stale-versus-fresh contrast within each
point and the training seed remain the valid comparison and replication units.

A useful fixed-window counterexample must occur beyond theta0, agree directionally across the
preselected MeanRank and at least one robust or standard secondary, and reproduce across cache
cohorts or new seeds. If 4+12 has no such result, 6+10 is the only predeclared split balance check.
Further post-result split or metric search does not support a cliff claim.

### 5.8 KuaiRand 4+12 progressive synchronization design

`kuairand_long_context_4plus12_progressive_sync_design_v1` fixes the current model and evaluation
task at theta11/D16. It compares cache source theta10, theta4, and theta0, representing one, seven,
and eleven checkpoint updates, on identical histories, positives, users, and the base-only
50,000-item prediction catalog. Source versions were selected before method evaluation to cover
low, medium, and high staleness; they are not chosen per user or by realized task quality.

The frozen formal split uses a seed-9151 permutation: 40 users fit one shared rank-32
`fresh - current-projection` residual for each source/target pair, 60 users are reserved for
label-free tier calibration, and every remaining evaluable user is held out for final reporting.
The action library is:

- stale reuse;
- current projection over cached old `Norm(x)`;
- the rank-32 residual compiled into that same affine projection;
- residual-delta prefix replay at depths 4, 8, and 12;
- exact current-model prefix recomputation.

The primary deployed ladder is unconditional compiled rank-32 synchronization, fixed p8 background
refinement, and exact recomputation. P4 and p12 are same-slot ablations or replacements, not extra
system contributions. Cache version is a batching and program-compilation key; the protocol does
not infer whether a cache version is safe to reuse. No recommendation labels are used for fitting,
tier admission, or per-version action selection.

Every action reports measured resident-GPU milliseconds and ratio to exact, relative K/V error and
fidelity recovery, hidden/score cosine, top-100 overlap, all ranking metrics, signed task difference
from fresh, absolute task deviation from fresh, and signed gain over stale reuse. Full
recomputation is exactly the cache-fidelity endpoint but remains only a paired reference for
ranking quality. Evaluation worker count may vary because inference records are independently
sharded and gathered; it must be recorded and is not a statistical replication unit. The frozen
user split, action library, batch size, and all other protocol settings may not vary.

The current 24-user, three-worker run is
`kuairand_long_context_4plus12_progressive_sync_design_v1` only in implementation semantics and is
stored with `status=diagnostic_complete` and `formal_protocol=false`. It uses four fit, four
reserved probe, and 16 test users plus one timing repeat. It validates action ordering and program
serialization but is not paper evidence and cannot be pooled with the formal run.

The formal two-worker run is complete in
`long_context_4plus12_progressive_sync_design_seed0.json`. It preserves the 40/60/582 split,
batch size four, three timing repeats, source versions, and complete action library above. Worker
count is two and is recorded; users are independently sharded, so this changes wall time rather
than the estimator or statistical unit.

Three separate design-search protocols reuse the identical endpoint and split but cannot be pooled
with the frozen rank-32 family:

- `kuairand_long_context_4plus12_compiled_rank_search_v1` fits a maximum rank-512 residual once,
  truncates it to ranks `16/32/64/128/256/512`, compiles every rank into the same dense affine
  shape, and selects one global rank by mean label-free fresh-score cosine over the three probe
  cohorts. Test users remain disjoint from fit and probe.
- `kuairand_long_context_4plus12_compiled_ridge_search_v1` directly solves full-affine candidates
  at ridges `1e-5/1e-4/1e-3/1e-2/1e-1`. It is a recorded negative search: the probe-selected
  `1e-2` program does not replace the `1e-3` incumbent on test fidelity.
- `kuairand_long_context_4plus12_attention_weighted_search_v1` fixes the rank-512 `1e-3`
  incumbent and weights fit tokens by the current HSTU latest-query-to-prefix-key activation.
  Mixes `0/0.25/0.5/0.75/1` all compile to the same affine operator; the global probe selects
  mix one. The weights use fresh model internals on fit users but no recommendation labels.

All three searches require exactly two evaluation workers, the same seed-9151 user permutation,
40 fit users, 60 probe users, 582 test users, batch four, 8,192 sampled fit tokens per layer,
three timing repeats, and the theta0/theta4/theta10-to-theta11 pairs. Every program contains
8,404,992 FP32 values. Later rounds were proposed after examining earlier test results, so the
selected attention-weighted program is exploratory development evidence. It must be frozen on new
training seeds or accepted external checkpoints before any confirmatory task-quality claim.

### 5.9 KuaiRand verified cohort compiler

`kuairand_long_context_4plus12_verified_compiler_v1` fixes the selected attention-weighted
full-affine programs and adds an independent label-free certification role. The seed-9151
permutation retains 40 program-fit users and 60 earlier hyperparameter-selection users. The next
60 users are certificate users, and the remaining 522 are final recommendation-test users.
Certificate users may not fit or tune projection parameters; final users may not change the
contract, action library, or selected plan.

The fixed candidate library is current projection, compiled full affine, structural replay at p4
and p8, and exact recomputation. Stale reuse is the zero-maintenance semantic reference, not a
publishable synchronization action. For each user, certification measures:

- relative K/V error;
- score error as one minus fresh-score cosine;
- top-100 error as one minus fresh top-100 overlap.

For each error view, recovery is the reduction from stale-reuse error toward exact-current-model
error. Every publishable primary action must satisfy a frozen 70% ratio-of-means recovery target,
a one-sided 90% bootstrap recovery lower bound of at least 70%, and at least 80% qualifying-user
coverage after a one-sided 90% Wilson lower bound. It must also cost at most 0.30x exact on the
same resident-GPU timing protocol. At least 50 valid certificate users and 1,000 deterministic
bootstrap samples are required. Recommendation labels and ranking metrics are forbidden during
certification.

The compiler publishes the minimum-cost action satisfying fidelity and budget. If none does, it
publishes the minimum-cost fidelity-certified budget-overflow action. The serialized plan includes
all certificates and an ordered, fidelity-certified fallback chain terminating in exact
recomputation. A fallback is an execution recovery path, not a prediction that stale reuse is safe.

The seed-0 two-worker run is complete with `status=verified_design_complete` and
`study_stage=adaptive_seed0_exploration`. It selects compiled full affine for all three source
cohorts, then evaluates only reuse, the published action, and exact on the 522 final users. The
new certificate split prevents direct label leakage into publication, but prior design rounds
inspected this seed, so confirmation still requires a frozen new seed or accepted external
checkpoint.

### 5.10 KuaiRand real-capsule system execution

`kuairand_long_context_4plus12_progressive_sync_system_v1` consumes a serialized design program
and materializes real old-version layerwise `Norm(x)` capsules from the same theta11/D16 histories.
The hot path measures a GPU-resident capsule. The warm path starts with FP16 capsules in pinned
host memory and includes H2D transfer, packed affine execution, D2H publication of target K/V, and
pipeline synchronization. Input reconstruction and checkpoint loading are excluded and reported
separately. Cache extents, rather than the 0.181B model, are assigned across GPUs.

Formal execution requires all four A40 GPUs, at least one warmup, at least three timing repeats,
real source theta0→theta11 rank-32 weights, and the predeclared user/batch settings. It reports
absolute throughput, bytes moved, completion time, load imbalance, scaling efficiency, padding,
program replication bytes, and packed-FP16 error against the FP32 operator. Exact current-model
recomputation is measured from pinned raw history through FP16 pinned-host K/V publication. The
current exact path is synchronous; a paper-grade endpoint must additionally compare against a
similarly pipelined full-recompute implementation.

The existing 12-user run on three currently idle GPUs is explicitly diagnostic. Its 1/2/3-GPU
points and resident-operator profile verify the real data path, not four-GPU scaling or
end-to-end serving benefit.

### 5.11 KuaiRand controlled mixed-version two-GPU system

`kuairand_long_context_4plus12_two_gpu_migration_system_v2` is a separate system family. It uses
the three programs and manifests already published by
`kuairand_long_context_4plus12_verified_compiler_v1`; it may not refit, recertify, or select an
action from system users. The system script consumes no recommendation labels.

The verified compiler's 522 final users are deterministically ordered with seed `43091 + seed`.
The first 32 users are system-layout search users. The next 64 are the disjoint final system
trace. Layout search maximizes median records/s and retains a setting only when it improves
throughput or padding. The frozen migration layout is batch one, 32-token length buckets, and
three in-flight batches while preserving source-version cohort boundaries. The independently
tuned exact baseline uses batch two, 32-token buckets, and three in-flight batches and may mix
raw histories across old source versions.

The final trace assigns theta0/theta4/theta10 anchors with a deterministic 20%/30%/50% controlled
mix, producing 13/19/32 records. This validates multi-program cohort dispatch and scheduling but
is not a complete update cohort. All 64 histories and token values are real. The source
assignment is independent of recommendation labels.

The migration boundary begins with FP16 old `Norm(x)` capsules in pinned host memory and ends
after complete FP16 target K/V has been copied into persistent pinned host extents. It includes
H2D, the fused or packed operator, D2H, stream synchronization, worker dispatch, and scheduling.
Program loading, capsule construction, checkpoint loading, and program fitting are excluded and
reported as setup/state. The exact boundary begins with pinned raw histories and ends at the same
FP16 pinned-host K/V representation. Exact BF16 and FP32 executions both use the pipelined
executor, persistent outputs, independent length batching, replicated current models, and
two-GPU LPT scheduling.

Formal defaults require two CUDA devices, source versions 0/4/10, target 11, 64 final records, one
warmup, three timing repetitions, the frozen layouts above, and training seed 0. The result must
report:

- all timing samples and medians;
- resident FP32, packed FP16, and fused FP16 operator error and latency;
- input/output bytes, records/s, tokens/s, and aggregate GiB/s;
- program and model replica bytes, normalized-capsule bytes, target extent bytes, and padding;
- per-device assigned work, cohort counts, load imbalance, and 1-to-2-GPU efficiency;
- BF16-published K/V error from FP32 and finite-value validation;
- speedup against the faster independently tuned two-GPU exact implementation.

The current formal-default run has `status=adaptive_system_complete` and
`study_stage=adaptive_seed0_system_development`. Timing repetitions quantify run stability on one
machine; they are not independent training replications. The controlled mix, layout search, and
seed-0 programs prevent a confirmatory systems claim. Its current successor is the deterministic
destination-v4 full-cohort update protocol. Request arrival, hotness, routing, and foreground
serving are not inferred from the available recommendation logs.

### 5.12 KuaiRand cohort-jagged and direct-HBM operator development

`kuairand_long_context_4plus12_cohort_jagged_system_v3` is a separate adaptive system family. It
uses the same verified programs, deterministic final-user ordering, 32-user label-free search
role, disjoint 64-user final role, and controlled 13/19/32 theta0/theta4/theta10 mix as system v2.
It may not fit, certify, or select a migration action. Its purpose is to test a migration-specific
layout hypothesis rather than to change algorithm quality.

The whole-record layout concatenates valid `Norm(x)` tokens only within a common source/target
version cohort. The paged layout splits each record into fixed cache pages, assigns every page an
explicit `(record, token_start, length)` identity, and compacts pages from one version cohort into
bounded valid-token tiles. The fused operator writes separate contiguous K/V directly; no padded
K/V tensor or dense-to-page conversion is permitted after the timed operator.

Whole-record search crosses token budgets `2,048/4,096/8,192/16,384` independently at two
publication boundaries. Page search crosses page sizes `128/256/512/1,024` with tile budgets
`2,048/4,096`. Both searches maximize median valid-token throughput on the first 32 users and use
no recommendation labels. The final trace may report the selected layouts only; final-user
results cannot be used to retune them.

The two timing boundaries must remain separate:

- `host_backed` starts with pinned-host FP16 capsules and ends with complete FP16 K/V in persistent
  pinned-host extents, including H2D, compute, D2H, worker, and scheduling time;
- `hbm_direct` starts with the same pinned-host capsules and ends with complete FP16 K/V in
  preallocated target-GPU HBM extents. Destination allocation and static page metadata are
  prepared before timing, and no D2H is included.

Host-backed migration may reference the v2 pipelined exact result only after checking identical
users, source counts, logical token count, and host publication semantics. Direct-HBM migration
must not be compared against that host-publishing exact result as an equal-boundary speedup. A
future HBM exact baseline requires its own protocol record.

The result must report all timing samples, selected and rejected search layouts, logical and
allocated tokens, page table and tile fill, per-device assignment, resident packed/fused latency,
and full valid-element correctness against dense fused FP16. Search repetitions and the 64 users
are systems diagnostics, not training replications.

The completed seed-0 result is a negative performance boundary. The selected 256-page/2,048-tile
layout is exactly equal to dense fused output but does not beat one-record HBM execution on the
disjoint final trace. It may support claims about implementation feasibility, exact layout, and
destination-placement cost; it may not support a positive cohort-compaction operator claim.

### 5.13 Destination-oriented out-of-core update system

`streamkv_destination_out_of_core_v4` defines the current system architecture and publication
semantics. It is a model-update-triggered batch job, not an online request scheduler. Training,
request arrival, user hotness, request routing, and foreground serving interference are outside
this protocol. Inputs are a fixed set of old capsules, already published migration programs, an
execution-device set, and one explicit destination. The output is one complete target-version K/V
manifest.

`scripts/run_streamkv_update_coordinator.py` is a thin orchestration entry point for this
protocol. Its default plan-only mode is not an experiment and does not define another result
family. With `--execute`, it resolves declared artifacts, groups source-version cohorts, invokes
the existing v4 engine, and returns that engine's manifest and metrics. It does not compile a
program, materialize source capsules, predict reuse safety, choose a destination, or introduce a
fourth system contribution.

The destination manifest protocol is `streamkv_destination_manifest_v1`. A valid publication must:

- declare the complete expected record-ID set before execution;
- preserve every extent's migration anchor and target K/V version separately;
- stage every record exactly once and reject unknown, duplicate, or missing records;
- expose no target manifest before all extents are complete;
- expose one immutable target manifest as the publication point;
- leave no visible target version after an aborted incomplete transaction.

The current implementation supports four interface families with different timing boundaries:

- `hbm_direct`: pinned-host capsules through H2D and migration execution into preallocated K/V on
  the destination CUDA workers; no D2H is included;
- `dram_host_staged`: pinned-host capsules through H2D, migration, D2H, and committed in-memory
  target extents;
- `posix_file_host_staged`: the DRAM path plus serialization, local file writes, optional `fsync`,
  and same-filesystem version-directory publication;
- `remote_object_host_staged`: the DRAM path plus immutable object upload and a manifest object
  written last as the commit marker.

An in-memory object store validates the remote interface but is not a network experiment.
Likewise, a filesystem path validates the POSIX backend but may not be called an SSD result unless
the physical device and mount are recorded. HBM direct execution currently requires compute on
the destination GPU; P2P publication to a different GPU is outside v4.

The current reference implementation has two additional boundaries. First, host-staged execution
accepts a caller-materialized sequence of CPU capsule batches. Its wave and queue limits bound
transformed outputs and pending publication, but not the memory occupied by the complete source
capsule sequence. Second, direct-HBM execution retains the complete target K/V in HBM and currently
runs the destination path as one job rather than applying the host-staged wave limit. These are
implementation facts, not full-cohort bounded-memory results.

The v4 engine currently executes published compiled-affine `MigrationProgram` objects. Residual
replay and exact recomputation exist elsewhere in the method/runtime code but are not yet routed
through the same destination transaction. A paper-grade exact baseline must therefore be added
under the identical source, destination, layout, dtype, durability, and manifest boundary before a
v4 algorithm speedup is reported.

The reference code may use tiny synthetic tensors to test interface and transaction semantics.
Such runs must be labeled `implementation_validation` and cannot support throughput claims. A
paper-grade performance result requires real fixed-cohort capsules and records:

- source and destination location, dtype, layout, and durability semantics;
- whether source-manifest scanning, capsule construction/materialization, serialization, `fsync`,
  network acknowledgement, coordinator overhead, and manifest commit are included;
- total records/tokens, logical and physical input/output bytes, and complete record coverage;
- wave size, publication queue depth, peak HBM, peak host staging, and backpressure;
- per-device assigned bytes, completion time, throughput, and 1/2/4-GPU scaling;
- numerical K/V error and failure-injection publication visibility.

The current manifest's `payload_bytes` is a logical tensor-footprint field. It must not be used as
physical filesystem or object-store traffic without separately measuring serialized bytes and
backend write amplification.

Compiled migration and full recomputation may be compared only within the same destination
boundary. HBM, DRAM, filesystem, and remote jobs answer different endpoint questions; their raw
times may be reported as a destination matrix but not converted into an algorithm or operator
speedup. SSD and remote results require actual reproducible hardware and their own protocol
records. Details of the initial implementation contract are in
`experiments/system/DESTINATION_OUT_OF_CORE_V4.md`.

### 5.14 KuaiRand mixed-version four-GPU scaling

`kuairand_long_context_4plus12_mixed_version_four_gpu_scaling_v1` is a frozen-layout systems
follow-up to v2/v3. It reuses the disjoint 64-user v3 final trace, theta0/theta4/theta10 counts
13/19/32, the three published verified programs, and the v3-selected 2,048-valid-token jagged
layout. It may not fit or certify a program, search a layout, inspect recommendation labels, or
change the final users.

Formal defaults use cuda:0–3, device counts 1/2/4, one warmup, five timing repetitions, three
in-flight batches, and greedy-LPT assignment. It reports three separately labeled paths:

- fused compiled migration from pinned-host capsules to persistent pinned-host FP16 K/V;
- fused compiled migration from pinned-host capsules to preallocated target-GPU HBM K/V;
- independently pipelined BF16 full recomputation from pinned histories to persistent pinned-host
  FP16 K/V.

Only the first and third paths share an endpoint and may be used for migration-versus-exact
speedup. HBM/DRAM completion-time ratios describe different destinations. The result must report
all timing samples, per-device assigned work and execution time, load imbalance, program/model
replica bytes, persistent target bytes, CUDA peak allocated/reserved memory, and 1/2/4 speedup and
efficiency.

The completed seed-0 run contains 98,252 valid tokens. Host-staged migration, direct-HBM
migration, and host-staged BF16 exact reach 1→4 speedups of 3.275x, 3.331x, and 3.592x,
respectively. Four-GPU assigned-work imbalance is at most 0.30%; timing coefficients of variation
are at most 1.10%. At the common host boundary, compiled migration remains 10.39x faster than
BF16 exact on four GPUs. These are controlled adaptive systems results, not independent training
replications or destination-v4 full-cohort evidence.

The separate tiny destination-v4 HBM validation distributes four extents across cuda:0–3,
commits one complete manifest, and has maximum absolute error `9.77e-4`. It is interface
correctness only. Details and commands are in
`experiments/system/FOUR_GPU_SCALING_V1.md`.

### 5.15 CohortKV single-configuration full-chain development

`cohortkv_single_config_full_chain_development_v1` is the only protocol for the current
single-configuration implementation round. Its Stage 0 contract is frozen in
`configs/cohortkv_single_config_v1/` and documented by
`experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md`. The blueprint and workload
manifest are plan/configuration artifacts, not empirical evidence.

The frozen data/model endpoint is KuaiRand 4+12, training seed 0, theta11/D16, 16 layers,
hidden/K/V width 512, and maximum history 2,048. The complete workload contains 682 unique
records and 1,087,785 valid prefix tokens. Record IDs are the positions of eligible one-based
prepared model user indices sorted ascending; each record separately retains its source-log raw
user ID. All source state and target K/V cover `history[:-1]`, while `history[-1]` remains the
current-model latest token. The existing seed-9151 and seed-27183 permutations retain disjoint 40
fit / 60 program-selection / 60 certificate / 522 final-test roles.

Every one of the 682 records enters the system job after method and layout selection is frozen.
The system path never reads recommendation labels. Source versions theta0/theta4/theta10 are a
predeclared, label-free controlled mix: largest-remainder counts from weights 20/30/50, followed
by a seed-58211 shuffle over canonical record order. The exact counts are 136/205/341. This is a
complete real-record workload with controlled anchors, not an organic cache-version trace.

All compared source representations use buffered POSIX shards on the `/data` ext4 tier. One
complete untimed warmup precedes three measured repetitions without explicit page-cache eviction,
and source reads remain inside job completion. The representation is action-specific and its
logical and physical bytes must be reported:

- compiled and cheap projection read FP16 normalized-state capsules;
- selective-contiguous reads old FP16 K/V, its selected FP16 transition hidden state, and raw
  history;
- residual-p reads raw history plus every old pre-block hidden state from layer `p` through the
  final layer;
- exact reads raw history;
- reuse/no-transform reads old FP16 K/V.

These paths share a physical source tier, not identical inputs. Source-shard creation, checkpoint
loading, and offline tuning are excluded from job completion and reported separately. The
residual-p hidden suffix is auxiliary state, not part of the default normalized capsule. It costs
12.45 GiB at p4 or 8.30 GiB at p8 over the full workload; the current theta0/theta10 p8 fallback
scope costs 5.83 GiB. If that state is absent, residual-p is not executable and a revised verified
plan must fall through to exact. Shard materialization requires a source-device/filesystem check
and at least 128 GiB free.

The earlier verified compiler result used in-memory FP32 layerwise state. Before the plan is
executable in this protocol, the unchanged certificate must pass again on serialized FP16 source
representations, prepared runtime programs, and FP16 output. This is not a new hyperparameter
search. Transport/layout correctness uses the same selected numeric path on the same serialized
input as its resident oracle, requires finite values, and uses `atol=0.02, rtol=0.02`; semantic
recovery remains measured against FP32 current-model exact K/V and score views.

The closest external baseline is `selective_contiguous`, an HSTU adaptation of DroidSpeak's
contiguous recomputation-group and transition-state semantics. For each
`m in {2,4,6,8,12}`, all legal contiguous intervals are evaluated on the 60 program-selection
records using the common label-free cache/score/top-100 views. Ties prefer lower measured GPU cost
and then the earlier start. The certificate role publishes the minimum-cost frozen interval that
passes the primary contract, with exact fallback. Final-test records cannot change the interval,
`m`, action, or contract. This is an adapted algorithm baseline, not a reproduction of
DroidSpeak's distributed serving runtime. The existing `migrate_contiguous_cache` helper is not
the reference implementation because it applies current projections outside the interval; the
baseline must instead copy source old K/V there. Candidate transition states are profiling-only on
program-selection records, while final system shards retain one frozen transition per cohort.
The frontier is complete per source-target pair: 53 selective intervals, p4/p8, compiled, cheap,
reuse, and exact, or 59 selection points per pair and 177 total. The aggregate must audit every
declared interval; selection and certification do not pool source versions.

The primary destination matrix is compiled/selective-contiguous/exact over HBM and pinned DRAM at
1/2/4 GPUs. Residual-p and no-transform are controls. Every method publishes the same contiguous,
unpadded, FP16 K/V extent layout with lengths/offsets and the same
`streamkv_destination_manifest_v1` coverage/visibility contract. Fresh target allocation,
source-manifest scan/read, decode/pinning, H2D, compute, D2H when required, staging, coverage
validation, coordinator overhead, and commit are inside completion time. HBM and DRAM remain
different endpoints. The one-GPU HBM point retains 33.20 GiB of target K/V and must pass a
pre-timing capacity check including model/program residency, maximum-batch temporary memory, and
allocator margin. An infeasible point triggers protocol revision rather than silent omission.
The DRAM path similarly probes pinned allocation and available host memory for its retained target
plus bounded queues. Each repetition destroys the previous target, allocates a fresh destination,
and reopens and decodes source shards; only the OS page cache, not decoded tensors, remains warm.
All methods use byte-weighted LPT with method-specific declared logical source plus target bytes.
Per-GPU record/token/byte totals, elapsed time, peak HBM, assigned-byte imbalance, and aggregate
source/staging/publication-queue peaks are required and must sum to the run aggregate.

Runtime tuning uses only the 60 program-selection records and is independent per method and
destination/GPU count. The frozen grid is batch size `1/2/4`, length bucket `16/32/64`, and
in-flight depth `2/3/4`; compiled additionally compares packed and fused FP16, and exact compares
BF16 and FP32 compute. Correctness precedes timing. After one complete source read establishes the
warm page-cache condition, every legal candidate receives one screen pass in seed-73421 order,
then the fastest three receive one warmup and three measured passes; every candidate result is
retained, and ties prefer lower peak memory then lower padding. The point-specific winner is frozen
before the complete workload.

Guard selection uses only program-selection records and no recommendation labels. It chooses the
lowest normal-job overhead mechanism that detects the frozen theta4 perturbation while preserving
unperturbed cohort certificates, and records reference bytes/time, no-fault overhead, false
escalations, and detection phase. If no runtime sentinel qualifies, an executable semantic
preflight is permitted and the claim is renamed before the mechanism is frozen.

Failure experiments predeclare artifact-hash mismatch, a structurally valid and
integrity-accepted semantic theta4-program perturbation, failure before the first extent, mid-wave,
during publication, and immediately before commit after complete coverage. The semantic
perturbation must be caught by semantic preflight or the runtime guard, escalate theta4 to its
published exact fallback, replace any already generated theta4 extents, and commit only a complete
corrected target. The other five jobs abort with the old current pointer preserved. No failure may
expose partial theta11. Each result records pointer/visibility state, staging reclamation, cleanup,
detection phase, and reworked records. Resume is optional until a journal validates
at-most-one-wave redo; atomic abort and prior-version visibility are mandatory.

The aggregate result must validate against
`configs/cohortkv_single_config_v1/result.schema.json`. Until that artifact exists with
`status=development_complete`, none of the Stage 0 files support a new speedup, fidelity,
full-cohort, failure-recovery, or capsule-economics claim. Even after completion, seed 0 remains
adaptive development evidence; timing repeats are not training replications.
The schema requires exactly one aggregate run for each of the 18 primary
method/destination/GPU-count combinations and one result for each predeclared failure. Controls
cannot satisfy a missing primary point.

RQ5 capture timing uses theta11, one GPU, and the 60 program-selection histories for matched
fresh-K/V-only, plus-device-capture, and plus-D2H/encode/buffered-POSIX-persist paths, each with one
warmup and three repetitions. INT8 is symmetric signed quantization with a per-record/per-layer
FP32 absmax scale and timed FP16 dequantization during staging; its frozen certificate is applied
on certificate users, with a complete 682-record one-GPU HBM run. Time break-even is
`ceil(capture_overhead / (exact - compiled - compiler_amortized))`; a nonpositive denominator is
reported as no break-even. Auxiliary transition/residual state is a separate row and cannot be
folded into the default capsule ratio.

## 6. Metrics and statistics

Primary quality views:

- Best Rank and Mean Rank: lower is better; gains are `reuse - method`.
- MRR, NDCG@10/100, Hit@10/100: higher is better; gains are `method - reuse`.
- Quality recovery: method gain divided by fresh-recompute gain, only when the denominator is
  sufficiently different from zero.
- Cache-fidelity recovery: relative reduction in K/V reconstruction error from stale reuse toward
  exact fresh K/V; unlike task-quality recovery, its full-recompute endpoint is exactly one.
- Staleness tax: cache-maintenance gain divided by full-compute streaming-training gain, computed
  within seed only when that denominator is positive.
- Full recompute is the cache-fidelity reference, but not a guaranteed upper bound on a ranking
  metric. Report paired method-minus-full quality differences whenever recovery is shown.

Statistical rules:

- training seed is the replication unit;
- aggregate users within a seed/window before cross-seed inference;
- methods on the same seed/window use paired differences;
- user-level bootstrap or correlations are descriptive diagnostics only;
- “interval contains zero” means the current run did not distinguish methods, not equivalence;
- recovery above 100% can arise from task-metric variation because full is a consistency reference,
  not a ranking upper bound; it is not a superiority result without paired multi-metric evidence.

## 7. Cost protocol

- Use CUDA events for GPU-resident batched migration and synchronize correctly.
- Normalize every configuration to full prefix K/V recomputation on the same batch and prefix.
- Report the measured ratio and absolute time; never use a hand-assigned projection cost.
- Report extra normalized/split-hidden state as both elements/bytes and a ratio to K/V capacity.
- For a calibrated operator, report adapter-fit size, one-time fit/compile time, shared parameter
  bytes, and an amortized or break-even cohort calculation separately from per-cache kernel time.
- Resident-kernel result families exclude host-device transfer, allocator, cache admission, and
  scheduler overhead and must remain labeled kernel-level. The two-GPU system-v2 family includes
  pinned host-device transfer, publication, worker, and scheduling overhead within its declared
  boundary, but still excludes lifecycle admission, storage below host DRAM, and foreground
  serving.
- Destination-v4 results include exactly the source-to-committed-manifest stages declared for
  their backend. Cross-destination timing differences are endpoint costs, not operator speedups.
  Filesystem and remote interface validation without identified physical I/O hardware remains a
  correctness artifact.
- Plan-only coordinator output is architecture metadata, not a timing or correctness result. If a
  later end-to-end protocol includes coordinator or source-reader overhead, that boundary must be
  declared symmetrically for compiled migration and exact recomputation.

## 8. Current artifacts

Motivation:

- `results/validity/core_seed{0,1,2,3}.json`
- `results/validity/multiseed_summary.json`
- `checkpoints/validity/core_seed{0,1,2,3}/theta_*.pt`

Six-layer method:

- `results/validity/core6l_seed{0,1,2,3}.json`
- `results/validity/layerwise6l_seed{0,1,2,3}.json`
- `results/validity/layerwise6l_multiseed_summary.json`
- `checkpoints/validity/core6l_seed{0,1,2,3}/theta_*.pt`

Three-layer sanity:

- `results/validity/layerwise_seed{0,1,2,3}.json`
- `results/validity/layerwise_multiseed_summary.json`

Terminal optimization and interval validation:

- `results/validity/interval_oracle_seed0.json`
- `results/validity/interval_validation_seed{1,2,3}.json`
- `results/validity/interval_validation_summary.json`

Streaming-training value control:

- `results/validity/streaming_control6l_seed{0,1,2,3}.json`
- `results/validity/streaming_control6l_summary.json`

Scaling-v1:

- `results/scaling/operator_cost_seed0.json`
- `results/scaling/sequence_length_seed{0,1,2,3}.json`
- `results/scaling/update_magnitude_seed{0,1,2,3}.json`
- `results/scaling/depth{3,9}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/movielens_seed{0,1,2,3}.json`
- `results/scaling/multiaxis_summary.json`
- `results/scaling/kuairand_data_coverage.json`
- `results/scaling/factorial_{more_data,larger_model,both}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/kuairand_factorial_summary.json`
- `results/scaling/top50k_{latest,all_chunks}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/top50k_{latest,all_chunks}_streaming_control_seed{0,1,2,3}.json`
- `results/scaling/kuairand_data_utilization_summary.json`
- `results/scaling/top50k_all_chunks_large_{core,method}_seed0.json`
- `results/scaling/top50k_all_chunks_large_streaming_control_seed0.json`
- `checkpoints/scaling/depth{3,9}_seed{0,1,2,3}/theta_*.pt`
- `checkpoints/scaling/movielens_seed{0,1,2,3}/theta_*.pt`
- `checkpoints/scaling/factorial_*_seed{0,1,2,3}/theta_*.pt`
- `checkpoints/scaling/top50k_{latest,all_chunks}_seed{0,1,2,3}/theta_*.pt`
- `checkpoints/scaling/top50k_all_chunks_large_seed0/theta_*.pt`

Dataset audits:

- `results/taobao/data_audit.json`
- `results/taobao/kuairand_matched_comparison.json`
- `results/dataset_audit/tenrec_qk.json`
- `results/dataset_audit/tenrec_qb.json`
- `results/dataset_audit/zhihurec.json`
- `results/dataset_audit/*_top50000_users5000_prepared.json`

Ordered-exposure reproduction:

- local `results/exposure/qb_{core,method,streaming_control}_seed{0..7}*.json`
- local `results/exposure/qk_{core,method,streaming_control}_seed{0..3}*.json`
- local ZhihuRec and 12L/H192 gate files under `results/exposure/`
- `results/exposure/{qb,qk,zhihu}_streaming_control_summary.json`
- `results/exposure/{qb,qk}_method_summary.json`
- `results/exposure/aligned_method_gate_summary.json`
- `results/exposure/qk_top5k_aligned_method_summary.json`
- `results/exposure/{kuai,qb_fixed_horizon,qk_top5k}_allages_streaming_control_summary.json`
- `results/exposure/cache_age_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_{cross_dataset,fine_cross_dataset}_summary.json`
- `results/exposure/long_context_opportunity_summary.json`
- `results/exposure/{qb_horizon256,long_context}_operator_cost_seed0.json`
- local `results/exposure/*cache_version_matrix_seed*.json`
- local `checkpoints/exposure/`

Compiled low-rank migration:

- local `results/validity/kuai_low_rank_migration_seed{0..3}.json`
- local `results/exposure/qb_fixed_horizon_low_rank_migration_seed{0..3}.json`
- local `results/exposure/qk_top5k_low_rank_migration_seed{0..3}.json`
- `experiments/migration/COMPILED_LOW_RANK_V1.md`

Capacity-tiered migration:

- local `results/motivation_scale/*_prefix_replay_seed*.json`
- local `results/motivation_scale/*_cohort_tiered_{discovery_,}seed*.json`
- `results/motivation_scale/progressive_prefix_replay_v1_summary.json`
- `results/motivation_scale/cohort_tiered_migration_v1_summary.json`
- `results/motivation_scale/structural_design_discovery_summary.json`
- `experiments/migration/PROGRESSIVE_PREFIX_REPLAY_V1.md`
- `experiments/migration/COHORT_TIERED_MIGRATION_V1.md`

KuaiRand 8+8 long-context bring-up:

- local `data/processed/kuairand_long_context_8plus8_v2.npz`
- `experiments/motivation/LONG_CONTEXT_8PLUS8_V2.md`
- local `checkpoints/kuairand_long_context_8plus8/seed0/theta_{0..8}.pt`
- local `results/motivation_scale/long_context_8plus8_training_seed0.json`
- local `results/motivation_scale/long_context_8plus8_motivation_seed0.json`
- local `results/motivation_scale/long_context_8plus8_motivation_all_pairs_seed0.json`
- pending local `results/motivation_scale/long_context_8plus8_method_seed0.json`

KuaiRand temporal-split exploration:

- local `data/processed/kuairand_long_context_4plus12_exploration_v1.npz`
- `experiments/motivation/LONG_CONTEXT_SPLIT_EXPLORATION_V1.md`
- local `checkpoints/kuairand_long_context_4plus12_exploration/seed0/theta_{0..12}.pt`
- local `results/motivation_scale/long_context_4plus12_training_exploration_seed0.json`
- local
  `results/motivation_scale/long_context_4plus12_motivation_all_pairs_exploration_seed0.json`
- diagnostic local
  `results/motivation_scale/long_context_4plus12_progressive_sync_design_diagnostic_seed0.json`
- local
  `results/motivation_scale/long_context_4plus12_progressive_sync_design_seed0.json`
- local
  `results/motivation_scale/long_context_4plus12_compiled_rank_search_seed0.json`
- local
  `results/motivation_scale/long_context_4plus12_compiled_ridge_search_seed0.json`
- local
  `results/motivation_scale/long_context_4plus12_attention_weighted_search_seed0.json`
- `experiments/migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md`
- local
  `results/motivation_scale/long_context_4plus12_verified_compiler_seed0.json`
- `experiments/migration/VERIFIED_COHORT_COMPILER_V1.md`
- local
  `checkpoints/kuairand_long_context_4plus12_exploration/seed0/verified_plans/theta{0,4,10}_to_theta11_verified.json`
- diagnostic local
  `results/system/kuairand_long_context_4plus12_progressive_sync_system_diagnostic_seed0.json`
- local
  `results/system/kuairand_long_context_4plus12_progressive_sync_system_seed0.json`
- local
  `results/system/kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json`
- `experiments/system/TWO_GPU_MIGRATION_SYSTEM_V2.md`
- local
  `results/system/kuairand_long_context_4plus12_cohort_jagged_system_seed0.json`
- `experiments/system/COHORT_JAGGED_SYSTEM_V3.md`
- local
  `results/system/kuairand_long_context_4plus12_four_gpu_scaling_seed0.json`
- local
  `results/system/streamkv_destination_hbm_4gpu_validation.json`
- `experiments/system/FOUR_GPU_SCALING_V1.md`
- `experiments/system/DESTINATION_OUT_OF_CORE_V4.md`
- `configs/cohortkv_single_config_v1/{blueprint,workload_manifest,result.schema}.json`
- `experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md`

`smoke.json` files, old `results/phase0`, and old `results/streaming` are not research artifacts.
Per-seed exposure JSON and all checkpoints are current local artifacts but ignored by Git. Taobao
UserBehavior is an action-only semantic boundary rather than the selected next stream.
