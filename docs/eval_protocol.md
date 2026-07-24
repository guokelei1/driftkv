# Current evaluation protocol

> This file defines the valid artifact boundary. Any material change to targets, data split,
> serving semantics, model family, or timing semantics requires a new protocol name and separate
> result files.

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
- Vocabulary: item IDs are fitted on the 14-day base period only.
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
- Current numbers exclude host-device transfer, allocator, cache admission, and scheduler overhead;
  they must be labeled kernel-level rather than end-to-end serving cost.

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

`smoke.json` files, old `results/phase0`, and old `results/streaming` are not research artifacts.
Per-seed exposure JSON and all checkpoints are current local artifacts but ignored by Git. Taobao
UserBehavior is an action-only semantic boundary rather than the selected next stream.
