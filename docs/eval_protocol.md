# Current evaluation protocol

> This file defines the valid artifact boundary. Any material change to targets, data split,
> serving semantics, model family, or timing semantics requires a new protocol name and separate
> result files.

The frozen D1 Stage 0–6 evidence ledger is
[`09_single_configuration_full_chain_plan.md`](09_single_configuration_full_chain_plan.md).
The D2 mechanism and live execution/evidence state are
[`future_design/DESIGN2_FINAL_PLAN.md`](future_design/DESIGN2_FINAL_PLAN.md) and
[`future_design/DESIGN2_DEVELOPMENT_STATUS.md`](future_design/DESIGN2_DEVELOPMENT_STATUS.md).
The D3 design-ready problem and exploration contract are
[`future_design/DESIGN3_FUTURE_DIRECTION.md`](future_design/DESIGN3_FUTURE_DIRECTION.md), with the
development execution route in
[`future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md`](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md).
These documents create no comparable evidence by themselves. D2 Stage A/B diagnostics and all
initial D3 mechanism work must be labeled as development artifacts. The current D2 execution uses a
strict double gate:
`stage_c_development_entry=go` has admitted and completed sample-only C0 wiring on W1/W2/W3 normal
plus W3 pre-commit abort with `scientific_result=false`, while
`stage_c_evaluation_entry=blocked` forbids formal Stage-C integrated evaluation and every paper
claim until the independent four-A40 W4 normal/hard-failure gate closes and
`stage_b_summary.json --check` passes. W3, C0, shared-GPU logical ranks, or Gloo cannot substitute
for W4. C0 uses a fixed 16-record fixture and records `formal_stage_c=false`, no timing/full-cohort
claim, `target_epoch_published=false`, and no capacity evidence; its development namespace pointer
is not a formal epoch publication. These artifacts are categorically ineligible for the Stage-B
summary.

Separate W3 mechanism-development families now include the integrated v1–v5 pilot/full682 runs,
full-payload validation, wave-embedding characterization, and the synthetic lookup contention
probe recorded in `DESIGN2_DEVELOPMENT_STATUS.md`. They may contain real full-cohort timings, but
remain `scientific_result=false` and `formal_stage_c=false`. They do not satisfy W4, freeze a
formal D2 protocol, publish a formal target epoch, close the segmented consumer/next-wave boundary,
or include the final plan/history preparation plus publication/commit/reclaim boundary. Their
valid use is mechanism discovery and protocol design, not Motivation-2 numbers or paper tables.

Design 3 has a frozen problem/exploration contract but no executable D2 handoff, implementation,
protocol family, or result. Its first readiness artifact must normalize, validate, serialize, and
hash the capacity-independent D2 owner/operator/compatibility/dependency/layout constraints; the
current single-rank wave adapter and capacity-specific W3 extents are not that artifact. A future
D3 protocol requires an ordinary-host-DRAM source and private target, bounded pinned staging,
per-rank usable-HBM admission, a fixed D1 ActionPlan, one common D3-facing D2 constraint hash, and
separate ResidencyPlans. Sequential capacity groups and an action-oblivious double buffer must
share the mixed action-required source-byte multiset; same-boundary all-exact reports its distinct
raw-history bytes. No existing normalized-capsule HBM/DRAM, destination-v4, or hot-HBM Stage-4.5
result may be relabeled as direct-old-K/V D3 evidence. H12 runs under an artificial per-rank HBM
cap are development-only capacity emulation; paper-facing physical oversubscription requires a
larger real-history workload.

Before any formal D2 integrated run or paper-facing D2 result is promoted, its action-plan
identity, configuration, metrics, timing boundary, communication accounting, baselines, and
artifact schema must be frozen here under a new D2 protocol name. A paper-facing D3 result
similarly requires a separate D3 protocol after the common D2 constraint exporter closes.
Existing protocol strings and result families must not be silently reused. The live
non-scientific D2 status and pending commands are recorded in
[`future_design/DESIGN2_DEVELOPMENT_STATUS.md`](future_design/DESIGN2_DEVELOPMENT_STATUS.md).

## 1. Protocol families

### D2 development families

D2 development is not one wildcard protocol. Directory names are storage organization and must
not be quoted as protocol strings. The current top-level artifact protocols include:

- `cohortkv_d2_stage_a_frozen_v1` and its separately named Stage-A characterization inputs;
- `cohortkv_d2_stage_b_distributed_primitives_v1`;
- `cohortkv_d2_stage_b_hard_failure_v1`;
- `cohortkv_d2_dev_w3_distributed_diagnostic_v1`;
- `cohortkv_d2_dev_w3_hard_failure_v1`;
- `cohortkv_d2_dev_c0_wave_v1` and aggregate `cohortkv_d2_dev_c0_status_v1`;
- `cohortkv_design2_integrated_w3_development_v1` through
  `cohortkv_design2_integrated_w3_development_v5`;
- `cohortkv_design2_integrated_full_payload_validation_v1`;
- `cohortkv_d2_embedding_resource_isolation_development_v1`.

Every artifact must retain its exact serialized protocol string and negative claim flags.
Different families and versions record implementation discovery and must not be pooled into a
confirmatory timing distribution.

The formal paper-facing families are not yet frozen. They will require separate names for:

1. a design-independent logical-to-physical Motivation-2 experiment; and
2. a fixed-action D2 physical-wave evaluation.

The first varies predeclared exact volume and shard count without using D1 outcomes as its
observation. The second fixes the D1 action hash and compares strong all-exact, naive sharded
fixed-action mixed, and D2 physical-sparse mixed. Neither family exists merely because this plan is
written; final names, hashes, timer, artifact schema, and run matrix must be frozen after W4.

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

### `cohortkv_single_config_organic_lifecycle_v1`

This is the one-seed, one-model, one-GPU growing-history lifecycle protocol for the KuaiRand 4+12
chain. It begins with exact theta0 K/V and recursively consumes the previous mixed cache over all
11 adjacent model updates. At endpoint `v`, theta-v history includes only the base and previously
completed canonical date partitions, and both mixed and all-exact predict the next unseen date
partition. The final policy refreshes approximately 20% of reusable continued prefixes using a
depth-four deadline followed by migration age and current-edge label-free K/V norm shift; cold,
re-entered, and zero-overlap prefixes are separate natural exact work.

The sole frozen output is
`results/system/cohortkv_single_config_full_chain_v1/stage4_7_organic_full_chain_seed0.json`.
`configs/cohortkv_single_config_v1/stage4_7_organic_summary.json` binds that ignored local result,
the compiler result, and the exact uncommitted implementation content hashes.
Its implementation and execution checks pass, but minimum q90 cache fidelity is `0.8744` against
the predeclared `0.90` gate. This family is completed mixed development evidence, not a passing
certificate, a strict raw-event/request-time trace, a selector-optimality result, or a replicated
claim. Exact settings, results, and the raw timestamp boundary are in
`experiments/system/COHORTKV_STAGE4_7_ORGANIC_LIFECYCLE_V1.md`.

### `cohortkv_single_config_stage4_9_rollout_boundary_v1`

This is the completed correction for the growing-history model-rollout boundary. It freezes two
independent rules: newly admitted behavior is appended
after prefix migration with the target model, while the GPU cost of that append is foreground
inference and is excluded from both the migration numerator and its paired exact denominator. The
primary cost is matched retained-prefix `migration-or-refresh / exact`, not a
foreground-inclusive lifecycle ratio. Stage-4.7/4.8 artifacts keep their recorded pre-migration
old-model append semantics and cannot be relabeled as this protocol. The formal v2 confirmation
runs the two frozen candidates sequentially on one A40 over all 11 recursive edges. It freezes
`staggered_renewal_h12` as the bounded-renewal deployment candidate and retains
`token_debt_total10` as the cost endpoint. Exact order, accounting, results, and scope boundaries
are in
`experiments/system/COHORTKV_STAGE4_9_ROLLOUT_BOUNDARY_V1.md`.

### `cohortkv_stage4_10_renewal_calibrated_h12_smoke_v1`

This is a non-scientific implementation successor to the frozen Stage-4.9 program lifecycle. H12
fixes the edge action partition first. The scheduled-exact cohort then supplies aligned
previous-actual K/V and current-model exact K/V pairs for one shared adjacent program, and the
same exact outputs refresh those records. No separate fit user is recomputed, no old serialized
program is loaded, and no semantic or task-quality gate changes the action partition. Program
fit, compile, FP16 cast, and prepare enter `U`; state movement and target-model append remain
separate ledgers.

The real smoke covers `theta0 -> theta1 -> theta2` for inverse-Norm ridge and direct-K/V residual
ridge with one repetition and no warmup. Both artifacts are `scientific_result=false`; neither
their timing nor the absence of nonfinite outputs selects a fit form or supports a task-quality
claim. The exact contract, artifacts, and open formal gate are in
`experiments/system/COHORTKV_STAGE4_10_RENEWAL_CALIBRATED_V1.md`.

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

This historical structural-baseline family freezes the optimized deepest suffix and changes one
KuaiRand axis at a time. Sequence length, model depth, and controlled parameter interpolation use
the same full-catalog target semantics as validity-v1. Depth 3/6/9 and every reported quality point
use seeds 0-3. No interval is selected from these result cells. The suffix is not the current D1
method.

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

### 3.1 Paper-facing unified motivation

The legacy setup above is one M1 staleness protocol, not the complete paper motivation. The
paper-facing logic has two separate evidence families:

1. **M1 semantic–compute dilemma.** Existing streaming-value, opportunity-regime,
   data/model-capacity, compiled-migration, and lifecycle protocols show that stale reuse preserves
   useful streaming value but leaves a recoverable version-consistency gap, while all-exact replays
   complete histories. Their metrics remain within their own protocol families and are not pooled.
2. **M2 logical-to-physical gap.** A new protocol must characterize exact maintenance under
   row-sharded embeddings before invoking D1: predeclared real-history exact-volume points,
   matched owner/layout/endpoint, shard-count sweep, and local/replicated exact control. It reports
   logical tokens alongside physical communication, padding, rank wait, and wave time. It does not
   require request-arrival or serving traces.

M2 may state that communication volume is large when bytes support it. It may state that
communication dominates only after matched timing attribution supports that percentage. A
hypothetical “20% exact work takes 40% time” is an experiment target, not a frozen result.

The bridge from motivation to design is evaluated separately with one immutable D1 action plan:

```text
strong all-exact
  → naive row-sharded fixed-action mixed
  → D2 wave-compiled segmented mixed
```

D1 decides what is compiled or exact. D2 preserves those requested actions and fixes owners,
operators, compatible physical pools, collective dependencies, segmented layout, and publication
semantics. A resident D2 executor may choose a legal batch/order/stream; capacity-specific cuts,
packing, and DRAM/HBM launch order belong to a D3 ResidencyPlan. Communication cost never changes
semantic action selection.

## 4. Historical predecessor-method setup

The original compact predecessor result uses the six-layer run:

- hidden 96, six layers, four heads, head dimension 24;
- sequence length 128, 5,000 items, 770,496 parameters;
- the same temporal split, training settings, 300-user limit, and seeds 0-3;
- cumulative theta-0 pairs at current versions 1, 3, and 5.

The original configurations are `reuse`, `cheap_all`, `cheap_plus_topN_full`, and `recompute`.
The optimized configurations use one-based names such as `interval_l5_l6`: full current blocks
run through all interval layers except the terminal layer, which runs only current
`Norm + Wk/Wv`. `[L1,L6]` must match current-model full prefix K/V recomputation. Arbitrary
intervals are an oracle ablation. The deepest suffix was retained within this historical protocol;
it is not the current D1 method.

The three-layer `layerwise_seed*` family is a correct sanity run, not the main method table.
The strongest table within this predecessor family is the separate top-50k/all-chunks six-layer
protocol in Section 5; it does not retroactively replace the original run's protocol or artifacts.

The compiled low-rank family is the historical cross-dataset successor. It retains the same old
normalized-state requirement as cheap refresh, learns one shared adapter per old/current
model-version pair, and precompiles it before cache migration. It does not execute a separate
adapter model per user. The current D1 route adds tiered actions, direct-old-K/V execution, renewal,
and exact fallback; those later protocols remain separate.

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

The frozen formal-default run has `status=adaptive_system_complete` and
`study_stage=adaptive_seed0_system_development`. Timing repetitions quantify run stability on one
machine; they are not independent training replications. The controlled mix, layout search, and
seed-0 programs prevent a confirmatory systems claim. Its historical successor was the
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

`streamkv_destination_out_of_core_v4` is a historical normalized-capsule destination prototype and
publication protocol. It no longer defines the current EvoKV architecture and is not the new D3
ordinary-DRAM direct-old-K/V protocol. It is a model-update-triggered batch job, not an online
request scheduler. Training, request arrival, user hotness, request routing, and foreground serving
interference are outside this protocol. Inputs are a fixed set of old capsules, already published
migration programs, an execution-device set, and one explicit destination. The output is one
complete target-version K/V manifest.

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

`cohortkv_single_config_full_chain_development_v1` is the frozen protocol for the historical D1
single-configuration implementation round. Its Stage 0 contract is frozen in
`configs/cohortkv_single_config_v1/` and documented by
`experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md`. The blueprint and workload
manifest are plan/configuration artifacts. The separate
`cohortkv_single_config_stage1_frontier_v1` raw result and its
`cohortkv_single_config_stage1_frozen_v1` checked-in summary are adaptive resident algorithmic
evidence under the same data/model/role contract. The separate
`cohortkv_single_config_stage2_compiler_v1` raw result and
`cohortkv_single_config_stage2_frozen_v1` checked-in summary are adaptive
deployed-representation compiler evidence. The separate
`cohortkv_single_config_stage3_operator_v1` raw result and
`cohortkv_single_config_stage3_frozen_v1` checked-in summary are adaptive resident
capsule/operator evidence under the common final-layout extent boundary. The separate
`cohortkv_single_config_stage4_system_v1` raw result and
`cohortkv_single_config_stage4_frozen_v1` checked-in summary are adaptive full-cohort normal-path
evidence under the HBM/DRAM destination boundary. Stage 2 and Stage 3 do not constitute the
complete 682-record job; Stage 4 does, but it does not include automatic fallback, fault
injection, cold-cache SSD timing, or final-test recommendation quality.

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
- selective-contiguous reads old FP16 K/V and raw history, plus one selected FP16 transition
  hidden state only when the frozen interval starts after layer zero; the Stage-1 frozen
  `m12/layers0-11` diagnostic therefore has zero transition-hidden bytes;
- residual-p reads raw history plus every BF16 old pre-block hidden state from layer `p` through
  the final layer;
- exact reads raw history;
- reuse/no-transform reads old FP16 K/V.

These paths share a physical source tier, not identical inputs. Source-shard creation, checkpoint
loading, and offline tuning are excluded from job completion and reported separately. The
residual-p BF16 hidden suffix is auxiliary state, not part of the default normalized capsule. It costs
12.45 GiB at p4 or 8.30 GiB at p8 over the full workload; the current theta0/theta10 p8 fallback
scope costs 5.83 GiB. If that state is absent, residual-p is not executable and a revised verified
plan must fall through to exact. Shard materialization requires a source-device/filesystem check
and at least 128 GiB free.

The earlier verified compiler result used in-memory FP32 layerwise state. Stage 2 has reapplied the
unchanged certificate to serialized FP16 capsules, prepared FP16 runtime programs, and FP16
output, without a new hyperparameter search or any final-test execution. All three primary plans
pass. Real shard materialization found that the optional unnormalized residual hidden suffix
exceeds the FP16 finite range, so that auxiliary representation alone is frozen as BF16 at the
same two bytes per element. Transport/layout correctness still uses the same selected numeric path
on the same serialized input as its resident oracle, requires finite values, and uses
`atol=0.02, rtol=0.02`. The Stage-3 `reference_fp32` operator widens the serialized FP16 capsule
and serialized FP16 runtime program for FP32 arithmetic, then writes the common FP16 extent. It is
an arithmetic and layout oracle for those deployed bytes, not the original FP32 fitted program
and not fresh current-model K/V. Semantic recovery remains measured separately against FP32
current-model exact K/V and score views.

Stage 3 uses only the 60 program-selection records and their predeclared theta0/theta4/theta10
assignments. It materializes unpadded FP16 normalized capsules without recommendation labels,
packs them into every `batch {1,2,4} × bucket {16,32,64}` dense layout, and compares the
FP32-arithmetic transport oracle, packed FP16, and fused FP16 through one preallocated output API.
The common operator endpoint is separate contiguous, unpadded FP16
`[layers, valid_tokens, kv_width]` K/V with lengths and offsets. Correctness covers every valid
element, finite values, dense padding zeros, exact dense-to-extent identity, and
destination-pointer preservation. Allocation, source I/O, H2D/D2H, publication, and commit remain
outside this resident boundary.

All 18 packed/fused layout candidates receive one screen pass in seed-73421 order after
correctness. The fastest three and the fastest per-operator stability controls receive one warmup
and three measured passes. The resident default is fused FP16, batch four, bucket width 32. It is
selected only when every measured run of the selected fused candidate is faster than every
measured run of the fastest packed control; otherwise the protocol falls back to packed. This
establishes the operator choice, not a stable ordering among close fused batch/bucket finalists.
The default is an input to Stage 4, not a replacement for Stage 4's independent
per-method/destination/GPU tuning. The prior jagged/page exactness and negative performance result
is retained without another layout search.

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
program-selection records; a final system shard retains one frozen transition only when its
selected interval begins after layer zero.

The frontier is complete per source-target pair: 53 selective intervals, p4/p8, compiled, cheap,
reuse, and exact, or 59 selection points per pair and 177 total. The aggregate must audit every
declared interval; selection and certification do not pool source versions.

The completed Stage-1 result adds a downstream distinction that was not known at Stage 0. No
selective interval passes the frozen 70% three-view contract for any of the three source pairs, so
its publishable action is exact. The highest-worst-view profiled action is nevertheless frozen on
program-selection users as `m=12, layers=0..11` for all three pairs. Stage 4 measures that action
through the common destination transaction only as a diagnostic external baseline and must report
`certificate_passed=false`; it may not call the resulting K/V a certified or publishable
synchronized target. Because the interval starts at layer zero, its executable source
representation is old FP16 K/V plus raw history, with zero transition-hidden bytes. The generic
transition-hidden requirement remains applicable to any future frozen interval whose start is
greater than zero.

The primary destination matrix is compiled/profiled-selective-diagnostic/exact over HBM and pinned
DRAM at 1/2/4 GPUs. Residual-p and no-transform are controls. The diagnostic selective row retains
the `selective_contiguous` artifact method name and its failed-certificate metadata. Every method
publishes the same contiguous, unpadded, FP16 K/V extent layout with lengths/offsets and the same
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

Stage 4 is frozen only when all 30 method/destination/GPU points exist: 18 primary points and
12 residual/no-transform controls. Every formal point performs one complete-workload transport
correctness pass, one untimed warmup, and three measured repetitions. Correctness covers all
17,822,269,440 valid FP16 K/V elements and verifies order, lengths, offsets, finiteness, and the
same selected numeric path at `atol=0.02, rtol=0.02`. Each of those five executions has an
independent capacity preflight. The HBM bound combines complete target residency, current
model/program residency, a full-cohort in-flight source/output wave, compute slack measured on
the tuning role, and a 512 MiB allocator margin. Compute slack is shared across equivalent GPUs;
the smaller tuning role's device assignment cannot be reused as a per-device full-workload bound.

The frozen Stage-4 result is a negative end-to-end gate for the current FP16 capsule source path.
Compiled beats the certificate-failed selective diagnostic at all six matched endpoints but
beats exact at zero of six. Compiled source read/decode/pinning consumes 91.35%–96.91% of
completion. This finding does not alter the Stage-2 semantic certificate or Stage-3 resident
operator measurements; it prohibits using them as an endpoint speedup.

The planned source/state-footprint redesign uses a new
`cohortkv_single_config_stage4_5_source_state_v1` protocol and does not mutate Stage 4. Its
method-level objective is fixed while its mechanism is open: deliver complete compiled K/V faster
than paired exact at the same source tier, HBM destination, FP16 extent layout, coverage, and
manifest boundary. Compression, residency, direct transformation from old K/V, decode fusion,
parallel source supply, and reclamation are candidates rather than required mechanisms.

Stage 4.5 begins with matched ceilings on the 60 program-selection records:

- `hbm_resident` gives compiled source state and exact raw history their respective complete
  inputs already resident on the assigned GPUs;
- `dram_resident` gives both paths decoded, pinned, pre-sharded host inputs and includes H2D;
- the Stage-4 buffered-POSIX path remains the frozen cold-source control.

These ceilings separately report source access, decode/dequantization, H2D, compute, allocation,
commit, and total completion. They determine whether source supply is sufficient to explain the
Stage-4 gap and define the time budget for later candidates. They are not endpoint wins by
themselves. A resident capsule must report its standing occupancy and lifecycle; exact may never
be left on a less favorable source tier.

Candidate screening uses only program-selection records and no recommendation labels. Every
candidate record contains:

- representation and placement, logical/physical/metadata bytes, source traffic, and creation
  path;
- capture/materialization, preload, eviction, decode/dequantization, H2D, compute, and complete
  update time;
- peak per-GPU HBM and host memory, including model, program, temporary, allocator margin, old
  K/V, source state, and new K/V overlap;
- transport error against its arithmetic oracle and semantic cache/score/top-100 recovery against
  current-model exact.

The first candidate families are resident FP16 ceilings, direct compiled transformation or
reparameterization from already retained old K/V, compact or INT8/FP8 normalized state, fused
decode/dequantization plus affine, and extent-wise overwrite/reclamation. Other mechanisms are
permitted when they preserve the same logical objective. Candidates are tested separately before
being combined, and only a candidate that changes the measured time/space Pareto frontier
survives. Once a representation is selected, the unchanged three-view contract is reapplied once
on the disjoint certificate role. A material semantic change requires its own frozen certificate
protocol; system timing cannot substitute for fidelity.

A candidate that passes numerical, certificate, standing-byte, and capacity gates may run the
complete cohort only at `compiled:hbm:1` and `compiled:hbm:4`. Paired exact is independently tuned
on program-selection records and rerun under the same new source-tier and destination boundary;
the Stage-4 exact numbers are only initial budgets. The formal representative experiment uses one
correctness run, one warmup, and at least five measured complete jobs per method and point.
Compiled passes a point only when its median is below paired exact and every measured compiled
completion is below every measured exact completion. The primary Stage-4.5 performance gate
requires both representative points to pass. A single passing, capacity-feasible point may define
only a scoped resident/hot operating regime whose policy sends every other cohort to exact.

Resident or compressed state is never free. Formal results must report update completion from the
declared steady-state tier, one-shot completion including preload when applicable, standing bytes,
capture/materialization cost, eviction/rebuild cost, and the reuse/update count required to
amortize setup. They must also report whether old K/V, source state, and new K/V coexist or are
reclaimed extent by extent. A candidate need not reduce stored bytes if another mechanism creates
a capacity-feasible Pareto point, but no storage or lifecycle cost may be omitted.

Other endpoints, one-GPU stress, and baselines are expanded only after a representative source
plan passes. The winning plan freezes representation, placement, lifecycle, decoder/operator,
reclamation, capacity preflight, and exact fallback before one final affected matrix is run.
Stage 4.6 was blocked until such a plan created an end-to-end Pareto point in its declared regime;
Stage 4.5 passed that gate and Stage 4.6 is now frozen. If no regime had passed after the bounded
candidate search, the negative result would have been frozen and the endpoint claim or method
route reconsidered; failure alone was not a lifecycle-stage admission.

Stage 4.5 is now complete under the frozen
`cohortkv_single_config_stage4_5_frozen_v1` aggregate. The selected source plan is
`compiled_old_kv`: for each source version, stack its old K/V projections, form their
minimum-norm right inverse, and compose it with the already frozen deployed capsule affine. The
result is a direct FP16 old-K/V-to-repaired-K/V affine. The three programs total 100,777,103 bytes
and are replicated per worker; the plan retains zero extra per-record source state and zero
`Norm(x)` bytes. The normal-path admission predicate requires a passing capacity preflight,
available existing old K/V, and a verified direct program. Any failed predicate selects exact.
This pure decision interface is implemented; automatic transactional exact execution remains a
Stage-5 obligation.

Operator selection uses only program-selection records. The unchanged three-view certificate is
then reapplied on the disjoint certificate role: all theta0/theta4/theta10 pairs select
`compiled_old_kv`, the minimum worst-view recovery is 0.8810287, the maximum measured cost ratio
to exact is 0.0367779, and exact terminates every fallback chain. A separate actual-data fused
transport traverses all 682 records, all four role partitions, and all 17,822,269,440 valid
elements. It has zero `atol=0.02, rtol=0.02` mismatches and maximum absolute error 0.01171875
against the deployed normalized-capsule program output. Final-test records participate only in
this label-free transport check and do not affect candidate, operator, policy, or threshold
selection.

The formal system boundary begins with complete existing old FP16 K/V already resident in HBM
for compiled and complete raw history already resident in HBM for exact, and ends with complete
replacement FP16 HBM extents plus atomic manifest commit. Each 1/2/4-GPU method point runs one
correctness pass, one warmup, and five measured complete jobs. Direct compiled medians are
0.929860/0.493566/0.254635 seconds; paired exact medians are
18.694872/9.729133/4.765541 seconds. Every compiled sample is below every exact sample at its
point. Capacity preflight includes the model, direct programs, complete old K/V, the maximum
replacement wave, and a 2-GiB allocator margin. Old extents are retired only after replacement
staging is accepted; final old-K/V bytes are zero, and peak old-plus-new K/V is
35.91/36.18/37.79 GB decimal at 1/2/4 GPUs.

The performance runs use shape-, dtype-, layout-, and occupancy-equivalent old-K/V values so that
five large repetitions are tractable; they do not establish numeric transport on their own. The
independent complete actual-data fused transport above supplies that evidence. Conversely,
transport correctness does not substitute for system timing. The result declares only an
existing-old-K/V hot-HBM regime. It does not claim cold filesystem, durable SSD, automatic tier
selection, organic mixed versions, or failure-safe fallback. The earlier pinned-DRAM normalized
capsule candidate is retained as a valid backup/negative economics result because it needs about
17.86 GB of extra host state and 24.7–39.5 seconds of preload. These artifacts admit Stage 4.6 but
do not satisfy it or Stage 5.

`cohortkv_single_config_stage4_6_lifecycle_development_v1` is the frozen protocol for the
single-configuration sequential-cache lifecycle round. Stage 4.5 starts every direct transform from exact
source-version K/V. It does not certify

```text
approximate C_hat_t -> direct(t -> t+1) -> approximate C_hat_(t+1)
```

and its reclamation path removes the original exact source extent after replacement. The
theta0/theta4/theta10-to-theta11 mix is therefore one controlled multi-source target job, not
evidence about repeated migration.

Stage 4.6 has exactly one development configuration: KuaiRand 4+12, seed 0, 16 layers,
hidden/K/V width 512, maximum history 2,048, one A40, and a hot-HBM source/target boundary. It
uses the actual sequence `theta0 -> theta1 -> ... -> theta11`: 12 checkpoints and 11 consecutive
updates. All 11 edges use the same Stage-0-frozen 682 histories/prefixes; only model version and
cache state evolve. This isolates accumulated version-migration error and is not an organic
growing-history/request trace. No other seed, dataset, model size, GPU count, DRAM, or SSD point
enters this stage. Exact current K/V is materialized at every version for offline evaluation. The
recursively migrated path must consume the previous step's actual output; regenerating exact K/V
before each measured edge is prohibited. At every update, every record receives exactly one
action:

- `migrate`: the direct-old-K/V affine, initially unchanged from the Stage-4.5 operator form;
- `exact`: current-model raw-history replay, which resets `state_kind=exact`,
  `last_exact_version=current`, and `migration_depth=0`.

There is no normal `reuse` action. A migrated result records `state_kind=migrated`,
`served_version=current`, its last exact version, incremented migration depth, risk components,
selected action, and program lineage. `served_version` never implies that an approximate state is
exact. A later migration uses the adjacent `served_version -> current_target` program; it cannot
pretend that the input is still exact at `last_exact_version`.

For `C_hat_t=C_t+e_t` and `T_t(y)=yB_t+c_t`, the measured lifecycle uses the decomposition

```text
C_hat_(t+1) - C_(t+1)
= [T_t(C_t) - C_(t+1)] + e_t B_t.
```

The first term is the current one-hop residual and the second propagates prior error. This
identity motivates the recursive risk budget but does not by itself provide a tight deployment
bound; both terms must be calibrated on disjoint exact trajectories.

The routing question is label-free current-model state fidelity and bounded maintenance load, not
per-user recommendation utility. Recommendation labels, realized task gain, and the retired
per-user drift/JVP/Fisher route are unavailable to the router. The first explored implementation
used two conditions:

```text
exact if migration_depth >= max_migration_depth or risk_score >= risk_threshold
migrate otherwise
```

Per-layer risk is

```text
calibrated_one_hop_error(normalized_correction_magnitude)
+ propagation_gain * previous_risk,
```

where `normalized_correction_magnitude=||T(KV)-KV||/(||KV||+eps)`. The 40 fit records build
exact-referenced recursive trajectories. Global correction magnitude has negligible one-hop
ranking value; a fused absolute-log K/V norm-ratio sketch supplies the strongest small threshold
diagnostic. That diagnostic beats matched-random p95 on the selection role but is not frozen: its
cumulative-only objective produces exact-refresh waves from 0% to 61.7% on selection and 0.15% to
65.1% on the diagnostic complete chain. This operational failure is preserved rather than hidden
by its acceptable cumulative cost and fidelity.

The frozen replacement is deterministic balanced age/deadline scheduling. For each adjacent edge,
the median fit-record one-hop cache-error q90 is a label-free program-level severity. Severity rank
maps the edge to a configured exact fraction in `[0.15,0.25]` around a 0.20 base. Caches already at
depth four are mandatory exact; remaining exact slots select greater migration depth first and use
a seeded SHA256 record hash only to break ties. Every other cache migrates. Exact resets risk and
depth to zero; migration increments depth. The action plan is produced before candidate execution,
so the frozen path does not compute and discard speculative migration candidates.

The resulting edge fractions are `25/19/15/17/22/16/20/18/23/24/21%`. The complete-cohort
contract permits one-record nearest-integer tolerance. This is an empirically selected bounded
development heuristic, not a global-optimality or learned per-user risk claim.

The current data has no defensible request-arrival or cache-hotness trace. Constructed hotness is
therefore not a routing feature in this protocol, and hotness would in any case weight operational
value rather than certify state fidelity.

Roles remain disjoint. The 40 fit records produce 11 adjacent-edge programs, edge severities, and
exact-referenced transition states. The 60 program-selection records share one precomputed
transition DAG to compare a bounded set of threshold, periodic, fixed-quota, and
severity-bounded-quota points. This is not a repeated full-cohort matrix. The independent 60
certificate records receive the frozen balanced policy once. The final experiment then runs
exactly one complete 682-record `theta0 -> theta11` balanced-policy chain plus its necessary
all-exact evaluation reference. The earlier threshold complete chain is explicitly a diagnostic
that exposed the missing per-step peak objective; it did not supply recommendation labels or
numeric parameters to the replacement policy. The 522 final-test records provide primary
recommendation results and cannot change the policy.

The selection trajectory contains `all_migrate` and `all_exact` endpoints and one
matched-exact-fraction random-refresh diagnostic for the threshold candidate. All candidate
points are derived from the same cached DAG; they do not trigger 1/2/4-GPU or HBM/DRAM matrices.
The desired operating region is approximately 20%–30% of cumulative all-exact
GPU cost with approximately 80%–90% or higher minimum label-free cache/score/top-100
exact-relative fidelity. Recommendation labels do not select the threshold. Real task metrics are
an independent outcome and should remain close to all-exact, but cannot be used to tune the
router. This is a target, not an admission to alter the data, metrics, or protocol when the result
misses it.

At each target, mixed policy and all-exact use the same current checkpoint, frozen history,
engaged positive, and existing stale-inference evaluator; only the prefix-K/V path differs. Every
one of the 11 targets reports MeanRank, Catalog AUC, NDCG@100, Hit@100, and paired differences.
Reuse-to-exact recommendation recovery is reported only when its denominator is stable;
otherwise absolute metrics and paired differences remain primary.
Cache error, score cosine, top-100 overlap, exact fraction, migration-depth distribution,
scheduler CPU time, per-step GPU cost ratio, and cumulative GPU cost ratio are also mandatory.
The all-record exact evaluation reference is isolated from the router and excluded from
mixed-policy cost. Mixed GPU cost includes only exact-subset gather/replay, records actually
migrated, and common publication; GPU cost is measured, never inferred from a handwritten
constant. Raw histories and old K/V use the declared hot-HBM boundary, so source preload is
outside this ratio. Scheduler CPU time is measured and disclosed separately rather than
mislabelled as GPU time.

If the exact-source program fails on migrated inputs, a migrated-input or depth-bucketed affine
may be fit behind the same runtime ABI, but that change requires a new program protocol and
certificate; the Stage-4.5 one-hop equivalence cannot be reused.

Stage 4.6 passes only if the complete recursive trajectory is reproducible, lineage is
unambiguous, all 11 steps have recommendation/fidelity/cost comparisons, a frozen policy passes
its predeclared per-step and terminal contract on certificate records, consecutive migration is
bounded, and per-step refresh/cost peaks are bounded. Beating matched random is insufficient for
an adaptive claim when an operational wave contract fails. Missing the desired 0.2–0.3× cost and
0.8–0.9 label-free fidelity region, or observing large independent task-metric gaps, is reported
as a real result rather than hidden. If no reasonable refresh fraction controls cumulative error,
the paper scope contracts to at-most-one migration per exact anchor.

The frozen outputs are
`configs/cohortkv_single_config_v1/stage4_6_lifecycle_policy.json` and
`stage4_6_lifecycle_summary.json`. The independent certificate records `0.2142×` cumulative GPU
cost, `0.2814×` maximum step cost, and minimum cache/score/top-100 values
`0.9613/0.999759/0.9898`. The complete 682-record chain records `0.2134×` cumulative cost,
`0.2543×` maximum step cost, `14.956%–25.073%` exact fractions after record rounding, and minimum
`0.9632/0.999950/0.9918`. All 7,502 lineage rows rebuild from the frozen policy, consume the
previous actual output, reset exact states, and stay at depth four or below. This fixed-history
protocol admits Stage 5 within its declared scope; the organic-history correction below is
classified separately.

`cohortkv_single_config_organic_lifecycle_v1` is a separate Stage-4.7 correction to the
fixed-history boundary above. It uses the same KuaiRand seed-0, 16L/H512, maximum-2,048, one-A40
configuration and the same deterministic 40/60/60/522 roles, but every endpoint has a
date-progressive history. Theta0 uses history through D4 to predict D5; after D5 is evaluated it
is admitted, theta1 uses history through D5 to predict D6, and this continues through theta11
predicting D16. Mixed and all-exact use the same current checkpoint, latest token, candidate
catalog, and engaged next-window positives. Previous actual mixed K/V must be consumed
recursively; current or future canonical date partitions cannot enter history early.

The causal authority is the prepared dataset's canonical date partition, not globally strict raw
`time_ms`. A suffix-membership gate proves that every resident history is drawn only from the base
and previously admitted partitions while permitting left crop and token-cap truncation. The
frozen diagnostic still records 147/8,167 resident record-windows and
3,521/11,797,055 history tokens at or beyond the next partition's raw-time minimum, with a maximum
lead of 6,128.618 seconds. Therefore this protocol supports a canonical-date growing-history
lifecycle claim, not a raw-event/request-arrival trace. Repairing timestamp order or deduplication
requires a new prepared-data/training protocol.

The router receives every reusable lightweight candidate and then applies a fixed 20% exact
budget. Depth-four caches are mandatory exact; remaining slots prefer greater migration age and
then greater current-edge q90 absolute log K/V norm shift. SHA256 is only an exact numeric
tie-break, and neither next-window labels nor future-edge severity may enter. Natural exact work
for cold, re-entered, and zero-overlap prefixes is outside that budget. Across the complete run,
1,344/6,711 reusable prefixes are selector exact (`20.0268%`), 5,367 migrate, 771 are natural
exact, and three are common-latest-only. Of the selector exact actions, 476 are depth deadlines
and 868 fill the norm-shift quota.

This is an algorithmic but weak secondary signal. Posthoc current-norm-shift versus candidate-error
Spearman averages `0.0341` across edges, selected candidates have higher mean error on 8/11 edges,
and mean top-error-oracle overlap is `23.46%` against a 20% random expectation. The protocol may
claim deterministic bounded age/deadline scheduling with norm-shift ranking; it may not claim a
strong adaptive failure predictor or selector optimality.

The one frozen full run completes all 12 endpoints and 11 updates. Cumulative update-only GPU cost
is `0.2703x` all-exact and maximum step cost is `0.2892x`; symmetric and common-inclusive
lifecycle ratios are `0.5069x` and `0.5372x`. Minimum score cosine and top-100 overlap are
`0.999876` and `0.97357`, but minimum q90 cache fidelity is `0.8744`. Thus the predeclared
cost, score, top-100, depth, and execution gates pass while the `0.90` cache-fidelity gate fails.
Across 4,368 final-role positive records, record-weighted mixed/all-exact ratios are
`0.999987/0.994590/0.997180` for catalog AUC/NDCG@100/Hit@100, with worst-window ratios
`0.999786/0.953324/0.977778`. `status=complete` means the fail-closed execution produced a
complete valid artifact; it does not turn the failed fidelity gate into a pass. Stage 4.7 is
completed mixed development evidence and cannot silently replace the frozen Stage-4.6 result.

`cohortkv_single_config_stage4_8_scheduler_sweep_v1` is the follow-up scheduler-development
protocol. It keeps every Stage-4.7 workload and recursive-state identity fixed, but does not use
K/V fidelity, norm shift, score cosine, top-100 overlap, migration age, or scheduler debt as a
scientific admission metric. The only quality-cost Pareto axes are record-weighted catalog AUC,
NDCG@100, Hit@100, symmetric GPU lifecycle cost, and common-inclusive GPU lifecycle cost.

The external exact reference is
`configs/cohortkv_single_config_v1/stage4_8_exact_baseline.json`. Every worker verifies its source
result, input, window, checkpoint, compiler, manifest, and program hashes. It reuses the frozen
eleven-edge exact GPU denominator and twelve exact task endpoints and must not execute the
independent exact reference. Natural exact and selector-scheduled exact remain real mixed-chain
actions. Every optional action is chosen before migration, so scheduled-exact records cannot also
pay candidate-transform cost.

The preregistered families and complete grids are work-balanced staggered renewal
`H={8,10,12,16}`, total exact-token cumulative debt
`b={0.10,0.12,0.14,0.16}`, AoI MaxWeight reusable-token budget
`beta={0.04,0.07,0.10,0.13}`, and label-free model-time staggered renewal
`H={8,10,12,16}`. All sixteen results are retained. Each reports strict cost gates against the
Stage-4.7 symmetric/common-inclusive ratios `0.5069011719265762/0.5372231748138118`; no posthoc
task-quality threshold is permitted. Four-GPU execution is a parameter-confounded development
screen, so any paper candidate requires later sequential same-device paired confirmation. Static
smoke and one-edge runtime smoke validate code paths only and are not scientific results.

All sixteen preregistered points completed. Under the frozen v1 accounting, update-only ratios
range from `0.110699` to `0.181674`, symmetric ratios from `0.398841` to `0.446337`, and
common-inclusive ratios from `0.435768` to `0.479905`; every point passes both v1 current-cost
gates. `token_debt/total10` is the minimum-cost point, while
`staggered_renewal/H=12` is the retained bounded-renewal quality candidate. These are development
screens, not same-device confirmation.

The completed Stage-4.8 execution first cropped the previous cache, appended the newly admitted
window with the source model, and then migrated or refreshed the resulting target-length prefix.
Its measured foreground append was added symmetrically to the two lifecycle diagnostics. Those
definitions remain part of the immutable v1 result meaning, but they are not the corrected primary
model-rollout boundary.

`cohortkv_single_config_stage4_9_rollout_boundary_v1` separates accounting from execution order.
For edge `theta_v -> theta_(v+1)`, let `R_v` be the retained suffix of the previously admitted
history after the crop required to make room for the newly observed window `Delta_(v+1)`.
The paired rollout comparison is:

```text
mixed: previous actual K/V(R_v) -> migrate-or-exact under theta_(v+1) -> stop timer
exact: raw R_v -> exact under theta_(v+1) -> stop timer
both:  append Delta_(v+1) under theta_(v+1), outside the rollout timers
```

These are two independent invariants:

- **Accounting:** target-model append of `Delta_(v+1)` is foreground inference. It is measured in
  a separate ledger but enters neither side of the primary rollout ratio, regardless of whether an
  alternate implementation performs it before or after rollout.
- **Ordering:** the primary growing-history path performs migration first and then appends
  `Delta_(v+1)` with the target model. It must not precompute that window with `theta_v`.

If `U_t` is measured migration plus selected exact refresh and matched output materialization for
the reusable retained prefix, `E_t` is exact current-model recomputation to the same destination,
and `A_t` is the common target-model append, the primary outcome is
`sum(U_t)/sum(E_t)`. `A_t` is measured in a separate ledger. A later final-state-ready systems
claim must separately compare measured `U_t+A_t` with the fastest measured exact path to the same
`R_v || Delta_(v+1)` output; exact cannot be forced through a slower decomposed path. That metric
is not migration speedup. Cold, re-entered, and zero-overlap construction is reported separately
and cannot be charged asymmetrically. If a nonempty retained prefix lacks its expected cache,
rebuilding it is natural exact charged to `U_t`; every aggregate reports reusable-prefix coverage.

After the untimed append, the exact branch must agree with a fresh target-model forward on
`R_v || Delta_(v+1)` within tolerance for K/V, hidden state, and task output; one-shot fresh is
the quality authority if they disagree. The crop defining `R_v` is fixed before routing from
causal history identity, the admitted window, and the maximum-length rule. Physically dropping old
K/V rows does not erase their influence from migrated deep-layer state, so no exactness claim is
made for the mixed branch. Both branches then predict the next unseen canonical window, and the
previous actual mixed output remains the recursive input. Because the synchronized prefix and
exact workload differ from v1, neither the frozen `346319.0015 ms` denominator nor the old
`token_debt/total10` value `0.110699` is formal Stage-4.9 evidence. The formal confirmation reruns
the two selected candidates and a paired exact reference sequentially on one A40 without
reopening the sixteen-point scheduler sweep.

The Stage-4.9 retained-prefix ABI and smoke-only runner have passed unit/static checks and one real
`theta0 -> theta1` GPU smoke covering migrate, scheduled exact, and natural exact. The smoke makes
zero source-model append calls and verifies target-model two-stage exact against one-shot exact.
It also injects one expected-but-missing cache: target-model exact reconstruction of its retained
prefix is charged to `U` and included in the paired exact population. Timed retained endpoints
share device-resident FP16, while FP32 is an independent parity-only branch. The latest-only
empty-prefix path is covered by a synthetic one-token equivalence test. It has no warmup, writes
no formal result, and is marked `scientific_result=false`; its timings must not enter any table or
claim.

The formal `cohortkv_single_config_stage4_9_same_device_confirmation_v2` result completes all
11 recursive edges for both candidates on the same physical A40, executes a fresh paired exact
reference on every edge, consumes the previous actual post-append mixed state, makes zero
source-model append calls, and passes FP32 exact-equivalence, lineage, capacity, provenance, and
old-denominator exclusion checks. Its frozen outcomes are:

- `token_debt_total10`: `sum(U)/sum(E)=0.071319`, 221 scheduled exact actions over 6,711 reusable
  record-edges, and record-weighted AUC/NDCG@100/Hit@100 recovery
  `1.000030/0.996890/0.999060`;
- `staggered_renewal_h12`: `sum(U)/sum(E)=0.100017`, 462 scheduled exact actions over 6,711
  reusable record-edges, and recovery `1.000039/0.997463/1.000000`.

The policy freeze retains `staggered_renewal_h12` because it is the preregistered bounded-renewal
candidate; `token_debt_total10` remains a cost endpoint without a per-cache deadline. This
decision does not use recommendation labels. The evaluator uses a CPU FP16 recursive store and
groupwise H2D/D2H staging for memory containment. H12 reports 662,869,804,944 logical movement
bytes separately outside `U/E`; consequently the result is not a full-cohort HBM-resident
lifecycle or end-to-end state-movement claim. Its horizon is 12 but the measured chain has only
11 updates, so a complete renewal cycle is not observed. Maximum observed migration depth 11
must not be confused with the fixed-history Stage-4.6 depth-four guarantee.

For D2, Stage-4.9 is an immutable action-plane input rather than a reusable performance
denominator. The v1 action-plan handoff contains:

```text
plan-level source/target version, policy, provenance, counts, and content SHA-256
per-record requested_action and requested_reason
old / retained / delta / latest / target-prefix / final extents
last-exact version, migration depth, cache-presence flags, and identity hashes
```

Program, owner, old-extent, and raw-history bindings are added by the D2 adapter and must be bound
separately; they are not v1 action-record fields.

On the first H12 D2 edge, 548 records request compiled, 46 request scheduled exact, and 88 are
natural exact. Thus `134/682 = 19.6%` is the runtime exact-route record fraction, not a policy-only
fraction and not a compute/communication ratio. The corresponding full post-append lookup ledger
is `347,062/934,917 = 37.1%`. D2 must not rerun the selector or change these requested actions
according to measured communication. Fixed safety fallback records requested action, final
action, and reason separately.

`cohortkv_stage4_10_renewal_calibrated_h12_smoke_v1` preserves the Stage-4.9 retained-prefix,
append, H12, and recursive-state definitions but changes the source of the per-edge direct
program. After action selection, and before any migrant transform:

1. assemble and crop the scheduled-exact records' previous actual K/V;
2. compute their current-model exact retained K/V once;
3. fit one shared program from those aligned pairs;
4. reuse the exact targets as those records' refreshed caches;
5. apply the new program only to the disjoint migrant set.

Calibration IDs must exactly equal scheduled-exact IDs. Natural-exact records, migrant exact
references generated only for evaluation, old 40-user fit records, and recommendation labels are
forbidden fit inputs. The action partition is immutable after calibration; this protocol has no
empirical semantic admission gate.

The runtime always receives one FP16 direct-old-K/V affine. `inverse_norm_ridge` estimates
`Norm(x)` from actual K/V with the source projection right inverse, fits the target residual, and
composes back to direct K/V. `direct_kv_residual_ridge` fits fresh-minus-actual K/V around an
identity prior. Both use centered ridge `0.001`, at most 8,192 deterministic paired tokens, and
the same fused operator ABI.

`U` includes scheduled source crop, scheduled exact replay, ridge/program construction, device
program preparation, migrant crop/transform, any missing-cache exact retained rebuild, and
retained materialization. `E` independently replays the same timed retained population.
Calibration H2D, migrant H2D, next-state D2H, and target append are measured outside primary
`U/E`; they may not be omitted from their separate ledgers.

The two real smoke artifacts cover only edges 0→1 and 1→2, use zero warmup and one repetition, and
run the two fit modes on different A40s. Direct K/V reports aggregate `U/E=0.128764`; inverse-Norm
reports `0.127694`. Per-edge program construction is 75–124 ms and is included in those
numerators. These numbers are execution diagnostics, not paper results. The smoke evaluates no
AUC, NDCG@100, Hit@100, score fidelity, or held-out migration quality, so it cannot select a
variant or replace the formal 11-edge Stage-4.9 result. A new protocol is required for that
comparison.

For that handoff, `post_retained_prefix_pre_append` is the guard hook and
`post_append_full_cache` is the only transaction-commit and recursive-state boundary. A retained
prefix is private intermediate state and must never be published as a complete user cache.
Expected cache IDs must come from the prior committed contract, while present IDs come from actual
store contents; deriving both from one set makes missing-cache fallback unobservable.

The frozen v1 Stage-5 amendment is
`experiments/system/COHORTKV_STAGE5_MINIMAL_CLOSURE_V1.md`. It freezes one job-level semantic
preflight instead of searching runtime-sentinel families. Before any target extent is produced,
the job validates artifact hash/version/shape, capacity, old-K/V presence, and program identity,
then applies one label-free canary and threshold frozen on the program-selection role. Failure
of program identity/shape, old-K/V presence, or the semantic canary routes the affected migration
cohort directly to exact. Artifact/version mismatch and copy-on-write capacity failure are fatal
admission errors before transaction creation, because exact cannot make either condition safe.
The result records preflight overhead and the final fallback reason, but makes no runtime
drift-detection or online-rework claim. Stage 4.10 does not inherit the semantic canary; only
integrity, shape, finite, capacity, lineage, and transaction checks remain in its current path.

Failure-safe evidence uses copy-on-write on one capacity-feasible representative GPU
configuration: all old extents remain readable until the complete target manifest commits. It
contains one normal integrated `theta0 -> theta1` job plus exactly three fallback/fault cases:

1. an integrity-accepted, shape-preserving perturbation of the actual `theta0 -> theta1`
   direct-old-K/V program, which the frozen canary executes and routes to exact before target
   execution, ending in one complete corrected commit;
2. a mid-job execution exception, which aborts and reclaims private staging;
3. an exception immediately before commit, which leaves the complete private target invisible.

For both abort cases, correctness requires reading every expected record through the old manifest
and checking version, shape, finite values, and checksum or old-K/V equivalence. Pointer equality
alone is insufficient because the one-GPU normal path may retire old extents. That reclaiming path
remains performance evidence but is not abort-safe. Artifact mismatch remains a unit/smoke check;
first-extent/publication fault grids, runtime invalidation/rework, journals, and resume are
post-v1 extensions.

The aggregate result must validate against
`configs/cohortkv_single_config_v1/result.schema.json`. The parent blueprint retains
`stage4_core_frozen` for compatibility with the immutable Stage-4 inputs and separately registers
`stage4_5_source_plan_summary.json` as a completed amendment. Together they support the complete
normal-path result, the negative normalized-source finding, and the scoped direct-old-K/V hot-HBM
Pareto point, but not failure recovery, automatic fallback execution, cold/durable SSD, or
capsule-economics claims. Even after later full-chain completion, seed 0 remains adaptive
development evidence; timing repeats are not training replications. The aggregate schema has been
amended to replace the former six-failure and capsule-economics requirements with the normal job,
three cases above, and the accounting ledger below. The 18 completed primary
method/destination/GPU-count combinations remain frozen; controls cannot satisfy a missing
primary point.

The source-state accounting audit performs no new representation experiment. It derives one table
from the frozen Stage-2/4/4.5 artifacts:

- direct old K/V has zero additional per-record source-state bytes and no independent capture,
  encode, or preload path;
- direct-program bytes, composition time, old/new peak overlap, and the prepublished-program
  data-plane boundary are reported;
- Stage-2 fit/runtime-prepare/certificate time and its existing resident amortization floor remain
  separate offline setup;
- the rejected FP16 normalized capsule retains its measured logical/physical bytes,
  preload/source time, and completed endpoint outcome.

INT8/FP8 implementation, capture/D2H/POSIX-persistence timing, quantized full-cohort execution,
time-break-even curves, physical SSD/GDS, remote storage, and automatic tier selection are
optional post-v1 extensions. A filesystem path remains a correctness interface and cannot be
called an SSD result.

The formal `cohortkv_single_config_stage5_full_cow_integration_v1` artifact binds the confirmed
`staggered_renewal_h12` action partition for `theta0 -> theta1`. Two A40s pass the copy-on-write
capacity preflight. The normal job commits and readback-validates all 682 target records; the
shape-preserving real-program perturbation fails the frozen canary, routes the migration cohort
to exact before target execution, and commits one complete corrected target. Mid-job and
pre-commit injections both abort with no partial target visible, reclaim private staging, and
readback-validate all 682 old records. Formal-candidate, input, source, capacity, normal,
fallback, abort, JSON Schema, and cross-field checks all pass. This is implementation-correctness
evidence, not a throughput, runtime-sentinel, online-rework, or durability claim.

`cohortkv_single_config_stage6_freeze_v1` is the final seed-0 assembly protocol. It performs no
new GPU experiment. The CPU-only deterministic assembler verifies 18 frozen Stage-1 through
Stage-5 source artifacts by path, protocol, status, size, and SHA-256; validates the amended
aggregate schema and whole-aggregate semantics; and atomically writes eight sidecars before
`final_summary_seed0.json`. Sidecars cover correctness, timing/memory, paper tables, paper
figures, artifact-to-claim binding, negative results, current-manuscript disposition, and the
code snapshot. All source-hash, candidate-binding, Stage-5-semantic, claim-binding, schema,
whole-aggregate, and TBD-disposition checks must pass. The result remains adaptive seed-0
development evidence. New training seeds and predeclared dataset/model-capacity cells belong to
Stage 7 and must not modify this frozen result family.

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

Formal D2 reports require two non-interchangeable ledgers.

Action plane:

- requested compiled, scheduled-exact, natural-exact, final-action, and fallback counts;
- retained/suffix/final tokens and exact-route record fraction;
- action-plan hash, reason counts, separate program binding/hash, and lineage/identity checks.

Physical plane:

- per-phase requested/unique/local/remote IDs and routed ID/returned-vector bytes;
- collective calls, exposed time, and per-rank wait/imbalance;
- `(R,S,F)` extent shapes, padded work, and physical exact pools;
- retained K/V read/write/P2P/rewrite bytes and suffix/segment bytes;
- source/target/transient HBM;
- plan/lowering/materialization, compute, validation, commit, reclaim, and optional contiguous
  consumer time.

Action-count fraction, lookup fraction, vector-byte fraction, and wall-time fraction must never be
substituted for one another.

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
- The formal D2 primary timer begins with an immutable ActionPlan, owner-resident old K/V, loaded
  model/program/embedding shards, and the declared raw-history tier ready. It includes any
  non-persisted wave lowering, row-sharded lookup/collectives, compiled/exact/suffix compute,
  target staging, segmented-manifest construction, validation, commit, reclaim, and any required
  synchronous contiguous materialization. It ends only when one complete post-append target epoch
  is visible.
- If a serialized WavePlan legitimately moves lowering outside the execution timer, report both
  execution-only and plan-inclusive single-wave cost plus reuse count/break-even.
- Formal D2 compares the same action hash and endpoint across strong all-exact, naive sharded
  fixed-action mixed, and D2 physical-sparse mixed. The integrated all-exact denominator is the
  faster measured one-shot/two-stage implementation; exact cannot be forced through a slower
  decomposed path.

## 8. Current artifacts

D2 mechanism-development diagnostics, ineligible for paper tables:

- `results/system/cohortkv_design2_integrated_w3_development_v*/`
- `results/system/cohortkv_design2_integrated_full_payload_development_v1/`
- `results/system/cohortkv_design2_resource_isolation_development_v1/`
- `configs/cohortkv_d2/development/`

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
- local
  `results/system/cohortkv_single_config_full_chain_v1/stage1_frontier_seed0.json`
- `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`
- `experiments/system/COHORTKV_STAGE1_FRONTIER_V1.md`
- local
  `results/system/cohortkv_single_config_full_chain_v1/stage2_compiler_seed0.json`
- local
  `checkpoints/kuairand_long_context_4plus12_exploration/seed0/single_config_v1/stage2_runtime/*.pt`
- `configs/cohortkv_single_config_v1/stage2_compiler_summary.json`
- `configs/cohortkv_single_config_v1/stage2_plans/*.json`
- `experiments/system/COHORTKV_STAGE2_COMPILER_V1.md`
- local
  `results/system/cohortkv_single_config_full_chain_v1/stage3_operator_seed0.json`
- `configs/cohortkv_single_config_v1/stage3_operator_summary.json`
- `experiments/system/COHORTKV_STAGE3_OPERATOR_V1.md`
- `configs/cohortkv_single_config_v1/{blueprint,workload_manifest,result.schema}.json`
- `experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md`

`smoke.json` files, old `results/phase0`, and old `results/streaming` are not research artifacts.
Per-seed exposure JSON and all checkpoints are current local artifacts but ignored by Git. Taobao
UserBehavior is an action-only semantic boundary rather than the selected next stream.
