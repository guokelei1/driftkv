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

### `streaming_value_control_v1_incremental_prefix_cache`

Six-layer seeds 0-3 compare theta-0 frozen serving, current-model full reuse of a theta-0 prefix,
and current-model full compute on the same history and future positives. The three contrasts are
total streaming-training value, value retained under stale reuse, and cache-maintenance value.

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

## 6. Metrics and statistics

Primary quality views:

- Best Rank and Mean Rank: lower is better; gains are `reuse - method`.
- MRR, NDCG@10/100, Hit@10/100: higher is better; gains are `method - reuse`.
- Quality recovery: method gain divided by fresh-recompute gain, only when the denominator is
  sufficiently different from zero.
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

`smoke.json` files, old `results/phase0`, old `results/streaming`, and checkpoints outside
`checkpoints/validity` or `checkpoints/scaling` are not research artifacts. Taobao UserBehavior is
an action-only semantic boundary rather than the selected next stream. The exposure-compatible
audits establish data capacity only; assign a new model protocol name before training on them.
