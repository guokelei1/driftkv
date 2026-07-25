# KuaiRand long-context 8+8 execution protocol

## Status

The prepared artifact, theta0-theta8 seed-0 training, both motivation matrices, and all three tiny
four-GPU smoke tests are complete as of 2026-07-25. The first evaluator stored the seven theta0
moving pairs and the fixed-theta7 row: 15 records but only 14 unique pairs, including one theta7
self-reference. It is retained as `kuairand_long_context_8plus8_motivation_v2`.

The active motivation protocol is now
`kuairand_long_context_8plus8_motivation_all_pairs_v3`. It evaluates all 28 distinct
older-cache/current-model pairs and writes a separate result. That v3 evaluation is complete; the
compiled method evaluation has not run.

The v3 matrix shows one dominant theta0-to-theta1 base-to-stream discontinuity. It does not show
repeated non-theta0 late cliffs across MeanRank, AUC, standard top-k metrics, or cache-fidelity
diagnostics. It therefore supports measurable stale-cache loss and a special training boundary,
but not a generic fixed-window-collapse claim.

This protocol supersedes the exploratory 12+4 seed-0 run. That run trained theta0 through theta3
and found increasing theta0-cache loss across its three moving endpoints: MeanRank loss was
1,327/1,597/2,284 and catalog-AUC loss was 2.65/3.19/4.57 percentage points. Those observations
were sufficient to preselect MeanRank as the primary metric and AUC as a robust secondary, but
three changing evaluation dates could not identify a causal cache-age jump. Its raw artifact,
checkpoints, and result JSONs were removed and cannot be pooled with this protocol.

## Frozen data

- Source: the two local KuaiRand-1K standard logs; the random-exposure log is excluded.
- Horizon: 2022-04-08 through 2022-04-23.
- Base period: D1-D8, 2022-04-08 through 2022-04-15.
- Online dates: D9-D16, 2022-04-16 through 2022-04-23.
- Cohort: 965 users with at least five base-period exposures. No users are added merely to reach
  1,000.
- Prediction vocabulary: the base-only top 50,000 video IDs.
- Context vocabulary: the prediction vocabulary plus 262,144 deterministic SplitMix64 buckets.
  Every selected exposure remains context, while an out-of-catalog item is input-only.
- History: an eight-day rolling window followed by deterministic tail truncation at 2,048 tokens.
- Training sequences: all chronological chunks with stride 2,047, dynamic padding, and length
  bucketing.

The fixed artifact is `data/processed/kuairand_long_context_8plus8_v2.npz`. It contains 5,820,867
exposures, 1,083,333 prediction-catalog rows, and 514,853 engaged in-catalog rows. Its SHA256 is
`2db3f76992ab490802fc586d47cbc2e1b4e38e1adc45a221c123dfe159633b36`.

Data-only enumeration gives this effective coverage per epoch:

| Phase | Sequences | Context tokens | Eligible next-item targets |
|---|---:|---:|---:|
| D1-D8 base | 1,940 | 2,914,284 | 422,097 |
| D9 update | 1,022 | 1,199,629 | 28,323 |
| D10 update | 969 | 1,137,993 | 18,170 |
| D11 update | 836 | 974,831 | 9,773 |
| D12 update | 803 | 935,536 | 7,603 |
| D13 update | 806 | 934,653 | 7,265 |
| D14 update | 816 | 945,381 | 6,459 |
| D15 update | 828 | 980,481 | 6,738 |
| D16 update | 867 | 1,040,321 | 8,259 |

## Frozen model and training

The simplified HSTU has 16 layers, hidden size 512, eight 64-dimensional heads, ReLU pointwise
unnormalized attention, and maximum sequence length 2,048. It has 181,082,112 parameters
(0.181B). FP32 parameter tensors contain 724,328,448 bytes, about 724 MB decimal or 690.8 MiB;
changing the context length does not change this parameter count.

Training uses FP32 AdamW, six base epochs at `3e-4`, and two epochs per online date at `1e-4`.
Each of four DDP workers receives a logical batch of four, executes it as four micro-batches of
one, and performs one optimizer update after gradient accumulation. The effective global batch
remains 16 without requiring a length-2,048 backward graph for four users at once.

| Version | Training boundary | Leak-free next-day evaluation |
|---|---|---|
| theta0 | D1-D8 base | D9 is a base diagnostic |
| theta1 | update on D9 | D10 |
| theta2 | update on D10 | D11 |
| theta3 | update on D11 | D12 |
| theta4 | update on D12 | D13 |
| theta5 | update on D13 | D14 |
| theta6 | update on D14 | D15 |
| theta7 | update on D15 | D16 |
| theta8 | update on D16 | none inside the frozen horizon |

Theta8 is saved so all eight stream increments are represented, but theta1-theta7 provide the
seven valid next-day quality points.

## Motivation contrasts

The active matrix evaluates every strictly older cache against every current version:

- theta0 cache with theta1-theta7;
- theta1 cache with theta2-theta7;
- theta2 cache with theta3-theta7;
- continuing through theta6 cache with theta7.

This gives `7 + 6 + ... + 1 = 28` distinct comparisons. Each theta-i row holds its current model,
next unseen evaluation date, histories, positives, and users fixed while cache versions vary from
theta0 through theta-(i-1). The theta0 column remains the seven-point moving curve, and the
theta7 row remains the fixed-D16 endpoint. A complete current-model forward inside every pair is
the fresh reference, so diagonal self-pairs are unnecessary.

The evaluator outputs all full-catalog ranking metrics and paired adjacent-cache-age summaries
within every current-version row. MeanRank is primary, catalog AUC is the robust secondary, and
NDCG@100/Hit@100 are standard secondaries.

Cache age means checkpoint-update distance, not the residence time of one physical snapshot.
Fresh, stale, and migrated paths consume the identical tail-cropped resident prefix; the complete
sequence never exceeds 2,048 and its cached prefix never exceeds 2,047. Literal D9 snapshot
survival, rolling eviction, and organically mixed per-token versions require a separate cache
lifecycle experiment.

The D16 dry run has 746 eligible users. Median history is 2,048; 392 users (52.5%) have more than
2,048 events inside the eight-day time window and are deterministically tail-truncated. The
controlled comparison remains token-aligned, but it must not be described as retaining every D9
token. Its unpadded FP32 prefix K/V is 77.58 GB decimal across the cohort; cached layerwise
normalized state adds 38.79 GB, making the compiled-method state 116.37 GB. These are logical
capacity measurements, not allocated-GPU peaks or timing results.

## Preliminary core method

The method command holds theta7 and D16 fixed. Forty label-independent users fit a rank-16 affine
residual from theta0 cached `Norm(x)` to fresh-minus-cheap theta7 K/V; all remaining users are held
out. It compares unmodified reuse, the compiled projection, and exact recomputation using ranking,
cache fidelity, and CUDA-event kernel time.

This remains a scale bring-up of the fast tier. It does not implement cohort admission, the quality
tier, organically mixed cache versions, transfers, or an end-to-end scheduler.

## Commands

Validate the prepared data and frozen model:

```bash
python scripts/prepare_kuairand_long_context.py --validate-existing
python scripts/train_kuairand_long_context.py --validate-only
python scripts/train_kuairand_long_context.py --data-dry-run
```

Training and the all-pairs motivation evaluation have completed. Run only the pending method
evaluation with:

```bash
torchrun --standalone --nproc_per_node=4 scripts/evaluate_kuairand_long_context_method.py
```

Expected local artifacts:

- `checkpoints/kuairand_long_context_8plus8/seed0/theta_{0..8}.pt`;
- `results/motivation_scale/long_context_8plus8_training_seed0.json`;
- `results/motivation_scale/long_context_8plus8_motivation_seed0.json`;
- `results/motivation_scale/long_context_8plus8_motivation_all_pairs_seed0.json`;
- `results/motivation_scale/long_context_8plus8_method_seed0.json`.

Training records the prepared artifact hash. Both evaluators reject a different data artifact,
incomplete training, or a mismatched protocol.

## Completed non-formal validation

All relevant unit tests pass. The tiny-model four-GPU training smoke test covers micro-batch
gradient accumulation and a partial final DDP step; replica parameters remain identical. The
motivation smoke test checks distributed record gathering, the expanded metric set, and fresh
incremental parity. The method smoke test fits on rank 0, broadcasts the compiled operator, and
evaluates disjoint shards on all ranks. None is research evidence.
