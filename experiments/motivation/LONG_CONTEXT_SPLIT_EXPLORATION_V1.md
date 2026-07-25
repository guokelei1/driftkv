# KuaiRand long-context split exploration

## Status

The 4+12 prepared artifact, formal 181M-parameter seed-0 training, and all 66 motivation pairs are
complete as of 2026-07-25. Four-GPU training produced theta0-theta12 in 860.7 seconds. This is an
exploratory family and remains separate from the frozen 8+8 v3 result.

The question is longitudinal: hold one cache-encoding checkpoint fixed and advance the deployed
model. The primary views are therefore the theta0 cache under theta1-theta11, the theta2 cache
under theta3-theta11, and the analogous later cache cohorts. A fixed-current-model matrix row is
still retained, but it is not the main view for this question.

## Primary 4+12 split

- Horizon: 2022-04-08 through 2022-04-23.
- Base: D1-D4; online updates: D5-D16.
- Cohort: 945 users with at least five base-period exposures.
- Selected context rows: 5,780,499.
- Base-only prediction catalog: top 50,000 items.
- Context-only tail: 262,144 deterministic hash buckets.
- History: eight-day window followed by deterministic truncation at 2,048 tokens.
- Model: 16 layers, hidden size 512, eight 64-dimensional heads, 181,082,112 parameters.
- Training: all chronological chunks, six base epochs, two epochs per daily update, four FP32 DDP
  workers, per-device logical batch four, micro-batch one.

Theta0 is trained on D1-D4. Updating on D5 produces theta1, continuing through theta12 after D16.
Theta1-theta11 have leak-free next-day endpoints on D6-D16. Theta12 is retained as the
post-horizon checkpoint and is not evaluated on already-ingested D16 labels.

The complete strict lower triangle has `1 + 2 + ... + 11 = 66` comparisons. It gives theta0 an
11-point fixed-cache trajectory and theta2 a nine-point trajectory. Every point compares stale
reuse with a complete fresh current-model forward on the identical users and resident prefix.
Cache version means the checkpoint used to encode that prefix, not survival of one immutable
physical snapshot.

The prepared artifact is
`data/processed/kuairand_long_context_4plus12_exploration_v1.npz`, with SHA256
`e03f3e80dacf9deccd5783d26a184d8ced7b339275bf13fa3b90de42a4b028b8`.

## Audit

Base training covers 1,301 sequences, 1,533,439 context tokens, and 261,814 eligible targets per
epoch. Daily updates cover 730-929 sequences after D5, with 4,327-24,072 eligible targets. The
final D16 endpoint contains 682 users. Its median history is 2,048, and 362 users exceed the token
cap after the time window.

The D16 logical unpadded FP32 prefix K/V is 71.29 GB decimal. Cached layerwise normalized state is
35.64 GB, for 106.93 GB of state under the compiled-method representation. These are logical
cohort totals rather than allocated-GPU peaks.

## Cliff screen and validity

The evaluator stores every ranking metric for every pair. It additionally emits
`fixed_cache_trajectories`: quality loss at each later current model, the extra loss from each
successive model update, and the largest late positive increment after at least two earlier
transitions.

A useful result is not merely a large theta0-to-theta1 transition. The 8+8 seed-0 matrix shows
that boundary is special. A credible fixed-window counterexample should instead satisfy all of
the following:

1. A non-theta0 cache remains relatively benign for at least two model updates and then has a
   materially larger positive loss increment.
2. The jump is visible in MeanRank and is directionally supported by AUC or a standard top-k
   metric, rather than being selected from one noisy metric after inspection.
3. Similar behavior occurs for more than one cache cohort or reproduces under new training seeds.
4. The claim is expressed as paired stale-versus-fresh loss within each endpoint. Longitudinal
   jump ratios remain exploratory because adjacent points use different next-day tasks.

Failure to find such a pattern is also informative: it means this dataset and update process do
not support a hard fixed-window failure claim, and the system motivation should be continuous
cost-quality control rather than a manufactured cliff.

## Result

The complete matrix does not support a universal delayed cliff. At fixed theta11/D16, cache age
strongly orders K/V drift but not ranking loss: age-to-K/V-drift Spearman is 0.817, while
age-to-MeanRank Spearman is approximately zero. After excluding the special theta0 base boundary,
age alone explains 6.15% of MeanRank variation and current-version identity explains 60.9%.
Adjacent one-update cohorts can even move in opposite directions.

The supported motivation is therefore narrower and more useful: a fixed age window is not a
sufficient quality policy, stale reuse can have a large harmful endpoint, and synchronization
must offer a bounded cost-fidelity action rather than infer that a version is safe. No generic
“two safe updates followed by collapse” claim is made.

The 6+10 balance check was not run. Once the research claim was frozen to age insufficiency rather
than a delayed cliff, another split search would not supply confirmatory evidence. Motivation is
now closed; subsequent work changes the migration design only.

The theta11/D16 full-user compiled design loop is documented separately in
`experiments/migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md`.

## Executed workflow

Run 4+12 seed 0 first. If it exposes a non-theta0 late jump, repeat only that frozen split with
seeds 1 and 2. If it does not, run the supported 6+10 seed-0 split as a balance check: it provides
nine evaluable versions while giving theta0 a larger base period. Do not search additional split
points or metrics after seeing results merely to create a cliff.

The one-command 4+12 workflow validates or prepares data, trains on four GPUs, validates all
checkpoints, and then runs the motivation matrix:

```bash
python scripts/run_kuairand_long_context_pipeline.py --base-days 4
```

If training completed but evaluation was interrupted, reuse the validated checkpoints:

```bash
python scripts/run_kuairand_long_context_pipeline.py --base-days 4 --reuse-training
```

The unused 6+10 balance-check entry point remains available through `--base-days 6`, but it is not
part of the completed result. Migration experiments remain in distinct protocol records.
