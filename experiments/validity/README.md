# Validity-v1 motivation repair

This experiment repairs the smallest set of issues that could invalidate the cache-version
motivation. It does not replace the simplified HSTU with a large external implementation.

## Repairs

- The training target is item `t+1` from hidden state `t`, not the current item.
- Only engaged targets and targets belonging to the current stream day contribute to loss.
- KuaiRand labels and sequence lengths propagate through histories and batches.
- Evaluation positives are engaged items, not all impressions.
- Padding positions are excluded from last-state selection and produce zero cached K/V.
- The item vocabulary can be fitted on base dates only, preventing future frequency leakage.
- Serving evaluation uses an old-version prefix cache plus the latest behavior token under the
  current model. Fresh is a complete current-model recomputation.

## Protocol

- Data: KuaiRand-1K standard logs, 14 base days and 17 streaming days.
- Vocabulary: top 5,000 items fitted only on the base period; 973 eligible users.
- Model: simplified HSTU, 3 layers, hidden size 128, 4 heads, head dimension 32, sequence length
  128, ReLU pointwise attention; 908,160 parameters.
- Training: 6 base epochs, then 3-day windows with 2 epochs per day.
- Evaluation: 300 users with engaged positives on the next unseen day, full-catalog ranking.
- Replication: seeds 0, 1, 2, and 3 on four A40 GPUs.
- Statistical unit: training seed. User-level bootstrap intervals remain within-window diagnostics.

Two staleness regimes are reported:

- `one_step`: prefix K/V from the model before the latest three-day update.
- `cumulative_theta0`: prefix K/V from the base model. This is a fixed-input cache-age stress test,
  not a mixed-version cache rollout.

## Results

The table reports the mean gain from fresh recomputation over stale reuse, averaged within each
seed and then across four seeds.

| Regime | Best-rank gain | Mean-rank gain | MRR gain | NDCG@100 gain | Hit@100 gain |
|---|---:|---:|---:|---:|---:|
| One-step stale | 4.15 | 6.26 | 0.00132 | 0.00086 | 0.00533 |
| Cumulative theta-0 stale | 63.39 | 109.12 | 0.00404 | 0.00527 | 0.04150 |

Across seeds, the cumulative best-rank gain has a 95% t interval of `[41.57, 85.20]`. Its
Spearman correlation with cumulative parameter distance is `0.975` across the five cache ages.
The one-step effect is much smaller and reverses on the fifth window, so it must not be described
as uniformly harmful at every update.

Raw relative K/V drift does not identify which users benefit from recomputation. Across 20
seed-window cells, the mean per-user Spearman correlation between K/V drift and rank-utility gain
is `0.020`, with a 95% interval of `[-0.012, 0.052]`. Drift-based selection is approximately
random at equal budgets, while an oracle based on realized quality gain has substantial headroom.

The supported conclusion is therefore:

1. Cache version age can produce a reproducible recommendation-quality loss.
2. A single update usually has a small effect, leaving room for reuse.
3. Unweighted `||delta KV||` is not yet a useful serving decision variable.

This passes the motivation gate. The latest fixed cheap-projection plus deepest-top-N full-layer
method is evaluated separately in [LAYERWISE_METHOD.md](LAYERWISE_METHOD.md). It does not require
per-user reuse selection or a JVP predictor.

## Artifacts

- `scripts/motivation_validity.py`: training and per-window evaluation.
- `scripts/summarize_validity.py`: seed-level aggregation.
- `results/validity/core_seed{0,1,2,3}.json`: complete per-user outputs.
- `results/validity/multiseed_summary.json`: aggregate statistics.
- `checkpoints/validity/core_seed{0,1,2,3}/`: reproducible checkpoint trajectories.

The four runs each take about 45-48 seconds after launch on an A40 with the current local dataset.
