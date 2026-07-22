# Streaming-training value control

This control tests whether streaming training is itself useful and how much of that value is lost
when the current model consumes a version-stale prefix cache.

## Conditions

All three conditions use the same current user history and future-day engaged positives:

- `frozen`: theta-0 model, theta-0-consistent prefix cache, and theta-0 scoring head;
- `full reuse`: current theta-t model, theta-0 prefix cache, and current scoring head;
- `full compute`: current theta-t model, current prefix cache, and current scoring head.

The contrasts have distinct meanings:

- `full compute - frozen`: total value of streaming model updates;
- `full reuse - frozen`: streaming value retained despite stale K/V;
- `full compute - full reuse`: value attributable to cache maintenance.

The experiment uses the existing six-layer validity-v1 checkpoints, 300 users per seed, full
5,000-item ranking, theta-0 to theta-1/3/5, and training seeds 0-3. Both current and frozen
prefix-plus-latest-token paths match their complete full-history forwards within `4.3e-6` maximum
absolute error.

## Results

Every entry is a positive-oriented gain. Best Rank is the number of ranks reduced; NDCG@100 is an
absolute increase. Brackets are seed-level 95% t intervals over four training seeds.

| Cache age | Streaming value: full compute over frozen | Streaming value retained by full reuse | Cache-maintenance value: full compute over reuse |
|---|---:|---:|---:|
| theta-1, Best Rank | 111.43 `[99.52, 123.34]` | 105.77 `[98.30, 113.23]` | 5.66 `[-1.39, 12.71]` |
| theta-1, NDCG@100 | 0.01315 `[0.01065, 0.01566]` | 0.01033 `[0.00671, 0.01394]` | 0.00283 `[0.00120, 0.00446]` |
| theta-3, Best Rank | 237.97 `[223.66, 252.29]` | 196.29 `[178.29, 214.29]` | 41.68 `[23.62, 59.74]` |
| theta-3, NDCG@100 | 0.02485 `[0.02107, 0.02864]` | 0.02141 `[0.01902, 0.02381]` | 0.00344 `[-0.00015, 0.00703]` |
| theta-5, Best Rank | 484.34 `[462.15, 506.54]` | 399.02 `[370.79, 427.26]` | 85.32 `[53.74, 116.91]` |
| theta-5, NDCG@100 | 0.02218 `[0.01840, 0.02596]` | 0.01525 `[0.01246, 0.01804]` | 0.00693 `[0.00295, 0.01090]` |

Streaming training is beneficial under both cache conditions at every evaluated age. Stale reuse
does not erase the update benefit, but the missing fraction grows with cache age:

| Cache age | Rank value lost to stale cache | NDCG@100 value lost to stale cache |
|---|---:|---:|
| theta-1 | 4.9% | 22.1% |
| theta-3 | 17.5% | 13.4% |
| theta-5 | 17.6% | 30.8% |

The ratios are descriptive because they divide two estimated quantities. At theta-5, their
seed-level intervals are `[11.5%, 23.7%]` for Best Rank and `[16.4%, 45.3%]` for NDCG@100.

MRR does not improve uniformly with cache maintenance: at theta-5 its mean full-compute-over-reuse
change is `-0.00251`, with interval `[-0.00607, 0.00105]`. The result therefore supports an
age-growing gap on the primary full-catalog Best Rank and NDCG views, not universal improvement of
every ranking metric.

## Supported conclusion

The correct motivation is not that a stale cache makes streaming training useless. Instead:

1. streaming updates create large, reproducible recommendation value over a frozen model;
2. full reuse preserves most of that value at low cost;
3. version-consistent cache maintenance recovers an additional component that becomes material at
   older cache ages.

This establishes the complete `streaming update -> cache inconsistency -> recoverable quality gap`
chain under the repaired protocol.

## Artifacts

- `scripts/streaming_value_control.py`: per-seed control evaluation.
- `scripts/summarize_streaming_value_control.py`: seed-level aggregation.
- `results/validity/streaming_control6l_seed{0,1,2,3}.json`: complete records.
- `results/validity/streaming_control6l_summary.json`: aggregate statistics and value partitions.
