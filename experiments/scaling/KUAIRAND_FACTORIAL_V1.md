# KuaiRand factorial-v1: data bundle and model scale

## Scope

This study asks whether the fixed optimized suffix survives a simultaneous increase in retained
KuaiRand data and model capacity. It uses the same 14 base dates, five three-day updates,
theta-0-to-theta-5 cache age, 300 full-catalog evaluation users, and training seeds 0-3 as the
earlier six-layer result. No interval is selected from these cells.

The four cells are:

| Cell | Fitted catalog / active length | Architecture | Parameters |
|---|---:|---:|---:|
| baseline | 5k / 128 | 6 layers, hidden 96, 4 heads | 770,496 |
| more-data bundle | 20k / 256 | 6 layers, hidden 96, 4 heads | 2,210,496 |
| larger model | 5k / 128 | 12 layers, hidden 192, 8 heads | 3,219,456 |
| both | 20k / 256 | 12 layers, hidden 192, 8 heads | 6,099,456 |

The “more-data” factor is deliberately a stress-test bundle: it changes both retained catalog and
active context length. Its larger embedding table also raises parameter count even when the HSTU
architecture is fixed. It is not a causal decomposition of item count and sequence length.

## Data boundary

The local KuaiRand-1K standard logs contain 11,713,045 rows, 1,000 users, and 4,369,953 distinct
items over 31 dates. Vocabulary fitting remains restricted to the first 14 dates. The original
top-5k trace retains 429,417 rows, or 3.67% of the standard logs; top-20k retains 957,866 rows, or
8.18%. Thus this experiment uses more KuaiRand, but emphatically not all of it. The random-exposure
log and KuaiRand-27K are not included.

## Does the maintenance problem survive?

The table reports current full recomputation over theta-0 cache reuse at theta-5. Intervals use the
training seed as the replication unit.

| Cell | Full Best Rank gain, 95% CI | Full NDCG@100 gain, 95% CI | Full GPU ms/user |
|---|---:|---:|---:|
| baseline | 85.32 [53.74, 116.91] | 0.00693 [0.00295, 0.01090] | 0.069 |
| more-data bundle | 76.83 [20.73, 132.92] | 0.00338 [0.00260, 0.00416] | 0.151 |
| larger model | 68.72 [50.31, 87.13] | 0.00622 [0.00468, 0.00776] | 0.237 |
| both | 49.54 [19.68, 79.40] | 0.00310 [0.00030, 0.00590] | 0.626 |

The gap remains positive on both primary views in every cell. Its absolute Best Rank scale is not
directly comparable across 5k and 20k catalogs; after division by catalog size it becomes smaller
in the larger-data cells. The combined cell is therefore evidence that the problem persists, not
that staleness becomes more severe with scale.

## Fixed suffix result

Each entry is measured cost relative to optimized full followed by the ratio of cross-seed mean
Best Rank gain to the full gain. The proportional suffixes contain approximately one third, two
thirds, and all but the first layer.

| Cell | cheap | one-third suffix | two-thirds suffix | all-but-first suffix |
|---|---:|---:|---:|---:|
| baseline | 0.187 / 72.9% | 0.373 / 84.6% | 0.636 / 101.3% | 0.768 / 103.0% |
| more-data bundle | 0.112 / 79.5% | 0.289 / 89.6% | 0.621 / 106.0% | 0.788 / 111.9% |
| larger model | 0.170 / 89.8% | 0.392 / 101.1% | 0.678 / 108.1% | 0.894 / 107.5% |
| both | 0.099 / 142.4% | 0.345 / 152.2% | 0.664 / 135.8% | 0.904 / 133.6% |

The cost side scales cleanly. Relative cheap cost falls below 0.1 in the combined cell while full
latency rises 9.1 times over the baseline. The optimized suffix therefore continues to expose
intermediate compute points after both axes grow.

The quality side requires a stricter interpretation than the earlier “above 100% is noise”
shorthand. In the combined cell, the one-third suffix improves Best Rank over full by 25.84 with a
paired seed interval [7.69, 44.00], while its NDCG@100 difference is 0.000004 with interval
[-0.00186, 0.00187]. Partial version mixing can therefore act differently from current-model
consistency on a task metric. Full recomputation remains the fidelity reference, but it is not a
mathematical upper bound on realized recommendation quality. No overall superiority claim is made.

## Decision

- The fixed structural operator passes this first data/model factorial stress test.
- Do not use cross-catalog raw Rank or recovery above 100% as a scale trend.
- Always report absolute quality, paired difference from full, and cache fidelity alongside
  recovery.
- The top-20k bundle still underuses KuaiRand. The follow-up in
  `KUAIRAND_DATA_UTILIZATION_V1.md` tests a larger catalog and chunked training rather than merely
  increasing the embedding table again.

## Artifacts

- `scripts/motivation_validity.py`
- `scripts/scaling_validity.py`
- `scripts/kuairand_data_coverage.py`
- `scripts/summarize_kuairand_factorial.py`
- `results/scaling/kuairand_data_coverage.json`
- `results/scaling/factorial_{more_data,larger_model,both}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/kuairand_factorial_summary.json`
- `checkpoints/scaling/factorial_{more_data,larger_model,both}_seed{0,1,2,3}/`
