# Fixed-task data/model-capacity motivation protocol

> Frozen before v2 seed-0 model training on 2026-07-23.

## Correction from v1

The rejected v1 screen changed catalog size, user count, width, and depth together. Its nine
seed-0 cells are diagnostic only: effective targets increased, but task difficulty also changed
and the 12L/H192 Tenrec cells had too little data. No additional v1 seeds are valid.

V2 holds each dataset's accepted prediction task and vocabulary fixed. Only nested training users
and model capacity increase. This implements the intended small-data/small-model,
medium-data/medium-model, and large-data/large-model comparison.

## Frozen scale points

| Tier | KuaiRand users | QB users | QK users | Model |
|---|---:|---:|---:|---|
| small | 250 | 1,000 | 1,000 | 3L, H64, 4 heads, head dim 16 |
| medium | 500 | 3,000 | 3,000 | 6L, H96, 4 heads, head dim 24 |
| large | all eligible, at most 1,000 | 5,000 | 5,000 | 9L, H128, 4 heads, head dim 32 |

KuaiRand and QB keep a 50,000-item base-fitted vocabulary. QK keeps the accepted dense 5,000-item
base-fitted vocabulary. The user cohorts are nested within each dataset and selected using base
activity only. Labels and model outcomes are never used for cohort selection.

Actual users, rows, tokens, eligible targets, and model parameters are acceptance checks. Effective
base and stream targets must both increase from small to medium to large within every dataset.

## Shared training and serving semantics

- Item `t+1` is predicted from hidden state `t`.
- Every observed exposure enters context; only observed positive feedback is a target.
- Vocabulary and cohort order are fitted before streaming updates.
- Sequence length is 128 and batch size is 32.
- Theta-0 uses six epochs at `3e-4`; every update uses two epochs at `1e-4`.
- There are 11 updates and a final unseen evaluation window.
- Full-catalog ranking evaluates at most 1,000 active users.
- Stale serving uses an old-version prefix cache and the current model for the latest token.
- Full recomputation is the version-consistent cache reference, not an assumed ranking ceiling.

KuaiRand uses 14 real base dates, 11 one-day updates, and complete chunked training. QB uses its
fixed raw-horizon top-50k cohort. QK uses the accepted top-5k base-activity cohort. Both Tenrec
tables use a 64-exposure base and 12 four-exposure ordinal windows. Calendar and ordinal age are
reported separately.

## Complete motivation artifacts

Every dataset-tier cell produces:

1. one-step and cumulative theta-0 cache comparisons during theta-1 through theta-11 training;
2. frozen/full-reuse/full-compute controls at theta-1/3/5/7/9/11;
3. a fixed theta-11 endpoint over stale theta-10 through theta-0 caches;
4. resident-GPU full-prefix timing at batch 32 and length 128.

Protocol strings:

- `motivation_capacity_v2_training`
- `motivation_capacity_v2_streaming_control`
- `motivation_capacity_v2_cache_version_matrix`
- `motivation_capacity_v2_operator_cost`

Seed 0 is the structural gate. Seeds 1-3 are replication units and are run only after the complete
matrix has been inspected for protocol failures, not to select a favorable metric or endpoint.

## Completion

The frozen matrix is complete: 9 dataset-tier cells, 4 independent training seeds per cell, and
108 matched core/control/cache-age artifacts. Resident-GPU cost was measured once per cell after
the quality protocol was frozen. The compact machine-readable summary is
`results/motivation_scale/capacity_v2_summary.json`; it can be regenerated with
`scripts/summarize_motivation_capacity_v2.py`.

All acceptance checks pass. Within every dataset, base and stream eligible targets increase from
small to medium to large:

| Dataset | Small base / stream targets | Medium base / stream targets | Large base / stream targets |
|---|---:|---:|---:|
| KuaiRand | 318,090 / 36,204 | 492,711 / 67,956 | 620,958 / 107,863 |
| QB | 37,494 / 29,596 | 126,610 / 94,182 | 187,711 / 138,235 |
| QK | 40,074 / 12,767 | 108,580 / 33,062 | 170,800 / 51,617 |

At batch 32 and length 128, seed-0 full-prefix latency grows from approximately 1.05 ms at 3L/H64
to 2.09 ms at 6L/H96 and 3.36 ms at 9L/H128. Cheap projection refresh costs 18.5%-20.4% of full
over these cells. These are resident-GPU operator measurements, not end-to-end serving latency.

## Four-seed motivation result

The primary table reports theta-11 full-catalog BestRank improvement. Positive values mean better
ranking. `Full`, `Reuse`, and `Maintenance` respectively denote full compute over frozen, full
reuse over frozen, and full compute over reuse. Tax is computed within each seed as
`Maintenance / Full`; brackets are seed-level 95% t intervals.

| Dataset | Tier | Full | Reuse | Maintenance | Staleness tax |
|---|---|---:|---:|---:|---:|
| KuaiRand | small | 1918.41 `[1341.70, 2495.13]` | 1649.41 `[1278.15, 2020.67]` | 269.01 `[-142.11, 680.12]` | 12.8% `[-7.7%, 33.4%]` |
| KuaiRand | medium | 3052.74 `[2687.65, 3417.82]` | 2732.45 `[2436.33, 3028.58]` | 320.29 `[-69.59, 710.16]` | 10.2% `[-1.5%, 21.9%]` |
| KuaiRand | large | 4415.56 `[3936.14, 4894.99]` | 2819.20 `[2388.48, 3249.91]` | 1596.37 `[1030.21, 2162.52]` | 36.0% `[25.3%, 46.7%]` |
| QB | small | 149.93 `[73.93, 225.94]` | 147.59 `[73.66, 221.53]` | 2.34 `[-1.10, 5.78]` | 1.5% `[-1.0%, 3.9%]` |
| QB | medium | 106.56 `[103.90, 109.21]` | 112.87 `[104.14, 121.59]` | -6.31 `[-17.15, 4.53]` | -6.0% `[-16.2%, 4.2%]` |
| QB | large | 141.87 `[36.99, 246.76]` | 66.70 `[0.37, 133.04]` | 75.17 `[22.06, 128.28]` | 54.8% `[38.3%, 71.3%]` |
| QK | small | 28.60 `[16.32, 40.87]` | 24.05 `[18.07, 30.03]` | 4.55 `[-4.30, 13.39]` | 12.6% `[-16.6%, 41.8%]` |
| QK | medium | 42.78 `[18.96, 66.60]` | 30.63 `[24.60, 36.66]` | 12.14 `[-10.89, 35.18]` | 23.3% `[-9.7%, 56.4%]` |
| QK | large | 44.93 `[19.87, 69.99]` | 40.27 `[35.38, 45.16]` | 4.66 `[-16.44, 25.75]` | -0.5% `[-66.4%, 65.5%]` |

The complete result supports three claims and rejects one stronger claim:

1. Streaming training has positive BestRank value in 4/4 seeds in all nine cells, and every
   seed-level interval excludes zero.
2. Theta-0 cache reuse also retains positive value in 4/4 seeds in all nine cells. Old K/V does
   not make the updated model fail immediately.
3. The interior quality opportunity becomes strong and replicated in large KuaiRand and large QB.
   Their maintenance gaps and taxes are positive in 4/4 seeds with intervals excluding zero.
   KuaiRand rank utility and NDCG@100 corroborate this result; large-QB rank utility does as well,
   while its NDCG interval touches zero.
4. Increasing data and model capacity together does not monotonically amplify cache invalidation
   on every dataset. QK medium has a positive BestRank maintenance sign in 4/4 seeds but a wide
   interval; QK large falls to 3/4 signs and conflicts on NDCG. QB medium also has a BestRank
   direction conflict even though rank utility favors maintenance in 4/4 seeds.

The paired large-minus-small BestRank-tax change is +23.2 percentage points on KuaiRand
`[-1.0, 47.4]`, +53.4 points on QB `[36.9, 69.8]`, and -13.1 points on QK
`[-81.4, 55.3]`. Therefore the defensible scale conclusion is that larger operating points can
create a substantially wider migration opportunity, not that capacity alone guarantees one.

## Cache-age implication

At the large operating point, the seed-level age/tax Spearman coefficient is positive in all four
seeds on KuaiRand, QB, and QK, with means 0.857, 0.784, and 0.448. Only KuaiRand and QB have
intervals excluding zero. QK medium is more stable than QK large, with all four correlations
positive and a mean of 0.616.

Seed 0 selected the largest local BestRank-tax transition before replication. Only large
KuaiRand's age 10 to 11 transition replicates cleanly on seeds 1-3: +16.8 percentage points
`[6.0, 27.5]`, positive in 3/3 seeds. The seed-0 transition does not replicate at a fixed age on
large QB or QK. This strengthens the narrower conclusion already suggested by the fine matrices:
cache age orders risk in some regimes, but a universal fixed-age trigger is not a calibrated
quality policy. It does not support claiming a common sudden-failure age across datasets or
capacities.

## Decision

V2 completes the requested multi-capacity motivation pass. It is positive for the two endpoint
facts needed by the paper across all nine cells: streaming updates are useful and indefinite stale
reuse leaves a nontrivial systems question. The recoverable maintenance gap is explicitly
regime-dependent. Large KuaiRand and large QB are the strongest method-development targets; the
accepted earlier 6L/H96 QK result and this medium-QK cell provide cross-dataset evidence, while
large QK is a scale boundary rather than a positive endpoint.

No migration operator was selected or tuned from this matrix.
