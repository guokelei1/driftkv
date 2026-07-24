# Compiled low-rank cache migration v1

> Status: four-seed cross-dataset method screen completed on 2026-07-23.
> Protocol: `compiled_low_rank_migration_v1`.

## 1. Design

For layer \(l\), let \(z_l\) be the cached old-version
\(\operatorname{Norm}(x_l)\). Cheap refresh produces

\[
\widehat C_l^{\mathrm{cheap}}
=
z_l [W_l^K,W_l^V]_{\theta_t}.
\]

On a small calibration set, full recomputation exposes the residual

\[
R_l
=
C_l(\theta_t)-\widehat C_l^{\mathrm{cheap}}.
\]

The method fits a shared, per-model-version, rank-\(r\) affine residual map

\[
R_l \approx (z_l-\mu_l)A_{l,r}B_{l,r}+b_l.
\]

This is not executed as two extra matrix multiplications for every cache. Because the correction
is linear in \(z_l\), it is folded once per model version into

\[
\widetilde W_l=[W_l^K,W_l^V]_{\theta_t}+A_{l,r}B_{l,r},
\qquad
\widetilde b_l=b_l-\mu_lA_{l,r}B_{l,r}.
\]

Online migration is therefore one prepacked, layer-batched projection
\(z_l\widetilde W_l+\widetilde b_l\). It has the same arithmetic structure as optimized cheap
refresh while using a calibration-derived projection rather than current \(W^K/W^V\) alone.

The adapter is fitted once for an `(old model version, current model version)` cohort. It is not
fitted per user, and it does not revive the retired user-level JVP/drift route.

## 2. Frozen selection rule

Each training seed uses three disjoint user sets selected by a seeded permutation independent of
labels and outcomes:

| Dataset | Adapter-fit users | Rank-selection probe | Held-out test |
|---|---:|---:|---:|
| KuaiRand | 40 | 60 | 200 |
| QB fixed horizon | 80 | 120 | 400 |
| QK top-5k | 80 | 120 | 400 |

Candidate ranks are `2, 4, 8, 16, 32, 64, 96`. The selected action is the smallest rank whose
probe-set relative K/V reconstruction error closes at least 50% of the stale-to-fresh cache gap.
If no rank passes, the ladder falls back to full recomputation. The 50% target was fixed during
seed-0 discovery; seeds 1-3 are `frozen_rule_replication`.

The resulting ranks vary by dataset and training seed:

| Dataset | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|---:|
| KuaiRand | 2 | 8 | 4 | 2 |
| QB fixed horizon | 8 | 4 | 2 | 16 |
| QK top-5k | 16 | 16 | 8 | 8 |

Thus the mechanism does not hide a fixed suffix depth or a fixed adapter rank.

## 3. Bounded architecture screen

Before freezing the compiled adapter, seed-0 discovery compared:

- current RMSNorm over cached old hidden states;
- a carried current-embedding delta with scales from 0.25 to 2;
- optimized suffix-2 and suffix-4;
- low-rank residual migration.

Current renormalization was nearly identical to cheap refresh despite higher cost. Embedding-delta
helped QB and QK at some scales but had little KuaiRand benefit and selected incompatible scales
across datasets. The low-rank residual was the only new candidate that materially improved the
quality/cost frontier on all three datasets, so only it passed to four-seed replication.

## 4. Held-out four-seed results

Intervals below are seed-level 95% t intervals. Training seed is the replication unit. Cache
recovery is relative reduction in K/V reconstruction error from reuse toward exact fresh K/V.
Task-quality gains are paired method-minus-reuse differences with metric direction corrected.

| Dataset | GPU time / full | Cache recovery | BestRank gain | Rank-utility gain | NDCG@100 gain |
|---|---:|---:|---:|---:|---:|
| KuaiRand | `0.123 [0.122, 0.124]` | `0.521 [0.499, 0.542]` | `79.93 [36.70, 123.16]` | `0.1687 [0.0932, 0.2441]` | `0.00626 [0.00334, 0.00918]` |
| QB fixed horizon | `0.115 [0.113, 0.117]` | `0.515 [0.483, 0.547]` | `20.74 [-0.88, 42.36]` | `0.0456 [0.0059, 0.0852]` | `0.00175 [0.00053, 0.00297]` |
| QK top-5k | `0.109 [0.106, 0.112]` | `0.536 [0.491, 0.582]` | `12.43 [-3.60, 28.46]` | `0.0384 [0.0052, 0.0716]` | `0.00341 [-0.00330, 0.01013]` |

BestRank, rank utility, and NDCG@100 are positive in all 4/4 individual seeds for every dataset.
The QB and QK BestRank intervals remain wide with only four training seeds; QK NDCG is also
unresolved. Rank utility is the only reported task-quality view whose seed interval is positive
on all three datasets.

The corresponding mean full-recompute BestRank gains are 79.23, 48.06, and 10.77. The migrated
gain can exceed full on a ranking metric, especially on KuaiRand and QK. Full recomputation is the
cache-fidelity reference, not a guaranteed ranking-quality upper bound, so no superiority or
equivalence claim is made from those ratios.

## 5. Calibration and amortization

The online numbers above are resident-GPU kernel time. The current unoptimized one-time
calibration includes collecting fresh targets and fitting the rank-96 basis before truncation:

| Dataset | Selected migration ms/user | Full ms/user | One-time calibration | Descriptive break-even cohort |
|---|---:|---:|---:|---:|
| KuaiRand | 0.00890 | 0.07246 | 303 ms | 4.8k caches |
| QB fixed horizon | 0.00722 | 0.06285 | 353 ms | 6.3k caches |
| QK top-5k | 0.00677 | 0.06213 | 405 ms | 7.4k caches |

Break-even divides one-time calibration time by per-cache kernel-time savings. It excludes cache
state reads/writes, transfers, scheduling, and adapter admission, and is not an end-to-end serving
claim. The current local evaluation cohorts are smaller than some of these break-even points; the
method is intended for a version cohort with many caches, where the shared fit can be amortized.
The current float32 prepacked projection is 111,744 elements, or 446,976 bytes, per model-version
pair. This shared state is separate from the per-user cached normalized states and must be counted
when multiple old-version cohorts coexist.

## 6. What this result supports

- A shared update-level calibration can replace an arbitrary fixed suffix or rank.
- The compiled operator gives a reproducible interior point between reuse and full recomputation
  on KuaiRand, QB, and QK.
- The same 50% fidelity target adapts rank across datasets and seeds while delivering approximately
  89% kernel-time savings relative to full recomputation.
- The result concerns theta-0 caches consumed by theta-5 on the aligned 6L/H96 protocols. It does
  not yet establish adjacent-version migration, organically mixed cache versions, or the stronger
  top-50k/length-512 KuaiRand operating point.

## 7. Required next checks

1. Freeze the 50% target and test the compiled operator on top-50k/all-chunks KuaiRand without
   another rank search.
2. Evaluate adjacent and mixed old/current version pairs, with one fitted adapter per version
   cohort.
3. Compare against periodic full recomputation and age-only thresholds at equal cost and equal
   quality.
4. Include normalized-state movement, adapter calibration/admission, throughput, and tail latency.
5. Sweep calibration-set size only at seed 0; replicate a new size only if it changes the
   end-to-end Pareto frontier.

## 8. Reproduction entry points

- Method and selector: `scripts/low_rank_migration_search.py`
- Migration implementation: `src/hstu_kvcache/migration/low_rank.py`
- Unit tests: `tests/test_layerwise.py`
- Raw local results:
  - `results/validity/kuai_low_rank_migration_seed{0..3}.json`
  - `results/exposure/qb_fixed_horizon_low_rank_migration_seed{0..3}.json`
  - `results/exposure/qk_top5k_low_rank_migration_seed{0..3}.json`
