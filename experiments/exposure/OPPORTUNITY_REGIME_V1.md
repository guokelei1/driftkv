# Long-context opportunity regime

> Status: protocol frozen before training or migration-quality evaluation on 2026-07-23;
> seed-0 screen completed as a negative gate on 2026-07-23.

## 1. Purpose

The method needs a non-trivial interior operating point:

1. streaming training must improve quality over a frozen model;
2. stale reuse must retain substantial value rather than fail completely;
3. version-consistent recomputation must recover a stable, non-negligible gap;
4. full-prefix recomputation must be expensive enough for a partial migration path to matter.

The first three conditions define quality opportunity. The fourth defines systems opportunity.
Migration quality is not used to select the data, update strength, cache age, or cohort.

For an oriented metric $G$:

$$
S=G(\mathrm{full\ compute},\mathrm{frozen}), \qquad
M=G(\mathrm{full\ compute},\mathrm{reuse}), \qquad
\tau=M/S.
$$

$S$ is streaming-training value, $M$ is the cache-maintenance denominator, and $\tau$ is the
staleness tax. The primary opportunity regime is:

- $S>0$ and $M>0$ on seed 0;
- $0.15 \le \tau_{\mathrm{BestRank}} \le 0.45$;
- stale reuse retains at least 55% of full-compute streaming value;
- MeanRank or NDCG@100 has the same maintenance direction;
- base and streaming losses remain finite without an optimizer change.

Passing seed-0 cells expand to four training seeds. The final gate requires the BestRank
streaming and maintenance seed intervals to exclude zero. A secondary metric is reported but is
not required to exclude zero in the seed-0 screen.

If endpoint tax is above 0.45, use a younger cache version from the fixed-current matrix. If it is
below 0.15, first increase cache age within the trained version chain. Optimizer strength is not
changed unless no available age enters the opportunity interval. This order prevents an
unnecessary learning-rate search.

If the oldest available cache still has tax below 0.15 while full-compute streaming value remains
positive, the frozen rescue order is stream learning rate `2e-4` and then `4e-4`, with epochs held
at two. Stop at the first accepted cell. Do not scan epochs and learning rate jointly. Reject a
cell if full-compute streaming value becomes non-positive or tax exceeds 0.45.

## 2. Frozen long-context cohorts

Both datasets retain observed negative exposures and the existing next-positive-item target.
Selection uses raw activity availability and retained base activity only, never future feedback
labels or migration outcomes.

| Axis | Tenrec QB | Tenrec QK |
|---|---:|---:|
| catalog | top-50k | top-20k |
| users | 1,000 | 1,000 |
| base raw exposures | 64 | 64 |
| stream | 12 × 16 raw exposures | 12 × 16 raw exposures |
| complete raw horizon | 256 | 256 |
| minimum retained base events | 48 | 48 |
| model | 6L/H96, four heads | 6L/H96, four heads |
| maximum sequence length | 256 | 256 |
| base optimization | 6 epochs, `3e-4` | 6 epochs, `3e-4` |
| stream optimization | 2 epochs/update, `1e-4` | 2 epochs/update, `1e-4` |

QB keeps top-50k because its earlier top-5k gate weakened the streaming denominator. QK uses
top-20k as a pre-outcome compromise: top-5k leaves only 117 users with minimum 48 retained base
events and at least 256 retained events, while top-50k previously produced a sparse 5.09M-parameter
model with an unresolved method denominator. The audit records 11,599 qualifying top-20k QK
sequences before the complete-horizon cohort restriction.

Raw sequence coverage also rules out a common 512-token primary cell: QB has only 470 raw users
with at least 512 events, and top-5k QK retains only three sequences of that length. Sequence 256
is the largest defensible shared first gate.

These are mechanism stress cohorts, not population estimates. Complete-horizon conditioning and
Tenrec's user-local ordinal time must accompany every claim.

## 3. Execution order

1. Materialize both frozen cohorts and record actual retained history lengths and target counts.
2. Train seed 0 with the unchanged optimizer.
3. At fixed current theta-11 and the twelfth unseen window, evaluate cache versions from newest to
   theta-0 using only frozen, reuse, and full compute.
4. Select the cache age closest to the center of the accepted tax interval without looking at any
   migration result.
5. Measure resident-GPU full, cheap, and frozen suffix costs at actual length quantiles and at
   synthetic lengths 128/256/512.
6. Only if the opportunity gate passes, run cheap and the frozen deepest-suffix candidates.
7. Expand a Pareto-changing candidate to seeds 1-3.

## 4. Stop conditions

Stop this branch before migration-quality evaluation if:

- no trained cache age has positive BestRank streaming and maintenance value;
- every age has tax below 0.15 or above 0.45 under the unchanged optimizer;
- full compute does not improve at least one secondary metric directionally;
- retained final histories remain too short to change measured full-recompute cost;
- the cohort becomes too small for seed-level evaluation.

Failure is retained as evidence that long raw horizon or larger catalog alone does not create a
useful cache-migration regime.

## 5. Seed-0 screen result

Both materialized cohorts are genuinely longer than the original Tenrec controls after catalog
filtering:

| Dataset | Retained rows | Retained history min / p50 / p90 / max |
|---|---:|---:|
| QB top-50k | 241,221 | 160 / 244 / 254 / 256 |
| QK top-20k | 202,636 | 112 / 206 / 229 / 250 |

The systems side of the opportunity gate passes. On one A40 at batch 32, the six-layer operator
takes `4.504 ms` for full recomputation and `0.512 ms` for cheap refresh at QB's median retained
length 244 (`0.114x`). At QK's median length 206, the corresponding times are `3.578 ms` and
`0.439 ms` (`0.123x`). These remain resident-GPU kernel measurements rather than end-to-end
serving latency.

The quality side does not pass:

| Dataset | Stream LR | Full compute over frozen BestRank | Maintenance BestRank | Staleness tax |
|---|---:|---:|---:|---:|
| QB | `1e-4` | 139.86 | 7.88 | 5.6% |
| QB | `2e-4` | 148.59 | 1.66 | 1.1% |
| QB | `4e-4` | 144.15 | 3.62 | 2.5% |
| QK | `1e-4` | 549.49 | 19.88 | 3.6% |
| QK | `2e-4` | 698.40 | 41.70 | 6.0% |
| QK | `4e-4` | 783.93 | 29.27 | 3.7% |

Every cell has positive BestRank streaming value and maintenance, but every oldest-cache tax is
below the frozen 15% lower bound. NDCG@100 maintenance is non-positive in five of six cells. The
learning-rate rescue is non-monotone, so parameter distance cannot repair the gate by itself.
Neither dataset proceeds to migration-quality evaluation or seed expansion.

The result separates two axes that should not be conflated:

- longer histories make full recomputation materially more expensive;
- they do not automatically make model-version cache incompatibility a larger fraction of useful
  streaming-training value.

Consequently, the primary joint opportunity regime remains the already replicated KuaiRand
top-50k/all-chunks cell: full compute over frozen is 3,837.67 BestRank, maintenance is 885.56,
staleness tax is 23.1%, reuse retains 76.9%, and cheap costs 0.058x full at length 512. The aligned
QB/QK coarse protocols remain cross-dataset motivation evidence, while this long-context branch is
a negative stress-cohort result rather than a replacement.

Tracked compact evidence is in
`results/exposure/long_context_opportunity_summary.json`. Per-seed core and cache-version matrices
remain local ignored artifacts.
