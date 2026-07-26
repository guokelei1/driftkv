# Exposure-compatible dataset expansion audit

> Status: data-capacity audit and frozen ordered-exposure materialization complete as of
> 2026-07-27. Model results and the accepted expansion boundary are reported separately in
> `experiments/exposure/ORDERED_EXPOSURE_V1.md`.

## 1. Decision

Taobao UserBehavior is not the primary second dataset. Its rows are user actions rather than a
served impression log, so it cannot supply true unclicked exposures without changing the task or
constructing synthetic negatives. The audit is retained as a semantic boundary, not as a failed
model experiment.

Tenrec QK-video, Tenrec QB-video, and ZhihuRec all contain observed impressions with negative
feedback and therefore preserve the KuaiRand task structure:

- every row may enter the user context;
- only observed positive feedback becomes a next-item training or evaluation target;
- a zero-feedback row remains a real exposure rather than a sampled negative.

Tenrec QK-video is the closest field-level match to KuaiRand because it is a video dataset with
click, follow, like, and share feedback. ZhihuRec is the cleanest temporal complement because it
has impression timestamps and comes from a different domain. Tenrec QB-video is small enough to
be the first loader and one-seed pipeline check. The accepted positive extensions are QB and QK,
reported explicitly as related tables from one Tenrec collection rather than two independent data
sources. ZhihuRec is retained as a documented negative maintenance boundary: its exposure
semantics pass, but the current stream construction does not yield a task-independent maintenance
signal. It should not be revived merely to increase the dataset count; doing so requires a new,
predeclared protocol that resolves that boundary.

## 2. Raw comparison

| Dataset | Ordering evidence | Rows | Users | Items | Positive rows | Users with length >=128 | Users with length >=512 |
|---|---|---:|---:|---:|---:|---:|---:|
| KuaiRand-1K standard | millisecond timestamps, 31 dates | 11,713,045 | 1,000 | 4,369,953 | 38.52% | 999 | 988 |
| Tenrec QK-video | official per-user file order, no timestamps | 493,458,970 | 5,022,750 | 3,753,436 | 30.01% | 1,156,391 | 39,615 |
| Tenrec QB-video | official per-user file order, no timestamps | 2,442,299 | 34,240 | 130,637 | 69.83% | 4,474 | 470 |
| ZhihuRec impressions | impression timestamps, 12 UTC dates | 99,978,523 | 798,086 | 554,976 | 26.99% | 476,339 | 0 |

The positive rule is any `click/like/follow/comment/forward/long_view` for KuaiRand, any
`click/follow/like/share` for Tenrec, and `click_timestamp > 0` for ZhihuRec. Tenrec
`watching_times` is retained as an optional feature but is not silently promoted to a positive
label.

ZhihuRec caps every user at 160 impressions and every audited user has at least ten clicks. It is
therefore strong for a length-128 cross-domain check but cannot replace KuaiRand's long-context
scaling evidence. Tenrec QK has ample raw long sequences, although catalog filtering reduces the
number that remain long.

## 3. Leak-free ordered-exposure audit

For each new dataset, `ordered_exposure_data_audit_v1` applies the same frozen rule:

1. The first 64 raw impressions of each user form the base prefix.
2. Item frequency, item vocabulary, and cohort selection use only that base prefix.
3. Catalog sizes 5k, 20k, 50k, 100k, and 250k are audited; top-50k is the primary setting.
4. Six consecutive windows contain eight raw impressions per user. Window 0 updates
   `theta_0 -> theta_1`; window 1 evaluates `theta_1` before it is ingested, and the pattern repeats
   through `theta_5` and window 5.
5. A user is never selected because it survives into a future window. Attrition is reported rather
   than removed with future information.

This is a real chronological protocol for ZhihuRec. For Tenrec it is an ordinal replay based on
the official within-user file order and must not be described as calendar-time drift. QK and QB
users are not stored in one contiguous block; the audit explicitly merges repeated blocks in
stable file order rather than assuming contiguity. The full scans contain 87,430,415 QK and 16,706
QB resumed user blocks, so a loader that assumes one contiguous block per user is incorrect.

ZhihuRec has no within-user impression-time reversal, but 58,379 clicked rows record a click time
earlier than the impression time. The primary rule uses only `click_timestamp > 0` as the label and
orders by impression time, so this anomaly does not alter the audit ordering; it should still be
reported and ignored or cleaned consistently in the eventual loader.

The current KuaiRand top-50k result uses a different boundary—14 real base days—so its retention
percentage is useful context but not a numerically matched split.

## 4. How much data is usable

### 4.1 Full top-50k capacity

| Dataset/protocol | Retained base rows | Retained stream rows | Retained total | Raw-row retention | Users with 64 retained base rows |
|---|---:|---:|---:|---:|---:|
| KuaiRand, first 14 base dates | 1,276,718 | 287,098 | 1,563,816 | 13.35% | 980 active users |
| Tenrec QK, first 64/user | 191,351,252 | 154,316,558 | 345,667,810 | 70.05% | 3,949 |
| Tenrec QB, first 64/user | 1,162,886 | 1,091,071 | 2,253,957 | 92.29% | 4,349 |
| ZhihuRec, first 64/user | 46,994,222 | 48,402,682 | 95,396,904 | 95.42% | 220,859 |

The strict last column requires all 64 retained base rows. It is intentionally conservative: a
single long-tail base item makes a user fail. QK still has 1,557,570 users with at least 48 retained
base rows, so 3,949 is not the actual scale ceiling. A top-250k QK vocabulary retains 89.54% of all
rows and gives 226,510 users with all 64 base rows, but the initial experiment should keep top-50k
for comparability and avoid increasing the embedding table before it is necessary.

### 4.2 Base-only candidate cohorts

A base-only top-5k-user cohort makes the three new streams similar in training volume without
using future activity. This capacity table counts all retained audit rows for those users; it is
not the exact 64+6x8 materialized input:

| Dataset | Base rows | Stream rows | Total rows | Base positive targets | Stream positive targets |
|---|---:|---:|---:|---:|---:|
| Tenrec QK | 318,949 | 312,802 | 631,751 | 158,489 | 122,134 |
| Tenrec QB | 319,349 | 408,550 | 727,899 | 232,629 | 313,996 |
| ZhihuRec | 320,000 | 349,000 | 669,000 | 94,900 | 96,574 |

These are already large enough for the current model. There is no reason to begin by training on
all 493 million QK rows or all 100 million ZhihuRec rows.

The implemented loader applies the exact 64+6x8 cap after fitting the base-only vocabulary and
cohort:

| Dataset | Selected users | Materialized base | Materialized stream | Stream positives | Theta-5 positive-active users |
|---|---:|---:|---:|---:|---:|
| Tenrec QK | 5,000 | 318,949 | 153,673 | 63,690 | 2,244 |
| Tenrec QB | 5,000 | 319,349 | 160,141 | 123,112 | 2,552 |
| ZhihuRec | 5,000 | 320,000 | 207,200 | 58,040 | 3,177 |

Exact split counts, positive counts, source provenance, and output sizes are in the tracked
`results/dataset_audit/*_prepared.json` files. The compact NPZ inputs are ignored by Git.

Under the stricter 64-retained-base cohort, the fifth ordered evaluation window also remains
well populated:

| Dataset | Selected users | Base rows | Full stream pool | Theta-5 eval rows | Theta-5 positive eval users |
|---|---:|---:|---:|---:|---:|
| Tenrec QK | 3,949 | 252,736 | 230,164 | 13,674 | 1,680 |
| Tenrec QB | 4,349 | 278,336 | 316,513 | 16,201 | 2,141 |
| ZhihuRec | 220,859 | 14,134,976 | 15,254,020 | 1,331,180 | 140,636 |

This establishes capacity only. It does not establish that streaming training is valuable or that
stale-cache maintenance has a measurable quality gap on any of these datasets.

## 5. Completed motivation-alignment gate

The first frozen top-50k gate was diagnostically useful but not the final motivation protocol. QB's
base-only cohort lost nearly half its active users by theta-5, confounding cache age with cohort
composition. QK used a 5.09M-parameter top-50k model for only 5,000 users and produced a
high-variance maintenance denominator.

The follow-up changed one non-label data axis per dataset before seed expansion:

- QB retains top-50k but requires the raw sequence to cover the complete 112-exposure horizon.
  Selection does not inspect positive labels or model outcomes. The six retained stream windows
  are stable at 37,079–37,366 rows and theta-5 has 4,730 positive-active users.
- QK uses a base-fitted top-5k catalog with the original base-activity cohort. The resulting model
  has 770k rather than 5.09M parameters; theta-5 has 1,588 positive-active users.

Over four seeds, theta-5 full-compute/full-reuse/maintenance BestRank gains are
`94.70/64.38/30.31` for QB and `47.34/34.34/13.00` for QK; all six theta-5 seed intervals exclude
zero. The cache-age/maintenance Spearman is `0.600 [0.005, 1.000]` on QB and
`0.700 [0.475, 0.925]` on QK, compared with `0.925 [0.845, 1.000]` on KuaiRand. This closes the
pre-design motivation gate on the three datasets.

Scope remains important. QB conditions on future activity availability, although not future
labels. QB/QK are related Tenrec tables, not independent collections, and have only within-user
ordinal order rather than global timestamps. Absolute BestRank gains are not comparable across
catalog sizes. In the fixed-endpoint matrix, within-seed BestRank staleness tax is
`0.176/0.315/0.276` on KuaiRand/QB/QK, only a 1.79x range; the fine matrix remains within 2.48x
and shows update-local jumps whose ages move most strongly on QK. This supports treating each
source/target version pair as its own calibration unit rather than using age as a universal rule;
it does not supply a reuse-safety trigger or prove that every tuned periodic window fails.
ZhihuRec remains a negative maintenance boundary.

The first fixed-suffix method result is mixed and separate. On the aligned top-5k QK setting, cheap refresh
costs 0.194x full and gains `9.17 [6.71, 11.63]` BestRank over four seeds, recovering 70.6% of
the mean full gap. Aligned fixed-horizon QB fails its seed-0 partial-method gate, and no aligned
suffix improves on QK cheap in the primary metric. Exact protocols, intervals, failed gates, and
commands are in `experiments/exposure/ORDERED_EXPOSURE_V1.md`.
The subsequent `compiled_low_rank_migration_v1` protocol learns a shared residual correction and
passes the aligned four-seed method screen on both QB and QK, while preserving the failed suffix
gate as a negative result. It is documented separately in
`experiments/migration/COMPILED_LOW_RANK_V1.md`.
The fixed-endpoint metric and transition analysis are in
`experiments/exposure/CACHE_VERSION_MATRIX_V1.md`.

Machine-readable audits are in `results/dataset_audit/`; the Taobao semantic boundary is in
`results/taobao/kuairand_matched_comparison.json`.

## 6. Long-context opportunity boundary

A later pre-frozen screen selected 1,000 users with a complete 256-raw-exposure horizon without
looking at feedback labels or model outcomes. QB top-50k retains 241,221 rows and has retained
history quantiles `160/226/244/254/256` at min/p10/p50/p90/max. QK top-20k retains 202,636 rows
with corresponding quantiles `112/173/206/229/250`.

These cohorts establish real long-context capacity and a larger recomputation cost. They do not
improve the quality-side cache-migration opportunity: across the three pre-specified stream
learning rates, oldest-cache BestRank staleness tax remains 1.1%-5.6% on QB and 3.6%-6.0% on QK.
The branch stops before migration-quality evaluation. This is a cohort boundary, not a
contradiction of the aligned coarse protocols: conditioning on much longer activity changes the
population and makes streaming-training value grow faster than cache-maintenance value.

The prepared-data audits are
`results/dataset_audit/tenrec_{qb_top50000,qk_top20000}_users1000_horizon256_prepared.json`; exact
screen rules and results are in `experiments/exposure/OPPORTUNITY_REGIME_V1.md`.
