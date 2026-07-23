# Exposure-compatible dataset expansion audit

> Status: complete data-capacity audit as of 2026-07-23. This document reports no model training,
> maintenance gap, or migration-quality result.

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
be the first loader and one-seed pipeline check. A strong paper should eventually use both the
Tenrec and ZhihuRec evidence rather than treating two Tenrec tables as two independent datasets.

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

### 4.2 Matched, manageable pilot cohorts

A base-only top-5k-user cohort makes the three new streams similar in training volume without
using future activity:

| Dataset | Base rows | Stream rows | Total rows | Base positive targets | Stream positive targets |
|---|---:|---:|---:|---:|---:|
| Tenrec QK | 318,949 | 312,802 | 631,751 | 158,489 | 122,134 |
| Tenrec QB | 319,349 | 408,550 | 727,899 | 232,629 | 313,996 |
| ZhihuRec | 320,000 | 349,000 | 669,000 | 94,900 | 96,574 |

These are already large enough for the current model. There is no reason to begin by training on
all 493 million QK rows or all 100 million ZhihuRec rows.

Under the stricter 64-retained-base cohort, the fifth ordered evaluation window also remains
well populated:

| Dataset | Selected users | Base rows | Full stream pool | Theta-5 eval rows | Theta-5 positive eval users |
|---|---:|---:|---:|---:|---:|
| Tenrec QK | 3,949 | 252,736 | 230,164 | 13,674 | 1,680 |
| Tenrec QB | 4,349 | 278,336 | 316,513 | 16,201 | 2,141 |
| ZhihuRec | 220,859 | 14,134,976 | 15,254,020 | 1,331,180 | 140,636 |

This establishes capacity only. It does not establish that streaming training is valuable or that
stale-cache maintenance has a measurable quality gap on any of these datasets.

## 5. Frozen next gate

The smallest defensible next experiment is:

1. Implement one shared ordered-exposure loader with top-50k, length 128, a 64-impression base,
   six 8-impression windows, and the feedback rules above.
2. Use Tenrec QB to check data parity and run one seed of `frozen / full reuse / full compute`.
3. If a maintenance gap is identifiable, run the same frozen control on a base-only 5k-user QK
   cohort and a base-only 5k-user ZhihuRec cohort.
4. Only then expand the successful cells to four seeds and evaluate the already frozen migration
   endpoints. Do not reopen arbitrary-layer search or tune the window split to obtain a positive
   result.

Exact machine-readable audits are in `results/dataset_audit/`; the Taobao semantic boundary is in
`results/taobao/kuairand_matched_comparison.json`.
