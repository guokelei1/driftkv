# Unified AUC diagnostics, six-layer Yambda, 2026-09-20

The [completed teacher-budget extension](../unified_auc_10k_teachers7144_20260920/README.md) is now the current nine-budget record through 7,144 teachers. It adds ten shared-read paths per edge while preserving every original score, comparison and result below.

Prospectively fixed in `configs/insight/unified_auc_10k_20260920.json` before unified scores. Motivation 1, Motivation 2 and the shared-read Insight use the same models, users, genuine feedback requests and release snapshots. The four admitted adjacent edges target V1–V4 at days 231, 245, 259 and 273; every evaluation window contains the following complete 14 days.

The [full run](diagnostic/summary.json) completed in 498.896 seconds, with 16.431 GiB peak GPU allocation and 212,773 known-item feedback requests across four edges. It used GPUs 0–3, global batch 64 and serial edges through `scripts/design/run_unified_auc_parallel.py`; CPU inputs were prepared once with 48 threads. The [runtime log](diagnostic.runtime.log) and all raw scores are retained. Execution-source hashes remained unchanged. The figure generator independently recomputed pooled AUC for all 27 paths on all four edges from raw scores.

At the fixed 256-user teacher budget, the two shared rules give the following results. Parent, Reuse and Exact columns are AUC; parentheses in the shared columns are un-clipped AUC gap recovery percentages. Cost is additional compatibility arithmetic as a percentage of one Exact rebuild for every selected user, including calibration and actual E14 reads.

| Edge | Parent | Reuse | Exact | Shared offset: AUC (recovery %) | + user response: AUC (recovery %) | Cost: offset / response (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0→V1 | 0.582773 | 0.595578 | 0.603686 | 0.601606 (74.35) | 0.604157 (105.80) | 3.0257 / 3.1082 |
| V1→V2 | 0.625008 | 0.627812 | 0.640335 | 0.624708 (-24.78) | 0.625940 (-14.95) | 3.0257 / 3.1072 |
| V2→V3 | 0.650717 | 0.657631 | 0.673957 | 0.663283 (34.62) | 0.663982 (38.91) | 3.0256 / 3.1055 |
| V3→V4 | 0.631304 | 0.627570 | 0.669788 | 0.637462 (23.43) | 0.661593 (80.59) | 3.0256 / 3.1050 |

The equal-edge mean recovery is 26.90% for the shared offset and 52.59% after adding the native user response. The latter improves over the offset on all four edges at this budget; both remain worse than Reuse on V1→V2. One-layer DroidSpeak reaches 93.39% recovery at 8.63% compute on V3→V4. These are fixed-state, single-seed development results.

Every method and budget is retained in the [complete quality summary](diagnostic/summary.json) and [cost ledger](../../../figures/out/unified_auc/cost_ledger.json). The [five-panel figure](../../../figures/out/unified_auc/combined_logx.pdf), [Motivation AUC table](../../../figures/tables/motivation_auc.tex) and [generator](../../../figures/src/unified_auc.py) use the same four-edge evidence. Calibration budgets remain 32/64/128/256 rather than selecting the best observed budget.

The cohort contains 10,000 development users with at least 1,024 events before day 217: 3,970 eligible original development users and 6,030 unused calibration-reserve users selected by frozen selector rank. The unchanged 256 fitting users, all 512 historical fitted users, 128 pilot users and both reserved confirmation groups are excluded. The original split is unchanged. This is a long-history development cohort. Each user has one frozen latest-1,024 prefix per cutover; actual E14 feedback items use their real query times against that same prefix. This is a fixed-state diagnostic.

| Current version | Known feedback requests | Users with known feedback | OOV requests |
| --- | ---: | ---: | ---: |
| V1 | 54826 | 5667 | 6996 |
| V2 | 53874 | 5567 | 9744 |
| V3 | 52309 | 5621 | 10854 |
| V4 | 51764 | 5637 | 12420 |

Primary quality is pooled real-feedback ROC AUC. Recovery is `(AUC_method-AUC_Reuse)/(AUC_Exact-AUC_Reuse)`, shown only when Exact minus Reuse exceeds 0.0001; absolute AUC and all unfavorable outcomes remain. These request counts came from fidelity metadata without labels or scores.

Fixed comparisons are Reuse, Current Exact and Parent Full; DroidSpeak with one calibration-selected interval for each width 1–6; tail replay of 32/64/128/256/512/1,024 events; KV translation with 1/2/3/4 source layers and 256 teacher users; and shared read correction using nested first-32/64/128/256 teacher users. Translation ranks individual source layers by same-head OLS R2 averaged over heads and K/V, then fits maps from all heads of the selected source layers. The two shared branches are `N*b` and `N*b+T*r`. Both use retained history length; the latter also uses the actual native history response. Candidate banks are fixed from all 256 source histories without labels. Normalization and fitted coefficients use only each teacher budget.

Compute is additional compatibility arithmetic over common native reading: one-time fitting/profiling/teacher work plus actual E14 read overhead, divided by one Exact rebuild per user snapshot. Existing Parent caches are common; Current teacher rebuilds are charged. It is not a per-request Exact rebuild denominator or an end-to-end timing ratio.

Initial planning estimate is 15–30 minutes on GPU 0 with a 32 GiB budget. Full 256-user translation with four source layers may peak near 20 GiB because it retains FP32 cache pairs and FP64 OLS views and design matrices. The canary uses the first 16 of the 121 mature users among the original 128 pilot users, on the first edge: 32 fitting users, shared budgets 8/16/32, DroidSpeak profiling on 8 users and translation on 32 users with 1–4 source layers. It checks causality, numerical agreement, request identity and measured resource use without changing the full-run budgets. All 128 original pilot users remain excluded from evaluation. Raw scores, method/budget choices and all outcomes are retained.

The original [canary](canary/summary.json) passed in 46.23 seconds with 4.51 GiB peak GPU allocation. A separate resource probe on 96 pilot users measured 2.90 seconds at batch 16 and 1.78 seconds at batch 64. These are throughput measurements used for execution planning; they are not the figure's compute axis. The [expanded serial estimate](expanded_resource_estimate.json) was 30–50 minutes. The initial single-GPU attempt was stopped during history loading, before calibration or scores, and remains in [serial_attempt](serial_attempt/configuration.json). The [initial estimate](resource_estimate.json) and serial estimate are retained.

The independent CPU preparation completed in 20.55 seconds and wrote mmap-readable snapshots for all four edges, in exact fit256/pilot121/evaluation10000 UID order. Its [metadata](prepared/metadata.json) includes input/output hashes and an exact five-array comparison against the unchanged old loader on three users across all four cutovers. The [four-GPU canary](canary_parallel/summary.json) compared 96 pilot users and 25 score paths against single-GPU execution: maximum logit difference 0 and maximum pooled AUC difference 0, with 26.34 seconds total elapsed. The [parallel resource estimate](parallel_resource_estimate.json) was 5–15 minutes with a 32 GiB GPU budget; the full run completed in 8.315 minutes. Only execution organization changed; the cohort, histories, methods, budgets and quality metric remained fixed.
