# Full-window random-ranking resolution audit

CPU-only diagnostic on the existing 512 development users and frozen 1M-item
catalog. No model was loaded, no GPU was used, and the current 3072-request
evaluation panels/configuration were not changed. The cohort remains fixed;
486/475/462 users have positive requests in the respective full windows.

| Window | All positive requests | Requests with known positives | All-OOV requests | Known-target coverage | Random NDCG@50 expectation | 3072-panel null99 | Full-window null99 | Full null99 / expectation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D19–21 | 29,027 | 17,046 | 11,981 | 36.35% | 0.000006026 | 0.000086014 | 0.000025653 | 4.26× |
| D22–24 | 25,694 | 10,854 | 14,840 | 22.43% | 0.000003704 | 0.000076631 | 0.000023863 | 6.44× |
| D25–27 | 25,180 | 10,194 | 14,986 | 20.77% | 0.000003456 | 0.000071976 | 0.000022020 | 6.37× |

The exact random-policy standard deviation falls to 0.330/0.342/0.350 of the
corresponding small-panel value. Full-window null99 is approximately 30% of the
small-panel value. A hypothetical score fixed at 5.7 times the matched random
expectation exceeds full-window null99 for D19–21, but remains slightly below it
for D22–24 and D25–27. The latter null tail probabilities are approximately
1.38% and 1.44%, compared with 8.50% and 7.86% on the small panels.

This supports considering a separately declared, matched full-window sensitivity
check. It does **not** establish that an existing model's score will persist on
the full window, that its sparse small-panel hits are representative, or that
it passes the gate. No model results are rescored here. The null concerns random
policy variation on fixed requests; actual model-test power and model
request-sampling/generalization uncertainty require additional information.
The original panel results and failures must remain visible.

Each positive request has equal weight. The original panel contains 1024
requests per day; the full windows contain different daily request volumes, so
their day mixture and analytic expectation differ slightly. This weighting
must be stated if a later full-window check is conducted.

| Day | Positive requests | With known positives | All-OOV requests | Users |
| --- | ---: | ---: | ---: | ---: |
| 19 | 9,990 | 6,929 | 3,061 | 456 |
| 20 | 10,171 | 5,805 | 4,366 | 448 |
| 21 | 8,866 | 4,312 | 4,554 | 426 |
| 22 | 9,039 | 4,014 | 5,025 | 437 |
| 23 | 8,523 | 3,501 | 5,022 | 423 |
| 24 | 8,132 | 3,339 | 4,793 | 424 |
| 25 | 7,920 | 3,318 | 4,602 | 422 |
| 26 | 8,233 | 3,349 | 4,884 | 422 |
| 27 | 9,027 | 3,527 | 5,500 | 421 |

Reproduction uses `scripts/recflow/full_window_random_power.py`, which calls the
existing exact without-replacement `draw_random_panel` implementation with
5000 trials, seed 20260918, and the same panel-derived seed convention as
`random_baseline.py`. Only NDCG@50 is retained in the local trial arrays. OOV
positives remain in IDCG and all-OOV requests score zero. Analytic mean/variance
and zero-hit probability supplement the Monte Carlo quantiles; null99 has only
about 50 simulated tail observations and is not an exact threshold.

`summary.json` retains daily target counts, UID ranges and hashes, source/data
hashes, and all comparison settings. `development_window_counts.npz` retains
only the fixed development request IDs, UIDs, days and target counts. The exact
finite-population variance formula was checked against all 21 positive-rank
subsets for a 7-item, 2-positive, Top3 toy problem. Total audit runtime was
14.93 seconds.
