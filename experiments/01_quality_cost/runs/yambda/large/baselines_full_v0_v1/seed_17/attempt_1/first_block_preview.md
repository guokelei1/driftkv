# First-block preview — Yambda Large V0→V1

Partial quality only: 4096 eligible users, 1890 with 20244 feedback events. All 24 fixed points; full run continues.

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)). Values are not clipped. Intervals are paired UID bootstrap, 500 draws, conditional on the fixed weights.

The Rebuild−Reuse AUC gap is 0.008005; its paired UID 95% interval is [−0.000064, 0.015480]. In 14/500 draws the gap is nonpositive. Recovery ratios are therefore unstable on this block, and no bounded recovery interval is reported.

| Point | AUC | Recovery (%) | UID 95% CI (%) |
|---|---:|---:|---|
| reuse | 0.658286 | 0.0 | unstable/undefined |
| rebuild | 0.666291 | 100.0 | unstable/undefined |
| layer_2layers | 0.666167 | 98.4 | unstable/undefined |
| layer_4layers | 0.666156 | 98.3 | unstable/undefined |
| layer_6layers | 0.666211 | 99.0 | unstable/undefined |
| layer_8layers | 0.666303 | 100.1 | unstable/undefined |
| tail_10pct | 0.659900 | 20.2 | unstable/undefined |
| tail_20pct | 0.660779 | 31.1 | unstable/undefined |
| tail_40pct | 0.662695 | 55.1 | unstable/undefined |
| tail_80pct | 0.667247 | 111.9 | unstable/undefined |
| deviation_10pct | 0.659526 | 15.5 | unstable/undefined |
| deviation_20pct | 0.660497 | 27.6 | unstable/undefined |
| deviation_40pct | 0.663030 | 59.3 | unstable/undefined |
| deviation_60pct | 0.663562 | 65.9 | unstable/undefined |
| query_10pct | 0.658940 | 8.2 | unstable/undefined |
| query_20pct | 0.659737 | 18.1 | unstable/undefined |
| query_40pct | 0.662289 | 50.0 | unstable/undefined |
| query_60pct | 0.665203 | 86.4 | unstable/undefined |
| history_conditioned_fit64 | 0.664488 | 77.5 | unstable/undefined |
| history_conditioned_fit128 | 0.664362 | 75.9 | unstable/undefined |
| history_conditioned_fit256 | 0.664551 | 78.3 | unstable/undefined |
| query_only_fit128 | 0.652317 | -74.6 | unstable/undefined |
| query_only_fit256 | 0.656845 | -18.0 | unstable/undefined |
| query_only_fit512 | 0.660591 | 28.8 | unstable/undefined |

This is a descriptive interim view, not the final paper result. The final output uses all eligible feedback and 1000 bootstrap replicates.

Layer points use the first 2/4/6/8 layers selected on prerelease fitting users and need no old hidden boundary cache. Rebuild is a once-per-release control, not a guaranteed AUC upper bound.

Reporting correction: the earlier generic 45% boundary-state statement did not apply to the selected prefix intervals. Numerical preview results are unchanged.
