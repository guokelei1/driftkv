# Interim preview: 8192 users — Yambda Large V0→V1

3759 users have 41073 feedback events. All fixed 24 points; GPU run continues.
Pooled event AUC across committed blocks; not a mean of per-block AUCs/recovery.

Reuse=0.648084; Rebuild=0.657275; AUC gap=0.009192. Reference gap UID 95% CI=[0.002950191475145045, 0.016948455026967835].
Nonpositive/invalid reference gap: 2/500 paired bootstrap draws. Intervals are conditional on these fitted weights.

| Point | AUC | Recovery (%) | UID 95% CI (%) |
|---|---:|---:|---|
| reuse | 0.648084 | 0.0 | unstable/undefined |
| rebuild | 0.657275 | 100.0 | unstable/undefined |
| layer_2layers | 0.657476 | 102.2 | unstable/undefined |
| layer_4layers | 0.657229 | 99.5 | unstable/undefined |
| layer_6layers | 0.657200 | 99.2 | unstable/undefined |
| layer_8layers | 0.657282 | 100.1 | unstable/undefined |
| tail_10pct | 0.649515 | 15.6 | unstable/undefined |
| tail_20pct | 0.651214 | 34.1 | unstable/undefined |
| tail_40pct | 0.654252 | 67.1 | unstable/undefined |
| tail_80pct | 0.658151 | 109.5 | unstable/undefined |
| deviation_10pct | 0.649841 | 19.1 | unstable/undefined |
| deviation_20pct | 0.651304 | 35.0 | unstable/undefined |
| deviation_40pct | 0.653828 | 62.5 | unstable/undefined |
| deviation_60pct | 0.654098 | 65.4 | unstable/undefined |
| query_10pct | 0.649238 | 12.6 | unstable/undefined |
| query_20pct | 0.650384 | 25.0 | unstable/undefined |
| query_40pct | 0.654146 | 66.0 | unstable/undefined |
| query_60pct | 0.656047 | 86.6 | unstable/undefined |
| history_conditioned_fit64 | 0.655306 | 78.6 | unstable/undefined |
| history_conditioned_fit128 | 0.655200 | 77.4 | unstable/undefined |
| history_conditioned_fit256 | 0.655527 | 81.0 | unstable/undefined |
| query_only_fit128 | 0.646652 | -15.6 | unstable/undefined |
| query_only_fit256 | 0.648745 | 7.2 | unstable/undefined |
| query_only_fit512 | 0.651703 | 39.4 | unstable/undefined |

Descriptive interim results only. Rebuild is not a guaranteed AUC upper bound; negative and >100% recovery values are retained. Selected prefix Layer points need no old hidden boundary cache. Final population and bootstrap results are pending.

Reporting correction: the earlier generic 45% boundary-state statement did not apply to the selected prefix intervals. Numerical preview results are unchanged.
