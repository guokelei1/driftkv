# Interim preview: 12288 users — Yambda Large V0→V1

5586 users contribute 59501 feedback events. All 24 fixed points; GPU run continues.
Pooled original feedback across blocks; not averages of block AUCs/recovery.

Reuse=0.645401; Rebuild=0.655390; reference gap UID 95% CI=[0.0041475643769383255, 0.016348476318219732]; nonpositive/invalid reference gap draws=0/500.

| Point | AUC | Recovery (%) | UID 95% CI (%) |
|---|---:|---:|---|
| reuse | 0.645401 | 0.0 | [0.0, 0.0] |
| rebuild | 0.655390 | 100.0 | [100.0, 100.0] |
| layer_2layers | 0.654916 | 95.3 | [80.2, 113.7] |
| layer_4layers | 0.655064 | 96.7 | [90.3, 103.6] |
| layer_6layers | 0.655119 | 97.3 | [92.4, 102.0] |
| layer_8layers | 0.655296 | 99.1 | [96.8, 101.6] |
| tail_10pct | 0.646842 | 14.4 | [8.0, 23.2] |
| tail_20pct | 0.648236 | 28.4 | [16.5, 43.5] |
| tail_40pct | 0.651089 | 56.9 | [39.5, 81.3] |
| tail_80pct | 0.655462 | 100.7 | [90.7, 117.8] |
| deviation_10pct | 0.647211 | 18.1 | [2.6, 34.0] |
| deviation_20pct | 0.648811 | 34.1 | [15.6, 54.1] |
| deviation_40pct | 0.651384 | 59.9 | [44.4, 83.8] |
| deviation_60pct | 0.651924 | 65.3 | [49.1, 80.1] |
| query_10pct | 0.646836 | 14.4 | [-9.6, 34.4] |
| query_20pct | 0.648265 | 28.7 | [-13.3, 58.6] |
| query_40pct | 0.652493 | 71.0 | [35.8, 95.9] |
| query_60pct | 0.654137 | 87.5 | [64.4, 104.1] |
| history_conditioned_fit64 | 0.652450 | 70.6 | [38.5, 82.7] |
| history_conditioned_fit128 | 0.652426 | 70.3 | [35.5, 82.7] |
| history_conditioned_fit256 | 0.652938 | 75.4 | [40.2, 86.9] |
| query_only_fit128 | 0.641860 | -35.5 | [-270.1, 57.9] |
| query_only_fit256 | 0.643653 | -17.5 | [-158.4, 43.0] |
| query_only_fit512 | 0.646127 | 7.3 | [-129.5, 74.4] |

Interim descriptive results only; 500 paired UID bootstrap draws, conditional on fixed correction weights. Negative and >100% recovery values retained. Selected prefix Layer points need no old hidden boundary cache. Full-population results are pending.

Reporting correction: the earlier generic 45% boundary-state statement did not apply to the selected prefix intervals. Numerical preview results are unchanged.
