# max v1_to_v2 baseline comparison

Window: days [245, 259); E14.

Quality and cost: 196,363 eligible users, 865,122 feedback events. Users with feedback: 89,406. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.686114 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.693613 | 0.007499 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 13.21 | 0.688184 | 0.002071 | 27.6 | [19.0, 39.0] |
| Layer | layers 1-4 | 25.61 | 0.691440 | 0.005326 | 71.0 | [62.5, 88.2] |
| Layer | layers 1-8 | 50.41 | 0.691670 | 0.005556 | 74.1 | [66.2, 89.9] |
| Layer | layers 1-12 | 75.21 | 0.693090 | 0.006977 | 93.0 | [86.4, 108.8] |
| Tail | 10.00% tokens | 14.83 | 0.687121 | 0.001007 | 13.4 | [10.1, 17.3] |
| Tail | 20.00% tokens | 26.92 | 0.687959 | 0.001845 | 24.6 | [19.3, 31.1] |
| Tail | 40.00% tokens | 49.29 | 0.689804 | 0.003690 | 49.2 | [40.9, 60.0] |
| Tail | 80.00% tokens | 87.33 | 0.693143 | 0.007030 | 93.7 | [83.7, 109.6] |
| Deviation | 10.00% tokens | 15.54 | 0.687791 | 0.001678 | 22.4 | [15.4, 31.1] |
| Deviation | 20.00% tokens | 28.70 | 0.687982 | 0.001868 | 24.9 | [17.4, 34.1] |
| Deviation | 40.00% tokens | 55.13 | 0.688557 | 0.002443 | 32.6 | [22.8, 43.0] |
| Deviation | 60.00% tokens | 81.57 | 0.689813 | 0.003699 | 49.3 | [39.9, 61.7] |
| Query | 10.00% tokens | 13.29 | 0.690548 | 0.004434 | 59.1 | [50.3, 71.0] |
| Query | 20.00% tokens | 26.45 | 0.691581 | 0.005467 | 72.9 | [62.3, 86.9] |
| Query | 40.00% tokens | 52.89 | 0.691660 | 0.005546 | 74.0 | [64.1, 87.3] |
| Query | 60.00% tokens | 79.32 | 0.691339 | 0.005225 | 69.7 | [60.3, 77.8] |
| Query-only | 128 fit users | 0.33 | 0.671412 | -0.014702 | -196.1 | [-252.3, -153.4] |
| Query-only | 256 fit users | 0.51 | 0.675124 | -0.010990 | -146.6 | [-195.3, -113.6] |
| Query-only | 512 fit users | 0.87 | 0.677269 | -0.008845 | -118.0 | [-161.6, -86.8] |
| History-conditioned | 64 fit users | 69.86 | 0.689381 | 0.003267 | 43.6 | [35.2, 55.8] |
| History-conditioned | 128 fit users | 70.20 | 0.689834 | 0.003720 | 49.6 | [41.9, 59.6] |
| History-conditioned | 256 fit users | 70.87 | 0.689565 | 0.003451 | 46.0 | [38.5, 53.7] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
