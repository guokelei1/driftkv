# large v4_to_v5 baseline comparison

Window: days [287, 301); E14_partial. Incomplete source horizon: 13 complete source days and partial source day 300; all available events; directional diagnostic, not complete 14-day evidence

Quality and cost: 75,807 eligible users, 347,575 feedback events. Users with feedback: 35,925. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.677488 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.685899 | 0.008411 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 21.05 | 0.682573 | 0.005085 | 60.5 | [16.3, 213.1] |
| Layer | layers 1-4 | 40.79 | 0.684793 | 0.007305 | 86.9 | [50.6, 213.1] |
| Layer | layers 1-6 | 60.54 | 0.685591 | 0.008103 | 96.3 | [65.5, 207.5] |
| Layer | layers 1-8 | 80.28 | 0.685279 | 0.007791 | 92.6 | [87.1, 105.3] |
| Tail | 10.00% tokens | 14.81 | 0.677972 | 0.000484 | 5.8 | [0.6, 15.2] |
| Tail | 20.00% tokens | 26.89 | 0.678885 | 0.001397 | 16.6 | [7.0, 45.7] |
| Tail | 40.00% tokens | 49.24 | 0.680539 | 0.003051 | 36.3 | [20.9, 76.0] |
| Tail | 80.00% tokens | 87.29 | 0.684395 | 0.006907 | 82.1 | [72.4, 102.5] |
| Deviation | 10.00% tokens | 16.86 | 0.681218 | 0.003730 | 44.3 | [21.7, 132.8] |
| Deviation | 20.00% tokens | 30.01 | 0.681831 | 0.004343 | 51.6 | [31.8, 122.6] |
| Deviation | 40.00% tokens | 56.41 | 0.683753 | 0.006265 | 74.5 | [47.2, 175.5] |
| Deviation | 60.00% tokens | 82.82 | 0.684931 | 0.007443 | 88.5 | [64.0, 180.4] |
| Query | 10.00% tokens | 13.28 | 0.682358 | 0.004870 | 57.9 | [23.0, 197.3] |
| Query | 20.00% tokens | 26.42 | 0.684974 | 0.007486 | 89.0 | [44.0, 269.6] |
| Query | 40.00% tokens | 52.83 | 0.686410 | 0.008922 | 106.1 | [59.1, 279.4] |
| Query | 60.00% tokens | 79.23 | 0.687083 | 0.009595 | 114.1 | [78.7, 249.7] |
| Query-only | 128 fit users | 0.59 | 0.670845 | -0.006643 | -79.0 | [-160.7, -3.5] |
| Query-only | 256 fit users | 1.01 | 0.667598 | -0.009890 | -117.6 | [-260.6, -66.1] |
| Query-only | 512 fit users | 1.85 | 0.675864 | -0.001623 | -19.3 | [-74.8, 59.2] |
| History-conditioned | 64 fit users | 70.15 | 0.673376 | -0.004112 | -48.9 | [-175.2, -7.7] |
| History-conditioned | 128 fit users | 70.92 | 0.674043 | -0.003445 | -41.0 | [-139.6, -4.8] |
| History-conditioned | 256 fit users | 72.47 | 0.674088 | -0.003400 | -40.4 | [-150.9, -2.3] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
