# large v2_to_v3 baseline comparison

Window: days [259, 273); E14.

Quality and cost: 75,807 eligible users, 323,362 feedback events. Users with feedback: 34,266. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.688928 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.702306 | 0.013379 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 21.04 | 0.696409 | 0.007481 | 55.9 | [51.2, 62.0] |
| Layer | layers 1-4 | 40.79 | 0.699457 | 0.010529 | 78.7 | [74.9, 83.0] |
| Layer | layers 1-6 | 60.53 | 0.700323 | 0.011395 | 85.2 | [82.2, 88.3] |
| Layer | layers 1-8 | 80.28 | 0.701562 | 0.012635 | 94.4 | [92.9, 96.3] |
| Tail | 10.00% tokens | 14.84 | 0.689950 | 0.001022 | 7.6 | [5.9, 9.6] |
| Tail | 20.00% tokens | 26.94 | 0.691562 | 0.002634 | 19.7 | [15.7, 24.0] |
| Tail | 40.00% tokens | 49.31 | 0.694715 | 0.005788 | 43.3 | [37.4, 49.9] |
| Tail | 80.00% tokens | 87.34 | 0.700228 | 0.011301 | 84.5 | [80.3, 89.3] |
| Deviation | 10.00% tokens | 16.88 | 0.690126 | 0.001198 | 9.0 | [5.2, 13.7] |
| Deviation | 20.00% tokens | 30.05 | 0.691088 | 0.002161 | 16.2 | [9.5, 23.7] |
| Deviation | 40.00% tokens | 56.50 | 0.694220 | 0.005293 | 39.6 | [30.8, 48.8] |
| Deviation | 60.00% tokens | 82.95 | 0.698753 | 0.009825 | 73.4 | [67.8, 79.4] |
| Query | 10.00% tokens | 13.30 | 0.691126 | 0.002199 | 16.4 | [9.9, 23.2] |
| Query | 20.00% tokens | 26.47 | 0.694316 | 0.005388 | 40.3 | [32.6, 47.8] |
| Query | 40.00% tokens | 52.92 | 0.699021 | 0.010093 | 75.4 | [70.1, 81.0] |
| Query | 60.00% tokens | 79.37 | 0.700774 | 0.011846 | 88.5 | [84.2, 92.5] |
| Query-only | 128 fit users | 0.63 | 0.681628 | -0.007299 | -54.6 | [-91.5, -27.4] |
| Query-only | 256 fit users | 1.08 | 0.681835 | -0.007092 | -53.0 | [-82.5, -31.6] |
| Query-only | 512 fit users | 1.99 | 0.687890 | -0.001038 | -7.8 | [-23.6, 6.5] |
| History-conditioned | 64 fit users | 70.15 | 0.696040 | 0.007112 | 53.2 | [44.4, 58.5] |
| History-conditioned | 128 fit users | 71.00 | 0.696499 | 0.007571 | 56.6 | [48.8, 62.0] |
| History-conditioned | 256 fit users | 72.69 | 0.696516 | 0.007588 | 56.7 | [49.4, 61.9] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
