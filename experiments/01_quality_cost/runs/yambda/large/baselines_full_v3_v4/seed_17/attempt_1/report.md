# large v3_to_v4 baseline comparison

Window: days [273, 287); E14.

Quality and cost: 75,807 eligible users, 335,971 feedback events. Users with feedback: 34,756. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.676937 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.695381 | 0.018444 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 21.05 | 0.685657 | 0.008720 | 47.3 | [37.1, 54.9] |
| Layer | layers 1-4 | 40.79 | 0.690392 | 0.013456 | 73.0 | [66.6, 78.8] |
| Layer | layers 1-6 | 60.54 | 0.691461 | 0.014524 | 78.7 | [73.6, 83.4] |
| Layer | layers 1-8 | 80.28 | 0.693970 | 0.017033 | 92.4 | [90.3, 94.1] |
| Tail | 10.00% tokens | 14.87 | 0.676968 | 0.000031 | 0.2 | [-3.3, 3.1] |
| Tail | 20.00% tokens | 26.98 | 0.677845 | 0.000908 | 4.9 | [-0.3, 9.4] |
| Tail | 40.00% tokens | 49.37 | 0.680873 | 0.003936 | 21.3 | [13.0, 27.5] |
| Tail | 80.00% tokens | 87.39 | 0.690050 | 0.013113 | 71.1 | [63.9, 76.9] |
| Deviation | 10.00% tokens | 16.91 | 0.682493 | 0.005556 | 30.1 | [24.7, 35.3] |
| Deviation | 20.00% tokens | 30.09 | 0.685236 | 0.008299 | 45.0 | [39.4, 50.9] |
| Deviation | 40.00% tokens | 56.59 | 0.689825 | 0.012888 | 69.9 | [64.2, 76.0] |
| Deviation | 60.00% tokens | 83.09 | 0.692099 | 0.015162 | 82.2 | [77.9, 86.7] |
| Query | 10.00% tokens | 13.32 | 0.681530 | 0.004593 | 24.9 | [19.5, 31.7] |
| Query | 20.00% tokens | 26.51 | 0.686513 | 0.009576 | 51.9 | [44.3, 62.3] |
| Query | 40.00% tokens | 53.01 | 0.691832 | 0.014895 | 80.8 | [73.8, 89.8] |
| Query | 60.00% tokens | 79.50 | 0.693759 | 0.016822 | 91.2 | [86.7, 97.1] |
| Query-only | 128 fit users | 0.62 | 0.655341 | -0.021596 | -117.1 | [-170.1, -80.4] |
| Query-only | 256 fit users | 1.07 | 0.663392 | -0.013545 | -73.4 | [-118.2, -42.0] |
| Query-only | 512 fit users | 1.97 | 0.670196 | -0.006741 | -36.6 | [-60.7, -18.8] |
| History-conditioned | 64 fit users | 70.18 | 0.676964 | 0.000027 | 0.1 | [-26.2, 18.5] |
| History-conditioned | 128 fit users | 71.01 | 0.677701 | 0.000764 | 4.1 | [-20.7, 21.3] |
| History-conditioned | 256 fit users | 72.67 | 0.677779 | 0.000842 | 4.6 | [-20.1, 21.8] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
