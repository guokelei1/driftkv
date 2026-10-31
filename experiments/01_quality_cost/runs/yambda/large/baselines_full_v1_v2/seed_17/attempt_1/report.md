# large v1_to_v2 baseline comparison

Window: days [245, 259); E14.

Quality and cost: 75,807 eligible users, 318,504 feedback events. Users with feedback: 34,080. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.659557 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.667397 | 0.007840 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 21.04 | 0.666258 | 0.006701 | 85.5 | [78.7, 91.5] |
| Layer | layers 1-4 | 40.79 | 0.667115 | 0.007558 | 96.4 | [92.8, 100.5] |
| Layer | layers 1-6 | 60.53 | 0.667217 | 0.007660 | 97.7 | [95.0, 100.7] |
| Layer | layers 1-8 | 80.28 | 0.667340 | 0.007783 | 99.3 | [98.0, 101.0] |
| Tail | 10.00% tokens | 14.80 | 0.660524 | 0.000967 | 12.3 | [8.4, 17.9] |
| Tail | 20.00% tokens | 26.87 | 0.661663 | 0.002106 | 26.9 | [19.6, 37.3] |
| Tail | 40.00% tokens | 49.22 | 0.664035 | 0.004478 | 57.1 | [48.2, 69.0] |
| Tail | 80.00% tokens | 87.28 | 0.667128 | 0.007571 | 96.6 | [91.6, 102.4] |
| Deviation | 10.00% tokens | 16.85 | 0.661896 | 0.002339 | 29.8 | [20.3, 43.6] |
| Deviation | 20.00% tokens | 29.99 | 0.662086 | 0.002529 | 32.3 | [19.4, 48.9] |
| Deviation | 40.00% tokens | 56.38 | 0.663315 | 0.003758 | 47.9 | [35.2, 62.0] |
| Deviation | 60.00% tokens | 82.77 | 0.664655 | 0.005098 | 65.0 | [53.8, 76.9] |
| Query | 10.00% tokens | 13.27 | 0.661412 | 0.001855 | 23.7 | [9.9, 37.6] |
| Query | 20.00% tokens | 26.41 | 0.663859 | 0.004302 | 54.9 | [39.1, 70.4] |
| Query | 40.00% tokens | 52.80 | 0.665997 | 0.006441 | 82.1 | [70.4, 93.1] |
| Query | 60.00% tokens | 79.19 | 0.666772 | 0.007215 | 92.0 | [84.4, 99.2] |
| Query-only | 128 fit users | 0.63 | 0.652098 | -0.007459 | -95.1 | [-166.4, -49.2] |
| Query-only | 256 fit users | 1.09 | 0.654337 | -0.005220 | -66.6 | [-105.4, -39.3] |
| Query-only | 512 fit users | 1.99 | 0.649979 | -0.009578 | -122.2 | [-183.0, -78.3] |
| History-conditioned | 64 fit users | 70.13 | 0.663708 | 0.004151 | 53.0 | [41.9, 59.6] |
| History-conditioned | 128 fit users | 70.99 | 0.664120 | 0.004563 | 58.2 | [47.4, 64.8] |
| History-conditioned | 256 fit users | 72.72 | 0.664154 | 0.004597 | 58.6 | [47.3, 65.2] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
