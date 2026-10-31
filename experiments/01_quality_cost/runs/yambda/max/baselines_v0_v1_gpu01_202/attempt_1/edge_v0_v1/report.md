# max v0_to_v1 baseline comparison

Window: days [231, 245); E14.

Quality and cost: 196,363 eligible users, 891,487 feedback events. Users with feedback: 89,121. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.678036 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.690710 | 0.012674 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 13.21 | 0.683180 | 0.005144 | 40.6 | [34.6, 47.1] |
| Layer | layers 1-4 | 25.61 | 0.685416 | 0.007380 | 58.2 | [48.2, 67.8] |
| Layer | layers 1-8 | 50.41 | 0.685673 | 0.007637 | 60.3 | [50.4, 69.5] |
| Layer | layers 1-12 | 75.21 | 0.688967 | 0.010931 | 86.2 | [81.6, 91.4] |
| Tail | 10.00% tokens | 14.78 | 0.680251 | 0.002215 | 17.5 | [14.3, 21.0] |
| Tail | 20.00% tokens | 26.85 | 0.681845 | 0.003809 | 30.1 | [25.7, 35.3] |
| Tail | 40.00% tokens | 49.19 | 0.685590 | 0.007554 | 59.6 | [53.0, 67.2] |
| Tail | 80.00% tokens | 87.25 | 0.690372 | 0.012336 | 97.3 | [93.1, 102.2] |
| Deviation | 10.00% tokens | 15.51 | 0.681813 | 0.003777 | 29.8 | [22.8, 37.0] |
| Deviation | 20.00% tokens | 28.63 | 0.684066 | 0.006030 | 47.6 | [38.5, 56.7] |
| Deviation | 40.00% tokens | 55.01 | 0.689136 | 0.011100 | 87.6 | [78.1, 99.9] |
| Deviation | 60.00% tokens | 81.38 | 0.691378 | 0.013342 | 105.3 | [98.5, 113.7] |
| Query | 10.00% tokens | 13.26 | 0.682963 | 0.004927 | 38.9 | [30.1, 48.5] |
| Query | 20.00% tokens | 26.39 | 0.684660 | 0.006624 | 52.3 | [43.0, 61.8] |
| Query | 40.00% tokens | 52.76 | 0.686986 | 0.008950 | 70.6 | [62.5, 78.7] |
| Query | 60.00% tokens | 79.13 | 0.687846 | 0.009810 | 77.4 | [69.7, 84.1] |
| Query-only | 128 fit users | 0.34 | 0.645236 | -0.032800 | -258.8 | [-326.0, -212.0] |
| Query-only | 256 fit users | 0.52 | 0.638831 | -0.039205 | -309.3 | [-390.2, -248.1] |
| Query-only | 512 fit users | 0.88 | 0.646462 | -0.031574 | -249.1 | [-319.6, -199.4] |
| History-conditioned | 64 fit users | 69.86 | 0.686171 | 0.008136 | 64.2 | [56.0, 71.7] |
| History-conditioned | 128 fit users | 70.20 | 0.685813 | 0.007777 | 61.4 | [52.8, 68.7] |
| History-conditioned | 256 fit users | 70.88 | 0.686297 | 0.008261 | 65.2 | [56.6, 72.4] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
