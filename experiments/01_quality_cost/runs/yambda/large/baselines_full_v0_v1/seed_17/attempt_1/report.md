# large v0_to_v1 baseline comparison

Quality and cost: 75,807 eligible users, 345,630 feedback events. Users with feedback: 34,190. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.634243 | 0.000000 | 0.0 | [0.0, 0.0] |
| Rebuild | anchor | 100.00 | 0.648931 | 0.014688 | 100.0 | [100.0, 100.0] |
| Layer | layers 1-2 | 21.04 | 0.646711 | 0.012467 | 84.9 | [78.0, 90.9] |
| Layer | layers 1-4 | 40.79 | 0.647954 | 0.013710 | 93.3 | [91.0, 95.6] |
| Layer | layers 1-6 | 60.53 | 0.648198 | 0.013955 | 95.0 | [93.3, 96.6] |
| Layer | layers 1-8 | 80.28 | 0.648605 | 0.014362 | 97.8 | [97.1, 98.5] |
| Tail | 10.00% tokens | 14.75 | 0.635814 | 0.001571 | 10.7 | [8.4, 13.2] |
| Tail | 20.00% tokens | 26.81 | 0.637321 | 0.003077 | 21.0 | [16.2, 25.2] |
| Tail | 40.00% tokens | 49.14 | 0.641133 | 0.006890 | 46.9 | [42.4, 52.1] |
| Tail | 80.00% tokens | 87.21 | 0.647012 | 0.012769 | 86.9 | [81.8, 91.3] |
| Deviation | 10.00% tokens | 16.82 | 0.636499 | 0.002256 | 15.4 | [10.1, 20.3] |
| Deviation | 20.00% tokens | 29.93 | 0.639096 | 0.004853 | 33.0 | [25.2, 42.6] |
| Deviation | 40.00% tokens | 56.26 | 0.642543 | 0.008300 | 56.5 | [48.0, 67.0] |
| Deviation | 60.00% tokens | 82.60 | 0.644266 | 0.010023 | 68.2 | [61.2, 76.9] |
| Query | 10.00% tokens | 13.24 | 0.636665 | 0.002421 | 16.5 | [10.0, 23.0] |
| Query | 20.00% tokens | 26.35 | 0.639826 | 0.005582 | 38.0 | [26.1, 49.9] |
| Query | 40.00% tokens | 52.69 | 0.644961 | 0.010718 | 73.0 | [62.7, 82.9] |
| Query | 60.00% tokens | 79.02 | 0.647232 | 0.012989 | 88.4 | [81.9, 94.1] |
| Query-only | 128 fit users | 0.65 | 0.629283 | -0.004960 | -33.8 | [-70.6, -2.4] |
| Query-only | 256 fit users | 1.10 | 0.632968 | -0.001275 | -8.7 | [-34.1, 12.8] |
| Query-only | 512 fit users | 2.01 | 0.633047 | -0.001197 | -8.1 | [-33.6, 15.9] |
| History-conditioned | 64 fit users | 70.15 | 0.644950 | 0.010707 | 72.9 | [65.9, 78.4] |
| History-conditioned | 128 fit users | 71.02 | 0.644974 | 0.010730 | 73.1 | [65.8, 78.9] |
| History-conditioned | 256 fit users | 72.77 | 0.645501 | 0.011257 | 76.6 | [69.4, 82.2] |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This first-edge full-population comparison does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
