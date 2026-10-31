# max v0_to_v1 baseline comparison

Window: days [231, 245); E14.

Quality and cost: 32 eligible users, 264 feedback events. Users with feedback: 19. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.731590 | 0.000000 | 0.0 | unbounded/undefined |
| Rebuild | anchor | 100.00 | 0.735948 | 0.004357 | 100.0 | unbounded/undefined |
| Layer | layers 1-2 | 106.67 | 0.742484 | 0.010893 | 250.0 | unbounded/undefined |
| Layer | layers 1-4 | 119.07 | 0.733333 | 0.001743 | 40.0 | unbounded/undefined |
| Layer | layers 1-8 | 143.87 | 0.733769 | 0.002179 | 50.0 | unbounded/undefined |
| Layer | layers 1-12 | 168.67 | 0.734205 | 0.002614 | 60.0 | unbounded/undefined |
| Tail | 10.00% tokens | 14.91 | 0.731590 | 0.000000 | 0.0 | unbounded/undefined |
| Tail | 20.00% tokens | 27.05 | 0.739869 | 0.008279 | 190.0 | unbounded/undefined |
| Tail | 40.00% tokens | 49.46 | 0.735076 | 0.003486 | 80.0 | unbounded/undefined |
| Tail | 80.00% tokens | 87.46 | 0.739434 | 0.007843 | 180.0 | unbounded/undefined |
| Deviation | 10.00% tokens | 15.61 | 0.738998 | 0.007407 | 170.0 | unbounded/undefined |
| Deviation | 20.00% tokens | 28.83 | 0.748584 | 0.016993 | 390.0 | unbounded/undefined |
| Deviation | 40.00% tokens | 55.38 | 0.755991 | 0.024401 | 560.0 | unbounded/undefined |
| Deviation | 60.00% tokens | 81.94 | 0.750327 | 0.018736 | 430.0 | unbounded/undefined |
| Query | 10.00% tokens | 13.35 | 0.735512 | 0.003922 | 90.0 | unbounded/undefined |
| Query | 20.00% tokens | 26.57 | 0.730719 | -0.000871 | -20.0 | unbounded/undefined |
| Query | 40.00% tokens | 53.12 | 0.711111 | -0.020479 | -470.0 | unbounded/undefined |
| Query | 60.00% tokens | 79.68 | 0.737255 | 0.005664 | 130.0 | unbounded/undefined |
| Query-only | 128 fit users | 1176.59 | 0.679739 | -0.051852 | -1190.0 | unbounded/undefined |
| Query-only | 256 fit users | 2257.57 | 0.644444 | -0.087146 | -2000.0 | unbounded/undefined |
| Query-only | 512 fit users | 4412.35 | 0.668845 | -0.062745 | -1440.0 | unbounded/undefined |
| History-conditioned | 64 fit users | 2373.00 | 0.743791 | 0.012200 | 280.0 | unbounded/undefined |
| History-conditioned | 128 fit users | 4374.18 | 0.740305 | 0.008715 | 200.0 | unbounded/undefined |
| History-conditioned | 256 fit users | 8406.52 | 0.737255 | 0.005664 | 130.0 | unbounded/undefined |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

All selected Layer intervals start at layer zero and recompute from raw events; they need no additional persistent old hidden boundary cache. Temporary reconstruction workspace is separate.

Intervals resample paired user clusters, conditional on these fitted weights. This independent-edge comparison alone does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
