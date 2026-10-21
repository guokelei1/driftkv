# large v0_to_v1 baseline comparison

Quality and cost: 8 eligible users, 5 feedback events. Users with feedback: 4. Independent adjacent edge; frozen backbone and correction seed 17.

| Method | Setting | Added FLOPs / Rebuild (%) | AUC | Gain over Reuse | Recovery (%) | UID 95% CI (%) |
|---|---|---:|---:|---:|---:|---|
| Reuse | anchor | 0.00 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Rebuild | anchor | 100.00 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Layer | layers 1-2 | 23.31 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Layer | layers 1-4 | 43.06 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Layer | layers 1-6 | 62.80 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Layer | layers 1-8 | 82.55 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Tail | 10.00% tokens | 14.65 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Tail | 20.00% tokens | 26.66 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Tail | 40.00% tokens | 48.93 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Tail | 80.00% tokens | 87.05 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Deviation | 10.00% tokens | 16.74 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Deviation | 20.00% tokens | 29.77 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Deviation | 40.00% tokens | 55.96 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Deviation | 60.00% tokens | 82.15 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Query | 10.00% tokens | 13.17 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Query | 20.00% tokens | 26.20 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Query | 40.00% tokens | 52.39 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Query | 60.00% tokens | 78.58 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Query-only | 4 fit users | 1.43 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| Query-only | 8 fit users | 2.30 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| Query-only | 16 fit users | 3.62 | 0.833333 | 0.000000 | 0.0 | unbounded/undefined |
| History-conditioned | 4 fit users | 72.43 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| History-conditioned | 8 fit users | 74.59 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |
| History-conditioned | 16 fit users | 78.46 | 1.000000 | 0.166667 | 100.0 | unbounded/undefined |

Recovery = 100 × (AUC(method) − AUC(Reuse)) / (AUC(Rebuild) − AUC(Reuse)); values are not clipped.

Complete added FLOPs include each method's calibration, selection, repair/mapping and query correction. The denominator is once-per-user/release Rebuild at the first history consumer. Common native reads/appends and old-cache initialization are reported separately in summary.json. All population events count toward costs, including users without feedback and events after the last scored request.

Layer assumes retained old block inputs (45% auxiliary state relative to native KV for ten layers); this is disclosed extra memory, and reconstruction would need a separate cost point.

Intervals resample paired user clusters, conditional on these fitted weights. This first-edge full-population comparison does not establish cross-edge, cross-dataset or training-seed stability. Execution progress times are not controlled paper timing results.
