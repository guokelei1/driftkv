# RecFlow Large / Max adjacent-version feedback

Two independent foundation chains, two epochs per version, two GPUs.

| Scale | Edge | Day | Parent AUC | Current AUC | AUC gain | LogLoss delta | NDCG@5 delta |
|---|---|---:|---:|---:|---:|---:|---:|
| large | v0_v1 | 20 | 0.662883 | 0.677917 | +0.015034 | -0.012782 | +0.000814 |
| large | v1_v2 | 21 | 0.676365 | 0.679098 | +0.002734 | -0.001761 | +0.001392 |
| large | v2_v3 | 22 | 0.669347 | 0.668508 | -0.000839 | -0.000651 | +0.003292 |
| large | v3_v4 | 23 | 0.668604 | 0.671441 | +0.002837 | -0.001100 | +0.001477 |
| large | v4_v5 | 24 | 0.676490 | 0.673520 | -0.002970 | +0.001536 | +0.001377 |
| max | v0_v1 | 20 | 0.656730 | 0.673240 | +0.016511 | -0.010420 | +0.000024 |
| max | v1_v2 | 21 | 0.669977 | 0.672617 | +0.002640 | -0.004964 | +0.001697 |
| max | v2_v3 | 22 | 0.665102 | 0.663196 | -0.001906 | +0.000557 | +0.002422 |
| max | v3_v4 | 23 | 0.664783 | 0.669109 | +0.004326 | -0.003286 | +0.001848 |
| max | v4_v5 | 24 | 0.673921 | 0.668502 | -0.005419 | +0.003913 | +0.001738 |

Positive AUC/NDCG delta and negative LogLoss delta indicate improvement. All outcomes are retained.
Per-edge reports contain the remaining metrics, known/hash strata and paired user confidence intervals.
