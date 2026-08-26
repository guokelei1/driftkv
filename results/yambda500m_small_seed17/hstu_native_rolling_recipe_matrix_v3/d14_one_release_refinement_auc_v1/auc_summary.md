# D14/E14 one-release EvoKV rolling AUC

The requested retained-gain columns preserve the earlier motivation denominator: D14 Full-only `Current − Parent` AUC. Reuse and Our deltas are measured on the same full-population rolling requests. Matched rolling ratios remain companions.
Equivalently, `retained(path) = 1 - (AUC_Recompute - AUC_path) / (AUC_CurrentFull - AUC_ParentFull)`, which is the pre-existing `(AUC_path - AUC_old) / (AUC_Recompute - AUC_old)` axis.

Fixed Our plan: parameter-only CAST of the old 384-position prefix, then GROUP recent 128 evidence into 64 Current PATCH carriers and SCALE each carrier by represented mass 2. This is one-hop only.

| Edge | Requests | Parent rolling AUC | Recompute AUC | Reuse AUC | Our AUC | Reuse gain retained | Our gain retained | Our − Reuse AUC (pp) | Reuse harm recovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 → v1 | 43186 | 0.669336 | 0.681769 | 0.678709 | 0.681432 | +74.5% | +97.2% | +0.272304 | +89.0% |
| v1 → v2 | 41655 | 0.664157 | 0.670493 | 0.668496 | 0.671611 | +68.1% | +117.9% | +0.311450 | +156.0% |
| v2 → v3 | 43092 | 0.611966 | 0.615001 | 0.613514 | 0.614609 | +52.1% | +87.3% | +0.109449 | +73.6% |
| v3 → v4 | 43945 | 0.620844 | 0.620994 | 0.619054 | 0.619960 | -318.7% | -123.2% | +0.090555 | +46.7% |
| v4 → v5 | 45706 | 0.549731 | 0.551858 | 0.551221 | 0.548563 | +71.1% | -49.4% | -0.265765 | -417.0% |

`Reuse harm recovered = (AUC_Our − AUC_Reuse) / (AUC_Recompute − AUC_Reuse)`. Values outside [0,100%] are retained rather than clipped.

All five formal E14 edges have `AUC_Recompute > AUC_Reuse`. The fixed plan improves Reuse on four edges and fails on v4 → v5. The v3 → v4 retained-gain ratio is unstable because the Full-only release-gain denominator is only 0.046331 AUC point.
