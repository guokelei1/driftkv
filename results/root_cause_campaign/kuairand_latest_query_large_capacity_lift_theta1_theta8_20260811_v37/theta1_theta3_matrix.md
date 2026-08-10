# KuaiRand large capacity-lift Recompute-over-Reuse matrices

Positive means Recompute is more accurate than Reuse. Values are unscaled relative percentages reported by the evaluator.

## NDCG@5

| current \ cache | theta1 | theta2 | theta3 |
|---|---:|---:|---:|
| theta1 | +0.000% | — | — |
| theta2 | +2.066% | +0.000% | — |
| theta3 | +2.912% | +6.345% | +0.000% |

All cells: 3/3 positive; mean +3.774%; median +2.912%.
Adjacent cells: 2/2 positive; mean +4.206%; minimum +2.066%.

## MRR

| current \ cache | theta1 | theta2 | theta3 |
|---|---:|---:|---:|
| theta1 | +0.000% | — | — |
| theta2 | +4.084% | +0.000% | — |
| theta3 | -1.227% | +1.612% | +0.000% |

All cells: 2/3 positive; mean +1.490%; median +1.612%.
Adjacent cells: 2/2 positive; mean +2.848%; minimum +1.612%.

## HR@5

| current \ cache | theta1 | theta2 | theta3 |
|---|---:|---:|---:|
| theta1 | +0.000% | — | — |
| theta2 | -4.407% | +0.000% | — |
| theta3 | +5.607% | +6.004% | +0.000% |

All cells: 2/3 positive; mean +2.401%; median +5.607%.
Adjacent cells: 1/2 positive; mean +0.798%; minimum -4.407%.

Development evidence only; no K/V perturbation or metric scaling is applied.
