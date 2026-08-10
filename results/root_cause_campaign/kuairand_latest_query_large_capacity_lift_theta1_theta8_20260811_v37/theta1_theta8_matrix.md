# KuaiRand large capacity-lift Recompute-over-Reuse matrices

Positive means Recompute is more accurate than Reuse. Values are unscaled relative percentages reported by the evaluator.

## NDCG@5

| current \ cache | theta1 | theta2 | theta3 | theta4 | theta5 | theta6 | theta7 | theta8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | +0.000% | — | — | — | — | — | — | — |
| theta2 | +2.066% | +0.000% | — | — | — | — | — | — |
| theta3 | +2.912% | +6.345% | +0.000% | — | — | — | — | — |
| theta4 | +39.688% | +22.646% | +11.515% | +0.000% | — | — | — | — |
| theta5 | +5.044% | +1.114% | -7.626% | +1.619% | +0.000% | — | — | — |
| theta6 | +15.104% | +6.862% | +2.227% | +6.119% | +1.613% | +0.000% | — | — |
| theta7 | +11.014% | +15.480% | +7.527% | -6.055% | +3.300% | +3.762% | +0.000% | — |
| theta8 | +25.106% | +18.408% | +10.237% | +12.618% | +6.361% | +7.708% | +15.128% | +0.000% |

All cells: 26/28 positive; mean +8.852%; median +6.612%.
Adjacent cells: 7/7 positive; mean +6.007%; minimum +1.613%.

## MRR

| current \ cache | theta1 | theta2 | theta3 | theta4 | theta5 | theta6 | theta7 | theta8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | +0.000% | — | — | — | — | — | — | — |
| theta2 | +4.084% | +0.000% | — | — | — | — | — | — |
| theta3 | -1.227% | +1.612% | +0.000% | — | — | — | — | — |
| theta4 | +17.108% | +12.605% | +4.839% | +0.000% | — | — | — | — |
| theta5 | -0.427% | +1.717% | -6.324% | +0.811% | +0.000% | — | — | — |
| theta6 | +11.942% | +5.683% | +3.941% | +6.192% | +1.512% | +0.000% | — | — |
| theta7 | +4.092% | +4.161% | -0.308% | -3.441% | +0.040% | +0.122% | +0.000% | — |
| theta8 | +12.373% | +5.014% | +4.080% | +5.203% | +2.771% | +4.991% | +5.317% | +0.000% |

All cells: 23/28 positive; mean +3.874%; median +4.082%.
Adjacent cells: 7/7 positive; mean +2.614%; minimum +0.122%.

## HR@5

| current \ cache | theta1 | theta2 | theta3 | theta4 | theta5 | theta6 | theta7 | theta8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | +0.000% | — | — | — | — | — | — | — |
| theta2 | -4.407% | +0.000% | — | — | — | — | — | — |
| theta3 | +5.607% | +6.004% | +0.000% | — | — | — | — | — |
| theta4 | +32.460% | +15.061% | +10.420% | +0.000% | — | — | — | — |
| theta5 | +5.273% | -4.613% | -3.661% | +0.696% | +0.000% | — | — | — |
| theta6 | +6.126% | +3.880% | -2.965% | -0.842% | +0.684% | +0.000% | — | — |
| theta7 | +12.447% | +18.444% | +14.624% | -2.914% | +6.175% | +7.243% | +0.000% | — |
| theta8 | +14.526% | +20.621% | +8.151% | +6.876% | +4.615% | +2.448% | +11.934% | +0.000% |

All cells: 22/28 positive; mean +6.961%; median +6.065%.
Adjacent cells: 6/7 positive; mean +4.653%; minimum -4.407%.

Development evidence only; no K/V perturbation or metric scaling is applied.
