# KuaiRand theta1–theta3 Reuse loss

Primary table split: all users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | projectionlow_anchor_e4 | +7.639% | +8.581% | +3.409% | [-0.00097, +0.00980] | [-0.00323, +0.01011] |
| theta2 | theta1 | 20220423→20220424 | projectionlow_continuation_e4 | +4.084% | +2.066% | -4.407% | [-0.00074, +0.00546] | [-0.00317, +0.00462] |
| theta3 | theta2 | 20220424→20220425 | projectionlow_continuation_e4 | -0.856% | +1.841% | +3.498% | [-0.00374, +0.00284] | [-0.00358, +0.00473] |

## 3x3 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +8.581% | — | — |
| theta2 | +7.972% | +2.066% | — |
| theta3 | -4.438% | -8.602% | +1.841% |

## 3x3 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +7.639% | — | — |
| theta2 | +5.258% | +4.084% | — |
| theta3 | -2.635% | -5.580% | -0.856% |

## 3x3 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +3.409% | — | — |
| theta2 | +5.816% | -4.407% | — |
| theta3 | -3.083% | -6.157% | +3.498% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.058233 | 0.062681 | +7.639% | 0.036776 | 0.039932 | +8.581% | +3.409% |
| theta2 | theta0 | 2 | 0.056518 | 0.059490 | +5.258% | 0.033900 | 0.036603 | +7.972% | +5.816% |
| theta2 | theta1 | 1 | 0.057155 | 0.059490 | +4.084% | 0.035862 | 0.036603 | +2.066% | -4.407% |
| theta3 | theta0 | 3 | 0.057882 | 0.056357 | -2.635% | 0.035007 | 0.033453 | -4.438% | -3.083% |
| theta3 | theta1 | 2 | 0.059687 | 0.056357 | -5.580% | 0.036602 | 0.033453 | -8.602% | -6.157% |
| theta3 | theta2 | 1 | 0.056843 | 0.056357 | -0.856% | 0.032848 | 0.033453 | +1.841% | +3.498% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
