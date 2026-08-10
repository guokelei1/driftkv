# KuaiRand theta1–theta3 Reuse loss

Primary table split: all users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | projectiontrain_anchor_e4 | +0.542% | +4.649% | +9.412% | [-0.00212, +0.00279] | [-0.00108, +0.00468] |
| theta2 | theta1 | 20220423→20220424 | projectiontrain_continuation_e4 | -1.877% | -5.213% | -4.405% | [-0.00480, +0.00248] | [-0.00701, +0.00274] |
| theta3 | theta2 | 20220424→20220425 | projectiontrain_continuation_e4 | -2.994% | -7.245% | -6.560% | [-0.00516, +0.00092] | [-0.00708, +0.00096] |

## 3x3 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +4.649% | — | — |
| theta2 | -0.070% | -5.213% | — |
| theta3 | +4.813% | -1.815% | -7.245% |

## 3x3 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +0.542% | — | — |
| theta2 | +1.522% | -1.877% | — |
| theta3 | +4.853% | -0.118% | -2.994% |

## 3x3 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +9.412% | — | — |
| theta2 | -0.340% | -4.405% | — |
| theta3 | +2.330% | -1.862% | -6.560% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.059967 | 0.060292 | +0.542% | 0.037905 | 0.039668 | +4.649% | +9.412% |
| theta2 | theta0 | 2 | 0.057490 | 0.058366 | +1.522% | 0.036654 | 0.036628 | -0.070% | -0.340% |
| theta2 | theta1 | 1 | 0.059482 | 0.058366 | -1.877% | 0.038642 | 0.036628 | -5.213% | -4.405% |
| theta3 | theta0 | 3 | 0.056538 | 0.059281 | +4.853% | 0.034409 | 0.036065 | +4.813% | +2.330% |
| theta3 | theta1 | 2 | 0.059352 | 0.059281 | -0.118% | 0.036732 | 0.036065 | -1.815% | -1.862% |
| theta3 | theta2 | 1 | 0.061111 | 0.059281 | -2.994% | 0.038882 | 0.036065 | -7.245% | -6.560% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
