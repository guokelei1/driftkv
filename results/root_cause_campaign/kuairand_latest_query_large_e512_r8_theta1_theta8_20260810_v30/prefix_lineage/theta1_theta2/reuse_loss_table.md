# KuaiRand theta1–theta2 Reuse loss

Primary table split: all users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | projectionlow_anchor_e4 | +15.361% | +27.365% | +21.558% | [+0.00345, +0.01365] | [+0.00277, +0.01573] |
| theta2 | theta1 | 20220423→20220424 | projectionlow_continuation_e4 | +4.220% | +0.463% | -5.565% | [-0.00180, +0.00667] | [-0.00507, +0.00540] |

## 2x2 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 |
|---|---:|---:|
| theta1 | +27.365% | — |
| theta2 | +1.889% | +0.463% |

## 2x2 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 |
|---|---:|---:|
| theta1 | +15.361% | — |
| theta2 | +4.587% | +4.220% |

## 2x2 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 |
|---|---:|---:|
| theta1 | +21.558% | — |
| theta2 | -1.093% | -5.565% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.056334 | 0.064988 | +15.361% | 0.033826 | 0.043083 | +27.365% | +21.558% |
| theta2 | theta0 | 2 | 0.056127 | 0.058701 | +4.587% | 0.034630 | 0.035284 | +1.889% | -1.093% |
| theta2 | theta1 | 1 | 0.056325 | 0.058701 | +4.220% | 0.035122 | 0.035284 | +0.463% | -5.565% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
