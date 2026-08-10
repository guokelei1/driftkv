# KuaiRand theta1–theta3 Reuse loss

Primary table split: all users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | projectionlow_anchor_e4 | +6.625% | +9.368% | +6.037% | [-0.00043, +0.00812] | [-0.00205, +0.00887] |
| theta2 | theta1 | 20220423→20220424 | large_projectionlow_kv4_e4 | +2.535% | +2.277% | -1.394% | [-0.00226, +0.00510] | [-0.00382, +0.00548] |
| theta3 | theta2 | 20220424→20220425 | large_half_kv4_e3 | -0.388% | +2.984% | +5.364% | [-0.00517, +0.00408] | [-0.00466, +0.00633] |

## 3x3 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +9.368% | — | — |
| theta2 | +4.596% | +2.277% | — |
| theta3 | +17.652% | -2.372% | +2.984% |

## 3x3 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +6.625% | — | — |
| theta2 | +3.662% | +2.535% | — |
| theta3 | +8.283% | -1.252% | -0.388% |

## 3x3 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +6.037% | — | — |
| theta2 | +4.236% | -1.394% | — |
| theta3 | +14.823% | -3.678% | +5.364% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.060157 | 0.064143 | +6.625% | 0.039351 | 0.043038 | +9.368% | +6.037% |
| theta2 | theta0 | 2 | 0.055529 | 0.057563 | +3.662% | 0.033888 | 0.035445 | +4.596% | +4.236% |
| theta2 | theta1 | 1 | 0.056140 | 0.057563 | +2.535% | 0.034656 | 0.035445 | +2.277% | -1.394% |
| theta3 | theta0 | 3 | 0.054380 | 0.058884 | +8.283% | 0.030978 | 0.036446 | +17.652% | +14.823% |
| theta3 | theta1 | 2 | 0.059631 | 0.058884 | -1.252% | 0.037332 | 0.036446 | -2.372% | -3.678% |
| theta3 | theta2 | 1 | 0.059114 | 0.058884 | -0.388% | 0.035390 | 0.036446 | +2.984% | +5.364% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
