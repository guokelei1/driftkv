# KuaiRand theta1–theta3 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | theta1_warmup_n8192_e2 | +116.282% | +229.276% | +166.835% | [+0.05813, +0.08080] | [+0.06420, +0.09002] |
| theta2 | theta1 | 20220423→20220424 | stationary_half_recent16 | +7.813% | +10.393% | +6.629% | [+0.00356, +0.01398] | [+0.00483, +0.01409] |
| theta3 | theta2 | 20220424→20220425 | stationary_upper_recent16 | +2.414% | +2.539% | +0.520% | [-0.00076, +0.00726] | [-0.00212, +0.00762] |

## 3x3 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +229.276% | — | — |
| theta2 | +100.771% | +10.393% | — |
| theta3 | +9.350% | +2.494% | +2.539% |

## 3x3 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +116.282% | — | — |
| theta2 | +61.929% | +7.813% | — |
| theta3 | +8.753% | +2.698% | +2.414% |

## 3x3 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +166.835% | — | — |
| theta2 | +68.742% | +6.629% | — |
| theta3 | +4.525% | -0.060% | +0.520% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.058938 | 0.127474 | +116.282% | 0.033630 | 0.110734 | +229.276% | +166.835% |
| theta2 | theta0 | 2 | 0.073173 | 0.118489 | +61.929% | 0.051311 | 0.103017 | +100.771% | +68.742% |
| theta2 | theta1 | 1 | 0.109903 | 0.118489 | +7.813% | 0.093318 | 0.103017 | +10.393% | +6.629% |
| theta3 | theta0 | 3 | 0.119591 | 0.130059 | +8.753% | 0.106745 | 0.116726 | +9.350% | +4.525% |
| theta3 | theta1 | 2 | 0.126642 | 0.130059 | +2.698% | 0.113886 | 0.116726 | +2.494% | -0.060% |
| theta3 | theta2 | 1 | 0.126993 | 0.130059 | +2.414% | 0.113836 | 0.116726 | +2.539% | +0.520% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
