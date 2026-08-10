# KuaiRand theta1–theta3 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | theta1_warmup_n8192_e2 | +116.282% | +229.276% | +166.835% | [+0.05813, +0.08080] | [+0.06420, +0.09002] |
| theta2 | theta1 | 20220423→20220424 | uniform2x_kv4x_n16384_e3 | +8.936% | +10.089% | +4.946% | [+0.00577, +0.01559] | [+0.00505, +0.01472] |
| theta3 | theta2 | 20220424→20220425 | theta3_kv006_e3 | +0.650% | +0.252% | -0.508% | [-0.00036, +0.00224] | [-0.00088, +0.00194] |

## 3x3 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +229.276% | — | — |
| theta2 | +1.115% | +10.089% | — |
| theta3 | +3.158% | +381.696% | +0.252% |

## 3x3 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +116.282% | — | — |
| theta2 | +1.166% | +8.936% | — |
| theta3 | +3.670% | +221.981% | +0.650% |

## 3x3 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +166.835% | — | — |
| theta2 | +2.308% | +4.946% | — |
| theta3 | +1.747% | +357.287% | -0.508% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.058938 | 0.127474 | +116.282% | 0.033630 | 0.110734 | +229.276% | +166.835% |
| theta2 | theta0 | 2 | 0.127896 | 0.129388 | +1.166% | 0.111788 | 0.113035 | +1.115% | +2.308% |
| theta2 | theta1 | 1 | 0.118774 | 0.129388 | +8.936% | 0.102676 | 0.113035 | +10.089% | +4.946% |
| theta3 | theta0 | 3 | 0.138494 | 0.143577 | +3.670% | 0.124214 | 0.128136 | +3.158% | +1.747% |
| theta3 | theta1 | 2 | 0.044592 | 0.143577 | +221.981% | 0.026601 | 0.128136 | +381.696% | +357.287% |
| theta3 | theta2 | 1 | 0.142649 | 0.143577 | +0.650% | 0.127814 | 0.128136 | +0.252% | -0.508% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
