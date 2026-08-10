# KuaiRand theta1–theta2 Reuse loss

Primary table split: all users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | projectionlow_anchor_e4 | +15.361% | +27.365% | +21.558% | [+0.00345, +0.01365] | [+0.00277, +0.01573] |
| theta2 | theta1 | 20220423→20220424 | rowrep_projectionlow_kv4_e4 | +5.490% | +11.908% | +10.943% | [-0.00159, +0.00780] | [-0.00156, +0.00958] |

## 2x2 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 |
|---|---:|---:|
| theta1 | +27.365% | — |
| theta2 | +12.723% | +11.908% |

## 2x2 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 |
|---|---:|---:|
| theta1 | +15.361% | — |
| theta2 | +6.452% | +5.490% |

## 2x2 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 |
|---|---:|---:|
| theta1 | +21.558% | — |
| theta2 | +13.295% | +10.943% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.056334 | 0.064988 | +15.361% | 0.033826 | 0.043083 | +27.365% | +21.558% |
| theta2 | theta0 | 2 | 0.056041 | 0.059657 | +6.452% | 0.033527 | 0.037792 | +12.723% | +13.295% |
| theta2 | theta1 | 1 | 0.056552 | 0.059657 | +5.490% | 0.033771 | 0.037792 | +11.908% | +10.943% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
