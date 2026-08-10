# KuaiRand theta1–theta4 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422+20220423→20220424 | rolling4_sequential_recent16_e2_kv005_initial | +118.563% | +231.107% | +178.041% | [+0.05727, +0.07961] | [+0.06723, +0.09169] |
| theta2 | theta1 | 20220422+20220423+20220424+20220425→20220426 | rolling4_sequential_recent16_e2_kv005 | +6.838% | +9.988% | +9.754% | [+0.00638, +0.01419] | [+0.00929, +0.01722] |
| theta3 | theta2 | 20220424+20220425+20220426+20220427→20220428 | rolling4_sequential_recent16_e2_kv005 | +0.961% | +1.959% | +2.823% | [-0.00214, +0.00433] | [-0.00174, +0.00599] |
| theta4 | theta3 | 20220426+20220427+20220428+20220429→20220430 | rolling4_sequential_recent16_e2_kv005 | +4.328% | +6.808% | +7.273% | [+0.00358, +0.00878] | [+0.00547, +0.01216] |

## 4x4 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +231.107% | — | — | — |
| theta2 | +377.015% | +9.988% | — | — |
| theta3 | +263.444% | +9.033% | +1.959% | — |
| theta4 | +424.249% | +8.537% | +1.206% | +6.808% |

## 4x4 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +118.563% | — | — | — |
| theta2 | +162.654% | +6.838% | — | — |
| theta3 | +134.548% | +6.959% | +0.961% | — |
| theta4 | +177.813% | +6.317% | -0.240% | +4.328% |

## 4x4 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +178.041% | — | — | — |
| theta2 | +296.990% | +9.754% | — | — |
| theta3 | +207.773% | +7.757% | +2.823% | — |
| theta4 | +310.289% | +8.701% | +3.954% | +7.273% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.057322 | 0.125285 | +118.563% | 0.034009 | 0.112607 | +231.107% | +178.041% |
| theta2 | theta0 | 2 | 0.058701 | 0.154180 | +162.654% | 0.030328 | 0.144670 | +377.015% | +296.990% |
| theta2 | theta1 | 1 | 0.144312 | 0.154180 | +6.838% | 0.131532 | 0.144670 | +9.988% | +9.754% |
| theta3 | theta0 | 3 | 0.062157 | 0.145788 | +134.548% | 0.036333 | 0.132049 | +263.444% | +207.773% |
| theta3 | theta1 | 2 | 0.136302 | 0.145788 | +6.959% | 0.121109 | 0.132049 | +9.033% | +7.757% |
| theta3 | theta2 | 1 | 0.144400 | 0.145788 | +0.961% | 0.129513 | 0.132049 | +1.959% | +2.823% |
| theta4 | theta0 | 4 | 0.054622 | 0.151746 | +177.813% | 0.026104 | 0.136850 | +424.249% | +310.289% |
| theta4 | theta1 | 3 | 0.142730 | 0.151746 | +6.317% | 0.126085 | 0.136850 | +8.537% | +8.701% |
| theta4 | theta2 | 2 | 0.152110 | 0.151746 | -0.240% | 0.135218 | 0.136850 | +1.206% | +3.954% |
| theta4 | theta3 | 1 | 0.145451 | 0.151746 | +4.328% | 0.128127 | 0.136850 | +6.808% | +7.273% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
