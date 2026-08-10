# KuaiRand theta1–theta4 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422+20220423→20220424 | rolling4_sequential_recent16_e2_kv003_initial | +104.799% | +199.643% | +157.388% | [+0.05289, +0.07767] | [+0.06272, +0.08710] |
| theta2 | theta1 | 20220422+20220423+20220424+20220425→20220426 | rolling4_sequential_recent16_e2_kv003 | +10.084% | +11.952% | +8.239% | [+0.00910, +0.01869] | [+0.00914, +0.02162] |
| theta3 | theta2 | 20220424+20220425+20220426+20220427→20220428 | rolling4_sequential_recent16_e2_kv003 | +0.078% | +0.685% | +2.015% | [-0.00229, +0.00227] | [-0.00239, +0.00431] |
| theta4 | theta3 | 20220426+20220427+20220428+20220429→20220430 | rolling4_sequential_recent16_e2_kv003 | +7.617% | +10.543% | +9.290% | [+0.00731, +0.01540] | [+0.00846, +0.02042] |

## 4x4 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +199.643% | — | — | — |
| theta2 | +291.451% | +11.952% | — | — |
| theta3 | +288.061% | +7.989% | +0.685% | — |
| theta4 | +265.400% | +17.774% | +12.169% | +10.543% |

## 4x4 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +104.799% | — | — | — |
| theta2 | +143.661% | +10.084% | — | — |
| theta3 | +133.326% | +7.260% | +0.078% | — |
| theta4 | +138.222% | +13.025% | +7.634% | +7.617% |

## 4x4 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +157.388% | — | — | — |
| theta2 | +222.251% | +8.239% | — | — |
| theta3 | +226.662% | +5.989% | +2.015% | — |
| theta4 | +199.213% | +16.273% | +13.038% | +9.290% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.061832 | 0.126632 | +104.799% | 0.037163 | 0.111356 | +199.643% | +157.388% |
| theta2 | theta0 | 2 | 0.062015 | 0.151107 | +143.661% | 0.035903 | 0.140541 | +291.451% | +222.251% |
| theta2 | theta1 | 1 | 0.137266 | 0.151107 | +10.084% | 0.125536 | 0.140541 | +11.952% | +8.239% |
| theta3 | theta0 | 3 | 0.060936 | 0.142180 | +133.326% | 0.032753 | 0.127103 | +288.061% | +226.662% |
| theta3 | theta1 | 2 | 0.132557 | 0.142180 | +7.260% | 0.117701 | 0.127103 | +7.989% | +5.989% |
| theta3 | theta2 | 1 | 0.142069 | 0.142180 | +0.078% | 0.126238 | 0.127103 | +0.685% | +2.015% |
| theta4 | theta0 | 4 | 0.066131 | 0.157539 | +138.222% | 0.039308 | 0.143630 | +265.400% | +199.213% |
| theta4 | theta1 | 3 | 0.139385 | 0.157539 | +13.025% | 0.121954 | 0.143630 | +17.774% | +16.273% |
| theta4 | theta2 | 2 | 0.146366 | 0.157539 | +7.634% | 0.128047 | 0.143630 | +12.169% | +13.038% |
| theta4 | theta3 | 1 | 0.146389 | 0.157539 | +7.617% | 0.129932 | 0.143630 | +10.543% | +9.290% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
