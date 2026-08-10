# KuaiRand theta1–theta4 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422+20220423→20220424 | two_day_sequential_recent16_e2_kv003_initial | +122.948% | +246.061% | +174.736% | [+0.05753, +0.07787] | [+0.06375, +0.08927] |
| theta2 | theta1 | 20220424+20220425→20220426 | two_day_sequential_recent16_e2_kv003 | +4.592% | +4.610% | +3.108% | [+0.00324, +0.00990] | [+0.00205, +0.00997] |
| theta3 | theta2 | 20220426+20220427→20220428 | two_day_sequential_recent16_e2_kv003 | -0.509% | +0.726% | +1.701% | [-0.00323, +0.00218] | [-0.00309, +0.00553] |
| theta4 | theta3 | 20220428+20220429→20220430 | two_day_sequential_recent16_e2_kv003 | +0.636% | -0.124% | -1.393% | [-0.00149, +0.00309] | [-0.00323, +0.00294] |

## 4x4 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +246.061% | — | — | — |
| theta2 | +434.322% | +4.610% | — | — |
| theta3 | +423.539% | +7.422% | +0.726% | — |
| theta4 | +366.468% | +8.230% | +1.325% | -0.124% |

## 4x4 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +122.948% | — | — | — |
| theta2 | +167.890% | +4.592% | — | — |
| theta3 | +153.209% | +5.508% | -0.509% | — |
| theta4 | +157.653% | +6.641% | +1.295% | +0.636% |

## 4x4 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +174.736% | — | — | — |
| theta2 | +329.861% | +3.108% | — | — |
| theta3 | +364.148% | +6.762% | +1.701% | — |
| theta4 | +278.806% | +6.741% | +0.389% | -1.393% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.055071 | 0.122780 | +122.948% | 0.030759 | 0.106443 | +246.061% | +174.736% |
| theta2 | theta0 | 2 | 0.056183 | 0.150508 | +167.890% | 0.025737 | 0.137517 | +434.322% | +329.861% |
| theta2 | theta1 | 1 | 0.143899 | 0.150508 | +4.592% | 0.131456 | 0.137517 | +4.610% | +3.108% |
| theta3 | theta0 | 3 | 0.056904 | 0.144086 | +153.209% | 0.025642 | 0.134244 | +423.539% | +364.148% |
| theta3 | theta1 | 2 | 0.136564 | 0.144086 | +5.508% | 0.124969 | 0.134244 | +7.422% | +6.762% |
| theta3 | theta2 | 1 | 0.144823 | 0.144086 | -0.509% | 0.133277 | 0.134244 | +0.726% | +1.701% |
| theta4 | theta0 | 4 | 0.058545 | 0.150843 | +157.653% | 0.029129 | 0.135880 | +366.468% | +278.806% |
| theta4 | theta1 | 3 | 0.141450 | 0.150843 | +6.641% | 0.125547 | 0.135880 | +8.230% | +6.741% |
| theta4 | theta2 | 2 | 0.148915 | 0.150843 | +1.295% | 0.134103 | 0.135880 | +1.325% | +0.389% |
| theta4 | theta3 | 1 | 0.149890 | 0.150843 | +0.636% | 0.136049 | 0.135880 | -0.124% | -1.393% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
