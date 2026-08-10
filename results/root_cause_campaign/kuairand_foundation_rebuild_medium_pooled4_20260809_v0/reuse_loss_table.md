# KuaiRand theta1–theta4 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422+20220423→20220424 | two_day_pooled_recent32_e2_kv003_initial | +114.286% | +195.272% | +130.610% | [+0.05125, +0.07238] | [+0.05276, +0.08017] |
| theta2 | theta1 | 20220424+20220425→20220426 | two_day_pooled_recent32_e2_kv003 | +0.551% | +1.591% | +2.651% | [-0.00528, +0.00836] | [-0.00454, +0.00934] |
| theta3 | theta2 | 20220426+20220427→20220428 | two_day_pooled_recent32_e2_kv003 | +2.171% | +2.264% | +1.041% | [+0.00048, +0.00544] | [+0.00033, +0.00592] |
| theta4 | theta3 | 20220428+20220429→20220430 | two_day_pooled_recent32_e2_kv003 | -0.202% | +0.385% | +1.033% | [-0.00140, +0.00087] | [-0.00113, +0.00190] |

## 4x4 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +195.272% | — | — | — |
| theta2 | +399.594% | +1.591% | — | — |
| theta3 | +391.365% | +8.098% | +2.264% | — |
| theta4 | +439.720% | +3.773% | +1.765% | +0.385% |

## 4x4 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +114.286% | — | — | — |
| theta2 | +170.197% | +0.551% | — | — |
| theta3 | +161.637% | +6.300% | +2.171% | — |
| theta4 | +175.657% | +4.062% | +0.666% | -0.202% |

## 4x4 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 | theta3 |
|---|---:|---:|---:|---:|
| theta1 | +130.610% | — | — | — |
| theta2 | +286.204% | +2.651% | — | — |
| theta3 | +289.115% | +6.752% | +1.041% | — |
| theta4 | +310.734% | +2.654% | +2.516% | +1.033% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.054645 | 0.117096 | +114.286% | 0.034069 | 0.100598 | +195.272% | +130.610% |
| theta2 | theta0 | 2 | 0.056821 | 0.153529 | +170.197% | 0.028362 | 0.141693 | +399.594% | +286.204% |
| theta2 | theta1 | 1 | 0.152688 | 0.153529 | +0.551% | 0.139474 | 0.141693 | +1.591% | +2.651% |
| theta3 | theta0 | 3 | 0.056827 | 0.148680 | +161.637% | 0.027830 | 0.136746 | +391.365% | +289.115% |
| theta3 | theta1 | 2 | 0.139868 | 0.148680 | +6.300% | 0.126502 | 0.136746 | +8.098% | +6.752% |
| theta3 | theta2 | 1 | 0.145521 | 0.148680 | +2.171% | 0.133719 | 0.136746 | +2.264% | +1.041% |
| theta4 | theta0 | 4 | 0.055914 | 0.154132 | +175.657% | 0.025537 | 0.137826 | +439.720% | +310.734% |
| theta4 | theta1 | 3 | 0.148116 | 0.154132 | +4.062% | 0.132815 | 0.137826 | +3.773% | +2.654% |
| theta4 | theta2 | 2 | 0.153112 | 0.154132 | +0.666% | 0.135435 | 0.137826 | +1.765% | +2.516% |
| theta4 | theta3 | 1 | 0.154444 | 0.154132 | -0.202% | 0.137297 | 0.137826 | +0.385% | +1.033% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
