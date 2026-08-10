# KuaiRand theta1–theta3 Reuse loss

Primary table split: held-out users.

## Adjacent-version measurements

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | theta1_warmup_n8192_e2 | +120.778% | +218.270% | +150.000% | [+0.04738, +0.09692] | [+0.04768, +0.10910] |
| theta2 | theta1 | 20220423→20220424 | stationary_half_recent16 | +12.390% | +13.345% | +6.667% | [+0.00554, +0.02481] | [+0.00152, +0.02543] |
| theta3 | theta2 | 20220424→20220425 | smooth_weak_recent16 | +0.057% | -2.754% | -5.882% | [-0.00683, +0.00604] | [-0.01002, +0.00329] |

## 3x3 triangular matrix: NDCG@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +218.270% | — | — |
| theta2 | +92.494% | +13.345% | — |
| theta3 | +40.890% | -1.509% | -2.754% |

## 3x3 triangular matrix: MRR relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +120.778% | — | — |
| theta2 | +55.445% | +12.390% | — |
| theta3 | +32.620% | +0.877% | +0.057% |

## 3x3 triangular matrix: HR@5 relative Recompute-over-Reuse

Positive means Recompute is better; negative means Reuse has the higher point estimate.

| current \ cache | theta0 | theta1 | theta2 |
|---|---:|---:|---:|
| theta1 | +150.000% | — | — |
| theta2 | +64.103% | +6.667% | — |
| theta3 | +23.077% | -4.478% | -5.882% |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.060054 | 0.132587 | +120.778% | 0.036261 | 0.115409 | +218.270% | +150.000% |
| theta2 | theta0 | 2 | 0.082156 | 0.127708 | +55.445% | 0.057509 | 0.110702 | +92.494% | +64.103% |
| theta2 | theta1 | 1 | 0.113629 | 0.127708 | +12.390% | 0.097668 | 0.110702 | +13.345% | +6.667% |
| theta3 | theta0 | 3 | 0.094777 | 0.125693 | +32.620% | 0.079410 | 0.111880 | +40.890% | +23.077% |
| theta3 | theta1 | 2 | 0.124601 | 0.125693 | +0.877% | 0.113594 | 0.111880 | -1.509% | -4.478% |
| theta3 | theta2 | 1 | 0.125622 | 0.125693 | +0.057% | 0.115049 | 0.111880 | -2.754% | -5.882% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
