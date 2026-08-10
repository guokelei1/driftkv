# KuaiRand theta1–theta8 Reuse loss

## Adjacent-version gate

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | theta1_canary_n8192_e2 | +33.831% | +53.648% | +36.364% | [+0.01650, +0.03088] | [+0.01658, +0.03292] |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.068868 | 0.092167 | +33.831% | 0.046065 | 0.070778 | +53.648% | +36.364% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
