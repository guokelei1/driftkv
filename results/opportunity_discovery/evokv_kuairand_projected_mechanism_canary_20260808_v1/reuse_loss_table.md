# KuaiRand theta1–theta8 Reuse loss

## Adjacent-version gate

| current | cache | update/eval | candidate | MRR | NDCG@5 | HR@5 | MRR CI | NDCG@5 CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| theta1 | theta0 | 20220422→20220423 | theta1_mechanism_canary_recent8_kv | +40.283% | +69.559% | +46.266% | [+0.01858, +0.03411] | [+0.02012, +0.03799] |

## Direct cache-age matrix

| current | cache | age | Reuse MRR | Recompute MRR | MRR loss | Reuse NDCG@5 | Recompute NDCG@5 | NDCG@5 loss | HR@5 loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta1 | theta0 | 1 | 0.065358 | 0.091686 | +40.283% | 0.041727 | 0.070753 | +69.559% | +46.266% |

All values are development measurements. Direct age-k is not recursive mixed-version append lineage.
