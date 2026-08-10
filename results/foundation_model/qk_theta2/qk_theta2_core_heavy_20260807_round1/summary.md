# QK theta2 core-heavy search

Status: `complete_no_preferred_candidate`

| Candidate | Rolling NDCG gap | NDCG CI+ | Rolling MRR gap | MRR CI+ | Fresh NDCG/anchor | Fresh MRR/anchor | Direct-K/V probe max | Primary pass |
|---|---:|---|---:|---|---:|---:|---:|---|
| theta2_relevance_e2_lr075_n32 | 1.4348% | True | 0.3291% | False | 1.0000 | 1.0000 | - | False |
| theta2_core_d150_p100_e025_n32 | 0.8471% | False | 0.0735% | False | 0.9890 | 0.9971 | 0.00377426 | False |
| theta2_core_d200_p100_e025_n32 | 0.4441% | False | -0.2834% | False | 0.9975 | 0.9983 | 0.00465826 | False |
| theta2_core_d150_p150_e025_n32 | 0.3981% | False | 0.0360% | False | 0.9973 | 1.0003 | 0.00371901 | False |
| theta2_core_d150_p100_e050_n32 | 0.9085% | False | 0.1824% | False | 0.9941 | 1.0004 | 0.00382332 | False |

Every search candidate uses the same theta1 source, n32 sampled objective, two epochs, window2 training data, optimizer-participant users, and unseen window3 evaluation protocol.
The primary development gate requires rolling full-catalog NDCG@10 in the frozen 5%-10% range, positive record-cluster intervals for both NDCG@10 and MRR, and at least 98% of anchor Recompute NDCG@10 and MRR.
No checkpoint is committed automatically, and qualification/final roles are not consumed.
