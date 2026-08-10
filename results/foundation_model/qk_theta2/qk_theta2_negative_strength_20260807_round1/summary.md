# QK theta2 negative-strength search

Status: `complete_no_preferred_candidate`

| Candidate | Rolling NDCG gap | NDCG CI+ | Rolling MRR gap | MRR CI+ | HR@10 gap | Boundary NDCG gap | Primary pass |
|---|---:|---|---:|---|---:|---:|---|
| theta2_relevance_e2_lr075_n32 | 1.4348% | True | 0.3291% | False | 2.0214% | 0.1784% | False |
| theta2_strength_e2_lr075_n64 | 0.2480% | False | 0.2033% | False | 0.3432% | 0.3331% | False |
| theta2_strength_e2_lr075_n96 | -0.3034% | False | 0.3342% | False | -0.7592% | -0.5180% | False |
| theta2_strength_e3_lr075_n64 | -0.6991% | False | -0.4910% | False | -0.4430% | 0.6029% | False |
| theta2_strength_e2_lr100_n64 | -0.5496% | False | -0.5681% | False | 0.0000% | -0.0425% | False |

The anchor and every search candidate use the same theta1 source, window2 training data, optimizer-participant users, and unseen window3 evaluation protocol.
The primary development gate requires rolling full-catalog NDCG@10 in the frozen 5%-10% range and positive record-cluster intervals for both NDCG@10 and MRR.
No checkpoint is committed automatically, and qualification/final roles are not consumed.
