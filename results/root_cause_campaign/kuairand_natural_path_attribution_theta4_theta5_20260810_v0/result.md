# KuaiRand natural update-path attribution

| intervention | NDCG@5 loss | MRR loss | HR@5 loss | Top-10 changed | score cosine loss | hidden history projection | score history projection |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding_only | +0.000% | +0.001% | +0.000% | 0.002% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection_plus_q | +0.000% | -0.000% | +0.000% | 0.002% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection_plus_kv | +3.592% | +2.042% | +4.122% | 19.519% | 2.287% | 0.6860 | 1.0538 |
| embedding_projection_plus_qkv | +2.661% | +0.921% | +4.237% | 18.457% | 1.856% | 0.7553 | 1.0194 |
| embedding_projection_plus_qkvo | +1.697% | +1.010% | +2.183% | 23.499% | 1.883% | 0.8498 | 1.1193 |
| full_without_qkvo | +0.637% | +0.576% | +0.330% | 1.415% | 0.017% | 1.0193 | 0.9766 |
| full_current | -3.377% | -3.217% | -1.726% | 32.203% | 4.049% | 0.8331 | 1.2945 |
| frozen_coordinate_positive_control | +5.053% | +3.673% | +4.712% | 7.918% | 0.176% | 0.9322 | 0.9559 |
