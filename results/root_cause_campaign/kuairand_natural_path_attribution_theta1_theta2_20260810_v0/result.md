# KuaiRand natural update-path attribution

| intervention | NDCG@5 loss | MRR loss | HR@5 loss | Top-10 changed | score cosine loss | hidden history projection | score history projection |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding_only | +0.000% | -0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection_plus_q | +0.000% | -0.002% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection_plus_kv | +5.650% | +4.373% | +4.947% | 10.106% | 4.805% | 0.8877 | 1.1786 |
| embedding_projection_plus_qkv | +5.650% | +4.371% | +4.947% | 10.091% | 4.797% | 0.8878 | 1.1787 |
| embedding_projection_plus_qkvo | +3.821% | +2.883% | +3.336% | 14.986% | 6.987% | 0.9108 | 1.1478 |
| full_without_qkvo | +0.000% | -0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0001 |
| full_current | +3.262% | +2.938% | +2.211% | 14.993% | 6.993% | 0.9120 | 1.1565 |
| frozen_coordinate_positive_control | -0.224% | +0.028% | -0.454% | 0.129% | 0.000% | 1.0006 | 1.0003 |
