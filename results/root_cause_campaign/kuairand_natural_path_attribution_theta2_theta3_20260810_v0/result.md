# KuaiRand natural update-path attribution

| intervention | NDCG@5 loss | MRR loss | HR@5 loss | Top-10 changed | score cosine loss | hidden history projection | score history projection |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding_only | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection_plus_q | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection_plus_kv | +2.983% | +2.176% | +3.278% | 14.492% | 6.160% | 0.7160 | 1.2604 |
| embedding_projection_plus_qkv | +2.422% | +1.926% | +2.587% | 12.778% | 4.660% | 0.7606 | 1.2599 |
| embedding_projection_plus_qkvo | +0.910% | +0.171% | +2.085% | 17.596% | 4.908% | 0.8139 | 1.2851 |
| full_without_qkvo | -0.058% | -0.003% | -0.098% | 0.013% | 0.000% | 1.0000 | 1.0002 |
| full_current | -0.096% | -0.352% | +1.147% | 17.110% | 4.587% | 0.8273 | 1.2829 |
| frozen_coordinate_positive_control | -0.186% | -0.443% | +0.498% | 8.398% | 1.330% | 0.9114 | 1.0148 |
