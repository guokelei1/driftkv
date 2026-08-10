# KuaiRand natural update-path attribution

| intervention | NDCG@5 loss | MRR loss | HR@5 loss | Top-10 changed | score cosine loss | hidden history projection | score history projection |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedding_only | +0.000% | +0.000% | +0.000% | 0.000% | 0.000% | 1.0000 | 1.0000 |
| embedding_projection | +0.000% | +0.001% | +0.000% | 0.006% | 0.000% | 0.9999 | 1.0001 |
| embedding_projection_plus_q | +0.000% | +0.005% | +0.000% | 0.010% | 0.000% | 0.9999 | 1.0001 |
| embedding_projection_plus_kv | +0.807% | +0.637% | +0.702% | 5.499% | 0.397% | 0.8980 | 1.0399 |
| embedding_projection_plus_qkv | +1.591% | +1.012% | +1.633% | 4.942% | 0.317% | 0.9117 | 1.0369 |
| embedding_projection_plus_qkvo | +1.012% | +0.795% | +0.917% | 7.259% | 0.398% | 0.9180 | 1.0712 |
| full_without_qkvo | -0.492% | -0.377% | -0.336% | 0.171% | 0.001% | 0.9968 | 1.0019 |
| full_current | +1.336% | +0.993% | +1.179% | 7.334% | 0.423% | 0.9165 | 1.0751 |
| frozen_coordinate_positive_control | +2.537% | +1.653% | +2.410% | 9.021% | 0.714% | 0.8846 | 1.0252 |
