# RecFlow-Large v2 / v3 D22

All eligible final-role exposure targets.

| Metric | v2 | v3 | v3 minus v2 |
|---|---:|---:|---:|
| log_loss | 0.623066 | 0.622415 | -0.000651 |
| Brier | 0.217258 | 0.217086 | -0.000172 |
| ROC_AUC | 0.669347 | 0.668508 | -0.000839 |
| effective_view_PR_AUC | 0.542551 | 0.541347 | -0.001204 |
| ineffective_view_PR_AUC | 0.760368 | 0.759664 | -0.000704 |
| user_macro_AUC | 0.539008 | 0.544492 | +0.005484 |
| within_exposure_request_NDCG@1 | 0.409399 | 0.413924 | +0.004525 |
| mixed_label_request_NDCG@1 | 0.506966 | 0.513286 | +0.006320 |
| within_exposure_request_NDCG@3 | 0.451430 | 0.456391 | +0.004961 |
| mixed_label_request_NDCG@3 | 0.565672 | 0.572601 | +0.006929 |
| within_exposure_request_NDCG@5 | 0.533313 | 0.536606 | +0.003292 |
| mixed_label_request_NDCG@5 | 0.680040 | 0.684638 | +0.004598 |

Paired UID 95% interval for AUC gain: [-0.001804362708352794, 7.875052673567416e-05].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
