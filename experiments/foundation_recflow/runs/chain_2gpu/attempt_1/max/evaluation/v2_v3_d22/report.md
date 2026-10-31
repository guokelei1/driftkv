# RecFlow-Max v2 / v3 D22

All eligible final-role exposure targets.

| Metric | v2 | v3 | v3 minus v2 |
|---|---:|---:|---:|
| log_loss | 0.625927 | 0.626485 | +0.000557 |
| Brier | 0.218607 | 0.218912 | +0.000305 |
| ROC_AUC | 0.665102 | 0.663196 | -0.001906 |
| effective_view_PR_AUC | 0.534947 | 0.533641 | -0.001306 |
| ineffective_view_PR_AUC | 0.758450 | 0.756173 | -0.002277 |
| user_macro_AUC | 0.539018 | 0.541014 | +0.001995 |
| within_exposure_request_NDCG@1 | 0.401738 | 0.405389 | +0.003651 |
| mixed_label_request_NDCG@1 | 0.496265 | 0.501365 | +0.005099 |
| within_exposure_request_NDCG@3 | 0.446597 | 0.449627 | +0.003030 |
| mixed_label_request_NDCG@3 | 0.558922 | 0.563153 | +0.004232 |
| within_exposure_request_NDCG@5 | 0.529896 | 0.532318 | +0.002422 |
| mixed_label_request_NDCG@5 | 0.675267 | 0.678650 | +0.003383 |

Paired UID 95% interval for AUC gain: [-0.0031102261576821527, -0.0007569198202812628].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
