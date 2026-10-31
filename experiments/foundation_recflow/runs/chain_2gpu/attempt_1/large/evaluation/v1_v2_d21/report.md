# RecFlow-Large v1 / v2 D21

All eligible final-role exposure targets.

| Metric | v1 | v2 | v2 minus v1 |
|---|---:|---:|---:|
| log_loss | 0.620234 | 0.618474 | -0.001761 |
| Brier | 0.215989 | 0.215270 | -0.000719 |
| ROC_AUC | 0.676365 | 0.679098 | +0.002734 |
| effective_view_PR_AUC | 0.551439 | 0.553399 | +0.001960 |
| ineffective_view_PR_AUC | 0.764087 | 0.766559 | +0.002472 |
| user_macro_AUC | 0.561403 | 0.561753 | +0.000350 |
| within_exposure_request_NDCG@1 | 0.416784 | 0.419913 | +0.003129 |
| mixed_label_request_NDCG@1 | 0.519231 | 0.523569 | +0.004338 |
| within_exposure_request_NDCG@3 | 0.461805 | 0.463990 | +0.002184 |
| mixed_label_request_NDCG@3 | 0.581637 | 0.584664 | +0.003028 |
| within_exposure_request_NDCG@5 | 0.540048 | 0.541440 | +0.001392 |
| mixed_label_request_NDCG@5 | 0.690091 | 0.692020 | +0.001930 |

Paired UID 95% interval for AUC gain: [0.0019515117645602848, 0.0035321234279617553].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
