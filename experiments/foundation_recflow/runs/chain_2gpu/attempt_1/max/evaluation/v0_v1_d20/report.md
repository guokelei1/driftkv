# RecFlow-Max v0 / v1 D20

All eligible final-role exposure targets.

| Metric | v0 | v1 | v1 minus v0 |
|---|---:|---:|---:|
| log_loss | 0.631763 | 0.621343 | -0.010420 |
| Brier | 0.220891 | 0.216527 | -0.004364 |
| ROC_AUC | 0.656730 | 0.673240 | +0.016511 |
| effective_view_PR_AUC | 0.528942 | 0.542360 | +0.013418 |
| ineffective_view_PR_AUC | 0.749895 | 0.766305 | +0.016410 |
| user_macro_AUC | 0.549847 | 0.552495 | +0.002648 |
| within_exposure_request_NDCG@1 | 0.410753 | 0.410703 | -0.000050 |
| mixed_label_request_NDCG@1 | 0.511541 | 0.511471 | -0.000070 |
| within_exposure_request_NDCG@3 | 0.457517 | 0.458256 | +0.000738 |
| mixed_label_request_NDCG@3 | 0.576365 | 0.577389 | +0.001023 |
| within_exposure_request_NDCG@5 | 0.538652 | 0.538676 | +0.000024 |
| mixed_label_request_NDCG@5 | 0.688834 | 0.688867 | +0.000034 |

Paired UID 95% interval for AUC gain: [0.014556168358354663, 0.01839411989537447].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
