# RecFlow-Max v4 / v5 D24

All eligible final-role exposure targets.

| Metric | v4 | v5 | v5 minus v4 |
|---|---:|---:|---:|
| log_loss | 0.617825 | 0.621738 | +0.003913 |
| Brier | 0.215304 | 0.216789 | +0.001485 |
| ROC_AUC | 0.673921 | 0.668502 | -0.005419 |
| effective_view_PR_AUC | 0.535293 | 0.531385 | -0.003908 |
| ineffective_view_PR_AUC | 0.771845 | 0.765277 | -0.006567 |
| user_macro_AUC | 0.534760 | 0.538579 | +0.003819 |
| within_exposure_request_NDCG@1 | 0.389116 | 0.393383 | +0.004267 |
| mixed_label_request_NDCG@1 | 0.486484 | 0.492556 | +0.006072 |
| within_exposure_request_NDCG@3 | 0.440732 | 0.443366 | +0.002634 |
| mixed_label_request_NDCG@3 | 0.559931 | 0.563678 | +0.003747 |
| within_exposure_request_NDCG@5 | 0.523016 | 0.524754 | +0.001738 |
| mixed_label_request_NDCG@5 | 0.677015 | 0.679488 | +0.002473 |

Paired UID 95% interval for AUC gain: [-0.006658181311443032, -0.004197018776988507].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
