# RecFlow-Max v3 / v4 D23

All eligible final-role exposure targets.

| Metric | v3 | v4 | v4 minus v3 |
|---|---:|---:|---:|
| log_loss | 0.623463 | 0.620177 | -0.003286 |
| Brier | 0.217450 | 0.215926 | -0.001524 |
| ROC_AUC | 0.664783 | 0.669109 | +0.004326 |
| effective_view_PR_AUC | 0.528974 | 0.533878 | +0.004904 |
| ineffective_view_PR_AUC | 0.761204 | 0.764609 | +0.003405 |
| user_macro_AUC | 0.541994 | 0.548499 | +0.006505 |
| within_exposure_request_NDCG@1 | 0.391829 | 0.394650 | +0.002821 |
| mixed_label_request_NDCG@1 | 0.494793 | 0.498810 | +0.004017 |
| within_exposure_request_NDCG@3 | 0.440772 | 0.441046 | +0.000274 |
| mixed_label_request_NDCG@3 | 0.564476 | 0.564866 | +0.000390 |
| within_exposure_request_NDCG@5 | 0.519676 | 0.521524 | +0.001848 |
| mixed_label_request_NDCG@5 | 0.676817 | 0.679447 | +0.002631 |

Paired UID 95% interval for AUC gain: [0.0032565516599837724, 0.005362638399847852].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
