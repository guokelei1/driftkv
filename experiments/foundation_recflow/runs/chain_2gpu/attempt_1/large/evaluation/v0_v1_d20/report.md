# RecFlow-Large v0 / v1 D20

All eligible final-role exposure targets.

| Metric | v0 | v1 | v1 minus v0 |
|---|---:|---:|---:|
| log_loss | 0.629356 | 0.616574 | -0.012782 |
| Brier | 0.219471 | 0.214455 | -0.005015 |
| ROC_AUC | 0.662883 | 0.677917 | +0.015034 |
| effective_view_PR_AUC | 0.538537 | 0.549650 | +0.011112 |
| ineffective_view_PR_AUC | 0.755638 | 0.769464 | +0.013827 |
| user_macro_AUC | 0.566344 | 0.564085 | -0.002258 |
| within_exposure_request_NDCG@1 | 0.423040 | 0.423341 | +0.000301 |
| mixed_label_request_NDCG@1 | 0.528573 | 0.528991 | +0.000417 |
| within_exposure_request_NDCG@3 | 0.466852 | 0.467977 | +0.001125 |
| mixed_label_request_NDCG@3 | 0.589305 | 0.590864 | +0.001559 |
| within_exposure_request_NDCG@5 | 0.544179 | 0.544992 | +0.000814 |
| mixed_label_request_NDCG@5 | 0.696495 | 0.697623 | +0.001128 |

Paired UID 95% interval for AUC gain: [0.013087530296420218, 0.016959609807981495].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
