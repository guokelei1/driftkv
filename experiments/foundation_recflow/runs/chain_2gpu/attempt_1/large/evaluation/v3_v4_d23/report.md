# RecFlow-Large v3 / v4 D23

All eligible final-role exposure targets.

| Metric | v3 | v4 | v4 minus v3 |
|---|---:|---:|---:|
| log_loss | 0.620281 | 0.619181 | -0.001100 |
| Brier | 0.215932 | 0.215370 | -0.000561 |
| ROC_AUC | 0.668604 | 0.671441 | +0.002837 |
| effective_view_PR_AUC | 0.533889 | 0.537645 | +0.003757 |
| ineffective_view_PR_AUC | 0.762574 | 0.764823 | +0.002249 |
| user_macro_AUC | 0.542984 | 0.545994 | +0.003010 |
| within_exposure_request_NDCG@1 | 0.397628 | 0.399457 | +0.001829 |
| mixed_label_request_NDCG@1 | 0.503050 | 0.505653 | +0.002603 |
| within_exposure_request_NDCG@3 | 0.442154 | 0.443456 | +0.001301 |
| mixed_label_request_NDCG@3 | 0.566444 | 0.568297 | +0.001852 |
| within_exposure_request_NDCG@5 | 0.521358 | 0.522835 | +0.001477 |
| mixed_label_request_NDCG@5 | 0.679211 | 0.681314 | +0.002102 |

Paired UID 95% interval for AUC gain: [0.001797273756340398, 0.00392814633577109].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
