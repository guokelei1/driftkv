# RecFlow-Max v1 / v2 D21

All eligible final-role exposure targets.

| Metric | v1 | v2 | v2 minus v1 |
|---|---:|---:|---:|
| log_loss | 0.627159 | 0.622195 | -0.004964 |
| Brier | 0.219116 | 0.217016 | -0.002099 |
| ROC_AUC | 0.669977 | 0.672617 | +0.002640 |
| effective_view_PR_AUC | 0.541039 | 0.544909 | +0.003870 |
| ineffective_view_PR_AUC | 0.760880 | 0.762798 | +0.001918 |
| user_macro_AUC | 0.553301 | 0.549862 | -0.003439 |
| within_exposure_request_NDCG@1 | 0.409169 | 0.410108 | +0.000939 |
| mixed_label_request_NDCG@1 | 0.508676 | 0.509977 | +0.001301 |
| within_exposure_request_NDCG@3 | 0.456390 | 0.456552 | +0.000163 |
| mixed_label_request_NDCG@3 | 0.574129 | 0.574355 | +0.000225 |
| within_exposure_request_NDCG@5 | 0.535851 | 0.537548 | +0.001697 |
| mixed_label_request_NDCG@5 | 0.684274 | 0.686626 | +0.002352 |

Paired UID 95% interval for AUC gain: [0.0016250204009838122, 0.0036437460893051525].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
