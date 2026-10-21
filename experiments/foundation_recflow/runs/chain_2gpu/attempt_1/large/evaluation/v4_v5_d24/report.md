# RecFlow-Large v4 / v5 D24

All eligible final-role exposure targets.

| Metric | v4 | v5 | v5 minus v4 |
|---|---:|---:|---:|
| log_loss | 0.616177 | 0.617714 | +0.001536 |
| Brier | 0.214540 | 0.215189 | +0.000649 |
| ROC_AUC | 0.676490 | 0.673520 | -0.002970 |
| effective_view_PR_AUC | 0.541363 | 0.539665 | -0.001698 |
| ineffective_view_PR_AUC | 0.773139 | 0.770094 | -0.003045 |
| user_macro_AUC | 0.539003 | 0.542268 | +0.003265 |
| within_exposure_request_NDCG@1 | 0.397533 | 0.399345 | +0.001812 |
| mixed_label_request_NDCG@1 | 0.498461 | 0.501040 | +0.002578 |
| within_exposure_request_NDCG@3 | 0.447794 | 0.449006 | +0.001212 |
| mixed_label_request_NDCG@3 | 0.569979 | 0.571704 | +0.001725 |
| within_exposure_request_NDCG@5 | 0.525797 | 0.527174 | +0.001377 |
| mixed_label_request_NDCG@5 | 0.680973 | 0.682932 | +0.001959 |

Paired UID 95% interval for AUC gain: [-0.004151038003262908, -0.0018100551404535649].

Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.
Metric selection based on this result remains exploratory; retain every negative outcome.
