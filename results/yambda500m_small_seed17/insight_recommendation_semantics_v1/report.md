# Recommendation semantics and real pairwise ranking

Scope: 217584 paired requests and 11124 real positive-negative pairs. No sampled negatives.

## Candidate modes

| edge | candidate_mode | requests | users | mean_release_benefit | mean_reuse_harm | mean_abs_probability_shift | current_minus_parent_ROC_AUC_pp | current_minus_reuse_ROC_AUC_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | novel_to_prefix | 19801 | 3392 | 0.012468 | 0.0013893 | 0.00245913 | 2.06375 | 0.442571 |
| v0_to_v1 | old_only_repeat | 5383 | 1836 | 0.00312154 | 2.78327e-05 | 0.00201609 | -0.471386 | -0.0457694 |
| v0_to_v1 | recent_repeat | 18002 | 3195 | -0.0044235 | -0.000331776 | 0.00229526 | 1.15484 | 0.289215 |
| v1_to_v2 | novel_to_prefix | 19585 | 3385 | 0.00223458 | 0.000698724 | 0.000990808 | 0.644705 | 0.270775 |
| v1_to_v2 | old_only_repeat | 4901 | 1807 | 0.000338322 | 0.000237746 | 0.000923897 | 0.575804 | 0.0516823 |
| v1_to_v2 | recent_repeat | 17169 | 3121 | 0.000967166 | -0.000200686 | 0.00102482 | 0.543253 | 0.139014 |
| v2_to_v3 | novel_to_prefix | 19657 | 3386 | -0.00138189 | 3.70432e-05 | 0.00075754 | 0.798038 | 0.356162 |
| v2_to_v3 | old_only_repeat | 5195 | 1744 | 0.00210548 | 0.000246033 | 0.000871705 | -0.283999 | -0.0555465 |
| v2_to_v3 | recent_repeat | 18240 | 3184 | 0.00216332 | 0.000351624 | 0.000844699 | 0.408921 | 0.129696 |
| v3_to_v4 | novel_to_prefix | 20479 | 3491 | 0.00360271 | 0.00140258 | 0.00270001 | -0.530714 | 0.255242 |
| v3_to_v4 | old_only_repeat | 4895 | 1684 | -0.00482156 | -0.000492243 | 0.00270845 | -0.963416 | 0.308705 |
| v3_to_v4 | recent_repeat | 18571 | 3163 | -0.00360916 | -0.00160726 | 0.00264777 | 0.103719 | 0.0812527 |
| v4_to_v5 | novel_to_prefix | 23391 | 3603 | -0.00168115 | 0.000123959 | 0.000830147 | 0.143914 | 0.0405892 |
| v4_to_v5 | old_only_repeat | 4648 | 1677 | -1.91044e-05 | 8.83402e-05 | 0.000760832 | 0.181853 | 0.110097 |
| v4_to_v5 | recent_repeat | 17667 | 3197 | 0.00237399 | -7.89054e-05 | 0.00082733 | 0.18611 | 0.00897777 |

## Real request-group ranking

| edge | real_positive_negative_pairs | request_groups | current_pairwise_accuracy | reuse_pairwise_accuracy | current_minus_reuse_pairwise_accuracy_pp | harmful_flip_fraction | beneficial_flip_fraction | mean_margin_erosion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 1375 | 231 | 0.498182 | 0.499636 | -0.145455 | 0.00654545 | 0.008 | -8.0099e-05 |
| v1_to_v2 | 1166 | 204 | 0.511149 | 0.51801 | -0.686106 | 0.0111492 | 0.0180103 | -6.38804e-05 |
| v2_to_v3 | 6449 | 209 | 0.538533 | 0.537293 | 0.12405 | 0.00232594 | 0.00108544 | -5.10209e-05 |
| v3_to_v4 | 960 | 241 | 0.50625 | 0.508333 | -0.208333 | 0.00208333 | 0.00416667 | -8.79832e-05 |
| v4_to_v5 | 1174 | 213 | 0.554514 | 0.554514 | 0 | 0.00170358 | 0.00170358 | -1.49786e-05 |

Novel-to-prefix candidates have larger Current-minus-Reuse ROC-AUC loss than recent repeats on all five edges. Parent-to-Current release gain does not consistently concentrate in the same candidate mode, so the supported claim is candidate-conditioned compatibility risk, not universal suppression of novel capability.

Real same-timestamp pairwise accuracy changes direction across edges; aggregate AUC harm must not be restated as a universal pair-inversion effect.

Feature correlations are in `semantic_correlations.csv`. Persistent behavior tokens are organic/non-organic listens; request like/dislike is used only as the evaluation label.
