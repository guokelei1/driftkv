# Embedding-origin analysis

This analysis scales the request-level probe to the first observed request of every active user on every edge. Item drift is measured for the fixed in-vocabulary mapping. OOV buckets are not interpreted as individual new-item embeddings.

## Edge summary

| edge | requests | users | exposed_iv_items | embedding_drift_p50 | embedding_drift_p75 | embedding_drift_p95 | fanout_p75 | high_drift_high_fanout_items | requests_exposed_to_high_high_fraction | spearman_embedding_vs_layer0_k | spearman_embedding_vs_layer0_v |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 4583 | 4583 | 216758 | 0.000155568 | 0.000432074 | 0.00151546 | 4 | 30352 | 0.999564 | -0.0563693 | -0.159858 |
| v1_to_v2 | 4533 | 4533 | 210547 | 0.000160635 | 0.000434101 | 0.00150025 | 4 | 29208 | 0.999338 | 0.0588094 | 0.0874623 |
| v2_to_v3 | 4585 | 4585 | 205298 | 0.000157714 | 0.000421584 | 0.00142003 | 4 | 28980 | 0.999564 | 0.135712 | 0.134709 |
| v3_to_v4 | 4579 | 4579 | 202985 | 0.000165343 | 0.000461936 | 0.00161548 | 4 | 29308 | 0.999563 | 0.0277363 | -0.0194396 |
| v4_to_v5 | 4771 | 4771 | 206443 | 0.000162423 | 0.000441223 | 0.0015093 | 4 | 29836 | 1 | 0.0526579 | 0.00768209 |

## Request-level correlations

| edge | scope | feature | requests | spearman_vs_reuse_harm | spearman_vs_abs_probability_shift |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | all | candidate_embedding_drift | 4583 | -0.0102827 | -0.0130615 |
| v0_to_v1 | all | prefix_mean_embedding_drift | 4583 | -0.0533794 | -0.111116 |
| v0_to_v1 | all | prefix_max_embedding_drift | 4583 | -0.0857175 | -0.15215 |
| v0_to_v1 | all | prefix_high_drift_fraction | 4583 | -0.0368235 | -0.0735626 |
| v0_to_v1 | all | prefix_high_drift_high_fanout_fraction | 4583 | -0.0624896 | -0.110592 |
| v0_to_v1 | all | candidate_side_geometry_drift | 4583 | -0.0062946 | -0.00676011 |
| v0_to_v1 | all | prefix_side_geometry_drift | 4583 | -0.00975011 | -0.0376194 |
| v0_to_v1 | all | total_geometry_drift | 4583 | -0.0149181 | -0.0323332 |
| v0_to_v1 | all | mean_layer0_k_drift | 4583 | 0.360479 | 0.466991 |
| v0_to_v1 | all | mean_layer0_v_drift | 4583 | 0.494985 | 0.616889 |
| v0_to_v1 | novel_to_prefix | candidate_embedding_drift | 2067 | -0.00293557 | -0.00632971 |
| v0_to_v1 | novel_to_prefix | prefix_mean_embedding_drift | 2067 | -0.044267 | -0.0782808 |
| v0_to_v1 | novel_to_prefix | prefix_max_embedding_drift | 2067 | -0.112978 | -0.14148 |
| v0_to_v1 | novel_to_prefix | prefix_high_drift_fraction | 2067 | -0.0236205 | -0.0462304 |
| v0_to_v1 | novel_to_prefix | prefix_high_drift_high_fanout_fraction | 2067 | -0.0554528 | -0.0792739 |
| v0_to_v1 | novel_to_prefix | candidate_side_geometry_drift | 2067 | 0.00330582 | 0.00126243 |
| v0_to_v1 | novel_to_prefix | prefix_side_geometry_drift | 2067 | -0.0369573 | -0.0576959 |
| v0_to_v1 | novel_to_prefix | total_geometry_drift | 2067 | -0.030536 | -0.0390397 |
| v0_to_v1 | novel_to_prefix | mean_layer0_k_drift | 2067 | 0.3622 | 0.412356 |
| v0_to_v1 | novel_to_prefix | mean_layer0_v_drift | 2067 | 0.530686 | 0.581923 |
| v1_to_v2 | all | candidate_embedding_drift | 4533 | 0.022949 | 0.00149009 |
| v1_to_v2 | all | prefix_mean_embedding_drift | 4533 | 0.0489534 | 0.0223305 |
| v1_to_v2 | all | prefix_max_embedding_drift | 4533 | 0.0194885 | -0.016089 |
| v1_to_v2 | all | prefix_high_drift_fraction | 4533 | 0.0395773 | 0.0269388 |
| v1_to_v2 | all | prefix_high_drift_high_fanout_fraction | 4533 | 0.0245209 | 0.0154788 |
| v1_to_v2 | all | candidate_side_geometry_drift | 4533 | 0.0296723 | 0.0100476 |
| v1_to_v2 | all | prefix_side_geometry_drift | 4533 | 0.0106229 | -0.00315474 |
| v1_to_v2 | all | total_geometry_drift | 4533 | 0.022993 | -0.00153995 |
| v1_to_v2 | all | mean_layer0_k_drift | 4533 | 0.0569975 | 0.087412 |
| v1_to_v2 | all | mean_layer0_v_drift | 4533 | -0.0251779 | 0.0201395 |
| v1_to_v2 | novel_to_prefix | candidate_embedding_drift | 2082 | 0.0173365 | -0.0298932 |
| v1_to_v2 | novel_to_prefix | prefix_mean_embedding_drift | 2082 | 0.0762776 | 0.0201248 |
| v1_to_v2 | novel_to_prefix | prefix_max_embedding_drift | 2082 | 0.0565916 | 0.0193243 |
| v1_to_v2 | novel_to_prefix | prefix_high_drift_fraction | 2082 | 0.0812411 | 0.0366627 |
| v1_to_v2 | novel_to_prefix | prefix_high_drift_high_fanout_fraction | 2082 | 0.0632614 | 0.0227294 |
| v1_to_v2 | novel_to_prefix | candidate_side_geometry_drift | 2082 | 0.0245271 | -0.0203106 |
| v1_to_v2 | novel_to_prefix | prefix_side_geometry_drift | 2082 | 0.0350756 | 0.0195572 |
| v1_to_v2 | novel_to_prefix | total_geometry_drift | 2082 | 0.0474561 | 0.00210908 |
| v1_to_v2 | novel_to_prefix | mean_layer0_k_drift | 2082 | 0.0678955 | 0.111994 |
| v1_to_v2 | novel_to_prefix | mean_layer0_v_drift | 2082 | -0.0056274 | 0.0636855 |
| v2_to_v3 | all | candidate_embedding_drift | 4585 | 0.00969633 | -0.00277549 |
| v2_to_v3 | all | prefix_mean_embedding_drift | 4585 | -0.0250918 | 0.0576602 |
| v2_to_v3 | all | prefix_max_embedding_drift | 4585 | -0.0482534 | 0.108348 |
| v2_to_v3 | all | prefix_high_drift_fraction | 4585 | -0.0168019 | 0.0415552 |
| v2_to_v3 | all | prefix_high_drift_high_fanout_fraction | 4585 | -0.0278089 | 0.0412139 |
| v2_to_v3 | all | candidate_side_geometry_drift | 4585 | 0.00895956 | 0.00817026 |
| v2_to_v3 | all | prefix_side_geometry_drift | 4585 | -0.0136965 | 0.0266522 |
| v2_to_v3 | all | total_geometry_drift | 4585 | -0.0123558 | 0.0199697 |
| v2_to_v3 | all | mean_layer0_k_drift | 4585 | -0.269896 | 0.550494 |
| v2_to_v3 | all | mean_layer0_v_drift | 4585 | -0.270652 | 0.560829 |
| v2_to_v3 | novel_to_prefix | candidate_embedding_drift | 2083 | 0.0173125 | -0.0427467 |
| v2_to_v3 | novel_to_prefix | prefix_mean_embedding_drift | 2083 | -0.0213766 | 0.0620697 |
| v2_to_v3 | novel_to_prefix | prefix_max_embedding_drift | 2083 | -0.0243031 | 0.0898759 |
| v2_to_v3 | novel_to_prefix | prefix_high_drift_fraction | 2083 | -0.00456616 | 0.0344276 |
| v2_to_v3 | novel_to_prefix | prefix_high_drift_high_fanout_fraction | 2083 | -0.0081627 | 0.0357941 |
| v2_to_v3 | novel_to_prefix | candidate_side_geometry_drift | 2083 | 0.0108412 | -0.026966 |
| v2_to_v3 | novel_to_prefix | prefix_side_geometry_drift | 2083 | 0.0110446 | 0.0226445 |
| v2_to_v3 | novel_to_prefix | total_geometry_drift | 2083 | 0.021867 | -0.00939182 |
| v2_to_v3 | novel_to_prefix | mean_layer0_k_drift | 2083 | -0.296201 | 0.549656 |
| v2_to_v3 | novel_to_prefix | mean_layer0_v_drift | 2083 | -0.301454 | 0.560307 |
| v3_to_v4 | all | candidate_embedding_drift | 4579 | 0.0214587 | 0.011171 |
| v3_to_v4 | all | prefix_mean_embedding_drift | 4579 | 0.121295 | 0.1146 |
| v3_to_v4 | all | prefix_max_embedding_drift | 4579 | 0.130743 | 0.147341 |
| v3_to_v4 | all | prefix_high_drift_fraction | 4579 | 0.109237 | 0.111814 |
| v3_to_v4 | all | prefix_high_drift_high_fanout_fraction | 4579 | 0.0757936 | 0.0728332 |
| v3_to_v4 | all | candidate_side_geometry_drift | 4579 | 0.0266575 | 0.0126957 |
| v3_to_v4 | all | prefix_side_geometry_drift | 4579 | 0.0274024 | 0.026759 |
| v3_to_v4 | all | total_geometry_drift | 4579 | 0.0417606 | 0.0329703 |
| v3_to_v4 | all | mean_layer0_k_drift | 4579 | -0.162615 | -0.202881 |
| v3_to_v4 | all | mean_layer0_v_drift | 4579 | -0.219699 | -0.282627 |
| v3_to_v4 | novel_to_prefix | candidate_embedding_drift | 2127 | 0.0453201 | 0.0295112 |
| v3_to_v4 | novel_to_prefix | prefix_mean_embedding_drift | 2127 | 0.0940861 | 0.0962501 |
| v3_to_v4 | novel_to_prefix | prefix_max_embedding_drift | 2127 | 0.12249 | 0.135981 |
| v3_to_v4 | novel_to_prefix | prefix_high_drift_fraction | 2127 | 0.098913 | 0.104415 |
| v3_to_v4 | novel_to_prefix | prefix_high_drift_high_fanout_fraction | 2127 | 0.0584589 | 0.0629581 |
| v3_to_v4 | novel_to_prefix | candidate_side_geometry_drift | 2127 | 0.0430868 | 0.023655 |
| v3_to_v4 | novel_to_prefix | prefix_side_geometry_drift | 2127 | 0.0116152 | 0.00985576 |
| v3_to_v4 | novel_to_prefix | total_geometry_drift | 2127 | 0.037048 | 0.033866 |
| v3_to_v4 | novel_to_prefix | mean_layer0_k_drift | 2127 | -0.207858 | -0.247557 |
| v3_to_v4 | novel_to_prefix | mean_layer0_v_drift | 2127 | -0.26349 | -0.3323 |
| v4_to_v5 | all | candidate_embedding_drift | 4771 | -0.00178606 | 0.00523886 |
| v4_to_v5 | all | prefix_mean_embedding_drift | 4771 | 0.0569875 | -1.3463e-06 |
| v4_to_v5 | all | prefix_max_embedding_drift | 4771 | 0.0431834 | 0.00733574 |
| v4_to_v5 | all | prefix_high_drift_fraction | 4771 | 0.0435397 | -0.00686603 |
| v4_to_v5 | all | prefix_high_drift_high_fanout_fraction | 4771 | 0.0171341 | -0.0138731 |
| v4_to_v5 | all | candidate_side_geometry_drift | 4771 | -0.00508701 | 0.00141586 |
| v4_to_v5 | all | prefix_side_geometry_drift | 4771 | 0.0186961 | -0.0223483 |
| v4_to_v5 | all | total_geometry_drift | 4771 | 0.0100428 | -0.0228539 |
| v4_to_v5 | all | mean_layer0_k_drift | 4771 | 0.0596194 | 0.207568 |
| v4_to_v5 | all | mean_layer0_v_drift | 4771 | 0.0366662 | 0.175793 |
| v4_to_v5 | novel_to_prefix | candidate_embedding_drift | 2321 | -0.00134658 | -0.013234 |
| v4_to_v5 | novel_to_prefix | prefix_mean_embedding_drift | 2321 | 0.0817417 | -0.0198021 |
| v4_to_v5 | novel_to_prefix | prefix_max_embedding_drift | 2321 | 0.0727753 | 0.0251039 |
| v4_to_v5 | novel_to_prefix | prefix_high_drift_fraction | 2321 | 0.0697139 | -0.0279018 |
| v4_to_v5 | novel_to_prefix | prefix_high_drift_high_fanout_fraction | 2321 | 0.047012 | -0.0105346 |
| v4_to_v5 | novel_to_prefix | candidate_side_geometry_drift | 2321 | -0.00543116 | -0.0199771 |
| v4_to_v5 | novel_to_prefix | prefix_side_geometry_drift | 2321 | 0.0163203 | -0.0175144 |
| v4_to_v5 | novel_to_prefix | total_geometry_drift | 2321 | -0.000670244 | -0.0356015 |
| v4_to_v5 | novel_to_prefix | mean_layer0_k_drift | 2321 | 0.0654524 | 0.252779 |
| v4_to_v5 | novel_to_prefix | mean_layer0_v_drift | 2321 | 0.0161879 | 0.209523 |

## Parameter groups

| edge | parameter_group | parameters | relative_l2_drift |
| --- | --- | --- | --- |
| v0_to_v1 | item_embedding | 100087680 | 0.0173732 |
| v0_to_v1 | behavior_embedding | 640 | 0.000470328 |
| v0_to_v1 | temporal_encoder | 4096 | 0.0103279 |
| v0_to_v1 | input_projection | 16384 | 0.0191249 |
| v0_to_v1 | transformer_blocks | 328192 | 0.0161513 |
| v0_to_v1 | query_encoder | 512 | 0.000645816 |
| v0_to_v1 | readout_and_norm | 257 | 0.00153123 |
| v1_to_v2 | item_embedding | 100087680 | 0.0170414 |
| v1_to_v2 | behavior_embedding | 640 | 0.0004817 |
| v1_to_v2 | temporal_encoder | 4096 | 0.00750514 |
| v1_to_v2 | input_projection | 16384 | 0.0120163 |
| v1_to_v2 | transformer_blocks | 328192 | 0.012097 |
| v1_to_v2 | query_encoder | 512 | 0.000287578 |
| v1_to_v2 | readout_and_norm | 257 | 0.000576645 |
| v2_to_v3 | item_embedding | 100087680 | 0.0165084 |
| v2_to_v3 | behavior_embedding | 640 | 0.000295644 |
| v2_to_v3 | temporal_encoder | 4096 | 0.00666885 |
| v2_to_v3 | input_projection | 16384 | 0.010857 |
| v2_to_v3 | transformer_blocks | 328192 | 0.00999866 |
| v2_to_v3 | query_encoder | 512 | 0.00024989 |
| v2_to_v3 | readout_and_norm | 257 | 0.000819731 |
| v3_to_v4 | item_embedding | 100087680 | 0.0171993 |
| v3_to_v4 | behavior_embedding | 640 | 0.000290255 |
| v3_to_v4 | temporal_encoder | 4096 | 0.0108974 |
| v3_to_v4 | input_projection | 16384 | 0.0123656 |
| v3_to_v4 | transformer_blocks | 328192 | 0.0112935 |
| v3_to_v4 | query_encoder | 512 | 0.000351249 |
| v3_to_v4 | readout_and_norm | 257 | 0.000812767 |
| v4_to_v5 | item_embedding | 100087680 | 0.0168961 |
| v4_to_v5 | behavior_embedding | 640 | 0.000399568 |
| v4_to_v5 | temporal_encoder | 4096 | 0.0073364 |
| v4_to_v5 | input_projection | 16384 | 0.00973621 |
| v4_to_v5 | transformer_blocks | 328192 | 0.00986183 |
| v4_to_v5 | query_encoder | 512 | 0.000273082 |
| v4_to_v5 | readout_and_norm | 257 | 0.000612543 |

Across 23,051 requests, raw candidate/prefix embedding drift and candidate-history embedding geometry have weak, sign-changing associations with Reuse harm. Mean item-embedding drift also has only -0.16 to 0.14 Spearman association with contextual layer-0 K/V drift. Isolated item drift is therefore not a supported request selector.

`top_drift_fanout_items.csv` is limited to 100 items per edge; the full expanded item table is intentionally not retained. Diagnostic correlations describe an origin candidate and do not authorize an embedding-aware scheduler.
