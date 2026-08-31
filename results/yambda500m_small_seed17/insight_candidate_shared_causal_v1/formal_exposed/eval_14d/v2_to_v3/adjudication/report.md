# Real-exposed signed causal quality: v2_to_v3

Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.

## Absolute quality by candidate width

| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_to_v3 | 2 | current_exact | 7680 | 1295 | 0.65419273 | 0.40965942 | 0.4965035 |
| v2_to_v3 | 2 | residual_only | 7680 | 1295 | 0.65148917 | 0.41027643 | 0.4965035 |
| v2_to_v3 | 2 | reuse | 7680 | 1295 | 0.65149052 | 0.41027671 | 0.4965035 |
| v2_to_v3 | 2 | shared_only | 7680 | 1295 | 0.65419011 | 0.40965972 | 0.48951049 |
| v2_to_v3 | 4 | current_exact | 3456 | 377 | 0.65803957 | 0.40431249 | 0.53142857 |
| v2_to_v3 | 4 | residual_only | 3456 | 377 | 0.65544828 | 0.40494217 | 0.52571429 |
| v2_to_v3 | 4 | reuse | 3456 | 377 | 0.6554439 | 0.40494235 | 0.52571429 |
| v2_to_v3 | 4 | shared_only | 3456 | 377 | 0.65802611 | 0.40431268 | 0.52 |
| v2_to_v3 | 8 | current_exact | 1944 | 134 | 0.59097364 | 0.53104187 | 0.46351931 |
| v2_to_v3 | 8 | residual_only | 1944 | 134 | 0.58701262 | 0.5323209 | 0.46351931 |
| v2_to_v3 | 8 | reuse | 1944 | 134 | 0.58701262 | 0.53232133 | 0.46351931 |
| v2_to_v3 | 8 | shared_only | 1944 | 134 | 0.59095959 | 0.53104229 | 0.46351931 |
| v2_to_v3 | 16 | current_exact | 1152 | 43 | 0.5390524 | 0.62423405 | 0.36430318 |
| v2_to_v3 | 16 | residual_only | 1152 | 43 | 0.53372459 | 0.62590883 | 0.36430318 |
| v2_to_v3 | 16 | reuse | 1152 | 43 | 0.53369512 | 0.62590963 | 0.36185819 |
| v2_to_v3 | 16 | shared_only | 1152 | 43 | 0.53904609 | 0.6242348 | 0.36430318 |

## Paired fidelity and log-loss delta to Current Exact

| edge | width | path | mean_abs_logit_gap_to_exact | event_path_minus_exact_log_loss | user_equal_path_minus_exact_log_loss |
| --- | --- | --- | --- | --- | --- |
| v2_to_v3 | 2 | reuse | 0.008094625 | 0.00061729004 | 0.00041119609 |
| v2_to_v3 | 2 | shared_only | 5.537377e-05 | 3.0160836e-07 | 6.686951e-07 |
| v2_to_v3 | 2 | residual_only | 0.0080947148 | 0.00061701155 | 0.00041058242 |
| v2_to_v3 | 4 | reuse | 0.0082892686 | 0.00062985447 | 0.00059238522 |
| v2_to_v3 | 4 | shared_only | 6.9718256e-05 | 1.9042579e-07 | 5.4992243e-07 |
| v2_to_v3 | 4 | residual_only | 0.008288861 | 0.00062967807 | 0.00059184939 |
| v2_to_v3 | 8 | reuse | 0.0080361246 | 0.0012794572 | 0.0004565657 |
| v2_to_v3 | 8 | shared_only | 7.3405818e-05 | 4.240105e-07 | 5.4306292e-07 |
| v2_to_v3 | 8 | residual_only | 0.0080348763 | 0.0012790268 | 0.00045603786 |
| v2_to_v3 | 16 | reuse | 0.0088962301 | 0.0016755848 | 0.0010145097 |
| v2_to_v3 | 16 | shared_only | 7.8286251e-05 | 7.5445997e-07 | 3.0995552e-07 |
| v2_to_v3 | 16 | residual_only | 0.0088956555 | 0.0016747763 | 0.0010141945 |

The shared/residual paths are oracle causal interventions, not executable migration actions.
