# Real-exposed signed causal quality: v4_to_v5

Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.

## Absolute quality by candidate width

| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v4_to_v5 | 2 | current_exact | 7056 | 1274 | 0.62709882 | 0.42499468 | 0.46540881 |
| v4_to_v5 | 2 | residual_only | 7056 | 1274 | 0.62704843 | 0.42488003 | 0.4591195 |
| v4_to_v5 | 2 | reuse | 7056 | 1274 | 0.6270489 | 0.42488004 | 0.46540881 |
| v4_to_v5 | 2 | shared_only | 7056 | 1274 | 0.62709921 | 0.42499468 | 0.46540881 |
| v4_to_v5 | 4 | current_exact | 3364 | 395 | 0.57948735 | 0.46428507 | 0.50810811 |
| v4_to_v5 | 4 | residual_only | 3364 | 395 | 0.58024093 | 0.46402717 | 0.4972973 |
| v4_to_v5 | 4 | reuse | 3364 | 395 | 0.5802403 | 0.46402735 | 0.5027027 |
| v4_to_v5 | 4 | shared_only | 3364 | 395 | 0.57948989 | 0.46428525 | 0.50810811 |
| v4_to_v5 | 8 | current_exact | 1824 | 125 | 0.48011907 | 0.50163535 | 0.47867299 |
| v4_to_v5 | 8 | residual_only | 1824 | 125 | 0.47842142 | 0.50161396 | 0.46919431 |
| v4_to_v5 | 8 | reuse | 1824 | 125 | 0.47842142 | 0.50161408 | 0.47393365 |
| v4_to_v5 | 8 | shared_only | 1824 | 125 | 0.48012535 | 0.50163549 | 0.47867299 |
| v4_to_v5 | 16 | current_exact | 1200 | 41 | 0.28492801 | 0.58651409 | 0.63576159 |
| v4_to_v5 | 16 | residual_only | 1200 | 41 | 0.28317228 | 0.58650838 | 0.62251656 |
| v4_to_v5 | 16 | reuse | 1200 | 41 | 0.28319909 | 0.58650822 | 0.62913907 |
| v4_to_v5 | 16 | shared_only | 1200 | 41 | 0.28491014 | 0.58651394 | 0.63576159 |

## Paired fidelity and log-loss delta to Current Exact

| edge | width | path | mean_abs_logit_gap_to_exact | event_path_minus_exact_log_loss | user_equal_path_minus_exact_log_loss |
| --- | --- | --- | --- | --- | --- |
| v4_to_v5 | 2 | reuse | 0.0077093832 | -0.00011463644 | -0.00010327201 |
| v4_to_v5 | 2 | shared_only | 2.7173146e-05 | -3.2745291e-09 | -1.3893379e-07 |
| v4_to_v5 | 2 | residual_only | 0.0077094497 | -0.00011464773 | -0.00010318695 |
| v4_to_v5 | 4 | reuse | 0.0072270535 | -0.00025772358 | 1.5329518e-05 |
| v4_to_v5 | 4 | shared_only | 3.1152924e-05 | 1.7768628e-07 | 4.0696516e-07 |
| v4_to_v5 | 4 | residual_only | 0.0072268138 | -0.00025789948 | 1.4925278e-05 |
| v4_to_v5 | 8 | reuse | 0.0066802391 | -2.1274946e-05 | -3.557611e-05 |
| v4_to_v5 | 8 | shared_only | 3.5751807e-05 | 1.359886e-07 | 1.820978e-07 |
| v4_to_v5 | 8 | residual_only | 0.0066796388 | -2.1389956e-05 | -3.5740067e-05 |
| v4_to_v5 | 16 | reuse | 0.0062807036 | -5.8624975e-06 | -0.00011991424 |
| v4_to_v5 | 16 | shared_only | 3.1981965e-05 | -1.49694e-07 | -1.8355036e-07 |
| v4_to_v5 | 16 | residual_only | 0.0062804568 | -5.7116207e-06 | -0.00011972906 |

The shared/residual paths are oracle causal interventions, not executable migration actions.
