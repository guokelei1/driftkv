# Real-exposed signed causal quality: v3_to_v4

Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.

## Absolute quality by candidate width

| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3_to_v4 | 2 | current_exact | 7810 | 1272 | 0.62822129 | 0.43704576 | 0.55191257 |
| v3_to_v4 | 2 | residual_only | 7810 | 1272 | 0.62538498 | 0.43612224 | 0.55191257 |
| v3_to_v4 | 2 | reuse | 7810 | 1272 | 0.62538915 | 0.43612222 | 0.55191257 |
| v3_to_v4 | 2 | shared_only | 7810 | 1272 | 0.62821712 | 0.4370457 | 0.55191257 |
| v3_to_v4 | 4 | current_exact | 3816 | 398 | 0.5800179 | 0.4335412 | 0.50526316 |
| v3_to_v4 | 4 | residual_only | 3816 | 398 | 0.57892993 | 0.43223881 | 0.51578947 |
| v3_to_v4 | 4 | reuse | 3816 | 398 | 0.57893472 | 0.43223893 | 0.51052632 |
| v3_to_v4 | 4 | shared_only | 3816 | 398 | 0.58000618 | 0.43354137 | 0.5 |
| v3_to_v4 | 8 | current_exact | 1928 | 133 | 0.50178309 | 0.50264821 | 0.41401274 |
| v3_to_v4 | 8 | residual_only | 1928 | 133 | 0.50465497 | 0.49959834 | 0.41401274 |
| v3_to_v4 | 8 | reuse | 1928 | 133 | 0.50464569 | 0.49959874 | 0.41401274 |
| v3_to_v4 | 8 | shared_only | 1928 | 133 | 0.50177938 | 0.50264873 | 0.41401274 |
| v3_to_v4 | 16 | current_exact | 1056 | 44 | 0.51096376 | 0.69057839 | 0.53614458 |
| v3_to_v4 | 16 | residual_only | 1056 | 44 | 0.51202356 | 0.68401997 | 0.53614458 |
| v3_to_v4 | 16 | reuse | 1056 | 44 | 0.51202825 | 0.68401962 | 0.53614458 |
| v3_to_v4 | 16 | shared_only | 1056 | 44 | 0.51097783 | 0.69057824 | 0.53614458 |

## Paired fidelity and log-loss delta to Current Exact

| edge | width | path | mean_abs_logit_gap_to_exact | event_path_minus_exact_log_loss | user_equal_path_minus_exact_log_loss |
| --- | --- | --- | --- | --- | --- |
| v3_to_v4 | 2 | reuse | 0.030361727 | -0.00092353905 | -0.00099733213 |
| v3_to_v4 | 2 | shared_only | 4.563965e-05 | -5.7712967e-08 | 1.7468285e-07 |
| v3_to_v4 | 2 | residual_only | 0.030360772 | -0.00092351832 | -0.0009975509 |
| v3_to_v4 | 4 | reuse | 0.031267884 | -0.0013022696 | -0.00087989536 |
| v3_to_v4 | 4 | shared_only | 5.560223e-05 | 1.7508662e-07 | -1.136863e-07 |
| v3_to_v4 | 4 | residual_only | 0.031266383 | -0.0013023888 | -0.00087979005 |
| v3_to_v4 | 8 | reuse | 0.032427719 | -0.003049469 | -0.0028799402 |
| v3_to_v4 | 8 | shared_only | 6.0771944e-05 | 5.228919e-07 | 7.0347062e-07 |
| v3_to_v4 | 8 | residual_only | 0.032426193 | -0.0030498711 | -0.0028805324 |
| v3_to_v4 | 16 | reuse | 0.027058327 | -0.0065587714 | -0.0054791012 |
| v3_to_v4 | 16 | shared_only | 4.9400172e-05 | -1.5002732e-07 | -1.5726605e-07 |
| v3_to_v4 | 16 | residual_only | 0.027054993 | -0.0065584235 | -0.0054787261 |

The shared/residual paths are oracle causal interventions, not executable migration actions.
