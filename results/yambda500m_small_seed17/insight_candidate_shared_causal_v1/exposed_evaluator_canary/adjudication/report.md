# Real-exposed signed causal quality: v0_to_v1

Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.

## Absolute quality by candidate width

| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 2 | current_exact | 14 | 4 | N/A | 0.09898305 | N/A |
| v0_to_v1 | 2 | residual_only | 14 | 4 | N/A | 0.10441763 | N/A |
| v0_to_v1 | 2 | reuse | 14 | 4 | N/A | 0.10441704 | N/A |
| v0_to_v1 | 2 | shared_only | 14 | 4 | N/A | 0.098982479 | N/A |
| v0_to_v1 | 4 | current_exact | 4 | 1 | N/A | 0.10513701 | N/A |
| v0_to_v1 | 4 | residual_only | 4 | 1 | N/A | 0.10876408 | N/A |
| v0_to_v1 | 4 | reuse | 4 | 1 | N/A | 0.10876437 | N/A |
| v0_to_v1 | 4 | shared_only | 4 | 1 | N/A | 0.10513741 | N/A |

## Paired fidelity and log-loss delta to Current Exact

| edge | width | path | mean_abs_logit_gap_to_exact | event_path_minus_exact_log_loss | user_equal_path_minus_exact_log_loss |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 2 | reuse | 0.056966092 | 0.0054339879 | 0.0044306893 |
| v0_to_v1 | 2 | shared_only | 0.000108506 | -5.7150749e-07 | -9.8709498e-07 |
| v0_to_v1 | 2 | residual_only | 0.0569696 | 0.005434575 | 0.0044316633 |
| v0_to_v1 | 4 | reuse | 0.035765827 | 0.0036273658 | 0.0036273658 |
| v0_to_v1 | 4 | shared_only | 2.8610229e-05 | 4.0487673e-07 | 4.0487673e-07 |
| v0_to_v1 | 4 | residual_only | 0.035762966 | 0.003627074 | 0.003627074 |

The shared/residual paths are oracle causal interventions, not executable migration actions.
