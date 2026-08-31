# Real-exposed signed causal quality: v0_to_v1

Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.

## Absolute quality by candidate width

| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 2 | current_exact | 7666 | 1300 | 0.69667492 | 0.41379902 | 0.46428571 |
| v0_to_v1 | 2 | residual_only | 7666 | 1300 | 0.69300771 | 0.41369644 | 0.48214286 |
| v0_to_v1 | 2 | reuse | 7666 | 1300 | 0.69300751 | 0.41369652 | 0.48214286 |
| v0_to_v1 | 2 | shared_only | 7666 | 1300 | 0.69667735 | 0.41379916 | 0.46428571 |
| v0_to_v1 | 4 | current_exact | 3136 | 381 | 0.67126692 | 0.39009531 | 0.39306358 |
| v0_to_v1 | 4 | residual_only | 3136 | 381 | 0.66756544 | 0.39013778 | 0.38728324 |
| v0_to_v1 | 4 | reuse | 3136 | 381 | 0.66756629 | 0.39013819 | 0.38728324 |
| v0_to_v1 | 4 | shared_only | 3136 | 381 | 0.67126947 | 0.39009569 | 0.39306358 |
| v0_to_v1 | 8 | current_exact | 1696 | 124 | 0.61210266 | 0.4612266 | 0.49480969 |
| v0_to_v1 | 8 | residual_only | 1696 | 124 | 0.61332553 | 0.46002642 | 0.49480969 |
| v0_to_v1 | 8 | reuse | 1696 | 124 | 0.61332053 | 0.46002657 | 0.49480969 |
| v0_to_v1 | 8 | shared_only | 1696 | 124 | 0.61208516 | 0.46122668 | 0.49480969 |
| v0_to_v1 | 16 | current_exact | 960 | 44 | 0.70518142 | 0.381741 | 0.54230769 |
| v0_to_v1 | 16 | residual_only | 960 | 44 | 0.70328463 | 0.38129308 | 0.54230769 |
| v0_to_v1 | 16 | reuse | 960 | 44 | 0.70327994 | 0.38129159 | 0.53846154 |
| v0_to_v1 | 16 | shared_only | 960 | 44 | 0.70527532 | 0.38173953 | 0.54615385 |

## Paired fidelity and log-loss delta to Current Exact

| edge | width | path | mean_abs_logit_gap_to_exact | event_path_minus_exact_log_loss | user_equal_path_minus_exact_log_loss |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 2 | reuse | 0.027079809 | -0.00010249586 | -0.00062318366 |
| v0_to_v1 | 2 | shared_only | 5.3235438e-05 | 1.4706149e-07 | 2.4488046e-08 |
| v0_to_v1 | 2 | residual_only | 0.027079385 | -0.00010257193 | -0.00062293516 |
| v0_to_v1 | 4 | reuse | 0.026578031 | 4.2877179e-05 | -0.00019239103 |
| v0_to_v1 | 4 | shared_only | 6.4888057e-05 | 3.7899038e-07 | -4.7880072e-08 |
| v0_to_v1 | 4 | residual_only | 0.026576767 | 4.247307e-05 | -0.00019235517 |
| v0_to_v1 | 8 | reuse | 0.021006011 | -0.0012000318 | -0.0021256668 |
| v0_to_v1 | 8 | shared_only | 5.9134485e-05 | 8.4125131e-08 | 4.4911958e-08 |
| v0_to_v1 | 8 | residual_only | 0.021005633 | -0.0012001785 | -0.0021257332 |
| v0_to_v1 | 16 | reuse | 0.019441244 | -0.00044940992 | -0.00080434964 |
| v0_to_v1 | 16 | shared_only | 6.3604489e-05 | -1.4710608e-06 | -1.1713086e-06 |
| v0_to_v1 | 16 | residual_only | 0.019440564 | -0.0004479217 | -0.00080315714 |

The shared/residual paths are oracle causal interventions, not executable migration actions.
