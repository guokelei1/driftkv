# Real-exposed signed causal quality: v1_to_v2

Raw rolling scores were sealed before labels were joined. Candidate groups are real same-UID, same-timestamp requests; no sampled negatives are introduced.

## Absolute quality by candidate width

| edge | width | path | requests | users | ROC_AUC | log_loss | within_bank_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1_to_v2 | 2 | current_exact | 7188 | 1284 | 0.68753411 | 0.45162552 | 0.51748252 |
| v1_to_v2 | 2 | residual_only | 7188 | 1284 | 0.68655672 | 0.45142985 | 0.51748252 |
| v1_to_v2 | 2 | reuse | 7188 | 1284 | 0.6865574 | 0.4514298 | 0.51048951 |
| v1_to_v2 | 2 | shared_only | 7188 | 1284 | 0.68753336 | 0.4516255 | 0.51748252 |
| v1_to_v2 | 4 | current_exact | 3136 | 361 | 0.66465126 | 0.42448752 | 0.5136612 |
| v1_to_v2 | 4 | residual_only | 3136 | 361 | 0.66344899 | 0.4243534 | 0.5136612 |
| v1_to_v2 | 4 | reuse | 3136 | 361 | 0.66344202 | 0.42435347 | 0.50273224 |
| v1_to_v2 | 4 | shared_only | 3136 | 361 | 0.66463576 | 0.42448768 | 0.5136612 |
| v1_to_v2 | 8 | current_exact | 1832 | 125 | 0.62431425 | 0.43115802 | 0.46062992 |
| v1_to_v2 | 8 | residual_only | 1832 | 125 | 0.62378728 | 0.43088082 | 0.46062992 |
| v1_to_v2 | 8 | reuse | 1832 | 125 | 0.62378843 | 0.43088044 | 0.46062992 |
| v1_to_v2 | 8 | shared_only | 1832 | 125 | 0.62432115 | 0.43115765 | 0.46062992 |
| v1_to_v2 | 16 | current_exact | 864 | 34 | 0.59743223 | 0.39618835 | 0.50446429 |
| v1_to_v2 | 16 | residual_only | 864 | 34 | 0.59785866 | 0.39622011 | 0.51785714 |
| v1_to_v2 | 16 | reuse | 864 | 34 | 0.59795086 | 0.39621944 | 0.52232143 |
| v1_to_v2 | 16 | shared_only | 864 | 34 | 0.59751291 | 0.39618759 | 0.50892857 |

## Paired fidelity and log-loss delta to Current Exact

| edge | width | path | mean_abs_logit_gap_to_exact | event_path_minus_exact_log_loss | user_equal_path_minus_exact_log_loss |
| --- | --- | --- | --- | --- | --- |
| v1_to_v2 | 2 | reuse | 0.011263071 | -0.00019571772 | -0.00024961111 |
| v1_to_v2 | 2 | shared_only | 5.546121e-05 | -1.8240724e-08 | 2.5980666e-07 |
| v1_to_v2 | 2 | residual_only | 0.011261838 | -0.00019566525 | -0.00024984125 |
| v1_to_v2 | 4 | reuse | 0.0090899621 | -0.00013404567 | 1.5944422e-05 |
| v1_to_v2 | 4 | shared_only | 6.3529428e-05 | 1.5888581e-07 | 4.7247611e-07 |
| v1_to_v2 | 4 | residual_only | 0.0090871073 | -0.00013412048 | 1.5535383e-05 |
| v1_to_v2 | 8 | reuse | 0.0077285755 | -0.00027758108 | -0.00024275352 |
| v1_to_v2 | 8 | shared_only | 6.8732224e-05 | -3.6449585e-07 | -6.365721e-07 |
| v1_to_v2 | 8 | residual_only | 0.0077240995 | -0.00027719915 | -0.0002421338 |
| v1_to_v2 | 16 | reuse | 0.0056567098 | 3.1092266e-05 | 0.00020377026 |
| v1_to_v2 | 16 | shared_only | 7.3245416e-05 | -7.6039284e-07 | -1.1915333e-06 |
| v1_to_v2 | 16 | residual_only | 0.0056449127 | 3.176105e-05 | 0.00020480266 |

The shared/residual paths are oracle causal interventions, not executable migration actions.
