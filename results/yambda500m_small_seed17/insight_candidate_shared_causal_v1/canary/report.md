# Candidate-shared signed causal canary

This canary uses signed, per-head HSTU prefix reads without candidate-wise normalization. The shared/residual paths are oracle diagnostic interventions, not executable cache actions.

Progression gate: **PASS**.

## Controlled width-64 score-gap intervention

| edge | path | mean_probability_gap | mean_gap_recovery |
| --- | --- | --- | --- |
| v0_to_v1 | full_delta | 6.4610504e-09 | 0.99999644 |
| v0_to_v1 | residual_only | 0.0041607804 | 3.3499673e-05 |
| v0_to_v1 | reuse | 0.0041606619 | 0 |
| v0_to_v1 | shared_only | 1.7335959e-05 | 0.99085393 |
| v1_to_v2 | full_delta | 7.1304385e-09 | 0.99999134 |
| v1_to_v2 | residual_only | 0.0016004479 | 0.015550325 |
| v1_to_v2 | reuse | 0.0016040904 | 0 |
| v1_to_v2 | shared_only | 2.1708052e-05 | 0.96693643 |
| v2_to_v3 | full_delta | 6.868504e-09 | 0.99998523 |
| v2_to_v3 | residual_only | 0.0012764132 | 0.010829736 |
| v2_to_v3 | reuse | 0.0012775724 | 0 |
| v2_to_v3 | shared_only | 2.7073547e-05 | 0.95346194 |
| v3_to_v4 | full_delta | 6.6647772e-09 | 0.99999782 |
| v3_to_v4 | residual_only | 0.0054183018 | 0.00060338154 |
| v3_to_v4 | reuse | 0.0054204522 | 0 |
| v3_to_v4 | shared_only | 1.4927413e-05 | 0.99621869 |
| v4_to_v5 | full_delta | 7.4505806e-09 | 0.99998864 |
| v4_to_v5 | residual_only | 0.0010573542 | 0.00060662627 |
| v4_to_v5 | reuse | 0.0010573455 | 0 |
| v4_to_v5 | shared_only | 2.171204e-05 | 0.97207335 |

## Signed head decomposition at the largest bank width

| bank_source | edge | signed_shared_energy_fraction | orthogonality_error |
| --- | --- | --- | --- |
| controlled | v0_to_v1 | 0.99806128 | 4.0518255e-08 |
| controlled | v1_to_v2 | 0.99881685 | 3.6040927e-08 |
| controlled | v2_to_v3 | 0.99814287 | 4.0862652e-08 |
| controlled | v3_to_v4 | 0.99793519 | 3.8451841e-08 |
| controlled | v4_to_v5 | 0.99789492 | 3.5934182e-08 |
| real_exposed_canary | v0_to_v1 | 0.99984439 | 1.9040279e-08 |
| real_exposed_canary | v1_to_v2 | 0.74945486 | 2.0301012e-08 |
| real_exposed_canary | v2_to_v3 | 0.99932719 | 1.9143929e-08 |
| real_exposed_canary | v3_to_v4 | 0.99966887 | 2.3475824e-08 |
| real_exposed_canary | v4_to_v5 | 0.9996464 | 1.8141652e-08 |

Maximum native trace error: 4.7683716e-07.
Maximum full-delta reconstruction error: 4.7683716e-07.
No label was read during the canary.
