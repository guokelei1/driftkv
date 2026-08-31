# Candidate-shared signed causal canary

This canary uses signed, per-head HSTU prefix reads without candidate-wise normalization. The shared/residual paths are oracle diagnostic interventions, not executable cache actions.

Progression gate: **PASS**.

## Controlled width-64 score-gap intervention

| edge | path | mean_probability_gap | mean_gap_recovery |
| --- | --- | --- | --- |
| v0_to_v1 | full_delta | 7.9429398e-09 | 0.9999967 |
| v0_to_v1 | residual_only | 0.0048331993 | 0.00086280686 |
| v0_to_v1 | reuse | 0.0048324748 | 0 |
| v0_to_v1 | shared_only | 2.0266765e-05 | 0.99157868 |
| v1_to_v2 | full_delta | 7.9814345e-09 | 0.99998732 |
| v1_to_v2 | residual_only | 0.0017079403 | 0.0073499396 |
| v1_to_v2 | reuse | 0.0017097576 | 0 |
| v1_to_v2 | shared_only | 2.6420687e-05 | 0.96598228 |
| v2_to_v3 | full_delta | 8.0562507e-09 | 0.99998436 |
| v2_to_v3 | residual_only | 0.0014364175 | 0.0093158885 |
| v2_to_v3 | reuse | 0.0014367486 | 0 |
| v2_to_v3 | shared_only | 2.8988187e-05 | 0.96154555 |
| v3_to_v4 | full_delta | 7.8637774e-09 | 0.99999658 |
| v3_to_v4 | residual_only | 0.0047024515 | 0.00046051129 |
| v3_to_v4 | reuse | 0.004704316 | 0 |
| v3_to_v4 | shared_only | 1.7086925e-05 | 0.99350095 |
| v4_to_v5 | full_delta | 7.8914066e-09 | 0.99997981 |
| v4_to_v5 | residual_only | 0.0011371055 | 0.0022093475 |
| v4_to_v5 | reuse | 0.0011376625 | 0 |
| v4_to_v5 | shared_only | 1.7206409e-05 | 0.94180483 |

## Signed head decomposition at the largest bank width

| bank_source | edge | signed_shared_energy_fraction | orthogonality_error |
| --- | --- | --- | --- |
| controlled | v0_to_v1 | 0.99813535 | 6.0640552e-08 |
| controlled | v1_to_v2 | 0.99790919 | 5.3368645e-08 |
| controlled | v2_to_v3 | 0.99792424 | 4.5906567e-08 |
| controlled | v3_to_v4 | 0.9978221 | 5.0858525e-08 |
| controlled | v4_to_v5 | 0.99770339 | 5.8494454e-08 |

Maximum native trace error: 9.5367432e-07.
Maximum full-delta reconstruction error: 9.5367432e-07.
No label was read during the canary.
