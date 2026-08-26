# Insight first pass: dilution and benefit/harm overlap

Scope: Yambda-500M Small, seed 17, D14/E14. This is descriptive discovery over existing sealed raw, not a new training result or a causal History Utility test.

## Release benefit and Reuse harm

| edge | requests | users | mean_release_benefit | mean_reuse_harm | release_winner_fraction | reuse_harmed_fraction | positive_harm_on_release_winners_fraction | positive_harm_concentration_lift | spearman_G_H |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 43186 | 4583 | 0.00426179 | 0.000502172 | 0.890682 | 0.795837 | 0.978094 | 1.09814 | 0.495035 |
| v1_to_v2 | 41655 | 4533 | 0.00148908 | 0.000273775 | 0.647941 | 0.720538 | 0.860059 | 1.32737 | 0.436451 |
| v2_to_v3 | 43092 | 4585 | 0.000539154 | 0.000195394 | 0.287223 | 0.318899 | 0.971514 | 3.38244 | 0.606474 |
| v3_to_v4 | 43945 | 4579 | -0.000383372 | -8.04342e-05 | 0.738833 | 0.831153 | 0.912376 | 1.23489 | 0.494595 |
| v4_to_v5 | 45706 | 4771 | 5.53251e-05 | 4.19225e-05 | 0.153087 | 0.523454 | 0.546398 | 3.5692 | 0.32253 |
| pooled_descriptive | 217584 | 7103 | 0.00117192 | 0.000183342 | 0.539088 | 0.63688 | 0.895199 | 1.66058 | 0.596752 |

Definitions: `G = loss(Parent Full) - loss(Current Full)`; `H = loss(Reuse) - loss(Current Exact Rolling)`. Concentration lift compares the share of positive harm on release winners with the winners' request share.

All five edges have positive G/H rank association (0.323 to 0.606) and positive-harm concentration lift above one (1.10x to 3.57x). This supports benefit/harm overlap as a discovery signal; it is not yet a causal History Utility result.

## Dilution outputs

The CSV files report paired mean Reuse harm and output divergence by cutover day, remaining-old-state bucket, and the remaining-state x append-count grid. Inspect the grid before distinguishing eviction from current-version anchoring.

| edge | requests | spearman_remaining_vs_abs_probability_shift | spearman_append_vs_remaining | mean_abs_probability_shift_remaining_gt_075 | mean_abs_probability_shift_remaining_zero |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 43186 | 0.704918 | -0.956146 | 0.00388079 | 7.60355e-05 |
| v1_to_v2 | 41655 | 0.73908 | -0.974858 | 0.00175904 | 4.16031e-05 |
| v2_to_v3 | 43092 | 0.691822 | -0.976837 | 0.00143871 | 1.94841e-05 |
| v3_to_v4 | 43945 | 0.733039 | -0.97517 | 0.00462707 | 0.000129139 |
| v4_to_v5 | 45706 | 0.573005 | -0.980721 | 0.00130485 | 4.63785e-05 |

Remaining-old fraction is positively associated with Current-Reuse output shift on every edge, but append count and remaining-old fraction are strongly coupled in the observational rolling trace. This table alone cannot distinguish eviction from current-version anchoring; the separate controlled experiment is required.

## Boundary

History Utility is not measured in this pass. History length or stale-state volume must not be presented as a utility proxy.
