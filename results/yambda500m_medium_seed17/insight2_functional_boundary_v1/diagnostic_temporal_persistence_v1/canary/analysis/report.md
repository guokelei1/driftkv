# Medium S4 temporal persistence: canary

The correction is a Current-Exact cutover oracle. It is frozen once and never refreshed; therefore this report tests the representation boundary, not an executable estimator or a 0--20% action.

## Fixed held-out panel, user-equal within edge

| edge | same_request_S4_oracle | frozen_cutover_S4 | coverage_scaled_frozen | gate_primary_positive |
| --- | --- | --- | --- | --- |
| v0_to_v1 | 0.9209 | -150.3363 | 0.0288 | True |
| v1_to_v2 | 0.9675 | -2.5319 | 0.7004 | True |
| v2_to_v3 | 0.9705 | -0.4872 | 0.6795 | True |
| v3_to_v4 | 0.9500 | -89.1635 | 0.2738 | True |
| v4_to_v5 | 0.9147 | -0.8771 | 0.3840 | True |

- Primary edge-equal recovery: 0.4133; positive edges: 5/5; Gate C: FAIL.
- Same-request S4 oracle remains at 0.9447 recovery.
- Unscaled frozen correction is -48.6792; it is retained as a negative companion rather than clipped.

## Correction drift

| edge | direction_cosine | norm_ratio | relative_l2 |
| --- | --- | --- | --- |
| v0_to_v1 | 0.9800 | 0.7606 | 81.9506 |
| v1_to_v2 | 0.9563 | 0.7862 | 2.7922 |
| v2_to_v3 | 0.9258 | 0.7841 | 1.3120 |
| v3_to_v4 | 0.9384 | 0.7972 | 6.5072 |
| v4_to_v5 | 0.9497 | 0.8067 | 0.6811 |

The edge-equal user-equal direction cosine is 0.9501, while the Current/cutover norm ratio is 0.7870. A stable direction alone is not a persistent offset: its amplitude evolves as Current events enter and Parent positions leave the cache.

## Time buckets for the preregistered coverage-scaled method

| time_bucket | recovery |
| --- | --- |
| [0d,1d) | 0.6229 |
| [1d,3d) | 0.6745 |
| [3d,7d) | 0.3383 |
| [7d,14d) | 0.1522 |

## Append buckets for the preregistered coverage-scaled method

| append_bucket | recovery |
| --- | --- |
| 0 | 0.3913 |
| >512 | 0.2187 |
| [1,8] | 0.8611 |
| [129,512] | 0.2663 |
| [33,128] | 0.4905 |
| [9,32] | 0.6126 |

## Real exposed-item companion

| edge | same_request_S4_oracle | coverage_scaled_frozen |
| --- | --- | --- |
| v0_to_v1 | 0.9247 | -0.0670 |
| v1_to_v2 | 0.9472 | 0.6835 |
| v2_to_v3 | 0.9636 | 0.6788 |
| v3_to_v4 | 0.8236 | 0.0614 |
| v4_to_v5 | 0.9343 | 0.3174 |

## Adjudication

- The S4 boundary continues to be causally sufficient when its correction is re-observed at the current request.
- The cutover direction remains highly aligned, but neither an unscaled offset nor the preregistered linear remaining-coverage decay is sufficient over the complete E14 timeline in this scope.
- No request was filtered for a small Reuse gap. Denominator quantiles and fixed-floor sensitivity are sealed in `summary.json`.
- This oracle result cannot authorize a migration action, estimator, refresh policy or confirmation read.
