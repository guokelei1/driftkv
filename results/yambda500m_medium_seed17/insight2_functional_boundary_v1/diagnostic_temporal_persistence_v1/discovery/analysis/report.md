# Medium S4 temporal persistence: discovery

The correction is a Current-Exact cutover oracle. It is frozen once and never refreshed; therefore this report tests the representation boundary, not an executable estimator or a 0--20% action.

## Fixed held-out panel, user-equal within edge

| edge | same_request_S4_oracle | frozen_cutover_S4 | coverage_scaled_frozen | gate_primary_positive |
| --- | --- | --- | --- | --- |
| v0_to_v1 | 0.9125 | -92.1860 | 0.2101 | True |
| v1_to_v2 | 0.9328 | -50.9902 | 0.2606 | True |
| v2_to_v3 | 0.9460 | -6.2568 | 0.5121 | True |
| v3_to_v4 | 0.9450 | -19.0498 | 0.3231 | True |
| v4_to_v5 | 0.9330 | -2.6116 | 0.3865 | True |

- Primary edge-equal recovery: 0.3385; positive edges: 5/5; Gate C: FAIL.
- Same-request S4 oracle remains at 0.9339 recovery.
- Unscaled frozen correction is -34.2189; it is retained as a negative companion rather than clipped.

## Correction drift

| edge | direction_cosine | norm_ratio | relative_l2 |
| --- | --- | --- | --- |
| v0_to_v1 | 0.9689 | 0.7650 | 14.9277 |
| v1_to_v2 | 0.9450 | 0.7528 | 5.3978 |
| v2_to_v3 | 0.9332 | 0.7844 | 7.8684 |
| v3_to_v4 | 0.9384 | 0.7847 | 6.7729 |
| v4_to_v5 | 0.9446 | 0.7616 | 1.9658 |

The edge-equal user-equal direction cosine is 0.9460, while the Current/cutover norm ratio is 0.7697. A stable direction alone is not a persistent offset: its amplitude evolves as Current events enter and Parent positions leave the cache.

## Time buckets for the preregistered coverage-scaled method

| time_bucket | recovery |
| --- | --- |
| [0d,1d) | 0.5204 |
| [1d,3d) | 0.5526 |
| [3d,7d) | 0.3509 |
| [7d,14d) | 0.2108 |

## Append buckets for the preregistered coverage-scaled method

| append_bucket | recovery |
| --- | --- |
| 0 | 0.5965 |
| >512 | -0.1053 |
| [1,8] | 0.6560 |
| [129,512] | 0.2994 |
| [33,128] | 0.5487 |
| [9,32] | 0.5153 |

## Real exposed-item companion

| edge | same_request_S4_oracle | coverage_scaled_frozen |
| --- | --- | --- |
| v0_to_v1 | 0.6805 | -0.9597 |
| v1_to_v2 | 0.8943 | 0.0477 |
| v2_to_v3 | 0.9389 | 0.4583 |
| v3_to_v4 | 0.8617 | -0.7713 |
| v4_to_v5 | 0.8856 | -0.4293 |

## Adjudication

- The S4 boundary continues to be causally sufficient when its correction is re-observed at the current request.
- The cutover direction remains highly aligned, but neither an unscaled offset nor the preregistered linear remaining-coverage decay is sufficient over the complete E14 timeline in this scope.
- No request was filtered for a small Reuse gap. Denominator quantiles and fixed-floor sensitivity are sealed in `summary.json`.
- This oracle result cannot authorize a migration action, estimator, refresh policy or confirmation read.
