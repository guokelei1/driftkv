# History Utility x State Staleness probe

Scope: first request for the first 256 active UIDs on each of five D14/E14 edges. No training.

| edge | utility | requests | positive_utility_fraction | mean_utility | spearman_utility_vs_harm | positive_harm_concentration_lift |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | utility_old_beyond_32 | 256 | 0.546875 | 0.00910131 | 0.103611 | 1.02704 |
| v0_to_v1 | utility_old_beyond_128 | 256 | 0.476562 | 0.000391031 | 0.152384 | 1.01225 |
| v1_to_v2 | utility_old_beyond_32 | 256 | 0.554688 | 0.00205172 | 0.0537616 | 0.950089 |
| v1_to_v2 | utility_old_beyond_128 | 256 | 0.472656 | -0.00395051 | -0.00681975 | 0.903738 |
| v2_to_v3 | utility_old_beyond_32 | 256 | 0.582031 | 0.00844075 | 0.0507289 | 0.987303 |
| v2_to_v3 | utility_old_beyond_128 | 256 | 0.507812 | 0.00046955 | 0.00371939 | 1.06389 |
| v3_to_v4 | utility_old_beyond_32 | 256 | 0.621094 | 0.00794443 | 0.123016 | 1.06926 |
| v3_to_v4 | utility_old_beyond_128 | 256 | 0.582031 | 0.00587464 | 0.053991 | 1.04961 |
| v4_to_v5 | utility_old_beyond_32 | 256 | 0.546875 | 0.00567196 | 0.0899427 | 1.26043 |
| v4_to_v5 | utility_old_beyond_128 | 256 | 0.546875 | 0.00113072 | 0.0593597 | 1.2262 |

Utility is the request log-loss increase from truncating Current Full history to recent-32 or recent-128. Staleness is the existing paired Reuse harm. This is a small causal history-ablation probe; it does not yet localize stale K/V to the ablated region.

The utility/harm association is weak and not stable enough to support selective migration: correlations are close to zero, and positive-harm concentration lift ranges around random rather than separating a consistently high-value cohort.

Recommendation-semantic correlations are in `recommendation_semantics.csv`. Current persistent history contains listen tokens with organic/non-organic behavior; likes/dislikes are request labels, not persistent action tokens in this implementation.

Repeat, diversity, organic fraction, and recent/old overlap change direction across edges. This probe therefore does not support a frozen recommendation-semantic risk rule.
