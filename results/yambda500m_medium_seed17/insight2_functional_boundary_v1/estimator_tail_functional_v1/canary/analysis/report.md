# Medium Tail-128 functional estimator canary

This is an executable, label-free estimator canary. It transiently performs dependency-closed Current Tail-128 replay against the Parent prefix, discards that mixed cache, and persists only a 1,152-scalar S4 sidecar. Current Exact is evaluation-only.

| edge | P1 | P2 | P4 |
| --- | ---: | ---: | ---: |
| v0_to_v1 | -0.0195 | -0.0239 | -0.0214 |
| v1_to_v2 | 0.0654 | 0.0649 | 0.0644 |
| v2_to_v3 | -0.3503 | -0.3458 | -0.3518 |
| v3_to_v4 | -0.1982 | -0.1900 | -0.1912 |
| v4_to_v5 | 0.0647 | 0.0622 | 0.0612 |

| probes | Exact-All compute | edge-equal recovery | min edge | positive edges | cosine | norm ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.2842% | -0.0876 | -0.3503 | 2/5 | 0.8834 | 0.1219 |
| 2 | 18.5303% | -0.0865 | -0.3458 | 2/5 | 0.8832 | 0.1217 |
| 4 | 19.0224% | -0.0878 | -0.3518 | 2/5 | 0.8834 | 0.1219 |

## Adjudication

- P4 uses 19.0224% of Exact-All and recovers -0.0878 edge-equal probability gap; 2/5 edges are positive.
- Focused family gate: FAIL.
- The preregistered stop rule retires this family. Width, positions, probes and scale are not changed, and no 512-user run is launched.
- The result does not authorize serving promotion, qualification labels or a new predictor.
