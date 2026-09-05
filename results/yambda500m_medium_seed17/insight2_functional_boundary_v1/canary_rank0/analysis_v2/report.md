# Medium Insight 2 rank-0 focused canary

This is a 32-user instrumentation and representation canary, not Design 1 qualification.
The correction is estimated on 32 anchor candidates and intervened only on 32 held-out candidates.
No label was read. `kv_prefix_contribution` is position-summed before injection and cannot qualify as a token-local action.

## Anchor-to-heldout rank-0 frontier

| stage | probability recovery | logit recovery | edges >=80% | edges >=90% | min edge | max edge | FP32 values/user | canary shape gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S3_positionwise_response_summed_for_injection | 0.9293 | 0.9291 | 5 | 4 | 0.8650 | 0.9772 | 1152 | FAIL |
| S4_aggregated_context | 0.9293 | 0.9291 | 5 | 4 | 0.8650 | 0.9772 | 1152 | PASS |
| S7_final_representation | 0.9165 | 0.9158 | 5 | 4 | 0.8856 | 0.9687 | 192 | PASS |
| S5_transformed_update | 0.8229 | 0.8234 | 4 | 0 | 0.6436 | 0.8924 | 1152 | PASS |
| S6_post_block_residual | 0.8229 | 0.8234 | 4 | 0 | 0.6436 | 0.8924 | 1152 | PASS |

## Per-edge held-out probability recovery

| stage | v0_to_v1 | v1_to_v2 | v2_to_v3 | v3_to_v4 | v4_to_v5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3_positionwise_response_summed_for_injection | 0.8650 | 0.9772 | 0.9095 | 0.9726 | 0.9220 |
| S4_aggregated_context | 0.8650 | 0.9772 | 0.9095 | 0.9726 | 0.9220 |
| S5_transformed_update | 0.6436 | 0.8555 | 0.8706 | 0.8924 | 0.8526 |
| S6_post_block_residual | 0.6436 | 0.8555 | 0.8706 | 0.8924 | 0.8526 |
| S7_final_representation | 0.8856 | 0.9687 | 0.9255 | 0.9013 | 0.9012 |

Passing this canary only unlocks low-rank query-conditioned instrumentation and a resource estimate.
It does not establish temporal persistence, an executable estimator, the 0%–20% cost gate, or task-label quality.
