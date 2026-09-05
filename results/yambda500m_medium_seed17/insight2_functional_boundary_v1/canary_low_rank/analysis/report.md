# Medium Insight 2 query-conditioned low-rank canary

All target corrections are anchor-side Exact oracles. This tests representation capacity, not an executable estimator.

| stage | rank | probability recovery | edges >=80% | edges >=90% | min edge | FP32 values/user | shape gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S4_aggregated_context | 8 | 0.9987 | 5 | 5 | 0.9964 | 20736 | PASS |
| S4_aggregated_context | 4 | 0.9987 | 5 | 5 | 0.9964 | 11520 | PASS |
| S7_final_representation | 8 | 0.9980 | 5 | 5 | 0.9971 | 3456 | PASS |
| S7_final_representation | 4 | 0.9975 | 5 | 5 | 0.9966 | 1920 | PASS |
| S4_aggregated_context | 2 | 0.9969 | 5 | 5 | 0.9929 | 6912 | PASS |
| S4_aggregated_context | 1 | 0.9925 | 5 | 5 | 0.9818 | 4608 | PASS |
| S7_final_representation | 2 | 0.9903 | 5 | 5 | 0.9842 | 1152 | PASS |
| S6_post_block_residual | 8 | 0.9902 | 5 | 5 | 0.9721 | 20736 | PASS |
| S5_transformed_update | 8 | 0.9902 | 5 | 5 | 0.9721 | 20736 | PASS |
| S6_post_block_residual | 4 | 0.9900 | 5 | 5 | 0.9722 | 11520 | PASS |
| S5_transformed_update | 4 | 0.9900 | 5 | 5 | 0.9721 | 11520 | PASS |
| S5_transformed_update | 2 | 0.9816 | 5 | 5 | 0.9583 | 6912 | PASS |
| S6_post_block_residual | 2 | 0.9816 | 5 | 5 | 0.9583 | 6912 | PASS |
| S7_final_representation | 1 | 0.9691 | 5 | 5 | 0.9459 | 768 | PASS |
| S5_transformed_update | 1 | 0.9584 | 5 | 5 | 0.9080 | 4608 | PASS |
| S6_post_block_residual | 1 | 0.9584 | 5 | 5 | 0.9080 | 4608 | PASS |
| S4_aggregated_context | 0 | 0.9293 | 5 | 4 | 0.8650 | 1152 | PASS |
| S7_final_representation | 0 | 0.9165 | 5 | 4 | 0.8856 | 192 | PASS |
| S5_transformed_update | 0 | 0.8229 | 4 | 0 | 0.6436 | 1152 | PASS |
| S6_post_block_residual | 0 | 0.8229 | 4 | 0 | 0.6436 | 1152 | PASS |

A positive rank is retained only if its held-out causal recovery improves materially over rank 0; in-sample anchor fit alone is not a selection criterion.
