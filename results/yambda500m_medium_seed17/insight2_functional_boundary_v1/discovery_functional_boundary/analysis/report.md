# Medium Insight 2 functional-boundary discovery

This is a 512-user, five-edge, label-free representation test. Target corrections come from Current-Exact anchor traces; no row below is an executable estimator result.

The primary recovery is `1 - sum(observed probability gap) / sum(Reuse probability gap)` within each edge, followed by an equal-weight mean across edges. This avoids giving nearly-zero-harm users disproportionate weight. The interval is a 2,000-replicate user-cluster bootstrap.

| stage | rank | recovery | 95% CI | min edge | edges >=90% | median user | users >=80% | FP32 values/user |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S4_aggregated_context | 0 | 0.9753 | [0.9743, 0.9764] | 0.9721 | 5 | 0.9795 | 0.9582 | 1152 |
| S4_aggregated_context | 1 | 0.9977 | [0.9976, 0.9978] | 0.9951 | 5 | 0.9981 | 0.9992 | 4608 |
| S4_aggregated_context | 2 | 0.9990 | [0.9989, 0.9991] | 0.9979 | 5 | 0.9993 | 1.0000 | 6912 |
| S4_aggregated_context | 4 | 0.9995 | [0.9994, 0.9995] | 0.9988 | 5 | 0.9998 | 1.0000 | 11520 |
| S4_aggregated_context | 8 | 0.9995 | [0.9995, 0.9995] | 0.9988 | 5 | 0.9998 | 1.0000 | 20736 |
| S5_transformed_update | 0 | 0.9410 | [0.9363, 0.9452] | 0.8976 | 4 | 0.9551 | 0.8367 | 1152 |
| S5_transformed_update | 1 | 0.9914 | [0.9909, 0.9918] | 0.9867 | 5 | 0.9932 | 0.9707 | 4608 |
| S5_transformed_update | 2 | 0.9958 | [0.9956, 0.9960] | 0.9937 | 5 | 0.9965 | 0.9922 | 6912 |
| S5_transformed_update | 4 | 0.9981 | [0.9979, 0.9982] | 0.9961 | 5 | 0.9988 | 0.9957 | 11520 |
| S5_transformed_update | 8 | 0.9981 | [0.9979, 0.9983] | 0.9962 | 5 | 0.9988 | 0.9961 | 20736 |
| S6_post_block_residual | 0 | 0.9410 | [0.9363, 0.9452] | 0.8976 | 4 | 0.9551 | 0.8367 | 1152 |
| S6_post_block_residual | 1 | 0.9914 | [0.9909, 0.9918] | 0.9867 | 5 | 0.9932 | 0.9707 | 4608 |
| S6_post_block_residual | 2 | 0.9958 | [0.9956, 0.9960] | 0.9937 | 5 | 0.9965 | 0.9922 | 6912 |
| S6_post_block_residual | 4 | 0.9981 | [0.9979, 0.9982] | 0.9961 | 5 | 0.9988 | 0.9957 | 11520 |
| S6_post_block_residual | 8 | 0.9981 | [0.9979, 0.9983] | 0.9962 | 5 | 0.9988 | 0.9961 | 20736 |
| S7_final_representation | 0 | 0.9718 | [0.9703, 0.9732] | 0.9606 | 5 | 0.9723 | 0.9227 | 192 |
| S7_final_representation | 1 | 0.9939 | [0.9935, 0.9942] | 0.9896 | 5 | 0.9940 | 0.9926 | 768 |
| S7_final_representation | 2 | 0.9974 | [0.9973, 0.9976] | 0.9964 | 5 | 0.9975 | 0.9992 | 1152 |
| S7_final_representation | 4 | 0.9992 | [0.9991, 0.9992] | 0.9988 | 5 | 0.9992 | 1.0000 | 1920 |
| S7_final_representation | 8 | 0.9994 | [0.9993, 0.9994] | 0.9990 | 5 | 0.9995 | 1.0000 | 3456 |

## Adjudication

- Earliest strong representation boundary: S4_aggregated_context, rank 0, recovery 0.9753, minimum edge 0.9721.
- Most compact strong representation boundary: S7_final_representation, rank 0, recovery 0.9718, 192 FP32 values/user.
- Best rank-1 response model: S4_aggregated_context, recovery 0.9977, minimum edge 0.9951.
- S5 transformed update and S6 post-block residual are algebraically equivalent under this same-hidden additive intervention; they are one observation, not independent confirmation.
- The representation gate passes. Estimator, 0--20% cost, persistence, and task-quality gates remain open; therefore neither Insight 2 nor Design 1 is frozen.
