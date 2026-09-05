# Medium Insight 2 functional-boundary discovery

This is a 512-user, five-edge, label-free representation test. Target corrections come from Current-Exact anchor traces; no row below is an executable estimator result.

The preregistered primary recovery is the mean of each user's unclipped `1 - observed gap / Reuse gap` within an edge, followed by an equal-weight mean across edges. Gap-weighted recovery is retained in CSV as a sensitivity analysis. The interval is a 2,000-replicate user-cluster bootstrap.

| stage | rank | recovery | 95% CI | min edge | edges >=90% | median user | users >=80% | FP32 values/user |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S4_aggregated_context | 0 | 0.9534 | [0.9493, 0.9571] | 0.9446 | 5 | 0.9795 | 0.9582 | 1152 |
| S4_aggregated_context | 1 | 0.9946 | [0.9940, 0.9951] | 0.9891 | 5 | 0.9981 | 0.9992 | 4608 |
| S4_aggregated_context | 2 | 0.9977 | [0.9974, 0.9979] | 0.9948 | 5 | 0.9993 | 1.0000 | 6912 |
| S4_aggregated_context | 4 | 0.9991 | [0.9990, 0.9992] | 0.9978 | 5 | 0.9998 | 1.0000 | 11520 |
| S4_aggregated_context | 8 | 0.9991 | [0.9990, 0.9992] | 0.9978 | 5 | 0.9998 | 1.0000 | 20736 |
| S5_transformed_update | 0 | 0.7982 | [0.7677, 0.8241] | 0.6117 | 2 | 0.9551 | 0.8367 | 1152 |
| S5_transformed_update | 1 | 0.9710 | [0.9668, 0.9744] | 0.9519 | 5 | 0.9932 | 0.9707 | 4608 |
| S5_transformed_update | 2 | 0.9874 | [0.9855, 0.9888] | 0.9796 | 5 | 0.9965 | 0.9922 | 6912 |
| S5_transformed_update | 4 | 0.9928 | [0.9915, 0.9939] | 0.9864 | 5 | 0.9988 | 0.9957 | 11520 |
| S5_transformed_update | 8 | 0.9930 | [0.9917, 0.9941] | 0.9867 | 5 | 0.9988 | 0.9961 | 20736 |
| S6_post_block_residual | 0 | 0.7982 | [0.7677, 0.8241] | 0.6117 | 2 | 0.9551 | 0.8367 | 1152 |
| S6_post_block_residual | 1 | 0.9710 | [0.9668, 0.9744] | 0.9519 | 5 | 0.9932 | 0.9707 | 4608 |
| S6_post_block_residual | 2 | 0.9874 | [0.9855, 0.9888] | 0.9796 | 5 | 0.9965 | 0.9922 | 6912 |
| S6_post_block_residual | 4 | 0.9928 | [0.9915, 0.9939] | 0.9864 | 5 | 0.9988 | 0.9957 | 11520 |
| S6_post_block_residual | 8 | 0.9930 | [0.9917, 0.9941] | 0.9867 | 5 | 0.9988 | 0.9961 | 20736 |
| S7_final_representation | 0 | 0.9274 | [0.9214, 0.9330] | 0.8949 | 4 | 0.9723 | 0.9227 | 192 |
| S7_final_representation | 1 | 0.9837 | [0.9818, 0.9854] | 0.9735 | 5 | 0.9940 | 0.9926 | 768 |
| S7_final_representation | 2 | 0.9936 | [0.9930, 0.9942] | 0.9908 | 5 | 0.9975 | 0.9992 | 1152 |
| S7_final_representation | 4 | 0.9981 | [0.9979, 0.9982] | 0.9974 | 5 | 0.9992 | 1.0000 | 1920 |
| S7_final_representation | 8 | 0.9986 | [0.9985, 0.9987] | 0.9979 | 5 | 0.9995 | 1.0000 | 3456 |

## Adjudication

- Earliest strong representation boundary: S4_aggregated_context, rank 0, recovery 0.9534, minimum edge 0.9446.
- Most compact strong representation boundary: S7_final_representation, rank 1, recovery 0.9837, 768 FP32 values/user.
- Best rank-1 response model: S4_aggregated_context, recovery 0.9946, minimum edge 0.9891.
- S5 transformed update and S6 post-block residual are algebraically equivalent under this same-hidden additive intervention; they are one observation, not independent confirmation.
- The representation gate passes. Estimator, 0--20% cost, persistence, and task-quality gates remain open; therefore neither Insight 2 nor Design 1 is frozen.
