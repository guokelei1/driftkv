# Medium release-functional-basis diagnostic: discovery

All rows are label-free oracle diagnostics. `exact_fixed_history_probe_mean` requires each target user's Current-Exact cache. Positive release-basis ranks also use each evaluation user's Exact target coefficients. Neither is an executable migration action.

| method | rank | recovery | min edge | edges >=80% | edges >=90% | cosine to candidate-anchor target | user fraction >=80% | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact_fixed_history_probe_mean | -1 | 0.9470 | 0.9358 | 5 | 5 | 0.9999 | 0.9513 | PASS |
| oracle_release_basis | 0 | -1.6855 | -3.2860 | 0 | 0 | 0.7164 | 0.1295 | FAIL |
| oracle_release_basis | 1 | -0.0884 | -1.2217 | 0 | 0 | 0.9437 | 0.4638 | FAIL |
| oracle_release_basis | 2 | 0.5967 | 0.4286 | 0 | 0 | 0.9700 | 0.6759 | FAIL |
| oracle_release_basis | 4 | 0.8947 | 0.7906 | 4 | 3 | 0.9937 | 0.9004 | PASS |
| oracle_release_basis | 8 | 0.9418 | 0.9316 | 5 | 5 | 0.9991 | 0.9429 | PASS |
| oracle_release_basis | 16 | 0.9463 | 0.9339 | 5 | 5 | 0.9998 | 0.9504 | PASS |
| oracle_release_basis | 32 | 0.9469 | 0.9358 | 5 | 5 | 0.9999 | 0.9513 | PASS |

## Adjudication

- Fixed history-query Exact ceiling: 0.9470 recovery, 5/5 edges >=80%.
- Best preregistered release-basis oracle ceiling: rank 32, 0.9469 recovery, 5/5 edges >=80%.
- Mean per-layer calibration-cohort rank@90: 1.53; mean rank-1/rank-2/rank-4 energy: 0.8884/0.9604/0.9893.
- Structural prerequisites for considering a separately authorized release-calibration design: PASS.
- Passing prerequisites does not relax the repository ban on adding predictor complexity to the current scale frontier and does not pass the 0--20% executable-estimator gate.
