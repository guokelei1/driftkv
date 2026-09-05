# Medium release-functional-basis diagnostic: canary

All rows are label-free oracle diagnostics. `exact_fixed_history_probe_mean` requires each target user's Current-Exact cache. Positive release-basis ranks also use each evaluation user's Exact target coefficients. Neither is an executable migration action.

| method | rank | recovery | min edge | edges >=80% | edges >=90% | cosine to candidate-anchor target | user fraction >=80% | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact_fixed_history_probe_mean | -1 | 0.9197 | 0.8752 | 5 | 3 | 0.9999 | 0.9417 | PASS |
| oracle_release_basis | 0 | -3.3982 | -6.6242 | 0 | 0 | 0.6726 | 0.1167 | FAIL |
| oracle_release_basis | 1 | 0.2948 | -0.0665 | 0 | 0 | 0.9369 | 0.5000 | FAIL |
| oracle_release_basis | 2 | 0.4341 | 0.1022 | 0 | 0 | 0.9635 | 0.6417 | FAIL |
| oracle_release_basis | 4 | 0.7380 | 0.4941 | 1 | 1 | 0.9822 | 0.8250 | FAIL |
| oracle_release_basis | 8 | 0.8775 | 0.7712 | 4 | 2 | 0.9959 | 0.9083 | PASS |
| oracle_release_basis | 16 | 0.8775 | 0.7712 | 4 | 2 | 0.9959 | 0.9083 | PASS |
| oracle_release_basis | 32 | 0.8775 | 0.7712 | 4 | 2 | 0.9959 | 0.9083 | PASS |

## Adjudication

- Fixed history-query Exact ceiling: 0.9197 recovery, 5/5 edges >=80%.
- Best preregistered release-basis oracle ceiling: rank 8, 0.8775 recovery, 4/5 edges >=80%.
- Mean per-layer calibration-cohort rank@90: 1.43; mean rank-1/rank-2/rank-4 energy: 0.8851/0.9692/0.9957.
- Structural prerequisites for considering a separately authorized release-calibration design: PASS.
- Passing prerequisites does not relax the repository ban on adding predictor complexity to the current scale frontier and does not pass the 0--20% executable-estimator gate.
