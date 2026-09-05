# Medium S4 temporal-coordinate diagnostic: canary

All coefficients below are least-squares projections of the current request's Current-Exact S4 correction onto the direction frozen at cutover. They diagnose representation geometry and are not executable estimators.

| edge | same-request oracle | frozen offset | global 1-scalar | layerwise 6-scalar |
| --- | ---: | ---: | ---: | ---: |
| v0_to_v1 | 0.9209 | -150.3363 | 0.2062 | 0.5766 |
| v1_to_v2 | 0.9675 | -2.5319 | 0.6347 | 0.7810 |
| v2_to_v3 | 0.9705 | -0.4872 | 0.8003 | 0.8277 |
| v3_to_v4 | 0.9500 | -89.1635 | 0.7733 | 0.7880 |
| v4_to_v5 | 0.9147 | -0.8771 | 0.4928 | 0.6635 |

## Gates

- One global temporal coefficient: recovery 0.5815, minimum edge 0.2062, positive 5/5 — FAIL.
- Six layerwise temporal coefficients: recovery 0.7274, minimum edge 0.5766, positive 5/5 — FAIL.
- Same-request full S4 shared correction ceiling: 0.9447.

## Representation conclusion

- Global/layerwise projection relative L2: 0.2058/0.1674.
- Adjudication: `fixed_cutover_response_basis_is_not_causally_sufficient_over_time`.
- A passing oracle coordinate gate supports a response-basis representation only. Design 1 still needs a legal <=20% coefficient estimator/update path before any action can be frozen.
- No label, confirmation user, serving action or target-KV fit is used.
