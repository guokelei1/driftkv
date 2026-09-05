# Medium S4 temporal-coordinate diagnostic: discovery

All coefficients below are least-squares projections of the current request's Current-Exact S4 correction onto the direction frozen at cutover. They diagnose representation geometry and are not executable estimators.

| edge | same-request oracle | frozen offset | global 1-scalar | layerwise 6-scalar |
| --- | ---: | ---: | ---: | ---: |
| v0_to_v1 | 0.9125 | -92.1860 | 0.3578 | 0.6396 |
| v1_to_v2 | 0.9328 | -50.9902 | 0.3333 | 0.4940 |
| v2_to_v3 | 0.9460 | -6.2568 | 0.5977 | 0.6982 |
| v3_to_v4 | 0.9450 | -19.0498 | 0.7389 | 0.7868 |
| v4_to_v5 | 0.9330 | -2.6116 | 0.4204 | 0.6332 |

## Gates

- One global temporal coefficient: recovery 0.4896, minimum edge 0.3333, positive 5/5 — FAIL.
- Six layerwise temporal coefficients: recovery 0.6504, minimum edge 0.4940, positive 5/5 — FAIL.
- Same-request full S4 shared correction ceiling: 0.9339.

## Representation conclusion

- Global/layerwise projection relative L2: 0.2099/0.1665.
- Adjudication: `fixed_cutover_response_basis_is_not_causally_sufficient_over_time`.
- A passing oracle coordinate gate supports a response-basis representation only. Design 1 still needs a legal <=20% coefficient estimator/update path before any action can be frozen.
- No label, confirmation user, serving action or target-KV fit is used.
