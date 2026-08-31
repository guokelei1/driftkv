# Lightweight PRO quality canary

Progression gate: **PASS**.

No label was read. All five checkpoint edges replayed the sealed Parent, Current and Reuse logits before any formal quality launch.

| edge | users | requests | Design 0 logit gap | PRO logit gap |
| --- | ---: | ---: | ---: | ---: |
| v0_to_v1 | 32 | 282 | 0.019558894 | 0.011553222 |
| v1_to_v2 | 32 | 370 | 0.01480137 | 0.0077171873 |
| v2_to_v3 | 32 | 428 | 0.018546246 | 0.0063619104 |
| v3_to_v4 | 32 | 292 | 0.029189036 | 0.027661675 |
| v4_to_v5 | 32 | 226 | 0.019214532 | 0.0096326906 |

The PRO action materialized zero version-translated prefix positions; the comparison does not turn Design 0 into a serving stage.
