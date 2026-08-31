# Matched-cost evidence-measure basis canary

Progression gate: **FAIL** (0/5 edges).

No label was read. Current, Reuse and Design-0 logits were replayed against the prior sealed full-population raw before comparing the new basis.

| edge | requests | Design 0 mean abs logit gap | evidence basis gap | basis not worse |
| --- | ---: | ---: | ---: | --- |
| v0_to_v1 | 282 | 0.019558878 | 0.023025714 | False |
| v1_to_v2 | 370 | 0.0148014 | 0.015947434 | False |
| v2_to_v3 | 428 | 0.018546245 | 0.018804059 | False |
| v3_to_v4 | 292 | 0.029189022 | 0.033628147 | False |
| v4_to_v5 | 226 | 0.019214531 | 0.021572638 | False |

The canary changes no model, action set or serving lineage.
