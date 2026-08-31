# Compact-probe AV broadcast-residual score canary

Progression gate: **PASS** (4/5 edges); labels read: **no**.

The sidecar is generated once from a fixed latest-history-item probe over a 32-carrier disposable Current source, then coverage-scaled and broadcast at every AV layer. It is not a target-K/V fit or a per-candidate selector.

| edge | requests | Design 0 logit gap | AV sidecar logit gap | relative change | not worse |
| --- | ---: | ---: | ---: | ---: | --- |
| v0_to_v1 | 368 | 0.017808326 | 0.011271247 | -36.71% | True |
| v1_to_v2 | 518 | 0.014596145 | 0.0066838904 | -54.21% | True |
| v2_to_v3 | 385 | 0.016330452 | 0.0060557691 | -62.92% | True |
| v3_to_v4 | 249 | 0.03230288 | 0.035847081 | +10.97% | False |
| v4_to_v5 | 285 | 0.017589901 | 0.0073851468 | -58.01% | True |

Maximum baseline/probe replay error: 7.1525574e-07.

Per the prospective contract, this canary does not launch formal quality or admit an action regardless of pass/fail; the result returns to expert discussion.
