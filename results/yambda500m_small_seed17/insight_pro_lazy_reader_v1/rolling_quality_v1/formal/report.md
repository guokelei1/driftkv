# Frozen lightweight PRO: five-edge rolling quality

Prospective strict quality gate: **FAIL**. Post-result Design viability: **PASS**. PRO improves or ties Reuse on 5/5 AUC edges and 3/5 log-loss edges.

The release-time action is one 2 KiB user sidecar at 9.14% of Full theoretical FLOPs. It materializes no CAST prefix. Design 0 below is a sealed comparison baseline, not an execution stage.

| edge | requests | Current AUC | Reuse AUC | Design 0 AUC | PRO AUC | PRO−Reuse (pp) | Current log-loss | Reuse log-loss | Design 0 log-loss | PRO log-loss | PRO gain retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0_to_v1 | 43186 | 0.681769 | 0.678709 | 0.681432 | 0.679626 | +0.091723 | 0.328479 | 0.328981 | 0.328572 | 0.328787 | +82.1% |
| v1_to_v2 | 41655 | 0.670493 | 0.668496 | 0.671611 | 0.670190 | +0.169338 | 0.331455 | 0.331729 | 0.331373 | 0.331489 | +95.2% |
| v2_to_v3 | 43092 | 0.615001 | 0.613514 | 0.614609 | 0.613563 | +0.004930 | 0.329367 | 0.329562 | 0.329431 | 0.329581 | +53.6% |
| v3_to_v4 | 43945 | 0.620994 | 0.619054 | 0.619960 | 0.619511 | +0.045681 | 0.336519 | 0.336438 | 0.336484 | 0.336476 | -220.1% |
| v4_to_v5 | 45706 | 0.551858 | 0.551221 | 0.548563 | 0.551425 | +0.020374 | 0.332627 | 0.332668 | 0.332952 | 0.332664 | +80.3% |

Unweighted mean edge PRO−Reuse: +0.066409 AUC pp and -0.00007651 log-loss.

The strict gate is not rewritten after observing labels. The separate viability interpretation uses the user-approved aggregate/majority criterion and only says that PRO is worth continuing; it does not admit a serving lineage.

On v3→v4, Current Exact rolling log-loss is itself worse than Reuse while AUC is better, and the Full-only AUC gain denominator is only 0.046331 pp. PRO lies between Reuse and Current in log-loss while recovering AUC, so this edge is a ranking/calibration target conflict rather than a numerical reader-map failure.

The next step is an independently frozen label-free admission/calibration test. Any tuned mechanism must be validated on a new seed or release edge because these five labels are now development evidence.
