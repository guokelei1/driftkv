# KuaiRand selected 8×8 MRR Recompute-over-Reuse matrix

Positive means Recompute is more accurate than Reuse. Values are relative percentages.

| current \ cache | theta1 | theta2 | theta3 | theta4 | theta5 | theta6 | theta7 | theta8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta2 | +4.084% | — | — | — | — | — | — | — |
| theta3 | -1.227% | +1.612% | — | — | — | — | — | — |
| theta4 | +17.108% | +12.605% | +4.839% | — | — | — | — | — |
| theta5 | -0.427% | +1.717% | -6.324% | +0.811% | — | — | — | — |
| theta6 | +11.942% | +5.683% | +3.941% | +6.192% | +1.512% | — | — | — |
| theta7 | +4.092% | +4.161% | -0.308% | -3.441% | +0.040% | +0.122% | — | — |
| theta8 | +12.373% | +5.014% | +4.080% | +5.203% | +2.771% | +4.991% | +5.317% | — |
| theta9 | -4.960% | -1.162% | -1.640% | +1.580% | +2.216% | +1.956% | +0.863% | +0.716% |

Adjacent: 8/8 positive, mean +2.377%, minimum +0.122%.
All cells: 28/36 positive, mean +3.001%, median +2.086%.
Age accumulation: 6/7 later rows contain an older cache with larger loss than the adjacent cache; 5/7 exceed it by at least one percentage point.

Development evidence only; no score scaling or K/V coordinate perturbation is applied.
