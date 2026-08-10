# KuaiRand selected 8×8 NDCG@5 Recompute-over-Reuse matrix

Positive means Recompute is more accurate than Reuse. Values are relative percentages.

| current \ cache | theta1 | theta2 | theta3 | theta4 | theta5 | theta6 | theta7 | theta8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta2 | +2.066% | — | — | — | — | — | — | — |
| theta3 | +2.912% | +6.345% | — | — | — | — | — | — |
| theta4 | +39.688% | +22.646% | +11.515% | — | — | — | — | — |
| theta5 | +5.044% | +1.114% | -7.626% | +1.619% | — | — | — | — |
| theta6 | +15.104% | +6.862% | +2.227% | +6.119% | +1.613% | — | — | — |
| theta7 | +11.014% | +15.480% | +7.527% | -6.055% | +3.300% | +3.762% | — | — |
| theta8 | +25.106% | +18.408% | +10.237% | +12.618% | +6.361% | +7.708% | +15.128% | — |
| theta9 | -8.311% | -4.318% | -1.047% | +5.813% | +2.892% | +3.760% | +1.134% | +2.524% |

Adjacent: 8/8 positive, mean +5.572%, minimum +1.613%.
All cells: 31/36 positive, mean +6.953%, median +5.428%.
Age accumulation: 6/7 later rows contain an older cache with larger loss than the adjacent cache; 6/7 exceed it by at least one percentage point.

Development evidence only; no score scaling or K/V coordinate perturbation is applied.
