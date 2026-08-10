# KuaiRand selected 8×8 HR@5 Recompute-over-Reuse matrix

Positive means Recompute is more accurate than Reuse. Values are relative percentages.

| current \ cache | theta1 | theta2 | theta3 | theta4 | theta5 | theta6 | theta7 | theta8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| theta2 | -4.407% | — | — | — | — | — | — | — |
| theta3 | +5.607% | +6.004% | — | — | — | — | — | — |
| theta4 | +32.460% | +15.061% | +10.420% | — | — | — | — | — |
| theta5 | +5.273% | -4.613% | -3.661% | +0.696% | — | — | — | — |
| theta6 | +6.126% | +3.880% | -2.965% | -0.842% | +0.684% | — | — | — |
| theta7 | +12.447% | +18.444% | +14.624% | -2.914% | +6.175% | +7.243% | — | — |
| theta8 | +14.526% | +20.621% | +8.151% | +6.876% | +4.615% | +2.448% | +11.934% | — |
| theta9 | -5.495% | -7.099% | -0.660% | +6.173% | -0.166% | +1.176% | -0.824% | +2.381% |

Adjacent: 7/8 positive, mean +4.369%, minimum -4.407%.
All cells: 25/36 positive, mean +5.289%, median +4.944%.
Age accumulation: 6/7 later rows contain an older cache with larger loss than the adjacent cache; 6/7 exceed it by at least one percentage point.

Development evidence only; no score scaling or K/V coordinate perturbation is applied.
