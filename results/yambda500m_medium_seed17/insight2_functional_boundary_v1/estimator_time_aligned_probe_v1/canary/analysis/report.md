# Medium time-aligned functional-probe canary

The only changed variable is the fixed probe's query-time delta: zero versus the known release cutover delta. Neither estimator reads labels, request candidates or Current-Exact K/V.

| carriers | probe time | Exact cost | recovery | min edge | edges >=80% | cosine | norm ratio | weighted sensitivity | gate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | cutover | 5.59% | -1.0590 | -2.8263 | 0 | 0.6256 | 0.4978 | 0.0934 | FAIL |
| 32 | zero | 5.59% | -1.2993 | -3.2415 | 0 | 0.6062 | 0.5138 | 0.1051 | FAIL |
| 64 | cutover | 10.05% | -1.1410 | -4.5654 | 0 | 0.6510 | 0.4761 | 0.1014 | FAIL |
| 64 | zero | 10.05% | -1.4062 | -5.1734 | 0 | 0.6247 | 0.4896 | 0.1174 | FAIL |

## Adjudication

- Correcting probe time modestly improves direction and average recovery but no edge reaches the 80% gate.
- More probes were already flat and denser carriers improve cosine without improving score recovery. The parameter-only map plus Parent-conditioned-carrier response estimator family is retired.
- This negative result does not invalidate the Exact-oracle S4 functional boundary; it isolates estimator bias as the open problem.
- No 512-user run is authorized from this failed canary.
