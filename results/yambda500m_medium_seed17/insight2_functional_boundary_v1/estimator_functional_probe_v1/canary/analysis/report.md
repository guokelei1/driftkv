# Medium executable functional-probe estimator: canary

No label, request candidate, future event or Current-Exact K/V enters the estimator. Current Exact is used only after construction as the evaluation reference and for correction-shape diagnostics.

| carriers | probes | Exact cost | recovery | min edge | edges >=80% | edges >=90% | cosine to oracle | norm ratio | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 1 | 2.30% | -1.1196 | -3.3569 | 0 | 0 | 0.5135 | 0.6327 | FAIL |
| 16 | 1 | 3.39% | -0.9495 | -1.8642 | 0 | 0 | 0.5565 | 0.5628 | FAIL |
| 8 | 2 | 3.53% | -1.0838 | -3.2848 | 0 | 0 | 0.5144 | 0.6256 | FAIL |
| 16 | 2 | 4.62% | -0.9288 | -1.8453 | 0 | 0 | 0.5573 | 0.5616 | FAIL |
| 32 | 1 | 5.59% | -1.2993 | -3.2415 | 0 | 0 | 0.6062 | 0.5138 | FAIL |
| 8 | 4 | 5.97% | -1.1255 | -3.3761 | 0 | 0 | 0.5138 | 0.6304 | FAIL |
| 32 | 2 | 6.81% | -1.2653 | -3.1995 | 0 | 0 | 0.6070 | 0.5150 | FAIL |
| 16 | 4 | 7.07% | -0.9430 | -1.8644 | 0 | 0 | 0.5570 | 0.5628 | FAIL |
| 32 | 4 | 9.26% | -1.2841 | -3.2700 | 0 | 0 | 0.6065 | 0.5150 | FAIL |
| 64 | 1 | 10.05% | -1.4062 | -5.1734 | 0 | 0 | 0.6247 | 0.4896 | FAIL |
| 8 | 8 | 10.87% | -1.1318 | -3.4517 | 0 | 0 | 0.5139 | 0.6276 | FAIL |
| 64 | 2 | 11.28% | -1.3635 | -5.0934 | 0 | 0 | 0.6251 | 0.4856 | FAIL |
| 16 | 8 | 11.96% | -0.9462 | -1.9059 | 0 | 0 | 0.5570 | 0.5644 | FAIL |
| 64 | 4 | 13.74% | -1.3843 | -5.2060 | 0 | 0 | 0.6245 | 0.4879 | FAIL |
| 32 | 8 | 14.17% | -1.2966 | -3.3207 | 0 | 0 | 0.6068 | 0.5171 | FAIL |
| 64 | 8 | 18.65% | -1.4084 | -5.2905 | 0 | 0 | 0.6251 | 0.4885 | FAIL |

## Adjudication

- No configuration passes Gate D on this canary; best is C16/P2 at -0.9288 recovery.
- This tests cutover construction only. Temporal persistence and task-label quality remain open even if the estimator gate passes.
