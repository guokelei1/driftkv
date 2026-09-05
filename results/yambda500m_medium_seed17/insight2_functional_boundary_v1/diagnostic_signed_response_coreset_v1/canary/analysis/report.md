# Medium signed attention-response coreset: canary

This is a fit-free Exact-state oracle. The complete Parent response is the control path; real held-out Current queries read paired positive-Current and negative-Parent midpoint atoms through the native attention kernel.

| R | edge-equal recovery | minimum edge | positive edges | >=80% edges | stored scalars | full-KV ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -2.7572 | -4.6163 | 1/5 | 0/5 | 36864 | 1.5625% |
| 16 | -1.2864 | -2.8975 | 1/5 | 0/5 | 73728 | 3.1250% |
| 32 | -1.0109 | -2.0547 | 1/5 | 0/5 | 147456 | 6.2500% |
| 64 | -0.0719 | -0.6908 | 3/5 | 0/5 | 294912 | 12.5000% |
| 128 | 0.0967 | -0.8886 | 3/5 | 1/5 | 589824 | 25.0000% |

## Adjudication

- Canary-to-discovery gate: FAIL.
- Compact operator gate: FAIL; smallest passing R: None.
- Interpretation: `fixed_midpoint_signed_coreset_family_retired`.
- No label, candidate-conditioned construction, response regression, confirmation user or executable-cost claim is used.
- Passing this oracle can only unlock a separately contracted sparse causal-replay constructor; it cannot admit Design 1.
