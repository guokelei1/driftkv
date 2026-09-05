# Medium attention-address signed response coreset: canary

This is a fit-free Exact-state oracle. It changes only landmark geometry: layer-0 paired Current/Parent key coverage replaces chronological midpoints; the signed native-query reader and held-out panel are unchanged.

| R | edge-equal recovery | minimum edge | positive edges | >=80% edges | address radius | address - chronological canary | full-KV ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -2.1787 | -4.1756 | 0/5 | 0/5 | 0.7760 | 0.5784 | 1.5625% |
| 16 | -0.7876 | -2.5183 | 2/5 | 0/5 | 0.5002 | 0.4988 | 3.1250% |
| 32 | -1.2751 | -2.9520 | 2/5 | 0/5 | 0.3230 | -0.2642 | 6.2500% |
| 64 | -0.4532 | -1.9112 | 2/5 | 0/5 | 0.1982 | -0.3814 | 12.5000% |
| 128 | 0.3719 | -0.1914 | 3/5 | 2/5 | 0.1146 | 0.2752 | 25.0000% |

## Adjudication

- Canary-to-discovery gate: FAIL; smallest passing R: None.
- Attention-address oracle gate: FAIL; smallest passing R: None.
- Storage-matched address-geometry gate: FAIL; selected improvement: None.
- Interpretation: `attention_address_coreset_hypothesis_retired`.
- Address clustering alone is not a Design contribution. This oracle cannot admit Design 1 because selected upper-layer Current K/V remain Exact.
- No labels, candidates in construction, response fitting, confirmation users or executable-cost claims are used.
