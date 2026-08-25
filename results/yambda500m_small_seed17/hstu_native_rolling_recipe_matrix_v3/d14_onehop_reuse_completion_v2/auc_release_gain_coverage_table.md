# D=14 AUC release gain and One-hop Reuse coverage

This table uses only already sealed artifacts; it does not run another evaluation.

- **Old → new gain** is `Current Full − Parent Full` ROC-AUC from the completed D=14 Full-only recipe matrix (pp). Positive means the new model is better.
- **Reuse loss** is `Current Exact Rolling − One-hop Reuse Rolling` ROC-AUC from the sealed cache diagnostic (pp). Positive means Reuse is worse.
- **Reuse / gain** is `Reuse loss / old→new gain`. It is reported as a percentage only for a positive old→new gain. For a non-positive gain it is `N/A`, since there is no positive release benefit to erase.
- These first-pass ratios are **cross-reference observations**, not the strict matched-rolling `rho_erase`: the Full-only gain and rolling Reuse paths have different post-eviction semantics. They are suitable for deciding where to investigate; a formal erasure claim needs the three matched rolling paths.

`—` means that the requested E had not been measured in the existing sealed artifacts; it is not a zero effect.

| Edge | E | Old → new gain (pp) | Reuse loss (pp) | Reuse / gain |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | 1 | +1.214635 | +0.408499 | +33.6% |
| v0 → v1 | 2 | +0.997980 | +0.294460 | +29.5% |
| v0 → v1 | 4 | +0.689972 | +0.316814 | +45.9% |
| v0 → v1 | 7 | +0.936550 | +0.314771 | +33.6% |
| v0 → v1 | 14 | +1.199879 | +0.306014 | +25.5% |
| v1 → v2 | 1 | +0.704082 | +0.366961 | +52.1% |
| v1 → v2 | 2 | +0.556320 | +0.270427 | +48.6% |
| v1 → v2 | 4 | +0.643578 | +0.225747 | +35.1% |
| v1 → v2 | 7 | +0.701488 | +0.255661 | +36.4% |
| v1 → v2 | 14 | +0.626216 | +0.199662 | +31.9% |
| v2 → v3 | 1 | +1.470283 | +0.510054 | +34.7% |
| v2 → v3 | 2 | +0.839614 | +0.311953 | +37.2% |
| v2 → v3 | 4 | +0.862297 | +0.318089 | +36.9% |
| v2 → v3 | 7 | +0.514729 | +0.074433 | +14.5% |
| v2 → v3 | 14 | +0.310144 | +0.148693 | +47.9% |
| v3 → v4 | 1 | +1.960758 | +0.469085 | +23.9% |
| v3 → v4 | 2 | +0.690239 | +0.091464 | +13.3% |
| v3 → v4 | 4 | +0.509668 | +0.099171 | +19.5% |
| v3 → v4 | 7 | -0.552281 | +0.076174 | N/A |
| v3 → v4 | 14 | +0.046331 | +0.193984 | +418.7% |
| v4 → v5 | 1 | -0.024634 | -0.080459 | N/A |
| v4 → v5 | 2 | +0.145598 | -0.024313 | -16.7% |
| v4 → v5 | 4 | +0.640660 | +0.187401 | +29.3% |
| v4 → v5 | 7 | +0.393310 | +0.108692 | +27.6% |
| v4 → v5 | 14 | +0.220611 | +0.063736 | +28.9% |
