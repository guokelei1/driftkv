# D=14 one-hop Reuse diagnostic

Comparison: the current HSTU query/head reads either its own exact rolling prefix KV (**Recompute**) or the parent-produced cutover KV followed by current-model appends (**One-hop Reuse**).
All values are post-hoc diagnostic observations, not release admission or recursive-lineage results.

## User-equal Reuse − Recompute log loss (positive = Reuse harms)

| Edge | E=1 | E=4 | E=7 | E=14 |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | +0.000626 | +0.000471 | +0.000242 | +0.000089 |
| v1 → v2 | +0.000404 | +0.000343 | +0.000252 | +0.000110 |
| v2 → v3 | -0.000166 | +0.000015 | — | — |
| v3 → v4 | -0.000418 | +0.000295 | — | — |
| v4 → v5 | +0.000004 | — | — | — |

## Event-weighted Reuse − Recompute log loss (positive = Reuse harms)

| Edge | E=1 | E=4 | E=7 | E=14 |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | +0.001426 | +0.000885 | +0.000735 | +0.000502 |
| v1 → v2 | +0.000787 | +0.000429 | +0.000400 | +0.000274 |
| v2 → v3 | +0.000929 | +0.000421 | — | — |
| v3 → v4 | +0.000260 | +0.000012 | — | — |
| v4 → v5 | -0.000200 | — | — | — |

## Current Recompute − Reuse ROC-AUC (pp; positive = Reuse harms)

| Edge | E=1 | E=4 | E=7 | E=14 |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | +0.408499 | +0.316814 | +0.314771 | +0.306014 |
| v1 → v2 | +0.366961 | +0.225747 | +0.255661 | +0.199662 |
| v2 → v3 | +0.510054 | +0.318089 | — | — |
| v3 → v4 | +0.469085 | +0.099171 | — | — |
| v4 → v5 | -0.080459 | — | — | — |

## Current Recompute − Reuse dislike PR-AUC (pp; positive = Reuse harms)

| Edge | E=1 | E=4 | E=7 | E=14 |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | +0.361795 | +0.289196 | +0.207484 | +0.177913 |
| v1 → v2 | +0.442107 | +0.352764 | +0.301279 | +0.181358 |
| v2 → v3 | +0.278177 | +0.195982 | — | — |
| v3 → v4 | +0.168059 | -0.068619 | — | — |
| v4 → v5 | +0.137073 | — | — | — |

## Reuse − Recompute Brier (positive = Reuse harms)

| Edge | E=1 | E=4 | E=7 | E=14 |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | +0.000219 | +0.000123 | +0.000103 | +0.000073 |
| v1 → v2 | +0.000155 | +0.000085 | +0.000082 | +0.000055 |
| v2 → v3 | +0.000182 | +0.000086 | — | — |
| v3 → v4 | -0.000001 | -0.000020 | — | — |
| v4 → v5 | -0.000027 | — | — | — |

## Bernoulli JS

| Edge | E=1 | E=4 | E=7 | E=14 |
| --- | ---: | ---: | ---: | ---: |
| v0 → v1 | +0.000035 | +0.000029 | +0.000024 | +0.000017 |
| v1 → v2 | +0.000006 | +0.000005 | +0.000004 | +0.000003 |
| v2 → v3 | +0.000006 | +0.000004 | — | — |
| v3 → v4 | +0.000049 | +0.000039 | — | — |
| v4 → v5 | +0.000005 | — | — | — |
