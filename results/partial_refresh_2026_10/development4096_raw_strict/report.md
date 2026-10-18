# Partial-refresh quality evidence

Selected users: 4096; active: 2562; requests: 51731.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.624779 | 0.63351 | 0.63351 | 100 | 0 |
| v4 | 0.600673 | 0.619945 | 0.61938 | 97.0665 | 0.0037233 |
| v5 | 0.623931 | 0.646838 | 0.645495 | 94.1375 | 0.0105536 |

Equal-release gain retention: 96.25176251888448%; UID95: [91.57042048408064, 101.16083351973269].
Preservation margin at 95%: {'estimate': 0.00021242587934985555, 'percentile_95_interval': [-0.0003622880864930999, 0.0007444296573885121], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

Theoretical FLOPs use recorded per-user service counters. Fixed30k complete costs and shared preparation are reported separately. No loaded-serving timing or throughput was measured.

## Predefined user groups

### initial_history_le256


Selected users: 458; active: 98; requests: 1248.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.552642 | 0.554002 | 0.554002 | 100 | 0 |
| v4 | 0.444564 | 0.474146 | 0.47115 | 89.8704 | 0.0218095 |
| v5 | 0.541092 | 0.600114 | 0.594258 | 90.0783 | 0.00567579 |

Equal-release gain retention: 90.15992924236956%; UID95: [63.797417208505244, 104.55799845104589].
Preservation margin at 95%: {'estimate': -0.0014514500155039453, 'percentile_95_interval': [-0.004711041783242837, 0.001817157802728828], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### initial_history_257_768


Selected users: 443; active: 187; requests: 2313.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.63359 | 0.635863 | 0.635863 | 100 | 0 |
| v4 | 0.572012 | 0.582108 | 0.58016 | 80.7054 | 0.00847082 |
| v5 | 0.670369 | 0.698894 | 0.698137 | 97.3451 | 0.00614721 |

Equal-release gain retention: 93.38470971510321%; UID95: [26.140159206227036, 154.74662159703723].
Preservation margin at 95%: {'estimate': -0.00022018735734433137, 'percentile_95_interval': [-0.0033836738326897945, 0.003448702931387076], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### initial_history_gt768


Selected users: 3195; active: 2277; requests: 48170.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.625493 | 0.634666 | 0.634666 | 100 | 0 |
| v4 | 0.606758 | 0.625667 | 0.625281 | 97.9589 | 0.00322653 |
| v5 | 0.627914 | 0.648301 | 0.647274 | 94.9599 | 0.011195 |

Equal-release gain retention: 97.08373468092158%; UID95: [92.10407442334748, 104.2055024654147].
Preservation margin at 95%: {'estimate': 0.00033665187725446903, 'percentile_95_interval': [-0.00029506418849899527, 0.0008747969212738911], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_inactive


Selected users: 1357; active: 549; requests: 9389.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.620176 | 0.636365 | 0.636365 | 100 | 0 |
| v4 | 0.48678 | 0.505749 | 0.504439 | 93.098 | 0.00502155 |
| v5 | 0.607545 | 0.646731 | 0.644943 | 95.4374 | 0.0062364 |

Equal-release gain retention: 95.83407666761524%; UID95: [80.19511856785051, 102.26337522855204].
Preservation margin at 95%: {'estimate': 0.00020669272268464267, 'percentile_95_interval': [-0.0013893105933568649, 0.0016937734334308088], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_active


Selected users: 2739; active: 2013; requests: 42342.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.626524 | 0.634203 | 0.634203 | 100 | 0 |
| v4 | 0.639437 | 0.655195 | 0.654888 | 98.0549 | 0.00342194 |
| v5 | 0.625501 | 0.645335 | 0.644214 | 94.3492 | 0.0122532 |

Equal-release gain retention: 96.7015071063276%; UID95: [89.01320985995595, 112.32281716231618].
Preservation margin at 95%: {'estimate': 0.000245414969293074, 'percentile_95_interval': [-0.0004363089788777908, 0.0008341706982406697], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
