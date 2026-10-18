# Partial-refresh quality evidence

Selected users: 4096; active: 2562; requests: 51731.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.624779 | 0.63351 | 0.63351 | 100 | 0 |
| v4 | 0.600673 | 0.619945 | 0.618885 | 94.5004 | 0.0208529 |
| v5 | 0.623931 | 0.646838 | 0.644517 | 89.8675 | 0.031361 |

Equal-release gain retention: 93.3591303204632%; UID95: [87.57174625724925, 106.14138011732835].
Preservation margin at 95%: {'estimate': -0.00027845791778838627, 'percentile_95_interval': [-0.0013792716738606889, 0.0009149189901653721], 'finite_draws': 1000, 'undefined_draws': 0}.
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
| v4 | 0.444564 | 0.474146 | 0.467735 | 78.3274 | 0.0505017 |
| v5 | 0.541092 | 0.600114 | 0.59635 | 93.6218 | 0.0110596 |

Equal-release gain retention: 88.68914767374656%; UID95: [55.80977918493747, 116.88060966599808].
Preservation margin at 95%: {'estimate': -0.0018925109085116062, 'percentile_95_interval': [-0.007433424212732679, 0.0032346811108521158], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### initial_history_257_768


Selected users: 443; active: 187; requests: 2313.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.63359 | 0.635863 | 0.635863 | 100 | 0 |
| v4 | 0.572012 | 0.582108 | 0.578631 | 65.5602 | 0.0250319 |
| v5 | 0.670369 | 0.698894 | 0.698036 | 96.9912 | 0.0159464 |

Equal-release gain retention: 89.39876213411274%; UID95: [3.0423671184015664, 176.73025855499878].
Preservation margin at 95%: {'estimate': -0.0007635294875964205, 'percentile_95_interval': [-0.005869316941114031, 0.003649845597091963], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### initial_history_gt768


Selected users: 3195; active: 2277; requests: 48170.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.625493 | 0.634666 | 0.634666 | 100 | 0 |
| v4 | 0.606758 | 0.625667 | 0.624916 | 96.0285 | 0.0202475 |
| v5 | 0.627914 | 0.648301 | 0.646291 | 90.1413 | 0.0337499 |

Equal-release gain retention: 94.30381774357193%; UID95: [87.95718476142964, 114.03117562447981].
Preservation margin at 95%: {'estimate': -0.00011247644226668359, 'percentile_95_interval': [-0.0012404937487999585, 0.0010637273600163264], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_inactive


Selected users: 1357; active: 549; requests: 9389.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.620176 | 0.636365 | 0.636365 | 100 | 0 |
| v4 | 0.48678 | 0.505749 | 0.503138 | 86.2383 | 0.020388 |
| v5 | 0.607545 | 0.646731 | 0.646138 | 98.4876 | 0.0127762 |

Equal-release gain retention: 95.69153092608678%; UID95: [85.2205222401046, 112.96893361974756].
Preservation margin at 95%: {'estimate': 0.00017136843108461752, 'percentile_95_interval': [-0.001784104715851628, 0.0020204729831071046], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_active


Selected users: 2739; active: 2013; requests: 42342.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.626524 | 0.634203 | 0.634203 | 100 | 0 |
| v4 | 0.639437 | 0.655195 | 0.65491 | 98.1947 | 0.0209608 |
| v5 | 0.625501 | 0.645335 | 0.642851 | 87.4793 | 0.0386777 |

Equal-release gain retention: 93.60346774218631%; UID95: [86.38129603552154, 132.89727387856735].
Preservation margin at 95%: {'estimate': -0.00020142726403761697, 'percentile_95_interval': [-0.001647835188760691, 0.0014360355470859531], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
