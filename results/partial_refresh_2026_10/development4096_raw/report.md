# Partial-refresh quality evidence

Selected users: 4096; active: 2562; requests: 51731.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.624779 | 0.63351 | 0.63351 | 100 | 0 |
| v4 | 0.600673 | 0.619945 | 0.618712 | 93.6011 | 0.0156007 |
| v5 | 0.623931 | 0.646838 | 0.642936 | 82.9686 | 0.0298999 |

Equal-release gain retention: 89.91465718190449%; UID95: [79.51346793187909, 98.04507980838969].
Preservation margin at 95%: {'estimate': -0.0008629899071367665, 'percentile_95_interval': [-0.002011191276321029, 0.00032766886872566116], 'finite_draws': 1000, 'undefined_draws': 0}.
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
| v4 | 0.444564 | 0.474146 | 0.469059 | 82.8033 | 0.0497115 |
| v5 | 0.541092 | 0.600114 | 0.591661 | 85.677 | 0.0157897 |

Equal-release gain retention: 84.94855112342069%; UID95: [45.50915631703364, 112.79329549167578].
Preservation margin at 95%: {'estimate': -0.003014248418733982, 'percentile_95_interval': [-0.007978413091410725, 0.002788580419991094], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### initial_history_257_768


Selected users: 443; active: 187; requests: 2313.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.63359 | 0.635863 | 0.635863 | 100 | 0 |
| v4 | 0.572012 | 0.582108 | 0.578631 | 65.5602 | 0.025031 |
| v5 | 0.670369 | 0.698894 | 0.693896 | 82.4779 | 0.0155891 |

Equal-release gain retention: 79.27522924880049%; UID95: [-75.57703034300356, 113.98181528138022].
Preservation margin at 95%: {'estimate': -0.002143512995110536, 'percentile_95_interval': [-0.00703494412180895, 0.0017479914270732788], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### initial_history_gt768


Selected users: 3195; active: 2277; requests: 48170.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.625493 | 0.634666 | 0.634666 | 100 | 0 |
| v4 | 0.606758 | 0.625667 | 0.624703 | 94.9035 | 0.0146361 |
| v5 | 0.627914 | 0.648301 | 0.644957 | 83.595 | 0.0319056 |

Equal-release gain retention: 91.11137982219049%; UID95: [78.63955901777783, 100.95202587363643].
Preservation margin at 95%: {'estimate': -0.0006282523849006506, 'percentile_95_interval': [-0.0019183834069924836, 0.0004956736125496128], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_inactive


Selected users: 1357; active: 549; requests: 9389.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.620176 | 0.636365 | 0.636365 | 100 | 0 |
| v4 | 0.48678 | 0.505749 | 0.503548 | 88.397 | 0.0189622 |
| v5 | 0.607545 | 0.646731 | 0.645012 | 95.6143 | 0.0133823 |

Equal-release gain retention: 94.72785502358637%; UID95: [82.11447714160863, 111.62950375733168].
Preservation margin at 95%: {'estimate': -6.744030653765831e-05, 'percentile_95_interval': [-0.0022714444090515735, 0.001985510960368755], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_active


Selected users: 2739; active: 2013; requests: 42342.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.626524 | 0.634203 | 0.634203 | 100 | 0 |
| v4 | 0.639437 | 0.655195 | 0.654333 | 94.5312 | 0.0148204 |
| v5 | 0.625501 | 0.645335 | 0.64075 | 76.8864 | 0.0364028 |

Equal-release gain retention: 87.4139072155505%; UID95: [58.28835758692216, 101.39309343596167].
Preservation margin at 95%: {'estimate': -0.0010941715851944412, 'percentile_95_interval': [-0.002561631159143359, 0.0004028516334375949], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
