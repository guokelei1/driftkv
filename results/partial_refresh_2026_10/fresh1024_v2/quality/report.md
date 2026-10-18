# Partial-refresh quality evidence

Selected users: 1024; active: 367; requests: 6809.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.714866 | 0.703421 | 0.703421 | undefined | 0 |
| v4 | 0.614645 | 0.568035 | 0.567561 | undefined | 0.0338006 |
| v5 | 0.638635 | 0.605518 | 0.606127 | undefined | 0.0501338 |

Equal-release gain retention: None%; UID95: [-30263.94455617429, 111.87464945503497].
Preservation margin at 95%: {'estimate': -0.001474351140716812, 'percentile_95_interval': [-0.004998569327068271, 0.0015539674382913234], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

Theoretical FLOPs use recorded per-user service counters. Fixed30k complete costs and shared preparation are reported separately. No loaded-serving timing or throughput was measured.

## Predefined user groups

### history_le256


Selected users: 163; active: 20; requests: 421.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | undefined | undefined | undefined | undefined | 0 |
| v4 | 0.846154 | 0.846154 | 0.846154 | undefined | 0.126406 |
| v5 | undefined | undefined | undefined | undefined | 0.212299 |

Equal-release gain retention: None%; UID95: None.
Preservation margin at 95%: {'estimate': None, 'percentile_95_interval': None, 'finite_draws': 0, 'undefined_draws': 1000}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### history_257_to768


Selected users: 162; active: 33; requests: 330.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.821895 | 0.763072 | 0.763072 | undefined | 0 |
| v4 | 0.761589 | 0.52649 | 0.523179 | undefined | 0.085321 |
| v5 | undefined | undefined | undefined | undefined | 0.0378627 |

Equal-release gain retention: None%; UID95: None.
Preservation margin at 95%: {'estimate': None, 'percentile_95_interval': None, 'finite_draws': 0, 'undefined_draws': 1000}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### history_gt768


Selected users: 699; active: 314; requests: 6058.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.711086 | 0.701407 | 0.701407 | undefined | 0 |
| v4 | 0.613753 | 0.570146 | 0.572288 | undefined | 0.0220292 |
| v5 | 0.640151 | 0.601946 | 0.604343 | undefined | 0.0332509 |

Equal-release gain retention: None%; UID95: [-54.60214409704192, 95.8863621958464].
Preservation margin at 95%: {'estimate': -1.1934856406728422e-05, 'percentile_95_interval': [-0.0029283204992525355, 0.0024432619911070527], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_inactive


Selected users: 467; active: 94; requests: 1967.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.619741 | 0.598058 | 0.598058 | undefined | 0 |
| v4 | 0.653649 | 0.556082 | 0.548784 | undefined | 0.051158 |
| v5 | 0.900242 | 0.777464 | 0.797658 | undefined | 0.0849756 |

Equal-release gain retention: None%; UID95: [-2010.998061336732, -77.35012550108966].
Preservation margin at 95%: {'estimate': 0.0002646961214613497, 'percentile_95_interval': [-0.009451649047308457, 0.008992874538012835], 'finite_draws': 925, 'undefined_draws': 75}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_active


Selected users: 557; active: 273; requests: 4842.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.729318 | 0.722242 | 0.722242 | undefined | 0 |
| v4 | 0.617361 | 0.57959 | 0.581425 | undefined | 0.0260932 |
| v5 | 0.597072 | 0.577235 | 0.574829 | undefined | 0.0363372 |

Equal-release gain retention: None%; UID95: [-73.27287886388062, 143.9440430312346].
Preservation margin at 95%: {'estimate': -0.0012684420711048805, 'percentile_95_interval': [-0.004300857140742275, 0.001264210768560395], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
