# Partial-refresh quality evidence

Selected users: 1024; active: 982; requests: 19397.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.59404 | 0.607285 | 0.607285 | 100 | 0 |
| v4 | 0.576072 | 0.622903 | 0.621863 | 97.7782 | 0.00375715 |
| v5 | 0.590024 | 0.657888 | 0.655392 | 96.322 | 0.00982488 |

Equal-release gain retention: 97.23580913840844%; UID95: [94.85923713082406, 99.7889687384036].
Preservation margin at 95%: {'estimate': 0.0009535047723920123, 'percentile_95_interval': [-4.658531209289617e-05, 0.0021061704844295703], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

Theoretical FLOPs use recorded per-user service counters. Fixed30k complete costs and shared preparation are reported separately. No loaded-serving timing or throughput was measured.

## Predefined user groups

### history_le256


Selected users: 44; active: 42; requests: 1014.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.864336 | 0.864336 | 0.864336 | undefined | 0 |
| v4 | 0.356114 | 0.498559 | 0.498559 | 100 | 0.00921337 |
| v5 | 0.568213 | 0.667489 | 0.665022 | 97.5155 | 0.0175058 |

Equal-release gain retention: 98.97962023515153%; UID95: [93.460045219005, 118.97453736017128].
Preservation margin at 95%: {'estimate': 0.0032065249456804806, 'percentile_95_interval': [-0.0014250252912498976, 0.012726399940057429], 'finite_draws': 987, 'undefined_draws': 13}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### history_257_to768


Selected users: 72; active: 65; requests: 969.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.453997 | 0.462443 | 0.462443 | 100 | 0 |
| v4 | 0.549337 | 0.562601 | 0.555484 | 46.3415 | 0.0146971 |
| v5 | 0.425532 | 0.488982 | 0.490426 | 102.275 | 0.0126137 |

Equal-release gain retention: 93.3376849850386%; UID95: [11.144275797984866, 128.50902245736256].
Preservation margin at 95%: {'estimate': -0.0004718792537362973, 'percentile_95_interval': [-0.007474473007079641, 0.00635502985480913], 'finite_draws': 998, 'undefined_draws': 2}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### history_gt768


Selected users: 908; active: 875; requests: 17414.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.598349 | 0.612546 | 0.612546 | 100 | 0 |
| v4 | 0.58607 | 0.630379 | 0.629909 | 98.9399 | 0.00262494 |
| v5 | 0.60526 | 0.670234 | 0.668083 | 96.6901 | 0.00899287 |

Equal-release gain retention: 97.87798723829899%; UID95: [95.61855547657657, 100.02845286992597].
Preservation margin at 95%: {'estimate': 0.0011845703568751018, 'percentile_95_interval': [0.0001885271865236728, 0.002460896722711438], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_inactive


Selected users: 208; active: 198; requests: 2865.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.566933 | 0.576553 | 0.576553 | 100 | 0 |
| v4 | 0.587081 | 0.66241 | 0.660583 | 97.5744 | 0.00781256 |
| v5 | 0.466832 | 0.539085 | 0.537675 | 98.048 | 0.00888454 |

Equal-release gain retention: 97.94048870713839%; UID95: [91.7123283960658, 105.89582244368935].
Preservation margin at 95%: {'estimate': 0.0015408306010615408, 'percentile_95_interval': [-0.0016407947314172973, 0.004825997777220593], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_active


Selected users: 816; active: 784; requests: 16532.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.597313 | 0.610955 | 0.610955 | 100 | 0 |
| v4 | 0.583092 | 0.625898 | 0.625143 | 98.2349 | 0.00313353 |
| v5 | 0.621783 | 0.684608 | 0.681489 | 95.035 | 0.0102203 |

Equal-release gain retention: 96.75131257879336%; UID95: [94.62581284444818, 98.48865122502397].
Preservation margin at 95%: {'estimate': 0.000696281078468225, 'percentile_95_interval': [-0.00010427202141365422, 0.0016068586490763804], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
