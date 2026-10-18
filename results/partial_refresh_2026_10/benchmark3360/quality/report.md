# Partial-refresh quality evidence

Selected users: 3360; active: 3225; requests: 69361.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.616461 | 0.637623 | 0.637623 | 100 | 0 |
| v4 | 0.610137 | 0.661214 | 0.660118 | 97.8534 | 0.00327216 |
| v5 | 0.626131 | 0.690915 | 0.686592 | 93.3269 | 0.0112792 |

Equal-release gain retention: 96.04483090243183%; UID95: [93.35100484750184, 97.46633101837787].
Preservation margin at 95%: {'estimate': 0.000477222844153212, 'percentile_95_interval': [-0.0006230369795040903, 0.0012227474800435931], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

Theoretical FLOPs use recorded per-user service counters. Fixed30k complete costs and shared preparation are reported separately. No loaded-serving timing or throughput was measured.

## Predefined user groups

### history_le256


Selected users: 126; active: 122; requests: 2894.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.524449 | 0.529064 | 0.529064 | 100 | 0 |
| v4 | 0.518556 | 0.557214 | 0.554742 | 93.6051 | 0.00798665 |
| v5 | 0.570066 | 0.650129 | 0.61791 | 59.7586 | 0.0410928 |

Equal-release gain retention: 71.87297365133821%; UID95: [40.62160672032374, 97.12423234995507].
Preservation margin at 95%: {'estimate': -0.009507894960240078, 'percentile_95_interval': [-0.0155185821600133, 0.0009187391870904652], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### history_257_to768


Selected users: 211; active: 193; requests: 2968.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.437418 | 0.458063 | 0.458063 | 100 | 0 |
| v4 | 0.452381 | 0.497495 | 0.491219 | 86.0878 | 0.0129959 |
| v5 | 0.522304 | 0.594944 | 0.586506 | 88.3847 | 0.00998598 |

Equal-release gain retention: 89.36866422797289%; UID95: [79.54832134934081, 96.14284383287254].
Preservation margin at 95%: {'estimate': -0.00259791192755382, 'percentile_95_interval': [-0.005561103223610596, 0.0005338642216920546], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### history_gt768


Selected users: 3023; active: 2910; requests: 63499.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.621402 | 0.642557 | 0.642557 | 100 | 0 |
| v4 | 0.618183 | 0.670374 | 0.669554 | 98.4288 | 0.00245867 |
| v5 | 0.636759 | 0.701231 | 0.698545 | 95.8341 | 0.0102499 |

Equal-release gain retention: 97.45614718145693%; UID95: [96.46387496502682, 98.35876552594044].
Preservation margin at 95%: {'estimate': 0.0011283304288485715, 'percentile_95_interval': [0.0005323392068626142, 0.0016309664309813669], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_inactive


Selected users: 652; active: 614; requests: 10059.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.694319 | 0.709349 | 0.709349 | 100 | 0 |
| v4 | 0.489216 | 0.519942 | 0.519047 | 97.0883 | 0.0056516 |
| v5 | 0.527345 | 0.606441 | 0.59377 | 83.9812 | 0.0120341 |

Equal-release gain retention: 89.13523335981884%; UID95: [80.35680842832251, 98.63161993216671].
Preservation margin at 95%: {'estimate': -0.002440757305804897, 'percentile_95_interval': [-0.006580084391329547, 0.0013993370701756516], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
### past72h_active


Selected users: 2708; active: 2611; requests: 59302.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.605977 | 0.627963 | 0.627963 | 100 | 0 |
| v4 | 0.627105 | 0.679996 | 0.678934 | 97.9935 | 0.00284754 |
| v5 | 0.647646 | 0.707817 | 0.704743 | 94.8908 | 0.0110114 |

Equal-release gain retention: 96.93774644182956%; UID95: [95.79196004974922, 97.79154271967467].
Preservation margin at 95%: {'estimate': 0.0008722964716519552, 'percentile_95_interval': [0.00026083404172530446, 0.0013772308803036847], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
