# Round 2: static16384_raw_strict

User target: ratio UID95% lower bound above 90%. Passed: True. Paired margin at that target: {'estimate': 0.002760964266987173, 'percentile_95_interval': [0.0012196264873180736, 0.0038801852663745418], 'finite_draws': 1000, 'undefined_draws': 0}.

The historical95% preservation test is retained below, separately from the current90% target.

# Partial-refresh quality evidence

Selected users: 3360; active: 3225; requests: 69361.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.616461 | 0.637623 | 0.637623 | 100 | 1.50229e-12 |
| v4 | 0.610137 | 0.661214 | 0.660118 | 97.8535 | 0.00327216 |
| v5 | 0.626131 | 0.690915 | 0.686592 | 93.3269 | 0.0112792 |

Equal-release gain retention: 96.04485058081607%; UID95: [93.35103566720606, 97.46633087279262].
Preservation margin at 95%: {'estimate': 0.0004772318321860855, 'percentile_95_interval': [-0.0006230439095552872, 0.0012227425731901472], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

Shared item-table preparation is charged for each run replica. Maintenance saving and deployment memory require the separate fixed30k ledger; no loaded-service timing is measured.

## history_le256

# Partial-refresh quality evidence

Selected users: 126; active: 122; requests: 2894.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.524449 | 0.529064 | 0.529064 | 100 | 3.00562e-12 |
| v4 | 0.518556 | 0.557214 | 0.554742 | 93.6051 | 0.00798665 |
| v5 | 0.570066 | 0.650129 | 0.61791 | 59.7586 | 0.0410928 |

Equal-release gain retention: 71.87297365133821%; UID95: [40.62160672032374, 97.12423234995507].
Preservation margin at 95%: {'estimate': -0.009507894960240078, 'percentile_95_interval': [-0.0155185821600133, 0.0009187391870904652], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## history_257_to768

# Partial-refresh quality evidence

Selected users: 211; active: 193; requests: 2968.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.437418 | 0.458063 | 0.458063 | 100 | 9.41925e-13 |
| v4 | 0.452381 | 0.497495 | 0.491219 | 86.0878 | 0.0129959 |
| v5 | 0.522304 | 0.594944 | 0.586506 | 88.3847 | 0.00998599 |

Equal-release gain retention: 89.36866422797289%; UID95: [79.54832134934081, 96.14284383287254].
Preservation margin at 95%: {'estimate': -0.00259791192755382, 'percentile_95_interval': [-0.005561103223610596, 0.0005338642216920546], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## history_gt768

# Partial-refresh quality evidence

Selected users: 3023; active: 2910; requests: 63499.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.621402 | 0.642557 | 0.642557 | 100 | 1.52526e-12 |
| v4 | 0.618183 | 0.670374 | 0.669554 | 98.4288 | 0.00245867 |
| v5 | 0.636759 | 0.701231 | 0.698545 | 95.8341 | 0.0102499 |

Equal-release gain retention: 97.4561601173969%; UID95: [96.46391836672757, 98.35885847163021].
Preservation margin at 95%: {'estimate': 0.0011283363714951686, 'percentile_95_interval': [0.0005323517893064761, 0.001630982509271358], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## past72h_inactive

# Partial-refresh quality evidence

Selected users: 652; active: 614; requests: 10059.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.694319 | 0.709349 | 0.709349 | 100 | 1.85585e-12 |
| v4 | 0.489216 | 0.519942 | 0.519047 | 97.0883 | 0.0056516 |
| v5 | 0.527345 | 0.606441 | 0.59377 | 83.9812 | 0.0120341 |

Equal-release gain retention: 89.13523335981884%; UID95: [80.35680842832251, 98.63161993216671].
Preservation margin at 95%: {'estimate': -0.002440757305804897, 'percentile_95_interval': [-0.006580084391329547, 0.0013993370701756516], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: False.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## past72h_active

# Partial-refresh quality evidence

Selected users: 2708; active: 2611; requests: 59302.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.605977 | 0.627963 | 0.627963 | 100 | 1.45647e-12 |
| v4 | 0.627105 | 0.679996 | 0.678934 | 97.9936 | 0.00284754 |
| v5 | 0.647646 | 0.707817 | 0.704743 | 94.8908 | 0.0110114 |

Equal-release gain retention: 96.93777122371576%; UID95: [95.79215862995858, 97.79152807396744].
Preservation margin at 95%: {'estimate': 0.0008723076274726668, 'percentile_95_interval': [0.00026085336312850406, 0.0013772280488386361], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
