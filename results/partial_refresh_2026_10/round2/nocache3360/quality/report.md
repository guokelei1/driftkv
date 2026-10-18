# Round 2: nocache_full

User target: ratio UID95% lower bound above 90%. Passed: True. Paired margin at that target: {'estimate': 0.0045674699584529604, 'percentile_95_interval': [0.0031719485528982575, 0.005751586467914578], 'finite_draws': 1000, 'undefined_draws': 0}.

The historical95% preservation test is retained below, separately from the current90% target.

# Partial-refresh quality evidence

Selected users: 3360; active: 3225; requests: 69361.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.616461 | 0.637623 | 0.637623 | 100 | 1.50694e-12 |
| v4 | 0.610137 | 0.661214 | 0.661214 | 100 | 2.29317e-13 |
| v5 | 0.626131 | 0.690915 | 0.690915 | 100 | 2.26654e-13 |

Equal-release gain retention: 100.00001114152145%; UID95: [99.99994788027281, 100.00008134996254].
Preservation margin at 95%: {'estimate': 0.002283737523651873, 'percentile_95_interval': [0.0015859808710044423, 0.0028757931415829435], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

Shared item-table preparation is charged for each run replica. Maintenance saving and deployment memory require the separate fixed30k ledger; no loaded-service timing is measured.

## history_le256

# Partial-refresh quality evidence

Selected users: 126; active: 122; requests: 2894.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.524449 | 0.529064 | 0.529064 | 100 | 2.57128e-12 |
| v4 | 0.518556 | 0.557214 | 0.557214 | 100 | 2.71824e-13 |
| v5 | 0.570066 | 0.650129 | 0.650129 | 100 | 3.75753e-13 |

Equal-release gain retention: 100.00000000000001%; UID95: [99.99999999999999, 100.00000000000001].
Preservation margin at 95%: {'estimate': 0.0020555809503780484, 'percentile_95_interval': [0.0006092515728399021, 0.0034253056386796205], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## history_257_to768

# Partial-refresh quality evidence

Selected users: 211; active: 193; requests: 2968.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.437418 | 0.458063 | 0.458063 | 100 | 9.43184e-13 |
| v4 | 0.452381 | 0.497495 | 0.497495 | 100 | 1.14304e-13 |
| v5 | 0.522304 | 0.594944 | 0.594944 | 100 | 1.20046e-13 |

Equal-release gain retention: 100.0%; UID95: [99.99999999999999, 100.00000000000001].
Preservation margin at 95%: {'estimate': 0.00230665692184312, 'percentile_95_interval': [0.0009320253108589545, 0.003835099972749001], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## history_gt768

# Partial-refresh quality evidence

Selected users: 3023; active: 2910; requests: 63499.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.621402 | 0.642557 | 0.642557 | 100 | 1.5344e-12 |
| v4 | 0.618183 | 0.670374 | 0.670374 | 100 | 2.36263e-13 |
| v5 | 0.636759 | 0.701231 | 0.701231 | 100 | 2.33522e-13 |

Equal-release gain retention: 100.00001615554481%; UID95: [99.99995987354825, 100.00010952469076].
Preservation margin at 95%: {'estimate': 0.002296959406841874, 'percentile_95_interval': [0.0015489883528037472, 0.002840145540944608], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## past72h_inactive

# Partial-refresh quality evidence

Selected users: 652; active: 614; requests: 10059.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.694319 | 0.709349 | 0.709349 | 100 | 1.82987e-12 |
| v4 | 0.489216 | 0.519942 | 0.519942 | 100 | 1.78353e-13 |
| v5 | 0.527345 | 0.606441 | 0.606441 | 100 | 1.51975e-13 |

Equal-release gain retention: 100.0%; UID95: [99.99999999999999, 100.00000000000001].
Preservation margin at 95%: {'estimate': 0.0020808648114680195, 'percentile_95_interval': [0.0011900596308383805, 0.002925878091724457], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.

## past72h_active

# Partial-refresh quality evidence

Selected users: 2708; active: 2611; requests: 59302.

Full denotes all current compact codes. Native first-request rebuilding is a separate reference.

| Release | Reuse AUC | Full-code AUC | Partial AUC | Gain retained (%) | Relative logit MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.605977 | 0.627963 | 0.627963 | 100 | 1.46509e-12 |
| v4 | 0.627105 | 0.679996 | 0.679996 | 100 | 2.38412e-13 |
| v5 | 0.647646 | 0.707817 | 0.707817 | 100 | 2.53154e-13 |

Equal-release gain retention: 100.000009256155%; UID95: [99.99993751143847, 100.0000883188538].
Preservation margin at 95%: {'estimate': 0.0022508055430890614, 'percentile_95_interval': [0.0015219299254065305, 0.002856080278532045], 'finite_draws': 1000, 'undefined_draws': 0}.
Noninferiority supported at this threshold: True.

Selected user panel and fixed models; no training-seed replication or adjustment for exploratory selection. A positive point retention is not a noninferiority result.
