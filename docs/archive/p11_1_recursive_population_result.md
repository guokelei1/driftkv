# Archived: P11.1 full-population recursive lineage result

P11.1 completed all six frozen cells: M0-F and M1, seeds 17/37/71,
8,229 users per cell. Each cell replayed 2,535,994 chronological events between
the two routine releases. No future labels were read, all raw artifacts were
sealed, and Recursive Exact matched Current Full exactly (`max_abs_logit=0`).

The true recursive No-op state remains stale. The deployable partial actions
recover a positive fraction of its Bernoulli-JS error in every model/seed cell:

| Model | Action | Mean recovery across seeds | Minimum seed recovery |
|---|---:|---:|---:|
| M0-F | Hybrid-Tail128 | 51.60% | 42.96% |
| M0-F | Layer0-Full | 57.61% | 17.02% |
| M0-F | Layer0-Middle | 51.33% | 17.56% |
| M0-F | Layer0-Recent128 | 33.93% | 6.43% |
| M1 | Hybrid-Tail128 | 55.62% | 52.83% |
| M1 | Layer0-Full | 97.45% | 96.48% |
| M1 | Layer0-Middle | 77.47% | 75.94% |
| M1 | Layer0-Recent128 | 54.13% | 53.82% |

M1 is structurally stable across all three seeds. M0-F is more heterogeneous;
seed 17 has much larger recursive mean JS (`0.004498`) and weaker Layer0-only
recovery, while Hybrid-Tail128 remains positive. This seed is retained rather
than filtered.

Risk remains concentrated: the top 10% of users contribute 30.8%–54.6% of
recursive No-op JS across the six cells. That supports retaining state-level
allocation as a development hypothesis, but does not replace the already
required same-cost policy comparison.

This result establishes that version debt and legal partial recovery survive a
true recursive rolling-cache lineage at full population scale. It is still
target-free development evidence. The next operation is to apply the frozen
P10 scheduler without retuning and compare it with frozen equal-cost baselines;
theta3 and paper qualification remain untouched.

Artifacts:

- contract: `configs/contracts/p11_1_recursive_population_contract_v1.yaml`
- adjudication: `results/p11/p11_1_recursive_population_adjudication_v1.json`
- raw cells: `results/p11/p11_1_recursive_population_raw/full/`
