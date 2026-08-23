# Archived: P11.4 recursive policy rolling-quality result

All six M0-F/M1 x seed 17/37/71 cells completed. Each contains 3,015 users and
18,959 heldout explicit-feedback requests. Raw action logits were sealed before
joining the already sealed P11.2 assignments. Recursive Exact matched Current
Exact exactly at the edge-2 state (`max KV difference = 0`).

The target-free fidelity result transfers to quality most clearly for M1. With
the primary 1% probe:

| Budget | M1 No-op minus Policy log-loss (seeds 17/37/71) | Mean | Positive seeds |
|---|---:|---:|---:|
| 5% | +0.000225 / -0.000048 / +0.000223 | +0.000133 | 2/3 |
| 10% | +0.000209 / +0.000008 / +0.000195 | +0.000137 | 3/3 |
| 25% | +0.000102 / +0.000121 / +0.000673 | +0.000299 | 3/3 |

M1 ROC-AUC also improves in 3/3 seeds at 10% and 25% budgets (mean absolute
improvements `0.000483` and `0.001641`). M0-F is more heterogeneous: aggregate
log-loss improvement is positive at all three budgets, but only 1/3--2/3 seeds
are positive depending on budget. The large M0-F seed-17 fidelity cell also has
the clearest quality recovery; it is retained alongside the other seeds.

The rare-class caveat remains mandatory. Dislike-only log-loss worsens for two
of three seeds in both models at every primary budget. Exact-All itself often
has the same direction of dislike-only degradation versus Recursive No-op, so
much of this is attributable to the current model's complete execution semantics
rather than solely to the mixed scheduler. Nevertheless, EvoKV may not claim
that aggregate fidelity guarantees every quality slice.

The correct development conclusion is:

> The frozen state-level scheduler retains a strong same-cost fidelity advantage
> under true recursive lineage and can recover small, repeatable aggregate
> quality effects for M1, while quality gains are not universal across model,
> seed, metric or rare-class slice.

This completes the recursive-lineage development validation. It does not open
theta3 or upgrade the result to paper qualification.

Artifacts:

- raw seal: `results/p11/p11_4_recursive_policy_quality_raw_seal_v1.json`
- adjudication: `results/p11/p11_4_recursive_policy_quality_v1.json`
