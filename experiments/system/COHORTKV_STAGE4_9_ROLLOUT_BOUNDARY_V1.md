# CohortKV Stage 4.9 rollout boundary v1

## Status and purpose

This is the frozen design for the single-configuration correction after the Stage-4.8 scheduler
screen. The retained-prefix ABI, smoke-only runner, unit/static smoke, one real-edge GPU smoke,
and the formal 11-edge same-device paired confirmation are complete. This stage changes neither
the migration operator nor the scheduler families. It
separates two questions that the earlier growing-history protocol combined:

1. whether GPU work for newly observed behavior belongs to model-rollout migration cost;
2. whether that behavior is encoded before or after the stale prefix is migrated.

These questions are orthogonal. Moving behavior append across the rollout boundary does not decide
whether its cost is charged to migration.

## Invariant A: migration-cost accounting

Let `A_t` be the target-model incremental forward that admits the newly observed canonical window,
`U_t` the mixed migrate-or-exact update of an already existing prefix, and `E_t` current-model
exact recomputation of the same prefix. `A_t` is foreground inference work and is outside both
sides of the model-rollout timer.

The primary cost outcome is:

`sum_t(U_t) / sum_t(E_t)`.

The same physical destination, dtype, retained-prefix population, and publication boundary are
required on both sides. Physical crop/repack, migration transforms, scheduled exact refresh, and
migration-side output materialization are charged to `U_t`; matched exact prefix execution and
its output materialization are charged to `E_t`. Scheduler CPU time, recommendation scoring, and
metric computation remain outside the GPU timer.

`A_t` must still be measured in a separate ledger so that its cost cannot disappear. If a later
claim concerns final-state-ready cutover rather than the migration component, it must use a
separate system metric: measured `U_t + A_t` versus the fastest measured current-model exact path
to the same final `R_v || Delta_(v+1)` state, including matched publication. Exact cannot be
forced through a slower two-stage implementation merely to share the migration decomposition.
This system metric is not migration speedup and cannot replace `U/E`.

Cold, re-entered, or zero-overlap records have no reusable stale prefix; their construction is
reported separately and cannot enter only one side of the paired migration ratio. If a nonempty
`R_v` should be retained but its source cache is missing, rebuilding `R_v` is natural exact work
charged to the mixed update. Reusable-prefix coverage is reported with every aggregate.

This accounting rule is independent of execution order. Even if an alternate system encodes a
new window before rollout, that common foreground inference is not thereby converted into
migration work.

## Invariant B: growing-history execution order

The primary Stage-4.9 path uses post-migration, target-model append. For adjacent versions
`theta_v -> theta_(v+1)`, let `H_v` be the causally admitted history represented by the previous
actual cache, let `Delta_(v+1)` be the just-observed window admitted for the next endpoint, and let
`R_v` be the retained suffix of `H_v` after applying the maximum-length crop needed before
`Delta_(v+1)` is appended.

Each edge executes in this order:

1. Start from the previous actual mixed cache for `H_v` and derive the retained old prefix `R_v`.
   `R_v` is fixed before routing from causal history identity, the admitted window, and the
   maximum-length rule; task outcomes and scheduler choices cannot change it.
2. Start the rollout timers.
3. On the mixed branch, route each reusable `R_v` to migration or exact refresh under
   `theta_(v+1)`. On the reference branch, exactly recompute the same `R_v` under
   `theta_(v+1)`.
4. Materialize the matched target-version retained-prefix outputs and stop the rollout timers.
5. Outside those timers, append the identical `Delta_(v+1)` to both branches with
   `theta_(v+1)`, never with `theta_v`.
6. Use the resulting states to evaluate the next unseen canonical window. The mixed output becomes
   the actual recursive input to the next edge.

In compact form:

```text
mixed: previous mixed K/V(R_v) -> migrate-or-exact with theta_(v+1)
       -> stop timer -> append Delta_(v+1) with theta_(v+1)

exact: raw R_v -> exact with theta_(v+1)
       -> stop timer -> append Delta_(v+1) with theta_(v+1)
```

The exact branch's `exact(R_v) + append(Delta)` output must match a single fresh
`theta_(v+1)` forward on `R_v || Delta_(v+1)` within the declared numerical tolerance for K/V,
hidden state, and task output; the one-shot fresh forward is the quality authority if they
disagree. The mixed branch is not claimed to be mathematically exact: migrated deep-layer state
may affect the subsequent append, and physically dropping earlier K/V rows does not remove their
historical influence from retained deep-layer state. Final AUC/NDCG@100/Hit@100 measure that
consequence.

## Relationship to Stage 4.7 and Stage 4.8

Stage 4.7 and Stage 4.8 v1 used the different sequence

```text
crop old cache -> append the newly admitted window with theta_v
-> migrate-or-exact the resulting longer prefix
```

and reported foreground-inclusive lifecycle ratios in addition to update-only cost. Those
artifacts remain valid only under their recorded v1 semantics and must not be overwritten or
silently relabeled.

That source-append-first order can describe a different hot-cache regime in which `theta_v`
continuously served and cached the new behavior before `theta_(v+1)` was published. It is retained
as an alternate systems boundary, not mixed with the post-migration-append primary protocol.

In particular, the Stage-4.8 `token_debt/total10` value `U/E=0.110699` is a useful old-order
diagnostic, but it is not a Stage-4.9 result: both the synchronized prefix and the matched exact
denominator change under post-migration append. Stage 4.9 therefore requires a same-device paired
exact measurement and cannot reuse the frozen `346319.0015 ms` Stage-4.7 denominator.

The first correction run stays on the existing KuaiRand seed-0, 16L/H512, maximum-2,048,
682-record configuration. It carries forward only explicitly selected Stage-4.8 scheduler
candidates, runs them sequentially on one physical GPU, and does not reopen the 16-point scheduler
search.

## Implementation checkpoint

The minimum implementation is:

- `src/hstu_kvcache/migration/rollout.py`: a label-free retained-prefix plan that separates
  `R_v`, `Delta_(v+1)`, and the latest query token before routing;
- `scripts/run_cohortkv_stage4_9_rollout_boundary.py`: static and real one-edge smoke modes only;
  it has no default or formal full-chain mode;
- `tests/test_rollout_boundary.py` and `tests/test_stage49_runner.py`: retained-prefix arithmetic,
  recursive mixed-state use, timing-ledger, CLI, and exact-equivalence checks.

The real one-edge smoke uses five records from `theta0 -> theta1` and covers two migrated records,
one scheduled exact record, one natural zero-overlap exact record, and one deliberately removed
expected cache. The missing cache performs target-model exact reconstruction of `R_v` inside
`U`, and the same record enters the paired exact population. Every timed retained endpoint is
materialized as device-resident FP16; a separate FP32 branch is used only for numerical parity.
The smoke makes zero source-model append calls, and every behavior append uses `theta1` after
retained-prefix update. Two-stage `exact(R_v) + append(Delta)` agrees with one-shot target-model
exact within maximum absolute differences of `6.20e-6` for K, `6.68e-6` for V, `4.30e-6` for
hidden state, and `1.43e-6` for scores; Top-100 is identical.

The latest-only path is covered separately by a synthetic one-token equivalence test because the
real first edge has no such record. The full trace does contain latest-only records on later
edges, so the formal runner must keep that path. In recursive execution, expected cache IDs come
from the prior commit contract while present IDs come from actual store contents; they must not
be inferred from the same set.

This smoke has no warmup, writes no result artifact, and is explicitly marked
`scientific_result=false`. Its timings are not performance evidence. In particular, it does not
confirm either selected scheduler, the full recursive 11-edge quality result, or a new `U/E`
ratio. Those claims come only from the formal result below.

## Formal same-device result

`scripts/run_cohortkv_stage4_9_formal_confirmation.py` runs the two frozen candidates
sequentially on one NVIDIA A40, executes a fresh paired exact reference for every edge, and
recursively consumes the previous actual post-append mixed cache. It does not rerun the
Stage-4.8 16-point search.

| Candidate | `sum(U)/sum(E)` | Scheduled exact / reusable | Record-weighted AUC recovery | NDCG@100 recovery | Hit@100 recovery | Role |
|---|---:|---:|---:|---:|---:|---|
| `token_debt_total10` | 0.071319 | 221 / 6,711 | 1.000030 | 0.996890 | 0.999060 | Cost endpoint |
| `staggered_renewal_h12` | 0.100017 | 462 / 6,711 | 1.000039 | 0.997463 | 1.000000 | Frozen deployment candidate |

Both raw artifacts and the same-device aggregate pass all 11-edge execution, new exact
denominator, post-migration target append, zero source-model append, FP32 exact-equivalence,
recursive lineage, provenance, and capacity checks. Recommendation labels do not select either
action. H12 is retained because it is the preregistered policy with a per-cache renewal horizon;
token debt has lower measured cost but no per-cache deadline.

The evaluator keeps the persistent FP16 recursive store on CPU and stages one group at a time.
H12 reports 662,869,804,944 logical H2D/D2H bytes separately outside `U/E`. This is a
device-resident retained-prefix result, not a full-cohort HBM-resident lifecycle or end-to-end
state-movement result. The measured trajectory has 11 updates against H12's 12-edge horizon, so
it does not observe a complete renewal cycle; maximum observed migration depth is 11.

Formal artifacts:

- `results/system/cohortkv_single_config_full_chain_v1/stage4_9_token_debt_total10_seed0.json`;
- `results/system/cohortkv_single_config_full_chain_v1/stage4_9_staggered_renewal_h12_seed0.json`;
- `results/system/cohortkv_single_config_full_chain_v1/stage4_9_same_device_confirmation_seed0.json`.

## Stage 5 handoff

The formal result admits and binds Stage 5. The guard hook is
`post_retained_prefix_pre_append`; `R_v` remains a private intermediate and cannot be published.
Only `post_append_full_cache` can be committed or used as the next recursive state. Formal
scheduler integration selects `staggered_renewal_h12` and is completed in
`COHORTKV_STAGE5_MINIMAL_CLOSURE_V1.md`, not the former runtime-sentinel or capsule-economics
matrix.
