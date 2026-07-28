# CohortKV Stage 4.8 scheduler sweeps v1

> Status: all sixteen preregistered development points completed on 2026-07-28. This document
> remains the frozen v1 protocol. Its old-model pre-migration append and foreground-inclusive
> lifecycle diagnostics must not be reinterpreted as the corrected rollout boundary in
> [Stage 4.9](COHORTKV_STAGE4_9_ROLLOUT_BOUNDARY_V1.md).

## Status and purpose

Stage 4.8 is a preregistered, single-configuration development sweep over refresh schedulers for
the organic growing-history lifecycle. It keeps the Stage-4.7 checkpoints, compiler programs,
causal windows, users, task endpoints, recursive-cache contract, and exact task reference fixed.
It changes only the decision that routes a reusable prefix to lightweight migration or
current-model exact replay.

The purpose is to measure a recommendation-quality versus GPU-lifecycle-cost frontier below the
Stage-4.7 20% reusable-record policy. It is not a search for a hidden K/V-error threshold, a new
cache-fidelity certificate, or permission to retain only the best-looking run.

## Frozen development scope

- dataset: KuaiRand-1K;
- stream: the same 4+12 canonical-date trajectory as Stage 4.7;
- training seed: 0;
- model: 16 layers, hidden/K/V width 512, maximum history 2,048;
- cohort: the same deterministic 682 users and frozen role assignment;
- execution: `theta0` exact followed by eleven recursive transitions that consume the previous
  mixed cache actually published by the policy;
- hardware: four NVIDIA A40 GPUs for parallel development screening;
- expansion explicitly excluded: additional datasets, model sizes, training seeds, storage
  backends, GPU counts, or compiler variants.

All causal-history, checkpoint, program, candidate-catalog, role-isolation, previous-actual-input,
lineage-coverage, and action-coverage checks from Stage 4.7 remain protocol-validity requirements.
They are not scientific admission metrics: a run that fails one is invalid and must not be
assigned an outcome.

## Frozen exact reference

The exact reference is the completed Stage-4.7 artifact:

`results/system/cohortkv_single_config_full_chain_v1/stage4_7_organic_full_chain_seed0.json`

Its frozen descriptor is:

- result protocol: `cohortkv_single_config_organic_recursive_chain_v1`;
- experiment protocol: `cohortkv_single_config_organic_lifecycle_v1`;
- SHA256:
  `e635c18844d05ce3948f6c8ac0ff3bd84e94c4b3a042bd4b04892198007c249d`;
- exact-prefix GPU time over eleven updates:
  `346319.0015463829 ms`;
- exact task endpoints: the twelve per-endpoint all-exact outputs bound to the same histories,
  checkpoints, candidates, positives, and role assignment.

The extracted baseline is frozen at
`configs/cohortkv_single_config_v1/stage4_8_exact_baseline.json`, SHA256
`29c0c3d6a6cee5521fe52d0d65ba4a96f739a58c7bf80b52858ef8ee781a6b51`. Every sweep process
must verify both fixed hashes, reconstruct the per-edge denominators and per-endpoint exact task
values from the Stage-4.7 result, reconstruct the incumbent gates from the frozen Stage-4.7
summary, and verify all workload/provenance identities before execution. It then reuses the
frozen exact task outputs and exact-prefix GPU denominator. It must not execute the independent
all-exact prefix reference again.

This omission applies only to the comparison baseline. Natural exact replay for cold, re-entered,
or zero-overlap records and selector-scheduled exact replay are still real actions in the mixed
chain. They must execute because their outputs become recursive state for later versions.

## Scientific outcomes

Only the following final outcomes determine the Stage-4.8 quality-cost frontier:

1. record-weighted catalog AUC relative to the frozen all-exact endpoint outputs;
2. record-weighted NDCG@100 relative to the frozen all-exact endpoint outputs;
3. record-weighted Hit@100 relative to the frozen all-exact endpoint outputs;
4. measured cumulative symmetric GPU lifecycle cost;
5. measured cumulative common-inclusive GPU lifecycle cost.

The three task metrics use the same final-role positive record-endpoints as Stage 4.7 and are
aggregated by record count, not by an unweighted mean over windows. Full recomputation is not
assumed to be a task-quality upper bound; ratios above one and signed mixed-minus-exact
differences remain valid.

K/V fidelity, K/V error, score cosine, top-100 overlap, norm shift, accumulated scheduler debt,
per-edge model drift, migration depth, and per-step action balance may be recorded for diagnosis
and mechanism inspection. None has an admission threshold, none may veto a point whose final task
and lifecycle outcomes are valid, and no correlation between one of these diagnostics and the
three recommendation metrics is claimed or required.

## Lifecycle boundary and current-cost gate

Each new mixed run measures its own foreground history maintenance, migration, routing,
selector-scheduled exact work, unavoidable natural exact work, publication, and common latest
token work on its assigned GPU. Compiler fit/compile, scheduler CPU time, catalog scoring,
task-metric computation, and host-to-device sequence construction remain outside the GPU boundary,
matching Stage 4.7.

Let:

- `F` be the run's measured foreground append/eviction work;
- `U` be its measured mixed-prefix update and publication work, including every exact action that
  the mixed policy actually performs;
- `C` be its measured common latest-token execution and publication;
- `E_frozen = 346319.0015463829 ms` be the frozen cumulative exact-prefix denominator.

The reported lifecycle ratios are:

- symmetric: `(F + U) / (F + E_frozen)`;
- common-inclusive: `(F + U + C) / (F + E_frozen + C)`.

Stage 4.7 measured `0.5069011719265762` symmetric and `0.5372231748138118`
common-inclusive. Every Stage-4.8 operating point is required to be strictly below both values to
pass the current-cost gate. A configured budget below 20% is not accepted as proof of lower GPU
cost; only the measured ratios decide this gate. Update-only cost and per-step maxima remain
diagnostics rather than substitutes for the two lifecycle outcomes.

No scalar combination of AUC, NDCG, Hit, and cost is defined. Among valid points that pass both
current-cost bounds, Pareto dominance is evaluated using lower lifecycle cost and higher values of
all three record-weighted recommendation metrics. Stage 4.8 does not impose a post-hoc task-quality
threshold.

## Common scheduler state

Every family maintains only information available at the current release:

- the version of the last exact refresh;
- a deterministic renewal phase, age, or service debt;
- the current canonical-history overlap and prefix length;
- an optional edge-level, label-free displacement computed directly from the currently deployed
  adjacent program.

Natural exact and selector exact update the affected record's renewal, age, or debt state exactly
as declared below. Migration carries that state to the next edge. Recommendation labels,
next-window positives, future checkpoint changes, future edge severities, exact final-role
candidate errors, and final-role task outcomes are prohibited from routing.

All ties use a stable record identity only after the declared priority keys are equal. Stable
identity is not a sampling policy. None of the four families has a separately tuned maximum
migration count or a fixed four-update reset deadline.

Every family makes its optional exact decisions before candidate execution. A record routed to
exact therefore does not also execute and discard a migration candidate.

## Preregistered family A: work-balanced staggered renewal

`work_balanced_staggered_renewal` is a deterministic renewal scheduler. On the first
`theta0 -> theta1` update, after causal foreground construction has produced the actual routable
prefix workload, reusable records are sorted by descending prefix tokens and then record identity.
Longest-processing-time placement assigns each record to the currently least-loaded one of `H`
phase bins, with phase index breaking a bin-load tie. The record's first due version is its initial
phase offset from the first update. Phase zero is due on that first edge.

When a scheduled record becomes due, it receives exact replay and its next due version advances by
exactly `H`. A natural exact action has already renewed the record on that edge and resets its next
due version to `current_target_version + H`. A migrated record retains its due version.

The renewal period is an external service-cost operating point, while LPT phase placement spreads
the initial exact token work instead of allowing all records to reach a common deadline. For a
continuously reusable record, the scheduled exact gap is `H`; there is no independent maximum-age
parameter, per-cache error threshold, random draw, or hash-based phase.

The four preregistered renewal horizons are:

| Tier | Renewal horizon `H` | Nominal scheduled rate |
|---|---:|---:|
| A1 | 8 | 12.5% |
| A2 | 10 | 10.0% |
| A3 | 12 | 8.33% |
| A4 | 16 | 6.25% |

All four horizons and their complete results are retained.

## Preregistered family B: total exact-token cumulative-debt SLA

`total_token_cumulative_debt` makes complete exact prefix tokens, including unavoidable natural
exact work, the resource. For edge `t`, let `W_t` be all resident prefix tokens and `N_t` the
natural-exact prefix tokens. With SLA fraction `b`, the signed global token balance is updated
before optional selection:

`G_t = G_(t-1) + b * W_t - N_t`.

Natural exact therefore consumes the SLA first. If it exceeds the accrued allowance, `G_t` becomes
negative and future allowance repays the deficit; natural correctness work is never suppressed.
Optional exact is permitted only while the balance is positive, and every scheduled exact prefix
is deducted from it.

Each reusable record with prefix-token cost `c_i` accrues deterministic service debt
`d_i <- d_i + b * c_i`. Eligible records are ordered by descending `d_i / c_i`, then descending
`age_i + 1`, then record identity. While the signed global balance is positive, the scheduler
serves the highest-priority record's complete prefix and deducts all `c_i` tokens even when that
makes the balance negative. Selection stops once the balance is nonpositive. Consequently an edge
can add at most one complete-prefix overshoot, and later SLA accrual repays that deficit before
more optional work is admitted. A scheduled exact action applies `d_i <- d_i - c_i`; natural exact
sets `d_i` to zero. Signed balance and record debt carry to the next edge.

This one-prefix borrowing rule treats exact replay as packetized service: a prefix is indivisible,
so a positive token balance authorizes the next complete packet rather than requiring an exact
fit. It prevents a long highest-priority prefix from causing head-of-line idle capacity and does
not bypass it in favor of a shorter, lower-priority record. The bounded packet overshoot changes
only finite-edge granularity; the long-run total exact-token SLA remains `b`.

The four total exact-token SLA tiers are:

| Tier | Cumulative total exact-token SLA `b` |
|---|---:|
| B1 | 10% |
| B2 | 12% |
| B3 | 14% |
| B4 | 16% |

The signed balance distinguishes this family from an independent per-edge cap: an organic
natural-exact burst is charged rather than hidden, and optional refresh subsequently adapts
without a hand-chosen reset threshold.

## Preregistered family C: AoI MaxWeight under a reusable-token budget

`aoi_maxweight` treats time since exact service as Age of Information. Before routing, record `i`
has post-migration age `a_i = migration_age_i + 1` and current complete prefix-token cost `c_i`.
Its deterministic MaxWeight priority is:

`I_i = a_i * (a_i + 1) / (2 * c_i)`.

The numerator is the triangular accumulated age that exact service removes; division by prefix
tokens prices that reduction by exact work. The optional capacity receives
`beta * sum_i(c_i)` reusable-prefix tokens per edge and carries a signed balance across edges.
Records are ordered by descending `I_i`, then descending `a_i`, then record identity. While the
balance is positive, the scheduler serves the highest-priority complete prefix and deducts its
full token cost even if this makes the balance negative. Selection then stops. Thus at most the
last prefix creates one complete-prefix overshoot on an edge, and later reusable-token accrual
repays it before additional optional service.

This is the same packetized deficit-service convention as family B. It avoids head-of-line
idling for heterogeneous prefix lengths without skipping an old, high-index record merely because
a younger short record fits the remaining fractional allowance. The carried deficit preserves the
long-run reusable-token budget `beta`; borrowing is not an additional refresh budget.

Natural exact does not consume this optional reusable-token budget and naturally resets age.
Under a finite cohort and bounded prefix length, the quadratic age term eventually raises an
unserved record's index, providing liveness without a fixed maximum migration count.

The four reusable-prefix token budgets are:

| Tier | Optional reusable-token budget `beta` |
|---|---:|
| C1 | 4% |
| C2 | 7% |
| C3 | 10% |
| C4 | 13% |

These tiers are operating points on one MaxWeight policy, not four fitted error thresholds.

## Preregistered family D: label-free model-time staggered renewal

`model_time_staggered_renewal` replaces integer version time with a causal clock derived from the
deployed adjacent direct program. For edge `t`, let `r_W` be the global RMS of the direct
program's weight displacement from identity and `r_b` the global RMS of its bias, each aggregated
over all layers and elements. The label-free severity is fixed as
`g_t = sqrt(r_W^2 + r_b^2)`. It uses only the current serialized program; it does not use
fit/final task labels, final-role candidate error, or a future program.

The causal mean includes the current edge:

`mu_t = mean(g_0, ..., g_t)`.

The model-time increment is zero when `mu_t` is zero and otherwise
`Delta_tau_t = g_t / mu_t`. The clock advances as
`tau_after = tau_before + Delta_tau_t`. Thus an average-sized release advances approximately one
unit, a larger release advances more, and repeated small releases still accumulate.

Initial reusable records use the same deterministic first-edge routable-prefix LPT placement as
family A.
Phase `p` receives initial due time `tau_before + p + 1`. If an edge crosses a record's due time,
the record receives exact replay once and its due time advances by `H` repeatedly until it lies
strictly after `tau_after`; it never executes exact more than once on one edge. Natural exact
resets due time to `tau_after + H`.

The four preregistered model-time renewal horizons are:

| Tier | Model-time horizon `H` |
|---|---:|
| D1 | 8 |
| D2 | 10 |
| D3 | 12 |
| D4 | 16 |

This family responds to model-release magnitude without a learned risk threshold or a fixed
version-count deadline. The direct-program severity formula, causal update order, LPT phases, and
horizons are frozen before task evaluation.

## Four-GPU execution and confounding

Each family has one launcher and four independent worker outputs. Within that launcher, the
mapping is fixed:

| GPU | Tier |
|---|---|
| `cuda:0` | family tier 1 |
| `cuda:1` | family tier 2 |
| `cuda:2` | family tier 3 |
| `cuda:3` | family tier 4 |

Exactly one worker owns each GPU. Output paths, temporary paths, CUDA devices, and process logs
must be unique. Device arguments must be four distinct explicit and available CUDA indices;
implicit aliases such as `cuda` are invalid. Families are launched separately; the four tiers of
one family may run concurrently.

This design perfectly confounds parameter tier with physical GPU. The four A40s have the same
model but are not assumed to have identical clocks, thermals, allocator state, or background
load. Consequently these parallel sweeps are development screens: small cost differences between
tiers or families are not paper evidence, and no variance estimate may be inferred from the four
devices.

Before a Stage-4.8 point becomes a paper result, every retained Pareto candidate must be rerun
sequentially on the same physical GPU under a separate confirmation protocol. That confirmation
must pair the candidate with a same-device exact denominator or an independently certified
same-device frozen denominator. The four-GPU screen alone cannot support a precise speedup claim.

## Retention, selection, and stopping rules

All sixteen preregistered points are preserved regardless of quality, cost, or failure. A point
that misses the current-cost gate is a measured negative result, not a reason to change its budget.
K/V or norm-shift diagnostics cannot be used to discard a point.

Stage 4.8 produces complete per-family quality-cost curves and a nondominated set; it does not
name or retain only one winner. Any later choice of a deployment default must be made under an
explicit external compute SLA and frozen before new-seed or cross-dataset evaluation. Changing a
family formula, budget grid, edge-displacement definition, role usage, exact reference, or
measurement boundary after observing these sixteen task outputs requires a new protocol version.

No Stage-5 admission, generalization claim, or paper-level speedup follows directly from this
single-configuration development sweep.

## Execution entry points

Run one family at a time from the repository root. Each command launches its four preregistered
tiers concurrently on `cuda:0..3`, writes one log and one result per tier, waits for all workers,
then writes a family summary:

```bash
python scripts/run_cohortkv_stage4_8_staggered_renewal_sweep.py
python scripts/run_cohortkv_stage4_8_token_debt_sweep.py
python scripts/run_cohortkv_stage4_8_aoi_maxweight_sweep.py
python scripts/run_cohortkv_stage4_8_model_time_renewal_sweep.py
```

The sequential wrapper runs the same four launchers in the declared order. It waits for one
family's four workers and summary to complete successfully before starting the next family, and
stops immediately on failure:

```bash
python scripts/run_cohortkv_stage4_8_all_sweeps.py
```

Do not launch two family commands concurrently. A completed worker is skipped on rerun only when
its frozen input, exact baseline, implementation hashes, repository commit, variant, and assigned
physical device all match the current invocation. Any other existing output fails closed;
`--force` is required to replace it. `--smoke-test` performs the CPU/provenance/policy preflight.
`--runtime-smoke-test --grid-index 0 --device cuda:N` runs one real edge without writing a formal
result.
