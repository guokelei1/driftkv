# Documentation index

The active documents describe one three-layer EvoKV architecture:

```text
semantic ActionPlan → distributed WavePlan constraints → capacity-bounded ResidencyPlan
```

Old design exploration is recoverable from Git history. Historical experiment records remain
tracked under `experiments/` because their measurements and negative results are useful, but they
do not define the current design.

## Precedence

When documents disagree, use this order:

1. [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md) — authoritative thesis,
   supported claims, current status, open work, and stop conditions.
2. [eval_protocol.md](eval_protocol.md) — authoritative result-family, timer, workload, and
   comparability boundary.
3. [future_design/DESIGN2_FINAL_PLAN.md](future_design/DESIGN2_FINAL_PLAN.md) — current D2
   mechanism and current D1→D2 starting interface.
4. [future_design/DESIGN2_DEVELOPMENT_STATUS.md](future_design/DESIGN2_DEVELOPMENT_STATUS.md) —
   implemented D2 state, non-scientific development evidence, pending W4/formal-evaluation work,
   and the D3 handoff.
5. [future_design/DESIGN3_FUTURE_DIRECTION.md](future_design/DESIGN3_FUTURE_DIRECTION.md) — D3
   problem definition, source/capacity/timer contract, candidate mechanisms, baselines, and
   go/no-go conditions.
6. [future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md)
   — executable two-card foundation benchmark, staged baseline closure, candidate search, and
   backtracking plan; it creates no protocol or evidence.
7. [09_single_configuration_full_chain_plan.md](09_single_configuration_full_chain_plan.md) —
   frozen D1 Stage 0–6 evidence ledger. It is history, not a current execution plan.
8. [dataset_expansion_audit.md](dataset_expansion_audit.md) — accepted and rejected dataset
   semantics, usable capacity, and generality boundaries.

Repository-local agent rules are in [../AGENTS.md](../AGENTS.md). Experiment records are indexed
by [../experiments/README.md](../experiments/README.md), active design documents by
[future_design/README.md](future_design/README.md), and paper/reference material by
[../paper/README.md](../paper/README.md).

## Current state

| Layer | Question | Output | Status |
|---|---|---|---|
| D1 | What should be translated, progressively repaired, or exactly recomputed? | immutable `ActionPlan` | frozen algorithm and single-configuration evidence |
| D2 | Where and in what physical distributed form should those fixed actions execute? | global D3-facing `WavePlan` constraints | mechanisms implemented; normalized exporter/hash and formal evidence open |
| D3 | How should an out-of-core two-GPU stack move and execute K/V? | initially a capacity-specific schedule; final interface open | flexible GPU0/GPU1 benchmark plan ready; implementation and evidence not started |

D1 resolves the semantic reuse–recompute trade-off. D2 converts the resulting logical sparsity into
physical savings through owner-local retained repair, row-sharded exact/append, `(S,R)`-aware
compiled ordering, merged exact pools, segmented destinations, and atomic publication. D3 begins
with an isolation track that preserves the current upstream snapshot while scheduling
ordinary-DRAM→GPU→ordinary-DRAM micro-waves. Mechanism discovery may also create a new globally
planned D1/D2/D3 `stack_revision`; it must rerun its own baselines and cannot be presented as a
ResidencyPlan-only ablation.

## What may be used as evidence

- A checked aggregate is valid only within its recorded protocol.
- Training seed is the replication unit. Users or samples inside one trained model are diagnostics,
  not independent repeats.
- Result families with different protocol strings must remain separate.
- D2 W3 integrated timings, full-payload validation, and synthetic lookup contention are
  `scientific_result=false` mechanism discovery.
- D3 currently has no result family. Destination-v4 correctness, normalized-capsule DRAM results,
  and hot-HBM D1 results cannot be renamed as D3 evidence.
- The frozen Markdown manuscript under `paper/cohortkv/` is a Stage-6 artifact dependency, not the
  current EvoKV manuscript or design source of truth.

## Historical evidence map

The records below remain useful within their own protocols:

- `experiments/validity/` — cache semantics, structural baselines, and early negative routes;
- `experiments/scaling/` and `experiments/motivation/` — scale, capacity, and workload
  characterization;
- `experiments/exposure/` — ordered-exposure dataset and cache-version controls;
- `experiments/migration/` — progressive, compiled, and cohort-tiered D1 method development;
- `experiments/system/` — D1 runtime/evidence chain and historical destination prototypes.

Some filenames retain `CohortKV` or `StreamKV`. Those names identify frozen protocols and artifacts;
they do not imply that the corresponding old architecture is current.

## D3 benchmark-first entry

The immediate implementation uses GPU0/GPU1 only:

1. export a minimal H12/W2 `WorkManifest`;
2. build pageable-DRAM source/target and a two-rank sequential path;
3. add a basic double buffer and phase metrics;
4. audit and construct the smallest real QK workload whose complete working set physically
   exceeds the two A40s;
5. use its profile to decide whether the best design is an isolated D3 scheduler or a cross-layer
   revision.

A normalized exporter, full transaction, 1/2/4-GPU matrix, and frozen protocol are later
paper-evidence tasks, not prerequisites for the first benchmark. Within one isolation revision,
variants still use the same `WorkManifest`; a cross-layer revision records changed actions,
owners, pools, layout, and source bytes and reruns its own baselines.

The concrete route is
[future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md).
H12 software capacity caps are development emulation only and cannot become paper evidence. All
D3 development artifacts remain non-scientific until a new protocol is frozen.

## Removed material

Redundant D2 stage-control and handoff documents, the pre-rewrite manuscript correspondence, early
paper drafts, and obsolete paper-process notes were removed. Their current facts were consolidated
into the D2 design/status pair, the D1 evidence ledger, and this index. Git history remains the
archive; deleted documents must not be cited as current state.
