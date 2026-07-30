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
| D3 | How should an out-of-core two-GPU stack move and execute K/V? | initially a capacity-specific schedule; final interface open | GPU0/GPU1 M0 S0 and the QK M1 data foundation are implemented; large-edge training, M1 S0/S1, and mechanism discovery remain open |

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
- D3 has a non-scientific M0 development family, not a frozen result family. Destination-v4
  correctness, normalized-capsule DRAM results, and hot-HBM D1 results cannot be renamed as D3
  evidence.
- The materialized QK M1 entity input is also non-scientific foundation state: it does not mean
  the H1536 model, new D1/D2 edge, 288-GiB K/V store, or M1 runtime has executed.
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

The active implementation uses GPU0/GPU1 only:

1. keep the completed minimal H12/W2 `WorkManifest`;
2. use the completed pageable-DRAM two-rank S0 as the reference;
3. add a basic double buffer and wait/bubble metrics;
4. use the materialized QK base-entity input and implemented sharded trainer to produce one
   H1536/24L `theta0→theta1` edge with an independent held-out window;
5. regenerate its D1/D2 snapshot, materialize the planned 288-GiB K/V working set, and run M1
   S0/S1/all-exact;
6. use its profile to decide whether the best design is an isolated D3 scheduler or a cross-layer
   revision.

A normalized exporter, full transaction, 1/2/4-GPU matrix, and frozen protocol are later
paper-evidence tasks, not prerequisites for the first benchmark. Within one isolation revision,
variants still use the same `WorkManifest`; a cross-layer revision records changed actions,
owners, pools, layout, and source bytes and reruns its own baselines.

The concrete route is
[future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md).
H12 software capacity caps are development emulation only and cannot become paper evidence. All
D3 M0 now includes a full682 GPU0/GPU1 sequential pass through pageable DRAM, one reusable pinned
slot, the real D2 mixed compute path, and private pageable writeback. It is a single-pass mechanism
profile under emulated capacity, not a paper result. D3 development artifacts remain
non-scientific until a new protocol is frozen.

The QK M1 input now contains 512 fit/calibration users plus a nested 2,048-user benchmark pool.
Its first-64 base-only entity table has 2,859,836 rows including padding; at H1536 the FP32
embedding is 16.364 GiB, and a 24-layer 2,048-record complete old/private-target FP16 K/V working
set is 288 GiB. `window_0` is the only update and `window_1` is held out. This is a materialized
data/capacity foundation, not a trained model or system measurement; the old H512 QK canary remains
only a functional diagnostic.

## Removed material

Redundant D2 stage-control and handoff documents, the pre-rewrite manuscript correspondence, early
paper drafts, and obsolete paper-process notes were removed. Their current facts were consolidated
into the D2 design/status pair, the D1 evidence ledger, and this index. Git history remains the
archive; deleted documents must not be cited as current state.
