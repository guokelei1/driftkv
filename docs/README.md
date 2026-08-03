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
3. [10_paper_experiment_blueprint.md](10_paper_experiment_blueprint.md) — paper-wide benchmark
   portfolio, baseline-first execution ledger, exact experiment matrix, physical scale budget,
   run count, and figure/claim map. It is authoritative only for the planned evaluation map; it
   neither overrides D2/D3 mechanism definitions nor makes development artifacts scientific.
4. [11_benchmark_qualification.md](11_benchmark_qualification.md) — registry of workload,
   capacity, topology, timer, baseline, and rolling-lifecycle checks required before formal
   promotion. It is not a protocol and does not block benchmark design or implementation.
5. [12_d1_d2_baseline_round.md](12_d1_d2_baseline_round.md) — selected large-QK streaming
   checkpoint/D1 development ledger, artifact retention, and D1→D2 result boundary.
6. [13_cross_dataset_stream_checkpoint_plan.md](13_cross_dataset_stream_checkpoint_plan.md) —
   selected QK/QB versions, manifest hashes, rebuild commands, retention/cleanup, and downstream
   reuse boundary; it creates no formal evidence.
7. [future_design/DESIGN2_FINAL_PLAN.md](future_design/DESIGN2_FINAL_PLAN.md) — current D2
   mechanism and current D1→D2 starting interface.
8. [future_design/DESIGN2_DEVELOPMENT_STATUS.md](future_design/DESIGN2_DEVELOPMENT_STATUS.md) —
   implemented D2 state, non-scientific development evidence, pending W4/formal-evaluation work,
   and the D3 handoff.
9. [future_design/DESIGN3_FUTURE_DIRECTION.md](future_design/DESIGN3_FUTURE_DIRECTION.md) — D3
   problem definition, source/capacity/timer contract, candidate mechanisms, baselines, and
   go/no-go conditions.
10. [future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md)
   — historical two-card mechanism ledger plus the flexible HET/XP/rolling successor foundation
   and backtracking plan; it creates no protocol or evidence.
11. [09_single_configuration_full_chain_plan.md](09_single_configuration_full_chain_plan.md) —
   frozen D1 Stage 0–6 evidence ledger. It is history, not a current execution plan.
12. [dataset_expansion_audit.md](dataset_expansion_audit.md) — accepted and rejected dataset
   semantics, usable capacity, and generality boundaries.

Repository-local agent rules are in [../AGENTS.md](../AGENTS.md). Experiment records are indexed
by [../experiments/README.md](../experiments/README.md), active design documents by
[future_design/README.md](future_design/README.md), and paper/reference material by
[../paper/README.md](../paper/README.md).

## Current state

| Layer | Question | Output | Status |
|---|---|---|---|
| D1 | What should be compiled or exactly recomputed? | immutable `ActionPlan`; progressive residual replay is D1-only supporting evidence | frozen algorithm and single-configuration evidence |
| D2 | Where and in what physical distributed form should those fixed actions execute? | global D3-facing `WavePlan` constraints | mechanisms implemented; normalized exporter/hash and formal evidence open |
| D3 | How should one live cache larger than HBM move and execute on a 1/2/4-rank stack? | capacity-bounded `ResidencyPlan` plus versioned rolling groups | historical two-rank M1 mechanism chain complete in development; HET/HOM, XP, rolling lifecycle, strongest generic baseline, rank-general runner, formal repeats and protocol remain open |

The retained local model inputs are frozen separately from the design documents: QK LR0.15
theta0--theta4 is primary, and QB `u30_e3` theta0--theta3 is secondary. Their machine registry is
`configs/evokv_foundation/selected_checkpoint_registry_development_v0.json`; verify it before use
with `python scripts/verify_evokv_selected_checkpoints.py`.

D1 resolves the semantic reuse–recompute trade-off. D2 converts the resulting logical sparsity into
physical savings through owner-local retained repair, row-sharded exact/append, `(S,R)`-aware
compiled ordering, merged exact pools, segmented destinations, and group-valid outputs. D3 begins
with an isolation track that preserves the upstream snapshot while scheduling
ordinary-DRAM→GPU→ordinary-DRAM micro-waves, then validates, commits, and reclaims each rolling
group. Mechanism discovery may also create a new globally planned D1/D2/D3 `stack_revision`; it
must rerun its own baselines and cannot be presented as a ResidencyPlan-only ablation.

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
- The fixed-512 QK M1 training, D1/D2 snapshot/characterizer, 144-GiB old-store materialization, fair S0,
  strong S1, fixed segmented-I/O, and route-aware ResidencyPlan runs are real development
  executions. They remain
  `scientific_result=false` and are historical full-private-target mechanism-discovery evidence,
  not the successor endpoint or paper results.
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

## Successor paper benchmark

The successor uses natural-length `X-QK-HET` from D1 through D3; `X-QK-HOM` reuses the same
records and valid histories in a masked 512-slot physical layout rather than selecting a different
population or a better alternative. XP fixes 2,859,835 base-period semantic rows plus one
padding row in a 2,859,836×4,096 physical FP32 table
(43.638 GiB) and an owner-side E4096→H1536 projection, making full single-card placement
capacity-inadmissible only after optimizer-updated active bytes pass the same bound and cover every
row in the all-comparator request union across both formal edges. The integrated route domain is
`compiled|exact`, and the same executor is
parameterized for 1/2/4 ranks. D3 keeps one live cache plus bounded shadow/staging and completes
`writeback → validation → group commit → old-group reclaim` for every byte-bounded group.

Formal baselines include independently tuned exact, S0/S1, fixed-FIFO segmented S2, and a
profile-aware work-conserving generic scheduler. Benchmark qualification is prepared alongside
this foundation and becomes binding only before protocol freeze, formal repeats, and paper-result
promotion.

## Historical D3 M0/M1 mechanism entry

The completed development implementation used GPU0/GPU1:

1. keep the completed minimal H12/W2 `WorkManifest`;
2. use the completed pageable-DRAM two-rank S0 as the reference;
3. retain the completed H1536/24L `theta0→theta1` compact result and positive held-out signal as
   the frozen M1 development boundary; its old D3-specific checkpoint copy was retired because
   the selected QK/QB registry now owns reusable model payloads;
4. retain its 410-exact/1,638-compiled D1 snapshot, D2 characterizer, complete 144-GiB old store,
   nine-group group-128 S0, and S1-paired 17-group group-64 S0;
5. retain the completed same-revision group-64 strong S1 and its wait/bubble profile;
6. retain the fixed-order segmented-I/O candidate as the causal predecessor;
7. use the implemented planner to bind route-specific I/C/O granularity, capacity, and a stable
   compiled/exact interleaving into one replayable plan;
8. retain the completed grouped E0/D1-only contribution diagnostics; same-binary tuned E0/S2,
   profile-aware generic, repeats, and sensitivity were unfinished in this revision and are not
   the current successor execution path.

A normalized exporter, rolling group lifecycle, segmented consumer, 1/2/4-rank runner, and frozen
protocol were not prerequisites for this historical mechanism pass; they are required by the
successor paper foundation. Within one isolation revision, variants still use the same
`WorkManifest`; a cross-layer revision records changed actions, owners, pools, layout, and source
bytes and reruns its own baselines.

The concrete route is
[future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md).
H12 software capacity caps are development emulation only and cannot become paper evidence. All
D3 M0 now includes a full682 GPU0/GPU1 sequential pass through pageable DRAM, one reusable pinned
slot, the real D2 mixed compute path, and private pageable writeback. It is a single-pass mechanism
profile under emulated capacity, not a paper result. D3 development artifacts remain
non-scientific until a new protocol is frozen.

The QK M1 boundary contains 512 fit/calibration users plus a disjoint 2,048-user benchmark pool.
Its first-64 base-only entity table has 2,859,836 rows including padding; at H1536 the FP32
embedding is 16.364 GiB. Two-rank training completed one `theta0→theta1` edge, and fixed held-out
NDCG@10 improves from 0.371468 to 0.380294. D1 fixes 410/2,048 records as exact; the D2
characterizer reports mixed lookup tokens and off-rank FP32 return bytes at about 25.0% of
all-exact.

The complete old K/V store is physically materialized in ordinary DRAM at 144 GiB. Full S0
processes all 2,048 records in nine sequential groups and writes a 144-GiB private target, for a
288-GiB old/target footprint. Its makespan is 53.497 seconds. On the makespan rank, ordinary-memory
staging, H2D, D2H, and publication total 26.397 of 50.017 phase seconds, or 52.8%, exposing the
DRAM staging/publication bottleneck that S1 should overlap. The first large group also required a
Triton int32→int64 pointer-index fix; a cold-cache execution crossing \(2^{31}\) completed. These
are non-scientific development diagnostics, not speedup or paper claims.

The first capacity-paired group-64 S0 took 54.577 seconds but included avoidable per-group Python
GC and CUDA allocator-cache flushing. It remains a fragmentation diagnostic, not the direct
runtime control. Remeasuring the same 2,048 records and 17 groups without that maintenance gives
the fair sequential S0: 48.238 seconds, complete coverage, and 20.146-GiB peak allocated HBM.

Strong group-64 S1 overlaps whole-group prefetch, execution, and drain with two bounded slots. It
finishes in 32.703 seconds, a 1.475x speedup over fair S0, but still exposes 6.20--6.79 seconds of
input-boundary wait. Segmenting only the input cuts that wait to 0.24--0.28 seconds yet moves the
bottleneck to output credits and reaches only 31.096 seconds. The historical v1 precursor
segments both directions: within each capacity group, pinned input components overlap CPU packing
with H2D, while pinned output components overlap D2H with ordinary-DRAM publication. It completes
in 28.885 seconds, 1.133x faster than strong S1 and 1.670x faster than fair S0. Both 77.3-GB/rank
targets are byte-identical to S1, and peak allocated/reserved HBM remains about
29.27/39.42 GiB. These are development results under
`evokv_design3_m1_qk_segmented_io_d3_development_v1`, not the current order-only control or a
frozen paper protocol.

The current planner independently represents per-route input, compute, and output granularity
around full-group GPU staging. It accepts only jointly measured same-source profiles, uses
max-rank stage service, discrete tail scaling, and the actual one-lookahead/one-drain recurrence.
Small stable-interleaving spaces are exhaustive; large spaces use Pareto-beam DP. A
global-min-anchored 3% resource tie chooses among near-equal predictions. The embedded-profile
plan binds code/compiler/program, Torch/CUDA, GPU UUID/PCI identity, store tier, groups,
checkpoints, HBM, and pinned capacity; both ranks preflight it.

Under one exact stack/hash, route-major `(8,8,8)` takes 28.514442098 seconds and selected order
`[13,0..11,14,12,15,16]` takes 28.147194647 seconds: 1.013047x, or 1.2879% lower wall time. The
selected result is 1.16186x over S1 and 1.71379x over fair S0; its 29.244944224-second prediction
is 3.90% high. Both 77,309,939,712-byte/rank targets are byte-identical to S1 with complete,
exactly-once coverage. The selected triple remains `(8,8,8)` for both routes, so asymmetric
granularity benefit is not established; input-16 and output-4 merely did not improve their
observed points. Plan/profile construction is outside the timer. These results use
`evokv_design3_m1_qk_route_aware_residency_d3_development_v3` and remain
`scientific_result=false`, `formal_design3=false`.

## Removed material

Redundant D2 stage-control and handoff documents, the pre-rewrite manuscript correspondence, early
paper drafts, obsolete paper-process notes, and completed one-off checkpoint-search launchers were
removed. Their current facts were consolidated into the D2 design/status pair, the D1 evidence
ledger, the selected-checkpoint registry, and this index. Reusable trainers/builders and compact
negative results remain. Git history is the source archive for tracked material; deleted local
checkpoint payloads are regenerable but not recoverable from a trash directory.
