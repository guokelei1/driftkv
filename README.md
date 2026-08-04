# EvoKV

EvoKV studies how to update model-version-stale HSTU prefix K/V after a generative recommender
advances to a new model version. Exact replay restores current-model state but repeats the complete
history computation. Reusing the old cache is cheap but version-inconsistent.

The core system premise is fixed: exactly one recommendation-model version serves at any instant.
Versions form a sequential update chain; they are not co-served and requests are not routed among
them. Old and new K/V groups may coexist only while one current model transitions persistent cache
state, after which committed old extents are reclaimed. EvoKV turns this single-model update job
into three successive system decisions:

```text
D1 ActionPlan
  what should be compiled or exactly recomputed
  (progressive residual replay is a D1-only supporting extension)
        ↓
D2 WavePlan constraints
  where that fixed work executes and how it becomes physical multi-GPU work
        ↓
D3 ResidencyPlan
  when each legal extent enters and leaves HBM under an out-of-core capacity bound
```

The authoritative research state is
[docs/08_core_insights_and_roadmap.md](docs/08_core_insights_and_roadmap.md). Experimental
comparability is defined only by [docs/eval_protocol.md](docs/eval_protocol.md). Start from
[docs/README.md](docs/README.md) for the complete document map.

The reusable local model inputs are the registered QK LR0.15 theta0--theta4 primary chain and QB
`u30_e3` theta0--theta3 secondary chain. Verify their manifests and payload bindings with
`python scripts/verify_evokv_selected_checkpoints.py`; exact paths, hashes, retention rules, and
QK/QB rebuild commands are in
[docs/13_cross_dataset_stream_checkpoint_plan.md](docs/13_cross_dataset_stream_checkpoint_plan.md).
The active recursive D1 design has completed QK theta1--theta4 as three consecutive maintenance
rounds and freezes the 10% rollout-aware method for QB theta1--theta3 confirmation; see
[docs/future_design/DESIGN1_RECURSIVE_KV_MIGRATION.md](docs/future_design/DESIGN1_RECURSIVE_KV_MIGRATION.md).

The successor paper benchmark uses natural-length `X-QK-HET` end to end and a same-record
masked-512 `X-QK-HOM` physical-shape control. XP fixes 2,859,835 base-period semantic rows plus
one padding row in a 43.638-GiB physical FP32 item table, with owner-side E4096→H1536 projection;
only optimizer-updated active bytes may force
distributed placement, and the all-comparator request union across both formal edges must be
active. The executor is
parameterized for 1/2/4 ranks.
D3 maintains one live cache plus bounded group shadow/staging, then validates, commits, and
reclaims each rolling group. The planned matrix and pre-promotion checks are
[docs/10_paper_experiment_blueprint.md](docs/10_paper_experiment_blueprint.md) and
[docs/11_benchmark_qualification.md](docs/11_benchmark_qualification.md).

## Current design

### D1: recursive reuse–recompute planning

D1 resolves the reuse–recompute trade-off and emits an immutable per-record `ActionPlan`. Its
large-model path is recursive: each migrated output becomes the next model update's input, with no
unreported exact reset between edges. D1 is selected by logical Exact valid-token work, recursive
K/V/task recovery, and lineage---not by GPU time.

- The primary RACT-KV (Rollout-Aware Cache Transport) branch fits one shared semantic affine on
  paired exact/deployed old-K/V states, preserving an exact-source anchor while learning the
  rollout distribution produced by preceding migrations.
- The current QK design point uses deterministic 10% valid-token Exact renewal; 0% and 20% remain
  mechanism/budget controls.
- A supporting D1-only quality extension can replay a current-model prefix and transport its
  boundary residual to deeper current projections; it is not a D2/D3 headline route.
- Exact current-model recomputation is the endpoint and the semantic reference.

The v0 artifact field `stability_certificate` is a conservative held-out rollout diagnostic, not
a mathematical proof of recommendation accuracy and not the current headline design gate.
Operational fallback belongs to implementation correctness; scheduled Exact renewal is the normal
D1 trade-off mechanism.

Version cohorts organize compilation and batching; they do not predict which user is safe to
reuse. Recommendation labels, per-user drift, JVP, and Fisher signals do not route caches.

### D2: wave-compiled segmented execution

D2 accepts D1 actions unchanged and lowers them onto row-sharded multi-GPU execution.

- compiled retained-prefix repair stays at the old-K/V owner and performs no embedding lookup;
- exact and append obtain only unavoidable item vectors from row-sharded embeddings;
- compiled work is ordered by `(suffix, retained, final)` shape before fixed-size resident extent
  cuts, while physically identical exact reasons share one pool;
- retained and suffix extents remain segmented, avoiding a full retained-prefix rewrite;
- collective dependencies, coverage, lineage, and group-valid output are explicit.

The D2→D3 contract is a global, capacity-independent `WavePlan` constraint view rather than a
capacity-specific launch schedule. Current code implements the constituent mechanisms and a
resident W3 schedule, but has not yet exported that normalized view as an independently validated,
content-hashed artifact. The three-A40 full-cohort results are development evidence only; the
exporter, formal W4, same-boundary 1/2/4-GPU evaluation, segmented-consumer closure, and complete
commit/reclaim timing remain open.

### D3: action-aware out-of-core pipeline

D3 is the active out-of-core implementation track. Its development runtime preserves the D1
`ActionPlan` and one common D2 owner/operator/compatibility/dependency/layout hash, then derives a
per-rank capacity-safe `ResidencyPlan` for:

```text
ordinary host DRAM
  → bounded pinned staging
  → GPU execution
  → bounded pinned staging
  → ordinary host DRAM replacement group
  → validation → group commit → old-group reclaim
```

The comparison includes sequential groups, strong double buffering, independently tuned
fixed-FIFO segmentation, a profile-aware generic scheduler, and all-exact under the same rolling
endpoint. The current fixed-512 GPU0/GPU1 QK M1 route-aware planner and complete-private-target
runtime are historical development diagnostics, not the successor endpoint or a frozen paper
result. SSD/database ingress, serving traces, hotness, and host-DRAM oversubscription remain
outside this boundary.

## Evidence boundary

- D1 has a frozen KuaiRand single-configuration vertical slice and broader motivation/method
  replications. The normalized-capsule source path is a measured negative result; the selected
  hot-HBM route transforms existing old K/V directly. The full QK three-edge Round A is complete
  as non-scientific development evidence: the 10% RACT-KV point recovers 99.985%--100.002% of the
  per-edge CE gap and more than 94.9% mean K/V fidelity. A revised locked QB two-edge confirmation
  remains the active unfinished D1 boundary.
- D2 implementation and mechanism discovery are far enough to define the physical lowering, but
  its current W3 timings are `scientific_result=false` and must not enter paper tables.
- D3 has a historical two-A40 QK M1 development chain. Existing destination-v4 and normalized-capsule
  DRAM experiments remain historical prototypes, not direct-old-K/V D3 evidence; those M1
  timings are also non-scientific until a formal protocol is frozen.
- The successor foundation now has a real 65,536-record natural-length QK manifest, same-record
  masked-512 control, 36–720-GiB capacity cohorts, a physical two-A40 E4096 owner-projection
  canary, and minimal rolling commit/reclaim/failure/replay canaries. The selected XP checkpoint
  passes the forced-sharding byte gate with 2,859,736 optimizer-active semantic rows; the exact
  request-union-to-active-bitmap join and the first real HET D1/D2 rolling run remain open. These
  artifacts are development evidence rather than a D2/D3 result.
- Historical protocol names such as `cohortkv_*` and `streamkv_*` remain unchanged because they
  identify immutable artifacts; the current paper/system name is EvoKV.

## Repository layout

```text
src/hstu_kvcache/
  data/          dataset loaders and temporal preparation
  models/        modular HSTU with first-class K/V output
  streaming/     leak-free streaming training and model-version utilities
  migration/     D1 operators, D2 planning/runtime, transactions, and historical backends
scripts/         experiment, validation, benchmark, and freeze entry points
configs/         frozen plans, manifests, schemas, and checked summaries
experiments/     protocol-scoped result records; not current design instructions
docs/            authoritative state, protocol, design contracts, and document index
paper/           writing references and the frozen pre-EvoKV D1 manuscript artifact
```

Historical result notes and writing resources are indexed by
[experiments/README.md](experiments/README.md) and [paper/README.md](paper/README.md).

## Development checks

```bash
pip install -e .
pytest
ruff check src tests scripts
```

Do not pool results with different protocol strings. New primary KuaiRand training should use
`training_sequences=all_chunks` and record effective target counts. GPU time must be measured; a
hand-written cost constant is not a substitute for execution.
