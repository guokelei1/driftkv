# EvoKV

EvoKV studies how to update model-version-stale HSTU prefix K/V after a generative recommender
advances to a new model version. Exact replay restores current-model state but repeats the complete
history computation. Reusing the old cache is cheap but version-inconsistent. EvoKV turns this
single update job into three successive system decisions:

```text
D1 ActionPlan
  what should be translated, progressively repaired, or exactly recomputed
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

## Current design

### D1: version-cohort tiered migration

D1 resolves the reuse–recompute trade-off and emits an immutable per-record `ActionPlan`.

- The fast tier fits the shared `fresh - cheap` K/V residual over cached old `Norm(x)` for one
  source/target version cohort; the selected data plane reparameterizes the compiled affine to run
  directly over existing old K/V.
- The quality tier can replay a current-model prefix and transport its boundary residual to deeper
  current projections.
- Exact current-model recomputation is the endpoint and the semantic reference.

Version cohorts organize compilation and batching; they do not predict which user is safe to
reuse. Recommendation labels, per-user drift, JVP, and Fisher signals do not route caches.

### D2: wave-compiled segmented execution

D2 accepts D1 actions unchanged and lowers them onto row-sharded multi-GPU execution.

- compiled retained-prefix repair stays at the old-K/V owner and performs no embedding lookup;
- exact and append obtain only unavoidable item vectors from row-sharded embeddings;
- compiled work is ordered by `(suffix, retained, final)` shape before fixed-size resident extent
  cuts, while physically identical exact reasons share one pool;
- retained and suffix extents remain segmented, avoiding a full retained-prefix rewrite;
- collective dependencies, coverage, lineage, and atomic target publication are explicit.

The D2→D3 contract is a global, capacity-independent `WavePlan` constraint view rather than a
capacity-specific launch schedule. Current code implements the constituent mechanisms and a
resident W3 schedule, but has not yet exported that normalized view as an independently validated,
content-hashed artifact. The three-A40 full-cohort results are development evidence only; the
exporter, formal W4, same-boundary 1/2/4-GPU evaluation, segmented-consumer closure, and complete
commit/reclaim timing remain open.

### D3: action-aware out-of-core pipeline

D3 is the next implementation target. Its first step is to freeze the D2 constraint exporter.
Subsequent scheduler variants preserve the D1 `ActionPlan` and one common D2
owner/operator/compatibility/dependency/layout hash, then derive a per-rank capacity-safe
`ResidencyPlan` for:

```text
ordinary host DRAM
  → bounded pinned staging
  → GPU execution
  → bounded pinned staging
  → ordinary host DRAM private target
  → atomic manifest publication
```

The initial comparison is against both sequential capacity groups and a strong action-oblivious
double buffer with the same action-required source bytes. SSD/database ingress, serving traces,
hotness, and host-DRAM oversubscription are outside the first D3 boundary. D3 has a frozen problem
statement and exploration contract, but no executable handoff, frozen protocol, scheduler
implementation, or result yet.

## Evidence boundary

- D1 has a frozen KuaiRand single-configuration vertical slice and broader motivation/method
  replications. The normalized-capsule source path is a measured negative result; the selected
  hot-HBM route transforms existing exact source-version old K/V directly.
- D2 implementation and mechanism discovery are far enough to define the physical lowering, but
  its current W3 timings are `scientific_result=false` and must not enter paper tables.
- D3 is design-ready only. Existing destination-v4 and normalized-capsule DRAM experiments are
  historical prototypes, not direct-old-K/V D3 evidence.
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
