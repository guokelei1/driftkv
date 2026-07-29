# Repository agent notes

## Environment

- 4x NVIDIA A40 (46GB each), CUDA 13.1, torch 2.12.1
- Python 3.13.12, numpy/pandas/scipy installed

## Commands

- Install package: `pip install -e .`
- Run tests: `pytest`
- Lint: `ruff check src tests scripts`
- Type check: `mypy src` only when explicitly requested; it is not installed by default

## Sources of truth

- `docs/08_core_insights_and_roadmap.md` is the authoritative research state and roadmap.
- `docs/eval_protocol.md` defines which experimental results are valid and comparable.
- `docs/future_design/DESIGN2_FINAL_PLAN.md` defines the D2 mechanism and D1→D2→D3 interface.
- `docs/future_design/DESIGN2_DEVELOPMENT_STATUS.md` is the only live D2 status ledger.
- `docs/future_design/DESIGN3_FUTURE_DIRECTION.md` is the flexible D3 problem/direction entry; it
  is not a frozen interface, protocol, or result.
- `docs/future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md` is the live two-card D3
  foundation, baseline, candidate-search, and backtracking plan; it is not a protocol or result.
- When documents conflict, follow the roadmap and evaluation protocol. Do not recover claims from
  deleted early documents or old result paths.

## Current research route

- The object is model-version-stale HSTU prefix K/V under streaming training.
- The active method is version-cohort tiered cache migration:
  - fast tier: fit a shared `fresh - cheap` K/V residual and compile it into one affine projection
    over cached old `Norm(x)`;
  - quality tier: replay a current-model prefix and transport its boundary residual delta to deeper
    current `Norm + Wk/Wv` projections;
  - endpoint: exact current-model K/V recomputation.
- Fixed suffix, progressive prefix, arbitrary intervals, and recent-token rectangles are baselines,
  not the active method. The cross-dataset capacity screen changed the earlier suffix decision:
  plain prefix is never selected in the unified library, all 54 matched recent-token partial
  actions are slower, and arbitrary intervals add negligible value.
- Fixed-task 3x3 data/model-capacity motivation and cohort-tiered method replication are complete
  over KuaiRand, QB, and QK. The compiled operator scales in kernel cost and K/V fidelity, but the
  strict task-quality gate passes 6/9 cells because some full-maintenance endpoints are near zero
  or negative. Task quality is not an admission oracle: every stale cohort receives unconditional
  compiled repair, then progressive residual replay and exact recomputation under budget. Version
  cohorts are used for compilation, not to predict whether reuse is safe. This frozen D1 determines
  what is compiled or exactly recomputed and exports an immutable action plan. Active D2 determines
  how that fixed work moves and executes: `(suffix, retained)` extent compilation, owner-local
  retained repair, row-sharded exact/append, segmented suffix-only destination, merged physical
  exact pools, collective dependencies, and atomic publication. The current D2→D3 paper
  decomposition suggests a global, capacity-independent WavePlan constraint view, but this
  normalized artifact is neither implemented nor a prerequisite for the first D3 benchmark. D3
  now starts benchmark-first on GPU0/GPU1 with a minimal H12/W2 `WorkManifest`, ordinary host
  DRAM, bounded staging, and HBM. Isolation-track runs keep one D1/D2 snapshot fixed. Mechanism
  discovery may also create a globally replanned D1/D2/D3 `stack_revision`; it must rerun its own
  baselines rather than compare against an older stack. Complex organic mixed versions remain a
  later feedback layer.
- The former per-user drift/JVP/Fisher/three-state route is retired. Do not reintroduce it as the
  project crux. Its only valid role is a clearly labeled negative result in the motivation.
- Single-configuration Stages 0–4.6 are frozen. The closest selective baseline fails its certificate;
  the deployed compiler plans pass on serialized FP16 capsule/program/output, while the optional
  residual hidden suffix is BF16 after measured FP16 overflow. Reference, packed, and fused
  operators share one unpadded extent API. The 30-point full-cohort normal-path matrix closes
  lazy source streaming and HBM/DRAM publication, but the current 17.82-GB physical FP16 capsule
  path loses to exact at all six matched endpoints and spends 91.35%–96.91% of compiled time in
  source processing. Stage 4.5 is now frozen separately: the primary hot-HBM plan composes the
  deployed affine into a direct transform over existing old K/V, retains zero extra per-record
  `Norm(x)`, reclaims old extents after replacement staging, preserves the certificate and full
  real transport, and beats paired HBM-resident raw-history exact at full-cohort 1/2/4-GPU points.
  This is not a cold-filesystem or SSD claim, and its equivalence/certificate assumes exact
  source-version K/V. Stage 4.6 now supplies the repeated-input evidence on the single frozen
  KuaiRand seed-0/16L-H512 configuration and one A40: exact theta0 K/V is recursively advanced
  through 11 updates under a balanced age/deadline policy with program-level label-free edge
  severity, 15%–25% exact budgets, and maximum migration depth four. The complete 682-record chain
  costs 0.2134x all-exact GPU time and preserves minimum cache/score/top100 fidelity
  0.9632/0.999950/0.9918. Its per-cache threshold predecessor is a negative result because it
  produced severe refresh waves; do not recover an adaptive-risk or optimal-selector claim.
  Recommendation labels never route caches. D1 Stages 0–6 and the declared Stage-5 guard/fallback/
  transaction closure are frozen.
- D2's paper story is logical-to-physical sparsity. `134/682 = 19.6%` exact-route records is an
  action-count statistic, while the full mixed wave still performs `347,062/934,917 = 37.1%` of
  all-exact lookup tokens. A three-A40 W3 development chain shows naive mixed losing to exact and
  the segmented/shape-aware/merged-exact lowering producing a positive full682 point; full payload
  correctness also passes. These are `scientific_result=false`, not paper evidence. Formal W4,
  a new D2 protocol, 1/2/4-GPU same-boundary results, publication/commit/reclaim timing, and a
  segmented consumer remain open. Synthetic lookup contention is supporting characterization,
  not a serving trace or D2 gate.
- D3 currently has a paper skeleton and design hypothesis only. It has no frozen protocol or paper
  result. Its immediate task is to run a minimal two-rank ordinary-DRAM→GPU0/GPU1→ordinary-DRAM
  benchmark: H12/W2 WorkManifest, sequential groups, then a basic double buffer. A normalized
  exporter, atomic publication/failure closure, and 1/2/4-GPU matrix are later formalization tasks,
  not mechanism-discovery prerequisites. Within one isolation revision, baselines share the work
  and source-byte snapshot; co-design revisions report changed actions/owners/source bytes and
  rerun baselines. SSD/database ingress, serving traces, host-DRAM oversubscription, and online
  hotness are out of scope. H12 capacity caps are development emulation because H12 fits two A40s.
  The preferred physical route audits a larger real-history QK workload.

## Code layout

- `src/hstu_kvcache/models/` — modular simplified HSTU; pointwise unnormalized attention and
  first-class K/V output are defining features.
- `src/hstu_kvcache/data/` — KuaiRand streaming trace and ML1m loader.
- `src/hstu_kvcache/streaming/` — leak-free next-item training and model-version utilities.
- `src/hstu_kvcache/migration/` — layerwise state capture and migration operators.
- `scripts/*validity.py`, `scripts/*scaling.py`, `scripts/*motivation_capacity_v2.py`, and
  `scripts/*migration*.py` — active experiment entry points.
- `scripts/evaluate_kuairand_long_context_sync_design.py` and
  `scripts/benchmark_kuairand_long_context_sync_system.py` — active 4+12 progressive-sync
  algorithm/operator/runtime entry points.
- `scripts/evaluate_cohortkv_stage1_frontier.py`, `scripts/compile_cohortkv_stage2.py`, and
  `scripts/benchmark_cohortkv_stage3_operator.py` — frozen single-configuration baseline,
  deployed-compiler, and common-layout operator entry points.
- `scripts/compile_cohortkv_stage4_6_edges.py`,
  `scripts/evaluate_cohortkv_stage4_6_lifecycle.py`,
  `scripts/run_cohortkv_stage4_6_full_chain.py`, and
  `scripts/freeze_cohortkv_stage4_6.py` — frozen continuous-lifecycle entry points.
- `scripts/materialize_cohortkv_stage4_sources.py`,
  `scripts/benchmark_cohortkv_stage4_system.py`, and `scripts/freeze_cohortkv_stage4.py` — frozen
  full-cohort source, normal-path system, and checked-summary entry points.
- `scripts/compile_cohortkv_stage4_5_oldkv.py`,
  `scripts/evaluate_cohortkv_stage4_5_oldkv_certificate.py`,
  `scripts/validate_cohortkv_stage4_5_oldkv_full_transport.py`,
  `scripts/benchmark_cohortkv_stage4_5_oldkv.py`, and
  `scripts/freeze_cohortkv_stage4_5.py` — frozen direct-old-K/V source-plan entry points.
- `src/hstu_kvcache/migration/design2_*.py`,
  `scripts/benchmark_cohortkv_design2_integrated_w3.py`,
  `scripts/validate_cohortkv_design2_integrated_full_payload.py`, and
  `scripts/benchmark_cohortkv_design2_resource_isolation.py` — active D2 development entry
  points; their current W3 artifacts are non-scientific mechanism discovery.
- `src/hstu_kvcache/migration/destination.py` and `out_of_core.py` are historical destination-v4
  implementation assets, not a D3 implementation or result. New D3 work must preserve the
  D2 constraint boundary, first materialize its normalized exporter/hash, and then create an
  explicit `ResidencyPlan` path.
- `results/validity/`, `results/scaling/`, `results/exposure/`, and
  `results/motivation_scale/` — current result families. Their protocol records must remain
  separate; raw per-seed files and checkpoints stay local and ignored.

## Experimental invariants

- Predict item `t+1` from hidden state `t`.
- Train only on targets from the current stream date and use engaged items as evaluation positives.
- Fit the item vocabulary on the base period only.
- Respect sequence lengths and zero padding in hidden-state and K/V computation.
- Evaluate stale serving as an old-version prefix cache plus the latest token under the current
  model; fresh is a complete current-model forward on the same history.
- Measure GPU cost; never substitute a hand-written cost constant.
- Treat training seed as the replication unit. User/sample-level statistics within one trained
  model are diagnostics, not independent experimental repeats.
- Do not mix result families with different protocol strings.
- New primary KuaiRand training runs should use `training_sequences=all_chunks` and record
  effective target counts. Existing latest-only results remain valid within their own protocols.
- Full recomputation is the cache-fidelity reference, not a guaranteed upper bound on ranking
  quality; report paired quality differences from full when recovery exceeds 100%.

## Datasets

- KuaiRand-1K source: `/home/gkl/fun-rec/data/dataset/kuairand/KuaiRand-1K/data/`; local copy in
  `data/kuairand/`.
- Only KuaiRand-1K standard logs are local. Top-5k/top-20k/top-50k retain 3.67%/8.18%/13.35% of
  their rows. The random-exposure log is excluded from training, and KuaiRand-27K is not present.
- ML1m hard_v5 source: `/home/gkl/LRM1/data/standardized/movielens_1m_hard_v5/pilot20`; local copy
  in `data/movielens/`.
- Taobao UserBehavior is local but rejected at the target-semantics gate because it lacks true
  unclicked impressions.
- Tenrec QB/QK are the positive ordered-exposure extensions; they are related tables from one
  collection and have ordinal rather than global calendar time.
- ZhihuRec is a documented negative maintenance boundary.

## Conventions

- No comments in code unless asked.
- Keep HSTU modules decoupled so layer/block/attention changes remain localized.
- Start new structural sweeps at one seed and small scale; only reproduce candidates that change
  the Pareto frontier.
