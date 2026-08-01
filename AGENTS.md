# Repository agent notes

## Environment

- 4x NVIDIA A40 (46GB each), CUDA 13.1, torch 2.12.1
- Current experiment availability is restricted to GPU0/GPU1. Do not schedule work on GPU2/GPU3
  or launch four-rank jobs until the user explicitly restores their availability.
- Python 3.13.12, numpy/pandas/scipy installed

## Commands

- Install package: `pip install -e .`
- Run tests: `pytest`
- Lint: `ruff check src tests scripts`
- Type check: `mypy src` only when explicitly requested; it is not installed by default

## Long-running experiment handoff

- Before starting any experiment expected to take five minutes or longer, first create and validate
  a user-runnable script. Do not start that experiment unless the user explicitly asks the agent to
  run it.
- Treat one handoff as one executable experiment round, not as one isolated command. Bundle all
  currently ready and mutually compatible training, baseline, evaluation, and validation jobs into
  one orchestration script, run them sequentially in dependency order, and avoid making the user
  invoke each job separately.
- The handoff script must freeze or record its configuration, perform relevant resource
  preflights, write logs and machine-readable results to explicit paths, fail without overwriting a
  valid result, and support safe resume or an unambiguous fresh run when practical.
- Reuse valid prerequisites within the round. After each job has produced and passed validation of
  its durable artifacts, release its transient GPU state, worker processes, shared-memory or mmap
  stores, and other regenerable large intermediates before starting the next job. Preserve
  only checkpoints likely to be reused, plus compact programs, plans, ledgers, logs,
  configurations, and result summaries needed for analysis or resume. Never retain full K/V
  payloads across stages; discard large intermediates that can be reconstructed from retained
  checkpoints and bindings.
- Give the user the exact command, estimated wall time and resource use, output paths, and the
  artifacts to return for analysis. End the turn and let the user run it; do not repeatedly poll a
  long experiment.
- Stop a round at a real result-dependent boundary: if later code, configuration, or experiment
  choice depends on interpreting this round's measurements, do not guess and append it. Analyze the
  returned artifacts, make the next implementation changes, and then provide the next round's
  orchestration script.
- The agent may directly run short canaries, unit tests, formatting checks, and experiments expected
  to finish in under five minutes. If a supposedly short run reveals that the remaining work will
  exceed five minutes, preserve the completed state and hand off the remainder through a script.

## Sources of truth

- `docs/08_core_insights_and_roadmap.md` is the authoritative research state and roadmap.
- `docs/eval_protocol.md` defines which experimental results are valid and comparable.
- `docs/10_paper_experiment_blueprint.md` defines the planned paper matrix, resource envelope,
  baseline-first ledger, and claim/figure map; it does not create evidence.
- `docs/11_benchmark_qualification.md` registers checks required before protocol freeze, formal
  repeats, or paper promotion; it is not a current design/implementation blocker.
- `docs/future_design/DESIGN2_FINAL_PLAN.md` defines the D2 mechanism and D1→D2→D3 interface.
- `docs/future_design/DESIGN2_DEVELOPMENT_STATUS.md` is the only live D2 status ledger.
- `docs/future_design/DESIGN3_FUTURE_DIRECTION.md` is the flexible D3 problem/direction entry; it
  is not a frozen interface, protocol, or result.
- `docs/future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md` preserves the historical
  two-card M0/M1 ledger and defines the flexible HET/XP/rolling successor foundation,
  baseline-first search, and backtracking path; it is not a protocol or result.
- When documents conflict, follow the roadmap and evaluation protocol. Do not recover claims from
  deleted early documents or old result paths.

## Current research route

- The object is model-version-stale HSTU prefix K/V under streaming training.
- The successor integrated ActionPlan uses `compiled|exact`. Progressive residual replay remains a
  D1-only supporting extension and is not a D2/D3 headline route.
- `X-QK-HET` is the primary D1→D2→D3 workload and preserves natural valid extents;
  `X-QK-HOM` reuses the same records and valid histories in a masked 512-slot physical layout.
  Capacity and grouping use valid K/V bytes, not padded records.
- XP fixes 2,859,835 base-period semantic rows plus one padding row in a
  2,859,836×4,096 physical FP32 table (43.638 GiB), owner-side
  E4096→H1536 projection and a 24L/H1536 core. Freeze the hardware HBM cap first; qualification
  validates this geometry without consulting EvoKV performance. Only optimizer-updated rows count
  toward forced sharding: the request union across both formal edges, all headline manifests,
  all-exact, and every frozen fixed-action exact/append/fallback path must be active and hashed.
  Active embedding bytes plus dense/projection bytes must exceed the single-card allocatable
  budget. The common-upstream base builder may use later-role users' base-period histories, but
  post-base roles remain user-disjoint.
- Foundation Review 0 is complete as development evidence. The full QK scan freezes disjoint
  roles and a 65,536-record HET/HOM universe; natural target length is median 153, p95 404, and
  only 2.1835% saturated. Full HET old/target valid K/V is 1.498/1.801 TB, with nested
  36/72/144/288/576/720-GiB cohorts. The two-rank physical E4096 owner-projection canary and
  HET/HOM rolling transaction canaries pass. The all-exact request union is 929,554 rows, but the
  optimizer-active forced-sharding gate remains pending and requires at least 2,840,105 active
  semantic rows. These artifacts are `scientific_result=false`; the rolling lifecycle canary does
  not yet execute D1/D2 numerics.
- The selected two-rank XP quality foundation is
  `quality_chain_stream_aligned_train16384_round1`: one warm-up edge followed by three ordinary
  stream edges, 16,384 training users, 4,096 disjoint qualification users, one epoch/update,
  dense/projection LR 1e-5, embedding LR 1e-4, 999 frozen negatives, and a common FP16-storage /
  FP32-consumption cache endpoint. Exact-over-Reuse CE gaps are 0.01846/0.01068/0.01340 with all
  record-cluster 95% intervals positive. This is development-selected, not formal replication.
  `selected_d1_bridge_round1` reproduces those endpoints and observes compiled gap recovery of
  63.9%/55.3%/70.0% at 0.162x/0.152x/0.146x Exact maintenance components; approximately 20%
  Exact mixed repair recovers 68.9%/62.3%/74.3%, while naive mixed component bounds remain
  0.764x/0.781x/0.731x Exact. This is D1→D2 causal development evidence, not end-to-end D2 timing.
  Keep the selected four checkpoints and compact programs/plans/results; the two rejected
  8,192-user checkpoint trees have been deleted and no full K/V payload is retained.
- D3 keeps one live cache plus bounded group shadow/staging and executes
  writeback→validation→group commit→old-group reclaim. Complete old+private-target COW remains a
  historical M1 endpoint, not the formal capacity definition.
- The successor runner is 1/2/4-rank-capable. XP 2/4-rank points are headline; X2/R-KR provide
  1-rank sanity. Qualification blocks only formal promotion, not foundation or mechanism work.
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
  compiled repair, with exact recomputation under budget in the integrated path. Progressive
  residual replay remains a separately evaluated D1 extension. Version cohorts are used for
  compilation, not to predict whether reuse is safe. This frozen D1 determines what is compiled or
  exactly recomputed and exports an immutable action plan. Active D2 determines how that fixed work
  moves and executes: `(suffix, retained)` extent compilation, owner-local retained repair,
  row-sharded exact/append, segmented suffix-only destination, merged physical exact pools,
  collective dependencies, and group-valid output. The current D2→D3 paper decomposition uses a
  global, capacity-independent WavePlan constraint view. The historical GPU0/GPU1 H12/QK paths
  remain development ledgers; the successor builds HET/HOM, XP, rolling groups, and a
  rank-parameterized runner. Isolation-track runs keep one D1/D2 snapshot fixed. Mechanism
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
  correctness also passes. These are `scientific_result=false`, not paper evidence. W4 is only
  the unfinished gate of the old family. The successor still needs XP/HET/HOM, a new D2 protocol,
  capacity-admitted 1/2/4-rank results, group validation/commit/reclaim timing, and a segmented
  consumer. Synthetic lookup contention is supporting characterization, not a serving trace or
  D2 gate.
- D3 has a real but historical two-A40 fixed-512 QK M1 out-of-core development chain, not a frozen
  protocol or paper result.
  Its 24L/H1536 model has a 16.364-GiB global FP32 embedding, and its fixed 2,048-record D1/D2
  snapshot contains 1,638 compiled and 410 exact actions. Complete old plus private target K/V is
  288 GiB in ordinary DRAM. On the shared 17-group boundary, fair S0 is 48.238 seconds, strong S1
  is 32.703 seconds, and the historical v1 fixed-order bidirectional runner is 28.885 seconds.
  The active development `ResidencyPlan` independently controls per-route input, compute, and
  output granularity over full-group GPU staging. It admits only compiled/exact profiles measured
  jointly from the same source, uses max-rank stage times with discrete tail scaling, and optimizes
  a stable route interleave under the runtime's one-group-input-lookahead/one-drain-credit flow
  model. Small order spaces are exhaustive; larger spaces use Pareto-beam dynamic programming. A
  global-min-anchored 3% tie prefers lower HBM, pinned memory, and segmentation cost. The plan
  embeds its profiles and binds compiler/program/source code, Torch/CUDA, GPU UUID/PCI identity,
  store tier, groups, and checkpoints; both ranks repeat HBM and pinned-capacity preflight.
  Under one exact stack/hash, the route-major `(8,8,8)` control takes 28.514442098 seconds and the
  selected stable order `[13,0..11,14,12,15,16]` takes 28.147194647 seconds: 1.013047x, or a
  1.2879% wall-time reduction. The selected result is 1.16186x over S1 and 1.71379x over fair S0;
  its 29.244944224-second prediction is 3.90% above observation. Both 77,309,939,712-byte rank
  targets are byte-identical to S1 with complete, exactly-once coverage. The chosen granularity is
  `(8,8,8)` for both routes, so route-asymmetric granularity benefit is not established. Compiled
  input-16 and output-4 only failed to improve their observed development points. Plan/profile
  construction is outside the timer. An adjacent identity-only revision observed 29.7169→28.0497
  (5.61%), but it is a variability diagnostic, not a frozen benefit. All results remain
  `scientific_result=false` and `formal_design3=false`. Grouped development E0 is now
  44.638644214 seconds sequentially and 33.548799294 seconds with the action-oblivious two-slot
  pipeline. Owner-local naive-staged D1-only is 57.597180375 seconds; the current-binary
  sequential D1+D2 rerun is 49.752669533 seconds. D1-only and D2 request the same 262,336 global
  lookup tokens but issue 852 versus 387 collectives/rank. These are contribution diagnostics,
  not a placement-oblivious owner-compute ablation or formal waterfall. Independently tuned E0,
  held-out qualification, formal repeats, HET/HOM and XP foundations, strongest generic
  baselines, rolling group lifecycle, segmented consumer, 1/2/4-rank successor runner, and a
  frozen protocol remain open.
  H12 is only a capacity-emulated semantic canary. SSD/database ingress, serving traces, host-DRAM
  oversubscription, and online hotness remain out of scope.

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
  implementation assets, not current D3 evidence.
- `src/hstu_kvcache/migration/design3_work.py`,
  `scripts/build_evokv_design3_m0_work.py`, and
  `scripts/benchmark_evokv_design3_m0.py` are the active flexible D3 M0 manifest/grouping and
  GPU0/GPU1 pageable-DRAM S0 entry points. Their artifacts are development diagnostics.
- `src/hstu_kvcache/migration/design3_residency.py`,
  `scripts/plan_evokv_design3_residency.py`, and
  `scripts/benchmark_evokv_design3_m1.py` implement the active hashed D3 planner and the real QK
  M1 route-aware out-of-core runtime. Their current artifacts are non-scientific development
  evidence.
- `src/hstu_kvcache/migration/foundation_{workload,projection,lifecycle}.py` and the matching
  `scripts/*evokv_foundation*.py` entry points implement the successor HET/HOM builder, physical
  owner-projection canary, and rolling transaction canary. The workload is real; the lifecycle
  payload is deterministic and does not yet execute D1/D2 numerics. All current artifacts are
  non-scientific Foundation Review evidence.
- `results/validity/`, `results/scaling/`, `results/exposure/`, and
  `results/motivation_scale/` — current result families. Their protocol records must remain
  separate; raw per-seed files and checkpoints stay local and ignored.

## Experimental invariants

- Predict item `t+1` from hidden state `t`.
- Train only on targets from the current stream date and use engaged items as evaluation positives.
- Fit the item vocabulary on the base period only.
- Respect sequence lengths and zero padding in hidden-state and K/V computation.
- For HET workloads, preserve valid old/retained/evicted/append/target extents end to end; HET
  capacity/throughput axes exclude padding. Physical admission always uses actual allocated bytes,
  including masked padding in the same-record HOM control.
- Formal out-of-core capacity is single-version valid K/V bytes plus bounded group shadow/staging,
  not complete old plus complete private target.
- Every rolling group carries explicit old/new version and lineage state. Validate before commit,
  reclaim only after commit, and make resume idempotent.
- QK HET is a trace-grounded heterogeneous cache snapshot under one fixed model edge; its
  per-user ordinal boundaries are not a cross-user co-temporal or calendar-time trace.
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
