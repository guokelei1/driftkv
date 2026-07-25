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
- `docs/paper_draft_intro_motivation.md` is the current advisor-facing story, not a source for
  implementation semantics.
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
  cohorts are used for compilation, batching, placement, and scheduling, not to predict whether
  reuse is safe. The next design problems are organic mixed versions and end-to-end state movement.
- The former per-user drift/JVP/Fisher/three-state route is retired. Do not reintroduce it as the
  project crux. Its only valid role is a clearly labeled negative result in the motivation.

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
