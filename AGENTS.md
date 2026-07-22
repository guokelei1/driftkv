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
- The active method is structure-aware cache migration:
  - cheap layer: cached old `Norm(x)` projected by current `Wk/Wv`;
  - full region: current blocks over the deepest suffix, with projection-only terminal execution;
  - all full layers: exact current-model K/V recomputation.
- The 21-interval search did not justify arbitrary-layer dynamic selection. Keep the optimized
  suffix unless larger-scale or cross-dataset evidence changes this result.
- Fixed-operator scaling across sequence length, batch size, depth, update magnitude, a
  top-5k/top-20k by 6L/12L factorial, and top-50k chunked training is complete. KuaiRand scaling is
  positive; the short MovieLens chain does not establish cross-dataset problem strength. Taobao
  UserBehavior is the selected next cross-dataset stream. Audit and freeze its temporal protocol
  before running the motivation control; mixed cache versions and end-to-end state movement follow.
- The former per-user drift/JVP/Fisher/three-state route is retired. Do not reintroduce it as the
  project crux. Its only valid role is a clearly labeled negative result in the motivation.

## Code layout

- `src/hstu_kvcache/models/` — modular simplified HSTU; pointwise unnormalized attention and
  first-class K/V output are defining features.
- `src/hstu_kvcache/data/` — KuaiRand streaming trace and ML1m loader.
- `src/hstu_kvcache/streaming/` — leak-free next-item training and model-version utilities.
- `src/hstu_kvcache/migration/` — layerwise state capture and migration operators.
- `scripts/*validity.py`, `scripts/interval_oracle.py`, `scripts/streaming_value_control.py`, and
  `scripts/*scaling.py` — active experiment entry points.
- `results/validity/`, `checkpoints/validity/`, `results/scaling/`, and `checkpoints/scaling/` —
  current result/checkpoint families. Their protocol records must remain separate.

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
- Taobao UserBehavior is planned but is not local and has no current protocol or result artifacts.

## Conventions

- No comments in code unless asked.
- Keep HSTU modules decoupled so layer/block/attention changes remain localized.
- Start new structural sweeps at one seed and small scale; only reproduce candidates that change
  the Pareto frontier.
