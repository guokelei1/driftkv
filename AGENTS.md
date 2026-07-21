# Opencode agent notes

## Environment
- 4x NVIDIA A40 (46GB each), CUDA 13.1, torch 2.12.1
- Python 3.13.12, numpy/pandas/scipy/scikit-learn installed
- `torch.func` (jvp, vmap, functional_call) is available — used for per-user JVP drift estimation.

## Commands
- Install package (editable): `pip install -e .`
- Run tests: `pytest`
- Lint: `ruff check src tests scripts`
- Type check: `mypy src` (not installed by default; only if requested)

## Architecture source of truth
- `docs/08_core_insights_and_roadmap.md` is the authoritative roadmap. Early docs (00-07) are superseded where they conflict.
- The single mathematical crux (Insight 4/5): can drift `||F(θ+Δθ,x) − F(θ,x)||` be estimated at cost << full KV recompute? per-user JVP (reverse-mode) costs ~2x a forward, so naive estimation is *more* expensive than recompute — the innovation is reducing that cost.

## Code layout
- `src/hstu_kvcache/models/` — HSTU architecture (modular, designed to be modified for research). Pointwise attention (elu+1, unnormalized) is the defining feature; KV cache is a first-class output `F(θ,x_u)`.
- `src/hstu_kvcache/data/` — KuaiRand streaming-trace loader + ML1m generative-rec loader.
- `src/hstu_kvcache/streaming/` — streaming training loop producing θ_0..θ_t checkpoints + Δθ extraction + oracle recompute.
- `src/hstu_kvcache/drift/` — drift estimation methods (naive JVP, Fisher-spectrum online lookup, cross-user low-rank).
- `src/hstu_kvcache/serving/` — three-state cache decision (reuse/migrate/recompute).

## Datasets
- KuaiRand-1K source: `/home/gkl/fun-rec/data/dataset/kuairand/KuaiRand-1K/data/` (ms-level timestamps, two 2-week windows — ideal for streaming drift). Copied to `data/kuairand/`.
- ML1m hard_v5 source: `/home/gkl/LRM1/data/standardized/movielens_1m_hard_v5/pilot20` (small generative-rec format). Copied to `data/movielens/`.

## Conventions
- No comments in code unless asked. Mimic existing style.
- Keep HSTU modules decoupled: each layer/block/attention is a standalone class so future design tweaks (U1/U2 in roadmap) are localized edits.
