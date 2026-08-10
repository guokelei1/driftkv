# Repository agent notes

## Environment

- Python 3.13.12, PyTorch 2.12.1, CUDA 13.1.
- Four NVIDIA A40 GPUs exist, but only GPU0/GPU1 are currently available.
- Use at most one two-rank job. GPU2/GPU3 and four-rank jobs require explicit new user authorization.

## Commands

- Install: `pip install -e .`
- Tests: `pytest`
- Lint: `ruff check src tests scripts`
- Do not run `mypy` unless explicitly requested.

## Sources of truth

- `docs/08_core_insights_and_roadmap.md` is the authoritative research state.
- `docs/eval_protocol.md` defines valid and comparable measurements.
- `docs/BASELINE_REPRODUCTION.md` defines the only supported rebuild path.
- `configs/evokv_root_cause/kuairand_large_baseline_registry_20260811_v0.json` is the machine-readable selected-baseline registry.
- `scripts/run_evokv_kuairand_large_baseline_rebuild.sh` is the canonical verify/resume/fresh entrypoint.
- Historical scripts and result directories are not current facts unless one of these sources names them.

## Current selected baseline

- Dataset: KuaiRand-1K standard logs only.
- Versions: θ1–θ8; θ0 is bootstrap only.
- Model: video/author, latest-item query, 8L/H512/8 heads, max length 512.
- Evaluation: one positive plus 99 frozen negatives; NDCG@5 primary, MRR and HR@5 reported.
- Capacity: one 23,396,297×512 physical embedding space, sharded over GPU0/GPU1; 47,960,055,552 parameter bytes or 44.666 GiB.
- Selected NDCG@5 matrix: 26/28 positive cells and 7/7 positive adjacent cells.
- This is single-seed development evidence: `scientific_result=false`, `formal_result=false`.

## Experimental invariants

- Exactly one current model serves at a time. Old version labels describe K/V lineage only.
- Train on an update date and evaluate on the next natural date; never train on evaluation targets.
- Compare Reuse and Recompute on identical users, histories, current model, query and candidates.
- Reuse means old-version prefix K/V plus current-model latest token/query. Recompute means a full current-model forward on the same valid history.
- Preserve natural sequence lengths and exclude padding from semantic K/V extents.
- Full recomputation is the cache-fidelity reference, not a guaranteed ranking-quality upper bound.
- Report absolute endpoints and `100 * (Recompute - Reuse) / Reuse` without clipping or scaling.
- Never perturb K/V, duplicate identical embeddings to claim capacity, delete negative cells, or combine incompatible protocols.
- Training seed is the replication unit; user-level samples within one model are diagnostics.

## Long experiments

- Before any experiment expected to exceed five minutes, create and validate a user-runnable orchestration script.
- Do not start such an experiment unless the user explicitly asks the agent to run it.
- Bundle dependent training, evaluation and validation into one resumable round with frozen configuration, resource preflights, explicit logs and machine-readable outputs.
- Preserve reusable checkpoints and compact records. Do not retain full K/V payloads between stages.
- Stop at genuine result-dependent boundaries; do not silently change the protocol after seeing results.
- Short canaries, lint and tests may run directly.

## Code layout

- `src/hstu_kvcache/models/`: HSTU model and first-class K/V output.
- `src/hstu_kvcache/streaming/kuairand_query_transition.py`: θ0 workload/training semantics.
- `src/hstu_kvcache/streaming/kuairand_projected_persistent.py`: sequential natural model chain.
- `src/hstu_kvcache/streaming/kuairand_capacity_lift.py`: function-preserving large-capacity lift.
- `scripts/verify_evokv_kuairand_large_baseline.py`: static, semantic and payload verification.

## Repository conventions

- Use `rg`/`rg --files` for search and `apply_patch` for edits.
- Preserve unrelated dirty-worktree changes.
- No code comments unless requested.
- Keep HSTU modules decoupled so attention, block and layer changes remain localized.
- Start structural sweeps at one seed and small scale; reproduce only candidates that change the frontier.
