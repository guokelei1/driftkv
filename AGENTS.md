# Repository agent notes

## Current direction

This repository is being rebuilt around the 37D EvoKV route. The active question is when a persistent user KV state can be reused across a model release, when it needs approximate or selective evolution, and when it must be recomputed exactly.

The current research route is defined by:

- `docs/current_route.md` — concise execution boundary and next steps;
- `docs/newset.md` — full 37D route specification;
- `docs/legacy/README.md` — what was deliberately removed and what may not be revived.

There is currently no frozen experiment contract, no active runner, and no formal result. Do not reconstruct or revive the deleted D1/D2/D3 exploration routes.

## Repository state

The repository intentionally keeps only:

- reusable HSTU model and persistent K/V primitives under `src/hstu_kvcache/models/`;
- raw KuaiRand loading and chronological streaming-plan primitives under `src/hstu_kvcache/data/`;
- small JSON/timing helpers under `src/hstu_kvcache/utils/`;
- a small amount of raw KuaiRand data under `data/kuairand/`;
- route documents and empty placeholders for future `configs/`, `scripts/`, and `tests/`.

Deleted results, checkpoints, logs, generated data, old configurations, experiment runners, and historical tests are not available inputs. Do not add compatibility shims for them.

## Code layout

- `src/hstu_kvcache/models/attention.py`: pointwise HSTU attention and cache-aware forward.
- `src/hstu_kvcache/models/block.py`: normalized attention block, gating and residual path.
- `src/hstu_kvcache/models/embeddings.py`: behavior, item and temporal encoders.
- `src/hstu_kvcache/models/hstu.py`: HSTU model with full and incremental KV execution.
- `src/hstu_kvcache/models/kv_cache.py`: persistent batched K/V value object.
- `src/hstu_kvcache/data/kuairand.py`: raw KuaiRand loader and batch utilities.
- `src/hstu_kvcache/data/streaming_plan.py`: chronological base/stream data plan.

Keep the core model modules independent of experiment orchestration. New profiler/controller code belongs in a new, minimal module only after its contract is defined in the current route.

## Development rules

- Use `rg` or `rg --files` for search.
- Use `apply_patch` for source and documentation edits.
- Preserve unrelated dirty-worktree changes.
- Read a file before deleting or substantially changing it; delete only code that is tied to a removed route or has no reusable API.
- Do not add experiment runners, result registries, checkpoint retention logic, or dataset-specific corpus builders before the 37D workload and evaluation contract are frozen.
- Do not fabricate or preserve historical result claims. Negative gaps, failed workload regions, and model-release regressions are valid observations, not values to be hidden.
- Keep model scoring raw and protocol decisions label-free; no post-hoc score mixing, metric scaling, selected-edge reporting, or target-KV fitting.

## Verification

The current minimal package should at least support:

```bash
PYTHONPATH=src python -c "from hstu_kvcache.models import HSTU, HSTUKVCache; from hstu_kvcache.data import KuaiRandTrace, StreamingDataPlan"
```

No current experiment suite is expected to run. Add focused tests only when a new 37D component has been specified and implemented.

## Resources and safety

- Do not launch long experiments unless explicitly requested.
- Use the minimum data and compute needed for a canary.
- Do not retain large checkpoints, logs, generated results, or processed datasets by default.
- Before any destructive cleanup, confirm the exact target and prefer a recoverable staging move before releasing space.
