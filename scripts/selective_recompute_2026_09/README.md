# Four selective-recomputation baselines

**Completed: all 15 edges, 60 curves, 295 measured method points and 12 figures;
exit code 0.** The original run was authorized on 2026-09-26.
Experiment record and panel statistics:
[results](../../results/selective_recompute_2026_09/README.md).

## Layout

| Method | Numerical implementation | Focused tests |
| --- | --- | --- |
| Layer | `src/hstu_kvcache/baselines/layer_recompute/` | `tests/selective_recompute/layer_recompute/` |
| Tail | `src/hstu_kvcache/baselines/tail_recompute/` | `tests/selective_recompute/tail_recompute/` |
| Deviation | `src/hstu_kvcache/baselines/deviation_recompute/` | `tests/selective_recompute/deviation_recompute/` |
| Query | `src/hstu_kvcache/baselines/query_recompute/` | `tests/selective_recompute/query_recompute/` |

Shared arbitrary-position replay is `baselines/sparse_recompute.py`. This
directory contains the common timeline, panel preparation, GPU workers,
calibration, arithmetic accounting and resumable queue. Existing historical
experiments and the unrelated K/V-translation baseline retain their own files.

## Retained commands

The queue is complete. The launch commands below document the execution
interface; a new formal run requires its own explicit launch authorization.

```bash
# CPU-only panel/file preflight; existing frozen panels are reused.
python scripts/selective_recompute_2026_09/run.py --prepare

# Authorized bounded checks: 2 profiling + up to16 canary users per scale,
# first edge only, four GPUs, maximum configured synthetic buffer shapes.
python scripts/selective_recompute_2026_09/run.py --probe

# Default launcher only describes the command; it does not start a job.
bash scripts/selective_recompute_2026_09/launch.sh

# Only AFTER the user's explicit OK:
bash scripts/selective_recompute_2026_09/launch.sh --start

# The same launch command resumes sealed user units after interruption.
python scripts/selective_recompute_2026_09/run.py --status
```

Formal startup checks `readiness.json` against execution source hashes, verifies
the selected18 weights and frozen panels, profiles32 separate users per edge,
then uses four independent GPU workers. Workers share models/history/ordinary
Reuse replay across the four methods, but save each method separately. No
baseline's repaired state is passed into another method or budget. Layer
profiling is paid once per standalone layer curve even though it is reused
across budget points during the experiment.

Each worker checkpoints all four methods together after256 users. Only completed
units with matching hashes, UIDs and source signatures are skipped on resume.
Logs, exit status, per-unit timings and progress are retained. The queue prints
a stage report every7200 seconds and after each edge. The GPU allocator is
capped at70%; an OOM reduces both user-cohort and query batches and retries the
same unit without dropping users. Startup requires75% free device memory.

After all15 edges, the plot generator exports four methods × three scales,
five edge curves each, with every raw point, a CSV and source hash manifest:

```bash
python figures/src/selective_recompute_2026_09.py \
  --input results/selective_recompute_2026_09/summary.json \
  --out figures/out/selective_recompute_2026_09
```

Relevant checks: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/selective_recompute`.
