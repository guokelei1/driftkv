# Four selective-recomputation baselines

All15 adjacent edges,60 curves and295 measured method points are complete.
Retained evidence and panel statistics:
[results](../../results/selective_recompute_2026_09/README.md).

| Method | Numerical implementation | Focused tests |
| --- | --- | --- |
| Layer | `src/hstu_kvcache/baselines/layer_recompute/` | `tests/selective_recompute/layer_recompute/` |
| Tail | `src/hstu_kvcache/baselines/tail_recompute/` | `tests/selective_recompute/tail_recompute/` |
| Deviation | `src/hstu_kvcache/baselines/deviation_recompute/` | `tests/selective_recompute/deviation_recompute/` |
| Query | `src/hstu_kvcache/baselines/query_recompute/` | `tests/selective_recompute/query_recompute/` |

Shared arbitrary-position replay is `baselines/sparse_recompute.py`.
This directory retains timeline/cohort, causal panel preparation, calibration,
numerical evaluation, arithmetic accounting and matched-score aggregation used
by current dependencies. The completed queue, GPU worker and resource probes
have retired; their historical inputs and per-user outputs are no longer complete.

Each method starts from the same rolling Reuse state and discards temporary
repairs after scoring. Real events append natively; repaired baseline state is
never fed into another method. Costs include selection, calibration and replay.

The final summary remains readable by the retained figure generator:

```bash
python figures/src/selective_recompute_2026_09.py \
  --input results/selective_recompute_2026_09/summary.json \
  --out figures/out/selective_recompute_2026_09
```

CPU checks: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/selective_recompute`.
New formal measurements require newly prepared inputs and launch authorization.
