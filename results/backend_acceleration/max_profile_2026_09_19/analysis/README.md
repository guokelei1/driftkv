# Max training performance diagnostic

CPU prepared fixture H2D + original FoundationForward/loss/backward/FSDP/AdamW; excludes original CPU row lookup/collate, startup, checkpoint save, trace overhead

| Variant | Max-rank step median (ms) | Mean (ms) | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|
| torch_default | 943.16 | 963.70 | 24490 | 38748 |
| auto_default | 759.47 | 804.86 | 12454 | 27758 |
| auto_fused | 690.45 | 709.53 | 12454 | 27758 |

| Comparison | Median speedup | Median time reduction | Mean speedup |
|---|---:|---:|---:|
| torch_default → auto_default | 1.242× | 19.48% | 1.197× |
| auto_default → auto_fused | 1.100× | 9.09% | 1.134× |
| torch_default → auto_fused | 1.366× | 26.79% | 1.358× |

Per-rank CUDA interval medians are shown below. Phases from different ranks, or separate phase medians, must not be added to reconstruct a whole-step wall time.

| Variant | Rank | H2D | Forward | Loss | Zero grad | Backward | Optimizer | Wall median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| torch_default | 0 | 0.58 | 387.49 | 0.02 | 0.00 | 510.61 | 43.38 | 942.15 |
| torch_default | 1 | 0.59 | 384.75 | 0.02 | 0.00 | 505.99 | 43.37 | 942.57 |
| torch_default | 2 | 0.56 | 382.49 | 0.02 | 0.00 | 512.50 | 43.39 | 943.05 |
| torch_default | 3 | 0.63 | 376.62 | 0.18 | 0.12 | 524.18 | 43.47 | 943.01 |
| auto_default | 0 | 0.61 | 259.91 | 0.02 | 0.03 | 447.26 | 43.38 | 758.38 |
| auto_default | 1 | 0.71 | 259.99 | 0.02 | 0.01 | 430.60 | 43.39 | 758.74 |
| auto_default | 2 | 0.65 | 259.55 | 0.02 | 0.00 | 449.01 | 43.39 | 759.11 |
| auto_default | 3 | 0.70 | 251.94 | 0.21 | 0.12 | 462.95 | 43.46 | 759.39 |
| auto_fused | 0 | 0.63 | 258.09 | 0.02 | 0.00 | 410.35 | 15.61 | 689.00 |
| auto_fused | 1 | 0.83 | 264.79 | 0.10 | 0.10 | 392.58 | 15.63 | 689.18 |
| auto_fused | 2 | 0.64 | 257.21 | 0.02 | 0.00 | 411.40 | 15.60 | 690.33 |
| auto_fused | 3 | 0.63 | 250.73 | 0.19 | 0.12 | 423.66 | 15.64 | 690.33 |

Profiler values below are milliseconds per profiled step. They describe the separate instrumented trace, not the clean timing run; kernel work may overlap across streams.

| Variant | Rank | Communication union | Noncommunication union | Overlap | Communication without concurrent noncommunication |
|---|---:|---:|---:|---:|---:|
| torch_default | 0 | 405.01 | 641.36 | 0.00 | 405.01 |
| torch_default | 1 | 502.90 | 584.40 | 0.00 | 502.90 |
| torch_default | 2 | 549.25 | 563.16 | 0.00 | 549.25 |
| torch_default | 3 | 623.18 | 520.29 | 0.00 | 623.18 |
| auto_default | 0 | 399.16 | 400.05 | 0.00 | 399.16 |
| auto_default | 1 | 489.26 | 340.90 | 0.00 | 489.26 |
| auto_default | 2 | 538.44 | 320.32 | 0.00 | 538.44 |
| auto_default | 3 | 607.30 | 276.73 | 0.00 | 607.30 |
| auto_fused | 0 | 434.87 | 372.18 | 0.00 | 434.87 |
| auto_fused | 1 | 481.84 | 312.94 | 0.00 | 481.84 |
| auto_fused | 2 | 567.33 | 292.32 | 0.00 | 567.33 |
| auto_fused | 3 | 635.27 | 248.83 | 0.00 | 635.27 |

NCCL durations include waiting for other ranks and are not pure network transfer time. Top kernels, collective CPU scope call counts, and inclusive prefix/query/embedding/operator work are retained per rank in summary.json. Inclusive operator totals and nested collective call counts cannot be added.

| Numerical comparison | Initial logits max abs | Global loss trajectory max abs | Parameter sample max abs | Sample relative L2 |
|---|---:|---:|---:|---:|
| torch_default → auto_default | 0 | 0.000145644 | 5.1856e-06 | 1.21475e-06 |
| auto_default → auto_fused | 0 | 0.000107694 | 3.45707e-06 | 6.53446e-07 |
| torch_default → auto_fused | 0 | 0.000145644 | 4.94719e-06 | 1.12982e-06 |

Parameter samples cover only a small deterministic subset of local shards; they do not establish full-state or long-term training equivalence.

Model context: 1,138,203,521 total parameters, of which the item embedding has 1,129,890,240 (99.27%). Parameter fraction is not a runtime fraction.

Original CPU collate median: 0.263 ms per local batch (prepared separately; excluded from clean step times).

No evaluation quality metrics were opened or used. These fixtures reproduce the first chronological batches of the earlier backend canary and do not establish later-window performance.

Input summary SHA256: `e94dd14b9c318d74c1366e9096eb8230539b088f72c8a3827614977847ee7084`
