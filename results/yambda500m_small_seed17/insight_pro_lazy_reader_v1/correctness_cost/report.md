# Lightweight PRO fused-reader correctness and cost

Progression gate: **FAIL**; labels or request scores read: **no**.

The action materializes zero version-translated prefix positions. A materialized prefix was used only as the sealed numerical reference for the fused AV identity.

| carriers | GFLOPs/user | of Full | reduction vs Full | logical Parent read | FP32 sidecar write |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.033 | 5.2% | 94.8% | 2.00 MiB | 2.0 KiB |
| 32 | 0.057 | 9.1% | 90.9% | 2.00 MiB | 2.0 KiB |

Maximum fused-reference or replay absolute error: `0.015625` (threshold `2e-05`).

| edge | carriers | direction cosine mean | direction cosine median | norm ratio median | relative L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 -> v1 | 16 | 0.9570 | 0.9701 | 0.9716 | 0.2699 |
| v0 -> v1 | 32 | 0.9989 | 0.9991 | 0.9731 | 0.0545 |
| v1 -> v2 | 16 | 0.9466 | 0.9529 | 1.0208 | 0.3258 |
| v1 -> v2 | 32 | 0.9992 | 0.9993 | 0.9942 | 0.0395 |
| v2 -> v3 | 16 | 0.9079 | 0.9645 | 1.0453 | 0.4507 |
| v2 -> v3 | 32 | 0.9992 | 0.9995 | 0.9975 | 0.0371 |
| v3 -> v4 | 16 | 0.9246 | 0.9587 | 1.0610 | 0.4431 |
| v3 -> v4 | 32 | 0.9976 | 0.9981 | 0.9524 | 0.0852 |
| v4 -> v5 | 16 | 0.8830 | 0.9495 | 1.0824 | 0.4842 |
| v4 -> v5 | 32 | 0.9990 | 0.9993 | 0.9865 | 0.0525 |

The legacy-sidecar comparison is diagnostic only: it measures the approximation introduced by Parent-conditioned carriers and does not admit score or task quality. The one-time version-map pseudoinverse and logical state bytes are reported separately from the per-user FLOP headline.
