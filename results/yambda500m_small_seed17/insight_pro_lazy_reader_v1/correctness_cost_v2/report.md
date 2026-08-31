# Lightweight PRO fused-reader correctness and cost

Progression gate: **PASS**; labels or request scores read: **no**.

The action materializes zero version-translated prefix positions. A materialized prefix was used only as the sealed numerical reference for the fused AV identity.

| carriers | GFLOPs/user | of Full | reduction vs Full | logical Parent read | FP32 sidecar write |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.033 | 5.2% | 94.8% | 5.00 MiB | 2.0 KiB |
| 32 | 0.057 | 9.1% | 90.9% | 5.00 MiB | 2.0 KiB |

Maximum fused-reference absolute error retained: `0.015625`; maximum relative L2: `4.7269427e-06`; maximum replay absolute error: `3.5762787e-07`.

Outside the release-time headline, bounded-horizon serving adds 512 coverage-scale multiplies per user request and 512 AV additions per candidate (4 layers x 128 width), plus sidecar I/O.

| edge | carriers | direction cosine mean | direction cosine median | norm ratio median | relative L2 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 -> v1 | 16 | 0.9646 | 0.9782 | 0.9698 | 0.2494 |
| v0 -> v1 | 32 | 0.9989 | 0.9992 | 0.9828 | 0.0510 |
| v1 -> v2 | 16 | 0.9419 | 0.9658 | 1.0330 | 0.3161 |
| v1 -> v2 | 32 | 0.9993 | 0.9993 | 0.9933 | 0.0364 |
| v2 -> v3 | 16 | 0.8870 | 0.9520 | 1.0583 | 0.4849 |
| v2 -> v3 | 32 | 0.9992 | 0.9994 | 0.9982 | 0.0367 |
| v3 -> v4 | 16 | 0.9103 | 0.9492 | 1.0007 | 0.4319 |
| v3 -> v4 | 32 | 0.9983 | 0.9991 | 0.9903 | 0.0649 |
| v4 -> v5 | 16 | 0.8761 | 0.9395 | 1.0718 | 0.4976 |
| v4 -> v5 | 32 | 0.9991 | 0.9995 | 0.9939 | 0.0456 |

The legacy-sidecar comparison is diagnostic only: it measures the approximation introduced by Parent-conditioned carriers and does not admit score or task quality. The one-time version-map pseudoinverse and logical state bytes are reported separately from the per-user FLOP headline.
