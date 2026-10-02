# Shared query/read correction arithmetic

CPU metadata-only estimate per release. Pilot calibration is charged once even though full evaluation reuses its parameters.

Shapes: 6 layers, 6 heads × 32, 1,024 history events; 256 calibration users × 16 candidates; 2560 evaluation users × 32 candidates.

One cache-only rebuild: 4.1971 G operations. One eager calibration prefill: 7.3477 G. Ordinary 32-candidate read: 228.530 M.

| Predictor | Pilot K (T ops) | Extra/user (M ops) | Extra/native read | Extra/rebuild | (K + 2560 extra)/(2560 rebuild) |
|---|---:|---:|---:|---:|---:|
| shared_query | 3.9593 | 2.658 | 1.163% | 0.063% | 36.913% |
| shared_query_response | 3.9723 | 16.961 | 7.422% | 0.404% | 37.374% |

K includes Current prefill on 256 calibration users and six one-layer teacher reads. Shared serving uses no evaluation-user teacher. Evaluation-user Exact measurements are research costs, excluded from K and serving. This diagnostic does not run a per-user oracle.

The last column counts additional compatibility computation, with one 32-candidate group per user. It is not an end-to-end time ratio. Native reading remains necessary; the JSON also reports the arithmetic ratio when ordinary reads are included. Repeated requests add correction costs.

Calibration includes both cache prefills, six full source reads, six one-layer teacher reads, 15 prior-layer corrections, and all six ridge fits including residual checks and prediction diagnostics. Empty read columns cost no floating arithmetic. Reductions/LU/special functions use the stated approximation; memory and execution overhead are excluded.

Historical generator (retired): `PYTHONPATH=src:scripts python scripts/design/analyze_query_read_probe.py`.
