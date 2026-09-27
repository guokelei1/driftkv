# Read-correction diagnostic arithmetic

This is an operation-count estimate from the retained runner and tensor shapes. It is not a population benchmark.

Six layers, width 192, history 1024; 256 calibration users × 16 candidates; 32 serving candidates per user.

Cache-only rebuild: 4.1971 G operations per user; literal calibration prefill: 7.3477 G; native 32-candidate read: 228.530 M.

| Shared predictor | Calibration K (T ops) | Extra read/user (M ops) | Extra/rebuild | Conditional U=30,000 ratio |
|---|---:|---:|---:|---:|
| shared_candidate | 3.9677 | 14.340 | 0.342% | 3.493% |
| shared_user_response | 3.9824 | 29.270 | 0.697% | 3.860% |

The conditional ratio is `(K + U*C_read)/(U*C_rebuild)`, assuming every user has 1,024 retained events and scores 32 candidates once. It includes both calibration prefills, response acquisition, previous-layer corrections, and ridge fitting. It excludes the native read common to both paths. It does not establish measured population FLOPs, speed, repeated-read amortization, or useful quality for the candidate-only arm.

Multiply-add counts as two; special functions count as one. Dense products follow the eager runner; reduction and LU costs are estimates. Memory traffic, transfers and dispatch are excluded. Evaluation-user teachers and per-user oracle fitting are research measurements, not inputs needed to serve either shared rule.

Reproduce: `PYTHONPATH=src:scripts python scripts/design/analyze_user_information_probe.py`.
