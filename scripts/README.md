# Script inventory

The repository is now organized around two roles: reproduce the frozen 4L
development round, and prepare the frozen 8L scale validation. A filename does
not authorize an experiment; `docs/current_route.md` and the matching contract do.

## Frozen evidence chain

- `build_p7_*`, `fit_p7_*`, `train_p7_*`, `eval_p7_*`, `seal_p7_*`,
  `adjudicate_p7_*`: N/R/F materialization, Frozen Base, theta0 and H.
- `build_p8_*`, `train_p8_*`, `eval_p8_*`, `seal_p8_*`, `adjudicate_p8_*`:
  R0/R1/R2 release chain and H/S.
- `eval/run/seal/adjudicate/analyze_p9_*`: tomography, dependency-closed actions,
  rolling lineage, full population, runtime and oracle frontier.
- `eval/run/seal/adjudicate_p10_*`: sparse target-free profiler, sealed policy
  quality, same-cost baselines and grouped executor.
- `eval/run/seal/adjudicate_p11_*`: true recursive lineage, version debt,
  frozen scheduler transfer and rolling quality.

These scripts are retained for audit/reproduction. They may not be used to tune
the frozen 4L method.

## Negative controls retained

- `cc_p5_seenmix_requalification.py` and
  `cc_p6_identifiability_adjudication.py` preserve the next-listen No-Go.
- Their `cc_*` dependencies remain only to reproduce that evidence.
- `train_yambda_two_edges.py` and `train_yambda_theta0_medium.py` remain because
  regression tests use their temporal/cache primitives; they are not the active
  experiment runner.

## 8L scale reproduction

The 8L/H256/context1024 path is controlled by
`configs/contracts/scale_8l_v1.yaml`:

- `audit_scale_8l_resources.py`: S1 model/history/resource audit;
- `eval_scale_8l_correctness_canary.py`: real-data cache/executor correctness;
- `eval_scale_8l_fsdp_preflight.py`: four-rank backward/Adam/checkpoint canary;
- `train_scale_8l_fsdp_theta0.py`: three-epoch M0-F/M1 theta0 training;
- `train_scale_8l_fsdp_release.py`: admitted R0/R1/R2 release training;
- `run_scale_8l_queue.py`: resumable scientific-gate-aware queue.

Inspect with `PYTHONPATH=src:scripts python scripts/run_scale_8l_queue.py
--status`. Long training occupies GPU 0–3 as one FSDP job; do not launch the
retained single-process helper entry points. The queue runs M0-F seed17 through
the H/release pilot gates first; it does not blindly train all six theta0 models.

## Retired tools

Pre-P7 Q_main, neutral-readout, metadata-controller, large-candidate and old
No-op/Exact frontier scripts were removed. The deletion boundary and retained
exceptions are documented in `docs/legacy/retired_pre_p7_tools.md`.
# 8L frozen-method scale replay

The remaining 8L M0-F seed17 method-validation stages are exposed as one
resumable fail-closed queue:

```bash
PYTHONPATH=src:scripts python scripts/run_scale_8l_method_full.py --status
PYTHONPATH=src:scripts python scripts/run_scale_8l_method_full.py --run
```

It trains the output-only R0 control, runs frozen legal-action canaries and
all-state cutover probes, seals raw cells, replays the fixed Ridge scheduler,
then joins quality only after assignments are sealed. It never reads theta3.
