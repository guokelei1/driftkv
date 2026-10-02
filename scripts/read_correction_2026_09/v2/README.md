# Correction v2 shared utilities and historical execution

The v2 execution queue is retired. Its shared calibration, fitting, reading
and cost functions remain dependencies of the selected nonlinear Q-v5 and
H-v4 probes. Current selection is Q C128/256/512 (45 points) plus unchanged
H C64/128/256 (45 points); see
[the final results](../../../results/read_correction_2026_09/motivation_final/README.md).

The user authorized preparation and tmux launch on 2026-09-28. This iteration
uses independent settings, artifacts and logs from v1. See the
[experiment record](../../../results/read_correction_2026_09/v2/README.md).

- `calibrate.py`: shared causal cache preparation and calibration dispatch.
- `query_only/fit.py`: query-only final-output distillation.
- `history_conditioned/fit.py`: H fitting with cached frozen lower-layer states.
- `aggregate.py`: matched-request AUC and complete cost aggregation.

The former worker, probe, resource-probe and launch/runner entry points are
retired. The completed run used the existing 3,000-user panels for all 15 edges.
Historical hashes and results remain unchanged; the old queue cannot resume
against those hashes in the cleaned source tree. Saved results remain plottable.
