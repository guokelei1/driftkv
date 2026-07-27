# Figure sources

All figures are paper-native SVGs. They are editable text/vector artifacts and do not depend on
external images.

- `01_problem_and_scope.svg`: conceptual problem and scope figure; no measured values.
- `02_architecture.svg`: implemented three-layer architecture and current contract.
- `03_evidence_ladder.svg`: values drawn from
  `cohort_tiered_migration_v1_summary.json`,
  `configs/cohortkv_single_config_v1/stage2_compiler_summary.json`,
  `kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json`, and
  `configs/cohortkv_single_config_v1/stage4_system_summary.json`.
- `04_execution_pipeline.svg`: current host-staged v2/v4 dataflow; the source-materialization
  limitation is shown in the figure.
- `05_admission_signals.svg`: Stage 0 result skeleton for the capacity and age-signal panels.
- `06_frozen_contract.svg`: Stage 0 result skeleton for the threshold sweep and seed replication.
- `07_pareto_frontier.svg`: measured adaptive seed-0 Stage-1 resident frontier; 177-point source
  artifact frozen by `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`.
- `08_full_cohort_breakdown.svg`: measured Stage-4 HBM/DRAM compiled-versus-exact wall-time and
  source-read decomposition from `configs/cohortkv_single_config_v1/stage4_system_summary.json`.
- `09_capsule_economics.svg`: Stage 0 result skeleton for precision and break-even measurements.

The evidence ladder deliberately labels each protocol class and states that its cards are not one
statistical series.

Files `05`, `06`, and `09` contain axes, panel roles, and artifact requirements only. Their
dashed placeholders are not data and must be replaced from protocol-valid JSON before submission.
Files `07` and `08` are measured development evidence and must be replaced or explicitly retained
as adaptive seed-0 if later replication changes the plotted claims.
