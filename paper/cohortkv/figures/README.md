# Figure sources

All figures are paper-native SVGs. They are editable text/vector artifacts and do not depend on
external images.

- `01_problem_and_scope.svg`: conceptual problem and scope figure; no measured values.
- `02_architecture.svg`: implemented three-layer architecture and current contract.
- `03_evidence_ladder.svg`: values drawn from
  `cohort_tiered_migration_v1_summary.json`,
  `long_context_4plus12_verified_compiler_seed0.json`, and
  `kuairand_long_context_4plus12_two_gpu_migration_system_seed0.json`.
- `04_execution_pipeline.svg`: current host-staged v2/v4 dataflow; the source-materialization
  limitation is shown in the figure.
- `05_admission_signals.svg`: Stage 0 result skeleton for the capacity and age-signal panels.
- `06_frozen_contract.svg`: Stage 0 result skeleton for the threshold sweep and seed replication.
- `07_pareto_frontier.svg`: Stage 0 result skeleton for the compiled/selective/residual frontier.
- `08_full_cohort_breakdown.svg`: Stage 0 result skeleton for separate DRAM/HBM completion stacks.
- `09_capsule_economics.svg`: Stage 0 result skeleton for precision and break-even measurements.

The evidence ladder deliberately labels each protocol class and states that its cards are not one
statistical series.

Files `05` through `09` contain axes, panel roles, and artifact requirements only. Their dashed
placeholders are not data and must be replaced from protocol-valid JSON before submission.
