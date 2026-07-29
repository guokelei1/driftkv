# Experiment records

This directory contains protocol-scoped result notes, including positive results, negative
results, controls, and implementation diagnostics. These files preserve what was measured; they
do not define the current EvoKV architecture or next task.

Use:

1. [../docs/08_core_insights_and_roadmap.md](../docs/08_core_insights_and_roadmap.md) for the
   current interpretation and roadmap;
2. [../docs/eval_protocol.md](../docs/eval_protocol.md) for valid protocol families and
   comparability;
3. the individual record for commands, artifacts, metrics, and limitations.

Historical identifiers such as `cohortkv_*` and `streamkv_*` are immutable protocol/artifact
names. They are not renamed when the paper/system name changes to EvoKV.

## Evidence classes

| Class | Permitted use |
|---|---|
| frozen/replicated result | claim only within the exact recorded protocol and population |
| adaptive seed-0 result | mechanism or configuration evidence; not independent confirmation |
| development diagnostic | debugging, characterization, and protocol design only |
| negative result | supports a scoped rejection or boundary; never silently discarded |
| interface/smoke validation | correctness of the exercised path only; no performance claim |

No current D3 result exists. In particular, destination-v4, normalized-capsule DRAM, and hot-HBM
direct-old-K/V records cannot be relabeled as D3 evidence.

## Validity and structural baselines

- [validity/README.md](validity/README.md)
- [validity/STREAMING_VALUE_CONTROL.md](validity/STREAMING_VALUE_CONTROL.md)
- [validity/LAYERWISE_METHOD.md](validity/LAYERWISE_METHOD.md)
- [validity/INTERVAL_ORACLE.md](validity/INTERVAL_ORACLE.md)

The suffix, interval, and early layerwise routes are historical structural baselines. They are not
the active D1 method.

## Scale, motivation, and dataset exposure

- [scaling/SCALING_V1.md](scaling/SCALING_V1.md)
- [scaling/KUAIRAND_FACTORIAL_V1.md](scaling/KUAIRAND_FACTORIAL_V1.md)
- [scaling/KUAIRAND_DATA_UTILIZATION_V1.md](scaling/KUAIRAND_DATA_UTILIZATION_V1.md)
- [motivation/JOINT_SCALE_V1.md](motivation/JOINT_SCALE_V1.md)
- [motivation/CAPACITY_V2.md](motivation/CAPACITY_V2.md)
- [motivation/LONG_CONTEXT_8PLUS8_V2.md](motivation/LONG_CONTEXT_8PLUS8_V2.md)
- [motivation/LONG_CONTEXT_SPLIT_EXPLORATION_V1.md](motivation/LONG_CONTEXT_SPLIT_EXPLORATION_V1.md)
- [exposure/ORDERED_EXPOSURE_V1.md](exposure/ORDERED_EXPOSURE_V1.md)
- [exposure/CACHE_VERSION_MATRIX_V1.md](exposure/CACHE_VERSION_MATRIX_V1.md)
- [exposure/OPPORTUNITY_REGIME_V1.md](exposure/OPPORTUNITY_REGIME_V1.md)

These records establish workload and generality boundaries. Weak or rejected cells remain useful
negative evidence and must not be pooled with another protocol.

## D1 method records

- [migration/PROGRESSIVE_PREFIX_REPLAY_V1.md](migration/PROGRESSIVE_PREFIX_REPLAY_V1.md) —
  structural baseline;
- [migration/COMPILED_LOW_RANK_V1.md](migration/COMPILED_LOW_RANK_V1.md) — compiled residual
  screen;
- [migration/COHORT_TIERED_MIGRATION_V1.md](migration/COHORT_TIERED_MIGRATION_V1.md) — active D1
  action library;
- [migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md](migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md) —
  long-context discovery;
- [migration/VERIFIED_COHORT_COMPILER_V1.md](migration/VERIFIED_COHORT_COMPILER_V1.md) — frozen
  compiler/certificate boundary.

## Frozen D1 system chain

- [system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md](system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md)
- [system/COHORTKV_STAGE1_FRONTIER_V1.md](system/COHORTKV_STAGE1_FRONTIER_V1.md)
- [system/COHORTKV_STAGE2_COMPILER_V1.md](system/COHORTKV_STAGE2_COMPILER_V1.md)
- [system/COHORTKV_STAGE3_OPERATOR_V1.md](system/COHORTKV_STAGE3_OPERATOR_V1.md)
- [system/COHORTKV_STAGE4_SYSTEM_V1.md](system/COHORTKV_STAGE4_SYSTEM_V1.md)
- [system/COHORTKV_STAGE4_5_SOURCE_PLAN_V1.md](system/COHORTKV_STAGE4_5_SOURCE_PLAN_V1.md)
- [system/COHORTKV_STAGE4_6_LIFECYCLE_V1.md](system/COHORTKV_STAGE4_6_LIFECYCLE_V1.md)
- [system/COHORTKV_STAGE4_7_ORGANIC_LIFECYCLE_V1.md](system/COHORTKV_STAGE4_7_ORGANIC_LIFECYCLE_V1.md)
- [system/COHORTKV_STAGE4_8_SCHEDULER_SWEEPS_V1.md](system/COHORTKV_STAGE4_8_SCHEDULER_SWEEPS_V1.md)
- [system/COHORTKV_STAGE4_9_ROLLOUT_BOUNDARY_V1.md](system/COHORTKV_STAGE4_9_ROLLOUT_BOUNDARY_V1.md)
- [system/COHORTKV_STAGE4_10_RENEWAL_CALIBRATED_V1.md](system/COHORTKV_STAGE4_10_RENEWAL_CALIBRATED_V1.md)
- [system/COHORTKV_STAGE5_MINIMAL_CLOSURE_V1.md](system/COHORTKV_STAGE5_MINIMAL_CLOSURE_V1.md)
- [system/COHORTKV_STAGE6_SINGLE_CONFIG_FREEZE_V1.md](system/COHORTKV_STAGE6_SINGLE_CONFIG_FREEZE_V1.md)

Stage 4's normalized source is a negative economics result. Stage 4.5 is a hot-HBM direct-old-K/V
result. Stages 4.7/4.8 use an older append accounting boundary. Stage 4.10 is an unselected
two-edge smoke. Consult the D1 evidence ledger before using any number:
[../docs/09_single_configuration_full_chain_plan.md](../docs/09_single_configuration_full_chain_plan.md).

## Historical system prototypes

- [system/STREAMKV_SYSTEM_PROTOTYPE_V1.md](system/STREAMKV_SYSTEM_PROTOTYPE_V1.md)
- [system/KUAIRAND_PROGRESSIVE_SYNC_V1.md](system/KUAIRAND_PROGRESSIVE_SYNC_V1.md)
- [system/TWO_GPU_MIGRATION_SYSTEM_V2.md](system/TWO_GPU_MIGRATION_SYSTEM_V2.md)
- [system/COHORT_JAGGED_SYSTEM_V3.md](system/COHORT_JAGGED_SYSTEM_V3.md)
- [system/FOUR_GPU_SCALING_V1.md](system/FOUR_GPU_SCALING_V1.md)
- [system/DESTINATION_OUT_OF_CORE_V4.md](system/DESTINATION_OUT_OF_CORE_V4.md)

These records document the route that exposed source-state, endpoint, layout, and transaction
issues. Their former compiler→capsule-operator→destination-engine contribution framing is
superseded. They do not define current D2 or D3.

## D2 development

Current D2 development artifacts are indexed in
[../docs/future_design/DESIGN2_DEVELOPMENT_STATUS.md](../docs/future_design/DESIGN2_DEVELOPMENT_STATUS.md).
They are intentionally not represented as a formal paper-result record because no D2 protocol is
frozen yet. W3 integrated timings, full-payload validation, and synthetic lookup contention remain
`scientific_result=false`.

## D3

There is no D3 experiment record. The first implementation must begin from
[../docs/future_design/DESIGN3_FUTURE_DIRECTION.md](../docs/future_design/DESIGN3_FUTURE_DIRECTION.md)
by exporting and hashing the capacity-independent D2 constraint view. It must create
development-only artifacts before any protocol is frozen.
