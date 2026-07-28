# Current documentation index

This directory contains only current research state, protocol, and implementation planning.
Historical design exploration is recoverable from Git history but is not kept beside active
instructions.

## Precedence

When documents disagree, use this order:

1. [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md) — authoritative research
   state, supported claims, open gates, and stop conditions.
2. [eval_protocol.md](eval_protocol.md) — authoritative experiment semantics, comparability, and
   valid artifact boundary.
3. [09_single_configuration_full_chain_plan.md](09_single_configuration_full_chain_plan.md) —
   active implementation sequence for the KuaiRand long-context vertical slice.
4. [10_target_manuscript_stage0_6_correspondence.md](10_target_manuscript_stage0_6_correspondence.md)
   — correspondence between the pre-rewrite target manuscript and the frozen Stage 0–6 evidence,
   including the resulting paper reframing and remaining gaps.
5. [Current CohortKV manuscript](../paper/cohortkv/manuscript_v3_target_en.md) — the
   evidence-bound English draft rewritten from the completed Stage 0–6 vertical slice.
6. [dataset_expansion_audit.md](dataset_expansion_audit.md) — dataset semantics, capacity, accepted
   ordered-exposure settings, and negative boundaries.

The current manuscript contains no `TBD` placeholders. Open replications and implementation gaps
are stated as limitations rather than filled with seed-0 numbers. It remains subordinate to the
roadmap and evaluation protocol when experimental semantics conflict.

## Current execution phase

The **single-configuration full-chain development** phase is complete. Its frozen configuration
was:

- KuaiRand 4+12 long-context data;
- the 16-layer, hidden/K/V-width-512 seed-0 model chain;
- theta0/theta4/theta10 to theta11 cohorts at the theta11/D16 endpoint;
- compiler, closest baselines, operator, full-cohort engine, migrate/exact lifecycle, and a
  minimal preflight/fallback/atomic-publication closure;
- HBM and DRAM as the primary destinations, plus an artifact-derived source-state accounting
  table. INT8/capture/SSD work is optional post-v1.

This package remains development evidence. A post-freeze Stage-4.10 amendment is now active before
Stage 7: H12 scheduled-exact records also calibrate the per-edge shared direct program, with build
cost included in `U` and no separate fit-only exact cohort or semantic admission gate. Its
two-edge smoke is complete, but full-chain task quality is not; new-seed and cross-capacity
replication waits for that formal closure. The completed stages and same-interface fallback rules
are in the active plan. Stage 0 is complete: the machine-readable
blueprint, 682-record workload manifest, and result schema are under
`configs/cohortkv_single_config_v1/`. Its re-audit separates internal/raw user identity, makes
residual hidden-suffix storage explicit, enforces all 18 primary system points, and requires HBM
capacity preflight. Stage 1 is also complete: the 177-point selective-contiguous frontier and
certificate are frozen in `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`.
No selective interval certifies; `m12/layers0-11` is retained only as a diagnostic external
baseline, while exact is its publishable fallback. Stage 2 is also complete: serialized deployed
certificates, FP16 runtime programs, threshold sweeps, and executable fallback plans are frozen in
`configs/cohortkv_single_config_v1/stage2_compiler_summary.json` and `stage2_plans/`. The measured
residual hidden suffix uses BF16 after real FP16 overflow; the primary capsule/program/output path
remains FP16. Stage 3 is also complete: reference, packed, and fused paths share one contiguous
unpadded output-extent API, all nine batch/bucket layouts pass full valid-element and padding
checks, and the resident development default is fused FP16 with batch 4 and bucket width 32. The
frozen record is `configs/cohortkv_single_config_v1/stage3_operator_summary.json`. Stage 4 is
also complete: all 30 full-cohort HBM/DRAM × 1/2/4-GPU method/control points pass capacity,
transport, and manifest checks. Compiled remains 2.70–3.49× faster than the certificate-failed
selective diagnostic but loses to exact at all six matched endpoints because the 17.82-GB FP16
capsule source path consumes 91.35%–96.91% of completion. The frozen record is
`configs/cohortkv_single_config_v1/stage4_system_summary.json`. Stage 4.5 is also complete and
frozen in `stage4_5_source_plan_summary.json`: a direct affine over the already resident old K/V
eliminates extra `Norm(x)` state, preserves the deployed certificate and full real transport, and
beats paired HBM-resident raw-history exact at the complete 1/2/4-GPU cohort points. The result is
limited to the declared existing-old-K/V hot-HBM regime. Stage 4.6 is now frozen in
`stage4_6_lifecycle_policy.json` and `stage4_6_lifecycle_summary.json`: one
KuaiRand/seed-0/one-A40 `theta0 -> theta11` chain uses a balanced 15%–25% exact-refresh schedule,
maximum migration depth four, and the actual previous output at every update. It costs 0.2134×
all-exact GPU time on the complete 682-record chain. The rejected per-cache threshold remains a
refresh-wave negative result. Stage 4.7 completed the canonical-date growing-history chain, and
Stage 4.8 completed all sixteen label-free scheduler development points. Stage 4.9 then completed
the corrected 11-edge same-device retained-prefix confirmation: `token_debt_total10` is the
`0.071319×` cost endpoint, and `staggered_renewal_h12` is the frozen bounded-renewal candidate at
`0.100017×`, with record-weighted AUC/NDCG@100/Hit@100 recovery
`1.000039/0.997463/1.000000`. Target-model append remains outside both timers. Its groupwise
host-staged evaluator reports state movement separately and does not support a full-cohort
HBM-resident or end-to-end movement claim.

The merged minimal Stage 5 formal two-A40 copy-on-write integration also passes: normal and
semantic-fallback jobs commit all 682 records, while mid-job and pre-commit faults expose no
partial target and preserve readback-valid old state. Stage 6 completes the single-configuration
package with a deterministic CPU-only assembly over frozen Stage-1 through Stage-5 artifacts,
publishing `final_summary_seed0.json` and eight checked sidecars without rerunning the old GPU
matrix. Runtime sentinel, online rework/resume, INT8/capture, and SSD benchmarks remain optional.
Stage 4.10 then adds non-scientific inverse-Norm and direct-K/V renewal-calibrated smokes over
`theta0 -> theta1 -> theta2`; neither is yet selected. The next experimental work is its
same-device full-chain quality/cost confirmation, followed by Stage 7.

## Key experiment records

These are navigation entry points, not an exhaustive artifact registry; the protocol remains the
source of truth.

Motivation and scale:

- [Validity overview](../experiments/validity/README.md)
- [Streaming value control](../experiments/validity/STREAMING_VALUE_CONTROL.md)
- [Interval oracle](../experiments/validity/INTERVAL_ORACLE.md)
- [Scaling v1](../experiments/scaling/SCALING_V1.md)
- [KuaiRand factorial](../experiments/scaling/KUAIRAND_FACTORIAL_V1.md)
- [KuaiRand data utilization](../experiments/scaling/KUAIRAND_DATA_UTILIZATION_V1.md)
- [Ordered-exposure reproduction](../experiments/exposure/ORDERED_EXPOSURE_V1.md)
- [Cache-version matrix](../experiments/exposure/CACHE_VERSION_MATRIX_V1.md)
- [Long-context opportunity boundary](../experiments/exposure/OPPORTUNITY_REGIME_V1.md)
- [Capacity v2](../experiments/motivation/CAPACITY_V2.md)
- [KuaiRand 4+12 split](../experiments/motivation/LONG_CONTEXT_SPLIT_EXPLORATION_V1.md)

Migration:

- [Compiled low-rank v1](../experiments/migration/COMPILED_LOW_RANK_V1.md)
- [Progressive prefix baseline](../experiments/migration/PROGRESSIVE_PREFIX_REPLAY_V1.md)
- [Cohort-tiered migration](../experiments/migration/COHORT_TIERED_MIGRATION_V1.md)
- [Long-context compiled search](../experiments/migration/LONG_CONTEXT_COMPILED_SEARCH_V1.md)
- [Verified cohort compiler](../experiments/migration/VERIFIED_COHORT_COMPILER_V1.md)

System:

- [Single-configuration full-chain v1](../experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md)
- [Stage 1 selective frontier](../experiments/system/COHORTKV_STAGE1_FRONTIER_V1.md)
- [Stage 2 deployed compiler certificate](../experiments/system/COHORTKV_STAGE2_COMPILER_V1.md)
- [Stage 4.5 direct old-K/V source plan](../experiments/system/COHORTKV_STAGE4_5_SOURCE_PLAN_V1.md)
- [Stage 3 capsule/operator](../experiments/system/COHORTKV_STAGE3_OPERATOR_V1.md)
- [Stage 4 full-cohort system](../experiments/system/COHORTKV_STAGE4_SYSTEM_V1.md)
- [Stage 4.7 growing-history lifecycle](../experiments/system/COHORTKV_STAGE4_7_ORGANIC_LIFECYCLE_V1.md)
- [Stage 4.8 scheduler sweeps](../experiments/system/COHORTKV_STAGE4_8_SCHEDULER_SWEEPS_V1.md)
- [Stage 4.9 rollout boundary](../experiments/system/COHORTKV_STAGE4_9_ROLLOUT_BOUNDARY_V1.md)
- [Stage 4.10 renewal-calibrated program](../experiments/system/COHORTKV_STAGE4_10_RENEWAL_CALIBRATED_V1.md)
- [Stage 5 minimal implementation closure](../experiments/system/COHORTKV_STAGE5_MINIMAL_CLOSURE_V1.md)
- [Stage 6 single-configuration freeze](../experiments/system/COHORTKV_STAGE6_SINGLE_CONFIG_FREEZE_V1.md)
- [Two-GPU controlled migration](../experiments/system/TWO_GPU_MIGRATION_SYSTEM_V2.md)
- [Cohort-jagged negative result](../experiments/system/COHORT_JAGGED_SYSTEM_V3.md)
- [Four-GPU controlled scaling](../experiments/system/FOUR_GPU_SCALING_V1.md)
- [Destination-oriented out-of-core v4](../experiments/system/DESTINATION_OUT_OF_CORE_V4.md)

Earlier experiment records remain valid only within their recorded protocol. They are historical
evidence, not current implementation instructions.

## Artifact boundary

`eval_protocol.md` is the only maintained list of valid result families. Do not duplicate that
inventory here.

- Raw per-seed files and checkpoints may remain local and ignored.
- Tracked aggregates must retain their protocol and evidence-level metadata.
- Smoke, plan-only, synthetic interface, and controlled trace outputs cannot be promoted to
  full-cohort or replicated evidence.
- Result families with different protocol strings cannot be pooled.

## Retired documentation

The former project-wide review snapshot and the large system-candidate exploration were removed
from the active directory. They mixed current tasks with speculative serving, second-hardware,
larger-model, compression, and alternate-paper directions that are outside the present scope.
Their useful current content is now represented by the authoritative roadmap, current manuscript,
and single-configuration plan.

The old background PNG was also removed after the paper-native problem-and-scope figure superseded
it. All removed tracked files remain recoverable from Git history and must not be cited as current
state.
