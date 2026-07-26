# Current documentation index

This directory contains only current research state, protocol, implementation planning, and
advisor-facing context. Historical design exploration is recoverable from Git history but is not
kept beside active instructions.

## Precedence

When documents disagree, use this order:

1. [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md) — authoritative research
   state, supported claims, open gates, and stop conditions.
2. [eval_protocol.md](eval_protocol.md) — authoritative experiment semantics, comparability, and
   valid artifact boundary.
3. [09_single_configuration_full_chain_plan.md](09_single_configuration_full_chain_plan.md) —
   active implementation sequence for the KuaiRand long-context vertical slice.
4. [CohortKV target manuscript](../paper/cohortkv/manuscript_v3_target_en.md) — the paper shape to
   test, not a source for unmeasured facts or implementation semantics.
5. [paper_draft_intro_motivation.md](paper_draft_intro_motivation.md) — advisor-facing narrative.
6. [dataset_expansion_audit.md](dataset_expansion_audit.md) — dataset semantics, capacity, accepted
   ordered-exposure settings, and negative boundaries.

The target manuscript may describe expected findings using `TBD`. Those statements remain
hypotheses until a protocol and result artifact listed by `eval_protocol.md` support them.

## Active execution phase

The current phase is **single-configuration full-chain development**:

- KuaiRand 4+12 long-context data;
- the 16-layer, hidden/K/V-width-512 seed-0 model chain;
- theta0/theta4/theta10 to theta11 cohorts at the theta11/D16 endpoint;
- compiler, closest baselines, capsule/operator, full-cohort engine, guard/fallback, failure
  semantics, and capsule economics;
- HBM and DRAM as the primary destinations; a named SSD only as a later backend measurement.

This phase is development evidence. After one complete frozen run, the order is new seeds on the
same configuration, then cross-dataset/model-capacity expansion. The detailed stages and
same-interface fallback rules are in the active plan. Stage 0 is complete: the machine-readable
blueprint, 682-record workload manifest, and result schema are under
`configs/cohortkv_single_config_v1/`. Its re-audit separates internal/raw user identity, makes
residual hidden-suffix storage explicit, enforces all 18 primary system points, and requires HBM
capacity preflight; Stage 1 is the next implementation step.

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
Their useful current content is now represented by the authoritative roadmap, target manuscript,
and single-configuration plan.

The old background PNG was also removed after the paper-native problem-and-scope figure superseded
it. All removed tracked files remain recoverable from Git history and must not be cited as current
state.
