# Frozen D1 single-configuration evidence ledger

> Historical status: the KuaiRand Stage 0–6 vertical slice is complete and frozen. This file keeps
> its evidence map and negative results; it is no longer an execution plan. The current paper
> architecture is D1 `ActionPlan` → D2 `WavePlan` constraints → D3 `ResidencyPlan`. Historical
> component numbers and destination prototypes in the frozen artifacts do not define current D2 or
> D3.

The authoritative current state is
[08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md). Result semantics are defined by
[eval_protocol.md](eval_protocol.md).

## 1. Frozen configuration

The single-configuration package uses:

- KuaiRand 4+12 long-context preparation;
- one 16-layer, hidden/K/V-width-512, seed-0 model chain;
- 682 final records;
- theta0/theta4/theta10 source cohorts and theta11/D16 endpoints for the original one-hop package;
- a theta0→theta11 adjacent-version lifecycle for repeated-input studies;
- disjoint 40/60/60/522 fit/selection/certificate/final user roles where applicable;
- explicit source, target, dtype, layout, timing, failure, and manifest schemas under
  `configs/cohortkv_single_config_v1/`.

The package is an adaptive single-configuration systems slice. It is not a multi-seed paper
replication.

## 2. Stage ledger

| Stage | Frozen output | Main conclusion | Evidence boundary |
|---|---|---|---|
| 0 | blueprint, workload manifest, result schema | record identity, extents, roles, source representations, capacity, and failure points are explicit | plan/schema only |
| 1 | selective frontier | no selective contiguous interval passes the certificate; exact is its publishable fallback | resident seed-0 selection/certificate |
| 2 | compiled programs and executable plans | serialized FP16 full-affine programs pass the declared three-view certificate | resident compiler/certificate |
| 3 | reference, packed, and fused operators | one unpadded extent API; fused B4/bucket32 is the resident default | resident operator only |
| 4 | complete 1/2/4-GPU HBM/DRAM matrix | normalized FP16 capsule loses to exact because source processing dominates | normal-path source/destination experiment |
| 4.5 | direct-old-K/V source plan | direct transform over existing exact old K/V removes extra per-record normalized state and wins in hot HBM | one-hop, exact-source-version K/V |
| 4.6 | bounded-renewal lifecycle | repeated approximation is bounded by exact reset and maximum depth four | fixed-history, one seed, one A40 |
| 4.7–4.8 | growing-history and scheduler-development traces | label-free deterministic renewal policies are feasible; early threshold routing creates undesirable waves | development protocols with older append accounting |
| 4.9 | corrected rollout boundary | retained migration and method-common target append are separated; H12 is the frozen D2 input | host-staged lifecycle, movement separate |
| 5 | minimal guard/fallback/COW integration | preflight, exact fallback, abort visibility, and atomic publication close on one representative edge | correctness, not throughput |
| 6 | checked aggregate and sidecars | deterministic assembly binds Stage 1–5 artifacts without rerunning GPU experiments | aggregate only |
| 4.10 | renewal-calibrated smoke amendment | scheduled-exact records can also calibrate a shared program without extra fit-only exact users | two-edge smoke, not selected or frozen into Stage 6 |

## 3. Key artifacts

### Stage 0

- `configs/cohortkv_single_config_v1/blueprint.json`
- `configs/cohortkv_single_config_v1/workload_manifest.json`
- `configs/cohortkv_single_config_v1/result.schema.json`
- [experiment record](../experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md)

Stage 0 distinguishes internal model user index from raw log user ID. It also records that
progressive residual repair requires a BF16 hidden suffix after measured FP16 overflow; that
auxiliary state is not implicit in the normalized capsule.

### Stages 1–3

- `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`
- `configs/cohortkv_single_config_v1/stage2_compiler_summary.json`
- `configs/cohortkv_single_config_v1/stage2_plans/`
- `configs/cohortkv_single_config_v1/stage3_operator_summary.json`
- [Stage 1 record](../experiments/system/COHORTKV_STAGE1_FRONTIER_V1.md)
- [Stage 2 record](../experiments/system/COHORTKV_STAGE2_COMPILER_V1.md)
- [Stage 3 record](../experiments/system/COHORTKV_STAGE3_OPERATOR_V1.md)

The strongest selective diagnostic is `m12/layers0-11`, but it fails the certificate and remains an
external baseline only. Compiled full affine is the deployed D1 action in all three primary source
pairs. Reference, packed, and fused implementations share the same valid-element and padding
contract.

### Stage 4 and 4.5

- `configs/cohortkv_single_config_v1/stage4_system_summary.json`
- `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json`
- [normalized-source system record](../experiments/system/COHORTKV_STAGE4_SYSTEM_V1.md)
- [direct-old-K/V record](../experiments/system/COHORTKV_STAGE4_5_SOURCE_PLAN_V1.md)

The 17.82-GB normalized-capsule source consumes 91.35%–96.91% of compiled completion and loses to
exact at every matched endpoint. This is a frozen negative result.

The Stage-4.5 route composes each deployed affine through a minimum-norm right inverse of the source
K/V projection. It adds no per-record `Norm(x)` state, preserves the certificate, and passes full
real transport over all 682 records. At the complete hot-HBM boundary, its 1/2/4-GPU times are
0.930/0.494/0.255 seconds versus 18.695/9.729/4.766 seconds for paired exact. The result assumes
exact source-version old K/V and cannot be reinterpreted as a DRAM or D3 result.

### Stages 4.6–4.9

- `configs/cohortkv_single_config_v1/stage4_6_lifecycle_policy.json`
- `configs/cohortkv_single_config_v1/stage4_6_lifecycle_summary.json`
- `configs/cohortkv_single_config_v1/stage4_7_organic_summary.json`
- `configs/cohortkv_single_config_v1/stage4_8_exact_baseline.json`
- `results/system/cohortkv_single_config_full_chain_v1/stage4_9_same_device_confirmation_seed0.json`
- `results/system/cohortkv_single_config_full_chain_v1/stage4_9_staggered_renewal_h12_seed0.json`
- [Stage 4.6 record](../experiments/system/COHORTKV_STAGE4_6_LIFECYCLE_V1.md)
- [Stage 4.7 record](../experiments/system/COHORTKV_STAGE4_7_ORGANIC_LIFECYCLE_V1.md)
- [Stage 4.8 record](../experiments/system/COHORTKV_STAGE4_8_SCHEDULER_SWEEPS_V1.md)
- [Stage 4.9 record](../experiments/system/COHORTKV_STAGE4_9_ROLLOUT_BOUNDARY_V1.md)

Stage 4.6 proves recursive consumption of the previous actual output, exact reset, maximum depth
four, and label-free routing. The frozen 682-record chain costs 0.2134× all-exact GPU time.

Stages 4.7 and 4.8 use growing histories but an older source-model-append accounting order. They
remain useful scheduler-development evidence and must not be pooled with Stage 4.9.

Stage 4.9 freezes the corrected boundary: migrate the retained prefix first, stop both maintenance
timers, then perform identical target-model append. Its `staggered_renewal_h12` action partition is
the immutable D2 input. The result is host-staged and reports movement separately.

### Stages 5–6

- `results/system/cohortkv_single_config_full_chain_v1/stage5_full_cow_theta0_theta1_seed0.json`
- `results/system/cohortkv_single_config_full_chain_v1/stage5_source_state_accounting_seed0.json`
- `results/system/cohortkv_single_config_full_chain_v1/final_summary_seed0.json`
- [Stage 5 record](../experiments/system/COHORTKV_STAGE5_MINIMAL_CLOSURE_V1.md)
- [Stage 6 record](../experiments/system/COHORTKV_STAGE6_SINGLE_CONFIG_FREEZE_V1.md)

Stage 5 commits normal and semantic-fallback jobs only after complete coverage. Mid-job and
pre-commit faults expose no partial target and preserve readback-valid old state. Stage 6 is a
CPU-only checked assembly; it creates no new GPU result.

### Stage 4.10 amendment

- `results/system/cohortkv_single_config_full_chain_v1/stage4_10_renewal_calibrated_h12_direct_kv_residual_ridge_2edges_seed0.json`
- `results/system/cohortkv_single_config_full_chain_v1/stage4_10_renewal_calibrated_h12_inverse_norm_ridge_2edges_seed0.json`
- [experiment record](../experiments/system/COHORTKV_STAGE4_10_RENEWAL_CALIBRATED_V1.md)

Both candidates are two-edge, one-repeat smokes marked `scientific_result=false`. Neither replaces
the frozen Stage-4.9 program/action inputs. A new full-chain paired quality-and-cost confirmation is
required before adoption.

## 4. Frozen negative results

- The closest selective layer baseline fails the certificate.
- FP16 residual hidden suffix overflows; BF16 is required when that action exists.
- The 17.82-GB normalized source representation destroys end-to-end economics.
- A per-cache threshold controller produces severe exact-refresh waves.
- Stage 4.7/4.8 append ordering is not the final paired maintenance boundary.
- Stage 4.10 has not selected a calibration form.

These negative results must remain visible because they explain the selected D1 source plan and
lifecycle. They do not define D2 or D3.

## 5. Claims this ledger does not support

- multi-seed confirmation of the complete vertical slice;
- cold-source, DRAM, SSD, filesystem, remote, or D3 performance;
- automatic destination selection;
- a per-user reuse-safety predictor;
- serving latency or SLO impact;
- end-to-end movement savings for the Stage-4.9 lifecycle;
- a tensor-parallel dense model;
- formal D2 multi-GPU physical-lowering performance.

The frozen pre-EvoKV Markdown manuscript remains at
`paper/cohortkv/manuscript_v3_target_en.md` because Stage-6 scripts and the blueprint reference its
path. It is an artifact dependency, not a current manuscript or source of design truth.
