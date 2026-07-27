# CohortKV Stage 4.5 direct-old-K/V source plan

> Status: frozen seed-0 single-configuration development evidence, 2026-07-27.
>
> Frozen aggregate:
> `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json`.

## Question and decision

Stage 4 showed that the deployed affine and resident fused operator were fast, but repeatedly
reading the 17.82-GB physical FP16 normalized capsule made compiled completion slower than exact
at all six HBM/DRAM endpoints. Stage 4.5 asks whether the arithmetic advantage can survive a
complete 682-record HBM update without hiding extra standing state or giving exact a worse source
tier.

The frozen answer is a scoped yes. The primary source plan no longer reads or retains
`Norm(x)`. It composes the deployed capsule affine through the source model's stacked K/V
projection and directly transforms the existing serving old K/V:

\[
[K_{\mathrm{new}},V_{\mathrm{new}}]
\approx
[K_{\mathrm{old}},V_{\mathrm{old}}]B_{v\rightarrow t}
+c_{v\rightarrow t}.
\]

The source projection is full row rank in every measured layer; its condition numbers span
5.97–10.74. The compiler uses its minimum-norm right inverse and publishes one FP16 direct
program per source/target pair. The three programs total 100,777,103 bytes. The source plan adds
zero per-record state and zero `Norm(x)` bytes.

## Frozen policy

The normal action is `compiled_old_kv` only when:

1. the capacity preflight passes;
2. the existing source-version old K/V is available in HBM; and
3. the direct program passes provenance and integrity verification.

Any failed predicate selects exact. The pure policy decision is implemented. Stage 4.6 has since
frozen the continuous migrate-or-exact lifecycle; automatic exact execution, semantic degradation
detection, replacement of already produced extents, and failure-safe publication remain Stage 5.

This policy and its certificate assume that the input is exact source-version K/V. The output is
an approximate target K/V, not an exact new anchor. Feeding it into a later adjacent direct
program is outside this Stage-4.5 result because prior error may be propagated by the new affine.
That boundary is now addressed separately by the Stage-4.6 balanced age/deadline policy and real
theta0-to-theta11 recursive chain; it does not retroactively change this one-hop artifact.

The declared operating regime is a complete existing-old-K/V hot HBM cache on one, two, or four
NVIDIA A40 GPUs. Paired exact receives its complete raw history already in HBM. This experiment
does not support a cold-filesystem, durable-SSD, automatic-tiering, or organic-cache-scheduling
claim.

## Selection and fidelity

The fused launch is selected only on the 60 program-selection records. The unchanged three-view
certificate is then reapplied on the disjoint 60-record certificate role.

| Source | Cache recovery | Score recovery | Top-100 recovery | Cost / exact | Worst recovery lower bound |
|---|---:|---:|---:|---:|---:|
| theta0 | 0.8810 | 0.9845 | 0.9479 | 0.03678 | 0.8500 |
| theta4 | 0.8897 | 0.9201 | 0.9046 | 0.03676 | 0.8345 |
| theta10 | 0.9365 | 0.9718 | 0.9470 | 0.03677 | 0.9229 |

All three pairs select `compiled_old_kv`; exact terminates each fallback chain. Minimum
worst-view recovery is 0.8810, and every certificate passes the existing 70% recovery,
80% coverage, and 30%-of-exact cost contract.

An independent actual-data fused transport covers all 682 records, all four role partitions,
1,087,785 prefix tokens, and 17,822,269,440 valid FP16 K/V elements. Against the deployed
normalized-capsule output it has zero mismatches at `atol=0.02, rtol=0.02`; maximum absolute
error is 0.01171875. The 522 final-test records enter only this label-free transport validation
and do not affect method, launch, policy, or threshold selection.

## Complete-cohort result

Each method/GPU point performs a capacity preflight, one complete correctness job, one warmup, and
five measured complete jobs. The boundary includes replacement allocation, transform, extent
staging, old-extent retirement, coverage validation, and atomic manifest commit.

| GPUs | Direct compiled | Paired exact | Speedup | Peak old + new K/V |
|---:|---:|---:|---:|---:|
| 1 | 0.9299 s | 18.6949 s | 20.11× | 35.91 GB |
| 2 | 0.4936 s | 9.7291 s | 19.71× | 36.18 GB |
| 4 | 0.2546 s | 4.7655 s | 18.72× | 37.79 GB |

Every compiled repetition is below every paired exact repetition. All capacity preflights pass,
all manifests cover 682 records and 1,087,785 tokens, and old K/V reaches zero bytes after commit.
The one-GPU preflight includes the current model, the complete old K/V, the direct programs, the
maximum replacement wave, and a 2-GiB allocator margin.

The performance repetitions use shape-, dtype-, layout-, and occupancy-equivalent old-K/V values
to isolate the full system critical path without repeatedly loading 35.64 GB from disk. They are
not the numeric fidelity evidence. The independent complete actual-data transport above verifies
the real fused values. Both records are required for the claim.

## Candidate history and lifecycle

Matched HBM- and pinned-DRAM-resident ceilings first confirmed that source supply explained the
Stage-4 loss. A full-cohort pinned-DRAM normalized-capsule implementation then reached
1.529/0.252 seconds on 1/4 GPUs and beat paired exact, but retained about 17.86 GB of extra host
state and required 39.5/24.7 seconds of preload. Its time break-even was three/six updates. It is
kept as a valid backup and negative state-economics result, not as the primary plan.

The direct route removes that additional state and preload. The old K/V is not newly captured for
Stage 4.5; it is the serving cache that the migration event is replacing. During execution, each
old extent is retired only after its replacement has been accepted by the destination transaction.
The only new standing source-plan artifact is the 100.78-MB direct program set replicated on each
worker.

## Artifacts and commands

- Compiler and selection transport:
  `results/system/cohortkv_single_config_full_chain_v1/stage4_5_oldkv_compiler_seed0.json`
- Semantic certificate:
  `results/system/cohortkv_single_config_full_chain_v1/stage4_5_oldkv_certificate_seed0.json`
- Complete actual-data transport:
  `results/system/cohortkv_single_config_full_chain_v1/stage4_5_oldkv_full_transport_seed0.json`
- One/four-GPU system result:
  `results/system/cohortkv_single_config_full_chain_v1/stage4_5_oldkv_system_seed0.json`
- Two-GPU expansion:
  `results/system/cohortkv_single_config_full_chain_v1/stage4_5_oldkv_system_expansion_seed0.json`

Freeze and verify:

```bash
python scripts/freeze_cohortkv_stage4_5.py
python scripts/freeze_cohortkv_stage4_5.py --check
pytest -q tests/test_stage45_resident.py tests/test_single_config_stage45.py
```

Stage 4 remains frozen as the negative normalized-capsule source result. Stage 4.5 is a separate
protocol and does not overwrite it.
