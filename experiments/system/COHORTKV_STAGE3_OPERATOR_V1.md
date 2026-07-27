# CohortKV Stage 3 capsule/operator v1

## Status

Stage 3 is complete and frozen under
`cohortkv_single_config_stage3_operator_v1` /
`cohortkv_single_config_stage3_frozen_v1`.

The raw local result is
`results/system/cohortkv_single_config_full_chain_v1/stage3_operator_seed0.json`.
The checked aggregate is
`configs/cohortkv_single_config_v1/stage3_operator_summary.json`.

## Boundary

This is a GPU-resident operator experiment on the 60 label-free
`program_selection` records. It starts with resident FP16 dense, length-bucketed capsules and
resident theta0/theta4/theta10-to-theta11 programs. It ends after each operator has written a
preallocated, contiguous, unpadded FP16 K/V extent:

```text
K, V: [layers, valid_tokens, kv_width]
metadata: record IDs, migration anchor, served K/V target, lengths, offsets
```

Source I/O, output allocation, H2D/D2H, destination staging, and manifest commit are not timed.
The result is not a 682-record job or an HBM/DRAM endpoint speedup.

## Implemented contract

`ReferenceMigrationOperator`, `PackedMigrationOperator`, and `FusedMigrationOperator` now expose
the same `execute_into(program, dense_capsule, contiguous_extent)` interface. Their earlier dense
output method remains available for padding-zero validation and existing executors.

- Reference widens the same serialized FP16 capsule and FP16 runtime program to execute the
  projection in FP32, then compacts valid FP16 K/V into the extent. It is the arithmetic/layout
  oracle for deployed bytes, not the original FP32 fitted program or fresh current-model K/V.
- Packed executes FP16 `baddbmm`, length masking, K/V splitting, and compact extent copies.
- Fused maps valid dense rows to offsets and writes K/V directly into the final extent from one
  Triton kernel.

The fused path therefore allocates no global temporary tensor after the destination is
preallocated. Packed retains a padded concatenated projection and compact-copy epilogue.
Before timing, every extent is checked for integer contiguous metadata, exact
length-to-offset/token agreement, and non-aliased input/output storage. The Triton stores also
mask destination rows to the allocated token range, so malformed metadata cannot produce an
out-of-bounds write.

## Full-distribution correctness

The 60 records contain 88,085 valid prefix tokens across theta0/theta4/theta10 counts 10/20/30.
Their sorted `(record_id, user_id, source_version, prefix_tokens)` identity hash is checked
directly against the frozen workload manifest.
Every one of the nine batch/bucket layouts was checked over all
1,443,184,640 valid FP16 K/V elements.

| Check | Result |
|---|---:|
| Layouts | 9 |
| Source padding nonzeros | 0 |
| Dense reference/packed/fused padding nonzeros | 0 |
| Nonfinite outputs | 0 |
| Reallocated destination pointers | 0 |
| Dense-versus-extent mismatches | 0 |
| Packed/reference transport failures (`atol=rtol=0.02`) | 0 |
| Fused/reference transport failures (`atol=rtol=0.02`) | 0 |
| Maximum packed/reference absolute difference | 0.0078125 |
| Maximum fused/reference absolute difference | 0.0078125 |

The prior jagged/page result remains exact and negative in performance. Stage 3 did not reopen
that layout search.

## Frozen resident selection

The frozen grid contains all 18 combinations:

- batch size `1/2/4`;
- bucket width `16/32/64`;
- packed FP16 or fused FP16.

Candidates received one screen pass in seed-73421 order after correctness. The fastest three,
plus the fastest packed control needed by the stability gate, received one warmup and three
measured passes.

The selected resident default is:

```text
fused_fp16, batch_size=4, bucket_width=32
```

Its complete-distribution samples are `30.142/31.070/31.154 ms`. The fastest packed control is
`packed_fp16, batch_size=4, bucket_width=64` at
`61.749/61.970/62.145 ms`. Every fused sample is below every packed sample, with a median 1.995×
advantage. The exact bucket choice is only a development default: the three finalist medians are
close, and Stage 4 must retune the complete grid.
Stage 4 still tunes the complete frozen grid independently for every method, destination, and GPU
count; this resident default cannot replace endpoint-specific tuning.

## Representative profile

The representative real shape contains four theta0 records, each length 2,047.

| Path | Median | Peak operator temporary |
|---|---:|---:|
| FP32-arithmetic transport reference to common extent | 14.610 ms | 1,073,766,400 B |
| Packed FP16 to common extent | 5.378 ms | 402,612,224 B |
| Fused FP16 direct extent write | 2.729 ms | 0 B |

For packed FP16, the median stage profile is:

- `baddbmm+bias`: 1.915 ms;
- in-place length mask: 1.017 ms;
- K/V split, valid compaction, and extent write: 2.386 ms.

The last stage is larger than the packed GEMM itself at this shape. The fused kernel combines
affine projection, bias, valid-length resolution, K/V split, and direct extent write in 2.596 ms
in the separately instrumented stage run. Its internal components are fused and are not claimed
as separately timed kernels.

## Decision

- Keep Triton as the Stage-4 compiled default because its advantage survives the complete
  development length distribution and the same output layout.
- Keep packed FP16 as the strong framework baseline and fallback implementation.
- Use the common unpadded extent API for both direct-HBM and host-staged engines.
- Do not interpret the 1.995× resident advantage as a full-cohort system speedup.
- Do not reopen jagged/page tuning without a newly measured padding or launch bottleneck.

## Commands

```bash
python scripts/benchmark_cohortkv_stage3_operator.py --validate-only
python scripts/benchmark_cohortkv_stage3_operator.py
python scripts/freeze_cohortkv_stage3.py --check
pytest -q
ruff check src tests scripts
```
