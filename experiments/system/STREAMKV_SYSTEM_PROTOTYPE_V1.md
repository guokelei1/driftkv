# StreamKV system prototype v1

## Status

This is a preliminary systems prototype result, not a replacement for any current algorithm
quality protocol. The algorithm section reuses the frozen
`cohort_tiered_migration_v1_seed_summary`; the operator and runtime sections use synthetic tensors
with the large capacity-v2 model shape. The three evidence families remain explicitly separated
inside the result artifact.

## Implemented prototype

The prototype now has one executable path across all three proposed layers:

1. `MigrationProgram` binds a source/target version pair to the existing compiled affine adapter.
2. `MigrationCapsuleBatch` stores contiguous old `Norm(x)` records and their migration anchor.
3. `PackedMigrationOperator` executes the affine transform in FP16 with `baddbmm` bias fusion,
   in-place padding masking, and packed K/V views.
4. `CohortBatchPlan` creates contiguous or length-bucketed logical extents and preserves record-ID
   mapping independently of physical order.
5. `CohortStreamingExecutor` overlaps pinned-host H2D, GPU transform, and D2H on three CUDA streams.
6. `MultiGPUCohortExecutor` greedily partitions extents and runs persistent workers concurrently.

Approximate migration output explicitly carries both `migration_anchor_version` and
`served_kv_target`. Producing target K/V does not silently advance the migration anchor.

## Protocol

- Command: `python scripts/streamkv_system_benchmark.py`
- Result: `results/system/streamkv_system_prototype_v1.json`
- Hardware: four NVIDIA A40 GPUs
- Torch/CUDA: 2.12.1+cu130 / CUDA 13.0
- Shape: 9 layers, hidden width 128, sequence width 128
- Capsule dtype: FP16
- Lengths: synthetic uniform samples from 64 through 128
- Operator batches: 8, 32, and 128 records
- System cohorts: 256, 512, and 1024 records
- System batch size: 32 records
- Timing: two warmups and five measured repetitions; tables report medians
- Operator boundary: resident GPU capsule and program, timed with CUDA events
- System boundary: pre-pinned host capsule through host K/V output
- System timing includes transfer, transform, output allocation, and worker execution
- Program distribution, tensor generation, and initial pinning are excluded

Synthetic records are systems workload units only. They are not quality samples or statistical
replications.

## Algorithm layer evidence

The frozen 27-seed validation remains the algorithm evidence:

| Metric | Result |
|---|---:|
| Selected family | compiled projection in 27/27 validation runs |
| Mean GPU cost relative to full | 0.121, 95% CI [0.112, 0.130] |
| Mean K/V fidelity recovery | 0.587, 95% CI [0.547, 0.627] |
| Primary fidelity target met | 25/27 seeds |
| Strict cell-level task gate | 6/9 cells |

The system prototype does not reinterpret the failed task cells. Version-cohort admission remains
an open algorithm/control gate.

## Operator layer

### End-to-end resident operator

| Batch | Reference FP32 | Packed FP16 | Speedup | FP16 relative error |
|---:|---:|---:|---:|---:|
| 8 | 0.200 ms | 0.102 ms | 1.95x | 3.28e-4 |
| 32 | 0.622 ms | 0.232 ms | 2.67x | 3.27e-4 |
| 128 | 2.251 ms | 0.775 ms | 2.90x | 3.28e-4 |

The estimated materialized-tensor footprint of the packed path is 0.25x the reference for all
three shapes. This estimate counts explicit cast/projection/output tensors, not allocator reserve
or total HBM traffic.

### Batch-32 stage profile

| Operator | Cast in | Projection + bias | Mask | Cast out | Total |
|---|---:|---:|---:|---:|---:|
| Reference FP32 | 0.059 ms | 0.301 ms | 0.147 ms | 0.103 ms | 0.613 ms |
| Packed FP16 | 0.010 ms | 0.141 ms | 0.078 ms | 0.002 ms | 0.232 ms |

The packed path removes most cast cost and folds bias into `baddbmm`. Padding mask handling is now
the largest remaining non-GEMM stage, so a true fused epilogue remains a justified next operator
step rather than a presumed optimization.

## Streaming and hardware layer

| Cohort | Bucketed payload / contiguous | Ref sync 1 GPU | Packed sync 1 GPU | + pipeline | + bucket | + 2 GPU | + 4 GPU | Best / ref |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.828 | 14.50 ms | 11.36 ms | 8.01 ms | 6.40 ms | 4.07 ms | 4.39 ms | 3.56x |
| 512 | 0.820 | 28.91 ms | 22.60 ms | 15.63 ms | 12.52 ms | 7.08 ms | 7.95 ms | 4.09x |
| 1024 | 0.813 | 57.73 ms | 45.16 ms | 30.98 ms | 24.61 ms | 12.97 ms | 15.41 ms | 4.45x |

The contributions compose rather than merely coexist:

- packed FP16 improves the synchronous one-GPU path by 1.28x;
- three-stage overlap adds 1.42–1.46x over packed synchronous execution;
- length bucketing removes 17–19% of capsule payload and adds 1.25–1.26x;
- two-GPU extent sharding adds 1.57–1.90x over the bucketed one-GPU path.

Every configuration preserved complete, unique record-ID coverage, and all invalid-token outputs
remained exactly zero.

Four GPUs do not outperform two GPUs in this host-backed setup. Per-device execution slows as
more GPUs issue host transfers concurrently even when estimated extent loads are balanced. This
is an initial negative result and a concrete next design target: the runtime needs topology/NUMA
aware source placement or adaptive active-GPU selection instead of assuming all available GPUs
should participate.

## Current claim boundary

This prototype supports a code-level and preliminary experimental skeleton for the three-part
paper design. It does not yet establish:

- real-checkpoint end-to-end full-recompute speedup under identical storage boundaries;
- an admission policy that resolves the current 6/9 strict quality gate;
- a fused GEMM epilogue or direct paged destination writer;
- foreground serving interference or a P99 SLO;
- SSD, GDS, RDMA, or cross-node behavior;
- multi-seed systems performance.

The immediate next experiments are real checkpoint/capsule replay, H2D/D2H and NUMA profiling,
and a foreground-load interference harness.
