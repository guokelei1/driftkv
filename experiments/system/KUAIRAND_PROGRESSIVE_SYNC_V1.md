# KuaiRand 4+12 progressive synchronization v1

> Historical disposition: superseded D1/runtime record. The former destination-v4 gate is closed
> as an experiment lineage and does not define current D2 or D3.

## Status

The algorithm/operator/runtime path is implemented. Its original bounded real-checkpoint
diagnostic used one training seed, 24 design users, 12 system records, and three idle GPUs because
physical GPU 2 was occupied. The frozen formal system command has now also completed on all four
A40 GPUs with 64 real records and 1/2/3/4-GPU points. The full-user design evaluation was later
completed and superseded by the verified compiler record.

The four-GPU v1 result is corroborating scaling evidence for a historical single-source packed
path, not the current primary system endpoint. Its then-active destination-v4 follow-up and the
frozen mixed-version fused four-GPU result are separate historical protocols.

The later `TWO_GPU_MIGRATION_SYSTEM_V2.md` supersedes this file's system-performance endpoint with
verified full-affine programs, a fused operator, controlled mixed versions, persistent
destinations, LPT, and pipelined BF16/FP32 exact baselines. The diagnostics below remain historical
implementation evidence.

## Frozen design

The endpoint is theta11 on D16. Source caches are theta10, theta4, and theta0, holding the current
task, histories, positives, and users fixed while covering one, seven, and eleven updates.

Every stale cohort follows one monotone ladder:

1. rank-32 compiled affine synchronization over cached old `Norm(x)`;
2. p8 current-prefix replay with boundary residual transport;
3. exact current-model recomputation.

P4 and p12 are same-slot ablations. There is no version-level or per-user reuse admission. Version
cohorts key shared programs, batching, placement, and work assignment only.

## Full-user design command

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/evaluate_kuairand_long_context_sync_design.py
```

The formal split is 40 fit users, 60 reserved label-free probe users, and every remaining user for
test. It writes one serialized rank-32 program per source/target pair. Worker count only controls
evaluation sharding, so the same frozen evaluation may use two workers:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_kuairand_long_context_sync_design.py
```

## Bounded design diagnostic

The diagnostic used four fit, four reserved probe, and 16 held-out users, one timing repeat, and
three workers. Its output is
`results/motivation_scale/long_context_4plus12_progressive_sync_design_diagnostic_seed0.json`.

| Cache source | Age | Rank-32 cost/full | Rank-32 K/V recovery | P8 cost/full | P8 K/V recovery |
|---|---:|---:|---:|---:|---:|
| theta0 | 11 | 0.061 | 0.610 | 0.550 | 0.846 |
| theta4 | 7 | 0.061 | 0.589 | 0.550 | 0.798 |
| theta10 | 1 | 0.061 | 0.636 | 0.551 | 0.793 |

For theta0→theta11, stale relative K/V error is 0.949, rank-32 reduces it to 0.370, p8 to 0.146,
p12 to 0.063, and exact to zero. Mean absolute MeanRank deviation from fresh falls from 4,577 for
reuse to 165 for rank-32 and 120 for p8. These user-level values validate implementation behavior
only; training seed remains the replication unit.

## Real-capsule system diagnostic

The system path materializes actual theta0 layerwise normalized state from D16 histories in FP16.
Hot capsules execute resident in HBM. Warm capsules begin in pinned host memory and traverse
overlapped H2D, packed affine execution, and D2H target-K/V publication. Cache extents are sharded;
the 0.181B model is not partitioned.

The bounded output is
`results/system/kuairand_long_context_4plus12_progressive_sync_system_diagnostic_seed0.json`.

| Point | Result |
|---|---:|
| Packed FP16 HBM speedup over FP32 reference | 3.97x |
| Packed relative K/V error | 3.53e-4 |
| 1-GPU warm throughput | 363.5 records/s |
| 2-GPU warm throughput | 623.5 records/s |
| 3-GPU warm throughput | 759.5 records/s |
| 2/3-GPU speedup | 1.72x / 2.09x |
| Synchronous full, same host boundary | 24.8 records/s |
| 1-GPU compiled over synchronous full | 14.7x |

The 3-GPU point has 22.4% assigned-work imbalance because six variable-length batches are divided
over three workers. It is evidence that the real multidevice path works, not a scalability claim.

## Completed frozen four-GPU system run

The formal command after the design step generated the default theta0→theta11 program was:

```bash
python scripts/benchmark_kuairand_long_context_sync_system.py
```

Its local output is
`results/system/kuairand_long_context_4plus12_progressive_sync_system_seed0.json`.

| Point | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs |
|---|---:|---:|---:|---:|
| Records/s | 364.5 | 735.3 | 1,059.2 | 1,318.1 |
| Speedup | 1.000x | 2.017x | 2.906x | 3.616x |
| Parallel efficiency | 100.0% | 100.9% | 96.9% | 90.4% |
| Assigned-work imbalance | 0.0% | 0.53% | 1.44% | 1.99% |

The trace contains 64 users, 107,247 valid tokens, and 1.65 GiB of FP16 capsules. Packed FP16 is
3.997x faster than the resident FP32 reference with relative K/V error `3.86e-4`. This protocol
uses one theta0→theta11 program and a synchronous full baseline, so the later mixed-version fused
result and independently pipelined exact comparison were the stronger evidence within that
experiment lineage.

The exact baseline starts with pinned raw histories and ends with FP16 pinned-host K/V, matching
the compiled path's input/output location. The later system-v2 result adds the independently
pipelined full-recompute baseline. The fixed-cohort destination job was this record's historical
successor; current work is defined by the roadmap.
