# KuaiRand 4+12 progressive synchronization v1

## Status

The algorithm/operator/runtime path is implemented and passes bounded real-checkpoint diagnostics.
The current result is not a paper result: it uses one training seed, 24 design users, 12 system
records, and three idle GPUs because physical GPU 2 was occupied by an unrelated process. The
full-user design evaluation and four-GPU system command below remain to be run.

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

## Formal system command

Run this only after the formal design command has generated the default theta0→theta11 program and
all four GPUs are available:

```bash
python scripts/benchmark_kuairand_long_context_sync_system.py
```

The exact baseline now starts with pinned raw histories and ends with FP16 pinned-host K/V, matching
the compiled path's input/output location. It is synchronous and therefore not the final strongest
baseline; the next systems step is a similarly pipelined full-recompute executor, followed by
organic mixed-version extents and foreground interference. SSD, cross-node storage, and model
partitioning are not current requirements.
