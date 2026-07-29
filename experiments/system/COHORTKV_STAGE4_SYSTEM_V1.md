# CohortKV Stage 4 full-cohort system v1

## Status

`cohortkv_single_config_stage4_system_v1` is complete and frozen on the controlled
KuaiRand long-context seed-0 configuration. It closes the normal-path full-cohort engine and
falsifies the expected end-to-end compiled speedup for the current serialized FP16 normalized
capsule source path.

The result is adaptive single-configuration development evidence. It is not a training-seed
replication, an online-serving result, a physical-SSD benchmark, or evidence for automatic
fallback and failure recovery.

The frozen artifacts are:

- `configs/cohortkv_single_config_v1/stage4_system_summary.json`;
- `results/system/cohortkv_single_config_full_chain_v1/stage4_system_seed0.json`;
- `checkpoints/kuairand_long_context_4plus12_exploration/seed0/single_config_v1/source_shards/source_manifest.json`;
- `scripts/benchmark_cohortkv_stage4_system.py`;
- `scripts/freeze_cohortkv_stage4.py`.

Regenerate and independently verify the tracked aggregate with:

```bash
python scripts/freeze_cohortkv_stage4.py
python scripts/freeze_cohortkv_stage4.py --check
```

The freeze check validates all source-shard sizes and SHA-256 digests, every tuning candidate,
the finalist and winner derivation, all complete-run byte totals, five capacity preflights per
point, transport correctness, publication manifests, and the full-phase implementation snapshot.

## Frozen workload and source materialization

The complete job contains 682 records and 1,087,785 valid prefix tokens. The label-free controlled
source assignment is 136 theta0, 205 theta4, and 341 theta10 records, all targeting theta11.
Every method publishes the same contiguous, unpadded FP16 K/V layout:

- 16 layers;
- K/V width 512;
- 35,644,538,880 logical target bytes;
- 17,822,269,440 valid FP16 K/V elements.

The source manifest contains 2,523 immutable checked shards. Method-specific source traffic is:

| Representation | Logical bytes | Physical bytes | Role |
|---|---:|---:|---|
| FP16 normalized capsule | 17,822,269,440 | 17,823,519,546 | compiled |
| FP16 old K/V | 35,644,538,880 | 35,645,917,202 | selective and no-transform |
| Raw history | 21,755,700 | 89,056,170 | exact, selective, residual |
| BF16 residual hidden suffix | 6,255,345,664 | 6,256,250,533 | theta0/theta10 residual-p |

The shards are persisted on `/data`, an ext4 mount on an identified Intel NVMe device, but this
protocol is not a cold-device SSD benchmark. Correctness and an untimed full source pass precede
measured repetitions, decoded tensors are destroyed between jobs, shards are reopened and decoded,
and the OS page cache is not explicitly evicted. The result therefore measures the complete
filesystem-open, deserialize, pin, H2D, transform, destination, and commit path under a warm
page-cache condition. It must not be described as raw NVMe bandwidth.

## Methods and matrix

The matrix contains five methods, two destinations, and three GPU counts:

`5 methods × 2 destinations × {1,2,4 GPUs} = 30 points`.

The methods are:

- `compiled`: the certified FP16 full-affine fast tier;
- `selective_contiguous`: the frozen `m=12, layers=0..11` certificate-failed diagnostic baseline;
- `exact`: complete current-model recomputation from raw history;
- `residual_p`: the p8 quality-tier control for theta0/theta10 with exact fallback for theta4;
- `no_transform`: old K/V movement without semantic repair.

HBM means the complete new target K/V remains on the assigned GPUs. DRAM means the target is copied
to and retained in pinned host memory. These names describe the output destination, not source
capsule residency.

Runtime tuning is independent for every method/destination/GPU point and uses only the 60
program-selection records. The common grid is:

- batch size 1, 2, or 4;
- length bucket width 16, 32, or 64;
- maximum in-flight depth 2, 3, or 4.

Compiled additionally compares packed and fused FP16 operators, and exact compares BF16 and FP32
compute. All legal candidates receive correctness and a screen pass in seed-73421 order. The
fastest three receive one warmup and three measured repetitions. The winner is frozen before the
complete 682-record run.

Every final point performs:

1. a full-cohort same-path transport/layout correctness pass;
2. one complete untimed warmup;
3. three complete measured repetitions;
4. a fresh capacity preflight before each of those five jobs;
5. atomic duplicate-free target-manifest commit.

## Complete-cohort results

Completion time is the median of the three measured jobs in seconds.

| Method | HBM 1 GPU | HBM 2 GPU | HBM 4 GPU | DRAM 1 GPU | DRAM 2 GPU | DRAM 4 GPU |
|---|---:|---:|---:|---:|---:|---:|
| Compiled | 27.083 | 18.943 | 13.707 | 22.567 | 12.231 | 15.662 |
| Selective diagnostic | 80.405 | 51.148 | 47.232 | 70.987 | 42.637 | 45.902 |
| Exact recompute | 18.881 | 9.644 | 5.742 | 18.886 | 9.391 | 5.448 |
| Residual-p | 25.457 | 13.472 | 7.454 | 25.196 | 13.557 | 7.399 |
| No-transform | 62.184 | 34.765 | 35.286 | 53.322 | 38.050 | 35.814 |

The compiled path beats the certificate-failed selective diagnostic at all six endpoints by
2.70–3.49×. It beats exact at none of the six endpoints:

| Destination | GPUs | Exact / compiled | Compiled slowdown |
|---|---:|---:|---:|
| HBM | 1 | 0.697× | 1.434× |
| HBM | 2 | 0.509× | 1.964× |
| HBM | 4 | 0.419× | 2.387× |
| DRAM | 1 | 0.837× | 1.195× |
| DRAM | 2 | 0.768× | 1.302× |
| DRAM | 4 | 0.348× | 2.875× |

This is not a compute-efficiency failure. At HBM 1/2/4 GPUs, the compiled transform's measured
critical-path compute components are 0.954/0.244/0.118 seconds, versus
18.024/9.176/5.356 seconds for exact. The compiled source-read component instead consumes
91.35%–96.91% of end-to-end completion across the six endpoints. Exact reads only the compact raw
history and remains compute-bound.

The no-transform control confirms the state-volume boundary. Moving old K/V without any numerical
repair takes 62.184/34.765/35.286 seconds at HBM 1/2/4 GPUs. Four GPUs do not improve over two
because the workers contend on the shared source and host path. Exact scales by 3.288× from one to
four GPUs on HBM and 3.467× on DRAM; compiled reaches only 1.976× and 1.441×.

All 30 points:

- are finite and allclose to their selected numeric-path oracle;
- cover all 17,822,269,440 valid elements;
- preserve record order, lengths, and offsets;
- pass complete and duplicate-free manifest validation;
- pass retained-target, source-wave, staging, model/program, transient, and allocator-margin
  capacity preflight.

The maximum measured source wave is 3,758,234,964 bytes and the maximum device staging wave is
3,219,653,472 bytes. Publication queue peak is zero because the HBM and pinned-DRAM transactions
stage extents directly into their retained target maps.

## Capacity-preflight amendment

The first formal run exposed a fail-safe preflight underestimation at
`selective_contiguous:dram:2`. The tuning subset assigned a smaller maximum extent to GPU 1, so its
device-specific transient estimate was 17 MB below the complete-cohort observed peak. No failed
point was committed.

The corrected preflight separates:

- the deterministic complete-cohort in-flight device source/output wave;
- the corresponding program-selection calibration wave;
- the maximum measured compute slack shared across identical device transforms;
- retained target, resident model/program, and allocator margin.

All previously measured points were invalidated by implementation hash and rerun. Every final run
records the same implementation snapshot. This amendment changes capacity accounting only; extent
planning, transformation, transfer, and publication semantics are unchanged.

## Historical interpretation and successor decision

Stage 4 closes the normal full-cohort execution path, but its end-to-end Pareto gate fails. The
current serialized FP16 normalized capsule is 50% of logical K/V size and is 200× larger in
physical bytes than the exact path's raw history. The resident operator and semantic-recovery
results remain valid; the current source-state representation and movement path do not.

At this point, Stage 5 guard/fallback and failure work was paused. The successor step, since
completed and frozen by Stage 4.5, was
`stage4_5_source_state_footprint_optimization`, with a deliberately narrow iteration loop:

1. establish matched HBM-resident and pinned-DRAM-resident ceilings for compiled and exact;
2. screen representation, placement, supply, and reclamation candidates on the frozen
   program-selection records;
3. run a complete cohort only at `compiled:hbm:1` and `compiled:hbm:4`, with paired exact starting
   from an equally favorable source tier;
4. expand to other endpoints only after a candidate changes the end-to-end Pareto frontier, then
   run one complete publication matrix after the source plan is frozen.

The one-GPU resident case tests the tight capacity regime. New K/V alone is about 33.2 GiB, and
adding the current FP16 normalized capsule reaches about 49.8 GiB before transient and allocator
headroom, beyond one A40. It therefore requires compression, streamed supply, or extent-wise
reclamation. The four-GPU old-K/V-plus-capsule-plus-new-K/V equivalent is about 20.75 GiB per GPU
and admits a complete hot-tier resident experiment.

Candidate mechanisms include capsule quantization, a smaller reconstructable latent, fused
decompression plus affine projection, direct transformation from already retained old K/V,
hot-HBM/warm-DRAM/cold-exact tiering, and extent-wise reclamation or replacement of old K/V.
These are hypotheses for a new protocol, not requirements or claims from the Stage 4 artifact.
The hard goal is a stable complete-job advantage over same-boundary exact with standing bytes,
capture/preload, and old/source/new-state overlap disclosed. Merely preloading the capsule and
excluding its lifecycle cost does not pass.
