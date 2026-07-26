# KuaiRand mixed-version four-GPU scaling v1

## Status

`kuairand_long_context_4plus12_mixed_version_four_gpu_scaling_v1` is a frozen-layout,
adaptive seed-0 systems follow-up. It extends the completed two-GPU v2/v3 execution path to
1/2/4 A40 GPUs without refitting a migration program, reselecting an action, changing the user
trace, or using recommendation labels.

The result is positive controlled scaling evidence, not the destination-v4 full-cohort result.
It retains the 64-user controlled theta0/theta4/theta10 mix and caller-materialized capsules.
Source-side lazy streaming, organic full-cohort source versions, physical SSD/network execution,
and a direct-HBM full-recompute baseline remain open.

## Frozen workload and boundaries

The final trace is exactly the disjoint v3 final trace:

- 64 users and 98,252 valid prefix tokens;
- theta0/theta4/theta10 source counts of 13/19/32;
- theta11 as the K/V target;
- 1.50 GiB of unpadded FP16 old `Norm(x)` capsules;
- the v3-selected 2,048-valid-token cohort-jagged layout;
- the published verified full-affine program for each source cohort;
- one warmup and five timing repetitions at 1/2/4 GPUs;
- greedy-LPT extent assignment and three in-flight batches.

Three paths are timed:

1. `host_staged_dram`: pinned-host FP16 capsule through H2D, fused migration, D2H, and complete
   FP16 K/V in a persistent pinned-host output pool;
2. `direct_hbm`: the same pinned-host capsule through H2D and fused migration into preallocated
   serving-native HBM K/V, with no D2H;
3. `bf16_full_recompute_host`: pinned raw history through current-model BF16 recomputation and
   complete FP16 K/V in the same host publication representation as path 1.

Only paths 1 and 3 have a common endpoint and may form an algorithm speedup. The HBM/DRAM ratio
describes two different destination completion boundaries and is not an operator speedup.

## Command

```bash
python scripts/benchmark_kuairand_four_gpu_scaling_system.py
```

The output is local:

```text
results/system/kuairand_long_context_4plus12_four_gpu_scaling_seed0.json
```

## Scaling result

| Path | 1 GPU | 2 GPUs | 4 GPUs | 1→4 speedup | 4-GPU efficiency |
|---|---:|---:|---:|---:|---:|
| Host-staged compiled migration | 466.4 records/s | 895.1 records/s | 1,527.5 records/s | 3.275x | 81.9% |
| Direct-HBM compiled migration | 1,064.7 records/s | 1,855.4 records/s | 3,546.1 records/s | 3.331x | 83.3% |
| Host-staged BF16 full recompute | 40.9 records/s | 80.5 records/s | 147.0 records/s | 3.592x | 89.8% |

At the common host publication boundary, compiled migration is 11.40x/11.12x/10.39x faster than
BF16 full recomputation at 1/2/4 GPUs. The small decrease with GPU count is expected: exact
recomputation has more per-record arithmetic and therefore amortizes fixed worker and transfer
overheads better.

Five-repeat timing coefficients of variation are at most 1.10%. Four-GPU LPT assigned-work
imbalance is 0.30% for both migration endpoints and 0.10% for exact. Thus the remaining
sublinearity is not caused by a visibly poor byte partition.

## Capacity and GPU memory

The logical FP16 old K/V and complete target K/V each occupy approximately 3.00 GiB. The old
normalized capsule occupies 1.50 GiB, exactly 50% additional state relative to logical FP16 K/V.

| Path at 4 GPUs | Maximum peak allocated per GPU | Aggregate peak allocated |
|---|---:|---:|
| Host-staged compiled migration | 0.43 GiB | 1.70 GiB |
| Direct-HBM compiled migration | 0.93 GiB | 3.71 GiB |
| Host-staged BF16 full recompute | 1.79 GiB | 6.88 GiB |

Direct-HBM uses more resident device memory because it retains the complete target K/V, but
sharding reduces the maximum device peak from 3.19 GiB on one GPU to 0.93 GiB on four. The
host-staged path keeps persistent target K/V in host memory and needs only the resident programs
and bounded transient device buffers.

## Topology and interpretation

The machine contains four 46 GiB A40 GPUs. GPU0/1 and GPU2/3 are two NVLink-connected pairs;
traffic between the pairs traverses the system interconnect and the machine has two NUMA nodes.
The runtime performs no cross-GPU K/V copy: programs are replicated and complete extents are
assigned independently. This topology and the short per-GPU completion time make fixed worker,
Python dispatch, and host-transfer costs plausible causes of the 82–83% migration efficiency.
No topology-specific policy was selected from the final trace.

An older frozen single-source packed path was also completed under
`kuairand_long_context_4plus12_progressive_sync_system_v1`. On 64 real users and 107,247 tokens it
scales from 364.5 records/s on one GPU to 1,318.1 records/s on four GPUs, a 3.616x speedup and
90.4% efficiency. This is useful corroboration but is not the current mixed-version fused path.

## Destination-v4 correctness

The tiny destination-v4 direct-HBM validation now publishes four extents across cuda:0–3 and
atomically commits one complete `streamkv_destination_manifest_v1`. Its maximum absolute K/V error
from the FP32 reference is `9.77e-4`. The HBM executor also trims excess idle workers when a job
contains fewer extents than declared devices; this behavior has a regression test.

This validation establishes four-GPU interface and publication correctness only. Its synthetic
tensors and sub-second setup are not throughput evidence. The performance table above uses real
capsules but the pre-v4 persistent host/HBM execution boundary, so it does not close source-side
streaming, manifest timing, or same-destination exact baselines for the full v4 engine.

## Claim boundary

Supported:

- the frozen mixed-version fused migration path scales positively from one to four local GPUs;
- LPT keeps assigned bytes nearly balanced;
- direct-HBM placement reduces completion time and shards target-resident HBM capacity;
- compiled migration retains a roughly 10.4x host-boundary advantage over a four-GPU BF16 exact
  implementation on this controlled trace;
- destination-v4 can commit a correct direct-HBM manifest spanning all four GPUs.

Not supported:

- production serving latency, online request scheduling, or foreground-SLO isolation;
- full-cohort destination-v4 performance or bounded total source-capsule memory;
- an HBM algorithm speedup without a same-HBM full-recompute baseline;
- physical SSD, remote network, P2P publication, or cross-node scaling;
- independent training-seed replication from repeated timing samples.
