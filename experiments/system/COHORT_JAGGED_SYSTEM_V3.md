# KuaiRand cohort-jagged migration system v3

> Historical disposition: frozen negative operator study. Its destination-oriented successor is
> no longer the current architecture; the exact-layout and negative performance findings remain
> valid only within this protocol.

## Status

`kuairand_long_context_4plus12_cohort_jagged_system_v3` is an adaptive seed-0
operator-development study. It tests whether cache-migration semantics create a stronger operator
opportunity than generic kernel fusion:

- all valid old `Norm(x)` tokens sharing a source/target version program can be concatenated;
- user boundaries are unnecessary during the affine migration;
- fixed-size K/V pages can be compacted into bounded token tiles;
- K/V can be published directly into packed host extents or persistent HBM extents.

The implementation and correctness gates pass. The performance hypothesis does not pass on the
current long-context trace: cohort compaction is neutral to slightly negative at the final HBM
boundary. This file records that boundary and must not be cited as a positive operator result.

## Protocol and command

- Command:
  `CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark_kuairand_cohort_jagged_system.py`
- Local result:
  `results/system/kuairand_long_context_4plus12_cohort_jagged_system_seed0.json`
- Model and programs: unchanged 16L/H512 theta0/theta4/theta10-to-theta11 verified programs
- Search trace: first 32 deterministic final-pool users, no labels
- Final trace: next 64 users, disjoint from search
- Final source mix: 13 theta0, 19 theta4, and 32 theta10 records
- Final logical prefix tokens: 98,252
- Devices: two NVIDIA A40 GPUs
- Timing: one warmup and three measured end-to-end repetitions

The system benchmark does not fit or select a migration program and consumes no recommendation
labels. Its quality source remains the verified compiler. Timing repetitions are stability
measurements, not independent training replications.

## Implemented mechanism

The library now contains:

1. a layer-major jagged capsule `[layers, valid_tokens, hidden]` with per-extent offsets;
2. packed-FP16 and fused-Triton jagged operators;
3. direct contiguous K/V publication without a padded `[user, sequence]` output;
4. persistent pinned-host and persistent HBM jagged K/V pools;
5. single- and two-GPU H2D/compute/D2H host executors;
6. single- and two-GPU H2D/compute direct-HBM executors;
7. fixed-size page packing with an explicit `(record, token_start, length)` page table.

For version cohort \(c=(v,t)\), the operator executes

\[
X_c^\ell =
\operatorname{concat}_{p\in c}\operatorname{Norm}_{v,p}^{\ell},
\qquad
[K;V]_c^\ell=X_c^\ell P_{v\rightarrow t}^{\ell}+b_{v\rightarrow t}^{\ell}.
\]

The selected page organization uses 256-token K/V pages compacted into at most 2,048-token
tiles. This is one migration data path, not a list of independent padding, batching, and layout
tricks.

## Correctness

The final trace contains 1,609,760,768 logical FP16 K/V elements. Both whole-record jagged
publication and 403-page publication were compared against the existing dense fused operator over
every valid element:

| Path | Records | Pages | Maximum absolute difference | FP16 mismatches |
|---|---:|---:|---:|---:|
| Whole-record jagged | 64 | — | 0 | 0 / 1,609,760,768 |
| Paged jagged | 64 | 403 | 0 | 0 / 1,609,760,768 |

The page table also covers every valid token exactly once. Padding is never published.

The fused and packed FP16 operators use different accumulation/code-generation paths. On the
representative 2,048-token tile, their relative K/V difference is `1.68e-5` with maximum absolute
difference `0.00390625`.

## Layout search

Whole-record token budgets `2,048/4,096/8,192/16,384` were compared separately for pinned-host and
direct-HBM publication. Both boundaries selected 2,048 tokens. Larger batches reduced launch count
but progressively reduced end-to-end throughput.

The page-layout search crossed page sizes `128/256/512/1,024` with tile budgets `2,048/4,096`.
The search trace selected page 256 and tile 2,048 at 2.977M valid tokens/s. It appeared about 5.6%
faster than the whole-record 2,048-token search point. That improvement did not transfer to the
disjoint final trace, so it is an adaptive-search diagnostic rather than a supported result.

On the final trace, page packing changes:

| Layout | Batches | Pages | Padding |
|---|---:|---:|---:|
| Dense length-bucketed | 64 | — | 0.509% |
| Whole-record jagged | 58 | — | 0 |
| Page-compacted jagged | 50 | 403 | 0 |

The selected page tiles have 95.95% mean fill.

## Resident operator result

The representative tile has eight pages and 2,048 valid tokens.

| Operator | Median | Valid tokens/s |
|---|---:|---:|
| Packed FP16 `baddbmm` plus contiguous K/V publication | 0.836 ms | 2.451M |
| Fused Triton direct K/V | 0.707 ms | 2.896M |

The fused kernel is 1.182x faster at the resident boundary. This repeats the v2 conclusion that
fusion is a real kernel optimization; it does not establish a similar end-to-end gain.

## Host-backed end-to-end result

This boundary starts with pinned-host FP16 capsules and ends with complete pinned-host FP16 K/V.

| Configuration | One GPU | Two GPUs |
|---|---:|---:|
| Dense fused | 140.64 ms | 71.50 ms |
| Whole-record jagged fused | 137.35 ms | 70.89 ms |
| Page-compacted jagged fused | 137.09 ms | 70.20 ms |

Page compaction improves the two-GPU dense result by only 1.019x. The old dense layout already has
only 0.509% padding, and the boundary moves approximately 1.5 GiB of capsules plus 3.0 GiB of
published K/V. Removing a small amount of padding cannot create a large end-to-end gain.

The host-backed jagged path remains about 11.21x faster than the independently tuned two-GPU BF16
full recomputation recorded by system v2 under the same host-to-host publication boundary. That
large advantage comes primarily from the compiled migration algorithm rather than jagged packing.

## Direct-HBM result

This boundary starts with pinned-host capsules and ends with K/V in persistent, target-GPU
HBM extents. Destination metadata and allocation are prepared before timing; target K/V is not
copied back to host.

| Configuration | One GPU | Two GPUs |
|---|---:|---:|
| One record per extent, fused | 60.06 ms | 31.98 ms |
| Whole-record cohort jagged, fused | 60.07 ms | 32.09 ms |
| Page-compacted cohort jagged, fused | 60.39 ms | 32.52 ms |
| Page-compacted cohort jagged, packed | 60.58 ms | 32.67 ms |

Direct HBM publication is 2.159x faster than host publication for the selected paged path because
it removes the complete D2H writeback. This is a placement-boundary result, not an operator
speedup, and it must not be compared with the host-publishing exact baseline as if the endpoints
were equal.

On the same HBM boundary, whole-record compaction is 0.997x and page compaction is 0.984x relative
to the one-record path. Fused versus packed page execution is only 1.005x end to end. The final
trace therefore rejects the claim that cohort token compaction materially accelerates this
workload.

## Interpretation

Three properties explain the result:

1. Mean history length is about 1,535 tokens, so even one user supplies a large GEMM dimension.
2. Length bucketing had already reduced padding to 0.509%.
3. Larger extents reduce pipeline and scheduling granularity. Fewer launches do not compensate
   for coarser H2D/compute overlap and final-device assignment.

The exact implementation should be retained as a destination layout and an enabling mechanism, but
the paper must not present cohort-jagged compaction as a major performance contribution on this
trace. FP16 Tensor Cores, Triton tiling, pinned transfers, and LPT remain generic support.

## Decision

- Retain direct packed K/V publication and the explicit host/HBM destination boundary.
- Retain page metadata because it is needed for complete manifest publication and record readback.
- Treat whole-record and page compaction as measured layout alternatives, not a claimed win.
- Do not spend another round tuning tile sizes on the current seed.
- Move the primary systems gate to a complete destination-oriented update job with bounded waves,
  identical-boundary exact baselines, atomic manifest publication, and 1/2/4-GPU completion time.
- If an operator contribution remains mandatory, it needs a new migration-specific bottleneck
  with a same-boundary strong baseline; this experiment rules out padding removal and mega-batch
  formation as that bottleneck at the current scale.

The historical successor is recorded in `experiments/system/DESTINATION_OUT_OF_CORE_V4.md`. Its
filesystem and remote-object paths remain interface/correctness implementations, not current D3
evidence.
