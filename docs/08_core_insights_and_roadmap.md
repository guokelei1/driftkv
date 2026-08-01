# Core insights and roadmap

> Status: authoritative as of 2026-07-31. This document replaces earlier project-wide problem
> statements, contribution layouts, and execution roadmaps. Experimental semantics remain
> subordinate to [eval_protocol.md](eval_protocol.md).

## 1. Thesis

Streaming recommendation training produces model versions

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_t.
$$

For a fixed history prefix \(x\), the cache

$$
C_v(x)=F(\theta_v,x)
$$

is derived from model version \(v\). After the model advances to \(t\), stale reuse is cheap but
version-inconsistent, while exact replay under \(\theta_t\) repeats the complete history
computation. EvoKV asks whether HSTU structure can update this persistent derived state at lower
cost while retaining a controlled approximation to current-model K/V.

The paper currently organizes one task into three successive system layers:

```text
D1: semantic ActionPlan
    primary integrated path decides compiled migration or exact recomputation

D2: distributed WavePlan constraints
    decide owner, operator, physical compatibility, communication dependencies, and output layout

D3: capacity-bounded ResidencyPlan
    decide legal capacity cuts, ordinary-DRAM↔GPU launch order, group commit, and reclaim
```

This is the current paper decomposition, not a permanent implementation firewall. For an isolated
D3 ablation, D1 actions and D2 execution stay fixed. During mechanism discovery, capacity and
transport measurements may motivate a new globally planned D1/D2/D3 `stack_revision`; that
revision must rerun its own baselines instead of comparing against an older stack. Existing frozen
D1 and D2 evidence is not retroactively changed.

### 1.1 D1 question: reuse versus recomputation

> Can a version cohort share a migration program that closes a useful portion of the stale-to-fresh
> K/V gap at materially lower GPU cost than exact history replay?

D1 emits a per-record `ActionPlan` that is immutable within one execution/revision. The
paper-core integrated action domain is:

- **compiled repair:** fit the shared `fresh - cheap` residual for a source/target version cohort
  and compile it into one affine transform over cached source-version state; the selected hot path
  composes this transform directly over existing old K/V;
- **exact:** recompute current-model K/V and reset approximation depth.

The already implemented **progressive residual repair** remains valid D1-only supporting evidence:
it replays a current-model prefix and transports a boundary residual when its auxiliary state is
available. It is not a primary D2/D3 route or headline end-to-end action. Adding it to a future
full-stack runtime requires a separately versioned action/source contract and new baselines.

Every stale cohort receives the declared repair. Version is a compilation, artifact, and batching
key, not a claim that all records at an age are safe to reuse. Recommendation labels and realized
task gain never route caches.

### 1.2 D2 question: logical versus physical sparsity

> In a capacity-forced row-sharded multi-GPU setting, can D1's fixed logical reduction survive
> embedding lookup, padding, suffix append, collectives, destination movement, synchronization,
> and group-valid publication?

The target D2 lowering is captured by global `WavePlan` constraints:

- compiled retained-prefix repair executes at the old-K/V owner and performs no embedding lookup;
- scheduled exact, natural exact, and append request only unavoidable row-sharded item vectors;
- compiled work is compatible by `(suffix, retained)` shape;
- semantically distinct but physically identical exact reasons retain provenance while sharing one
  execution pool;
- retained and suffix output remain segmented instead of rewriting the retained prefix;
- collective order, coverage, lineage, group-valid output, commit, abort, and reclaim are
  explicit.

D2 may choose owner placement, batching, physical pool construction, execution order within its
dependency rules, and segmented layout. The current D2 design does not change
`compiled|exact`. A later cross-layer revision may regenerate both the D1 plan and D2
lowering before a run; it is then a new stack rather than a D2-only ablation. The mechanisms are
implemented in the current runtime, but their capacity-independent D3-facing constraint view has
not yet been separately serialized and hashed.

### 1.3 D3 question: working set versus HBM capacity

> When one live cache version exceeds per-rank usable HBM, can the current D1/D2 work execute from
> ordinary host DRAM with bounded in-flight replacement state, without losing its gains to
> capacity cuts, CPU staging, PCIe movement, collective stalls, or pipeline bubbles?

D3 derives a separate `ResidencyPlan`:

```text
ordinary host DRAM live versioned cache
  → bounded pinned input staging
  → H2D
  → D2-constrained GPU execution
  → D2H
  → bounded pinned output staging
  → ordinary host DRAM group shadow/replacement extent
  → validation → group commit → old-group reclaim
```

D3 owns per-rank admission, legal bin/pool slices, independent route-specific
input/compute/output granularity, prefetch/execute/writeback order, pinned-buffer credits,
backpressure, and NUMA/PCIe staging placement. Its current development runtime stages a complete
capacity group on GPU, then executes D2 with an independently chosen compute batch and drains the
result with an independently chosen output segment. The planner accepts only compiled and exact
profiles obtained jointly from the same source execution, takes each stage's maximum across ranks,
and applies discrete segment-count scaling to tail groups. It then searches stable interleavings
under the runtime's one-group input lookahead and one-drain-credit recurrence: exhaustive search
for small spaces and Pareto-beam dynamic programming for large spaces. A global-min-anchored 3%
tie prefers lower HBM, pinned memory, and segment count.

The replayable plan embeds its selected profiles and binds the group plan, checkpoints,
compiler/program, relevant source code, Torch/CUDA versions, GPU UUID/PCI identities, store tier,
capacity limits, and launch order. Both ranks independently repeat the HBM and pinned-memory
preflight before creating the target and issuing collectives. This narrow path is the isolation
track. If measurement shows that owner placement, action granularity, pool construction, or target
layout must change to exploit out-of-core execution, the project may explore a global cross-layer
revision and assign final ownership only after the mechanism is understood.

## 2. Scope

The primary object is model-version-stale HSTU prefix K/V produced by streaming training. The
current system boundary begins after target-model checkpoint publication and ends when every
selected group has reached a consumer-readable target-version extent. During the wave, each group
has an explicit old/new version and lineage; no global blue-green epoch switch is required.

In scope:

- HSTU models with pointwise unnormalized attention and first-class per-layer K/V;
- model-version cohorts, compiled programs and exact replay in the integrated path; progressive
  state only in the declared D1 supporting extension;
- row-sharded embedding lookup and multi-GPU execution;
- live group-version state, bounded replacement state, validation, group commit, abort, and reclaim;
- ordinary host DRAM, bounded pinned staging, PCIe/NVLink/NCCL, and GPU HBM for D3.

Out of scope for the current three designs:

- predicting per-user safe reuse from drift, JVP, Fisher, task labels, or online reward;
- online recommendation request scheduling, hotness, or serving SLOs;
- training/inference colocation and synthetic foreground traces as a primary workload;
- automatic HBM/DRAM/SSD tier selection;
- SSD, database, object-store, or network ingress into host DRAM;
- host-DRAM oversubscription;
- a dense HSTU trunk that itself requires tensor parallelism;
- durable cross-host transactions.

The existing X2 development model fits on one A40 and remains a mechanism/quality bridge. The
paper-scale XP configuration fixes 2,859,835 base-period semantic item rows plus one padding row
in a 2,859,836×4,096 physical FP32 table (43.638 GiB), an owner-side E4096→H1536 projection,
and the 24L/H1536 core. Table plus
dense/projection state is about 44.725 GiB and exceeds single-A40 Torch allocatable bytes before
program/runtime state; qualification validates capacity,
trainability, and 2/4-rank execution before any EvoKV timing rather than selecting geometry from
performance. D2's formal distributed problem therefore comes from a capacity-forced placement
rather than an asserted industrial convention. The capacity claim counts only embedding rows that
received a real base-period optimizer update: the semantic-request union across both formal edges,
all headline manifests, all-exact, and every frozen fixed-action exact/append/fallback path must be
active and hashed, and
active embedding bytes plus dense/projection bytes must already exceed the frozen single-card
allocatable budget. XP uses a separately frozen 4-rank row-sharded sparse checkpoint builder with
row-wise/offloaded optimizer state. This common-upstream \(\theta_0\) builder may consume
base-period histories from later-role users but no update/final windows; post-base roles remain
user-disjoint. Cold allocation cannot manufacture the boundary.

## 3. Current status

Current execution availability (2026-08-01): development rounds are restricted to the GPU0/GPU1
NVLink pair. Four-rank and GPU2/GPU3 experiments are deferred until the user explicitly restores
those devices; this changes scheduling only, not the planned paper matrix.

| Layer | State | What is closed | What remains |
|---|---|---|---|
| D1 | frozen | method, direct-old-K/V source plan, bounded renewal, Stage-5 guard/fallback/transaction, Stage-6 aggregate | broader replication and optional Stage-4.10 successor |
| D2 | implementation and mechanism discovered; paper evidence open | Stage A, W1/W2, W3 diagnostics, C0 wiring, segmented/shape-aware/merged-exact development path, full-payload development correctness; XP owner-side E4096→H1536 two-rank physical canary | optimizer-active XP checkpoint, D3-facing constraints, 1/2/4-rank runner, strong placement/exact baselines, segmented consumer, full integrated group timing, frozen formal protocol |
| D3 | successor foundation running; mechanism and paper evidence open | historical M0/M1 chain; real 65,536-record HET/HOM manifests and capacity ledger; minimal rolling validate/commit/reclaim/failure/replay canary; fixed-512 D1/D2 environment regression | real HET ActionPlan overlay and numeric rolling runner, active-row XP edge, 36/72-GiB problem-existence baselines, strongest generic scheduler, held-out qualification/repeats, frozen formal protocol and paper evidence |

The two-A40 XP quality foundation is complete as development evidence. After one explicit
bootstrap-to-streaming-objective warm-up edge, the selected chain trains 16,384 user-disjoint
update records for one epoch on each of four contiguous eight-exposure windows and evaluates the
ordinary `theta1→theta2`, `theta2→theta3`, and `theta3→theta4` edges on 4,096 fixed qualification
users and the next unseen window. The model remains 24L/H1536 with a 43.638-GiB global FP32
embedding table. All cache endpoints use FP16 storage followed by FP32 consumption, and every
edge uses the same frozen 999-negative candidate binding. The selected learning rates are
`1e-5` for dense/projection and `1e-4` for embedding. Checkpoint admission reads no ranking
metric.

The resulting Exact-over-Reuse sampled-CE gaps are 0.01846, 0.01068, and 0.01340; record-cluster
95% intervals are `[0.01520, 0.02167]`, `[0.00917, 0.01223]`, and
`[0.01073, 0.01603]`. All three current-model Exact endpoints also improve over the frozen-model
control in CE. This configuration was selected from three development candidates; it is not a
formal replication or untouched-test result. Its four checkpoints are retained under
`quality_chain_stream_aligned_train16384_round1`; the two rejected candidate checkpoint trees
were deleted after their compact results and hashes were preserved.

The follow-on `evokv_xp_d1_quality_development_v1` bridge is also complete on the three edges.
The analytic direct-old-K/V compiled path closes 63.9%, 55.3%, and 70.0% of the paired CE gap;
adding a label-free approximately 20% retained-token Exact route closes 68.9%, 62.3%, and 74.3%.
The compiled maintenance component is 0.162x, 0.152x, and 0.146x Exact. Conversely, the naive
mixed batches still pay component bounds of 0.764x, 0.781x, and 0.731x Exact because many batches
execute both routes. This is development evidence for the D1→D2 causal boundary, not an
end-to-end mixed-runtime speedup and not a replacement for the cross-dataset fitted-residual D1
headline. The bound artifacts and summary are in
`results/baseline_rounds/quality_chain/selected_d1_bridge_round1/`; no full K/V payload is
retained.

D3 development uses the current D1 plan and implemented D2 runtime. M0 uses a minimal two-rank
`WorkManifest`; the QK M1 revision binds its action snapshot, owner map, group plan, checkpoints,
and source store directly. A normalized exporter is not a prerequisite for mechanism discovery.
Within one `stack_revision`, baselines and candidates share the recorded work snapshot. Cross-layer
revisions are allowed during discovery but must rerun their own baselines. All early outputs remain
non-scientific until a D3 protocol is frozen. D2 paper claims remain blocked independently;
starting D3 does not promote W3 evidence.

The new paper workload is `X-QK-HET`, which preserves natural
old/retained/evicted/append/target lengths
from D1 through D3 and defines capacity by valid K/V bytes. `X-QK-HOM` uses the same HET record IDs
and valid histories but materializes a masked 512-slot physical layout; it is a shape-regularity
control, not a different user population or an alternative chosen after seeing performance.
Existing fixed-512 QK M1 artifacts remain immutable historical development evidence and do not
define the successor formal endpoint.

Foundation Review 0 has now materially constructed the successor input. A complete QK scan
freezes 65,536 mutually isolated final records plus separate model-edge, fit, profile, and
qualification roles. Their natural target length has median 153 and p95 404; only 2.1835% reach
512. The full HET old/target valid K/V inventories are 1.498/1.801 TB, and the frozen
36/72/144/288/576/720-GiB target points require
1,416/2,815/5,625/11,272/22,544/28,192 records. The same-record HOM allocation is much larger
and is retained only where independently capacity-admitted.

The all-exact valid-target request union contains 929,554 mapped rows and is entirely inside the
base catalog, but optimizer activity remains unmeasured. The actual XP forced-sharding threshold
is 2,840,105 optimizer-updated semantic rows (99.3101%): global FP32 embedding plus the dense core
is 48,023,005,184 bytes versus 47,699,722,240 single-card Torch allocatable bytes. A real
GPU0/GPU1 physical canary successfully allocates the modulo shards and performs owner-side
E4096→H1536 lookup/projection with a numerical oracle, but it intentionally does not claim a
trained checkpoint. HET and HOM short/mid/long/saturated lifecycle canaries pass full-payload
length/hash validation, commit-before-reclaim, failure isolation, idempotent replay, and
exactly-once coverage; they do not yet execute D1/D2 numerics. All of these are development
artifacts, not a protocol or performance result.

The current M0 full pass is `scientific_result=false`, `formal_design3=false`, and uses a logical
payload cap rather than physical oversubscription. It partitions H12 into 26 groups and writes
all 682 records (30.64 GB including small metadata/hidden outputs) exactly once. The latest
17.73-second single-pass S0 profile records 7.02–7.79 seconds of D2 execution including embedding
collectives/rank wait and 9.93–10.01 seconds across pageable→pinned, H2D, D2H, and pinned→pageable
phases on the two ranks. Lookup collectives account for 0.72–1.64 seconds; subtracting only that
measured component leaves a 6.15–6.30-second non-collective estimate. Peak allocated HBM is
18.53/13.90 GB and the one-slot pinned footprint is 1.53/1.56 GB. This supports proceeding to
overlap exploration; it is not a speedup, capacity, or paper claim.

The QK M1 boundary is now a completed and frozen development snapshot, not merely a planned
capacity point. Its 2,560 users comprise 512 fit/calibration users and one disjoint 2,048-record
benchmark pool. The first 64 raw exposures per QK user produce 2,859,835 base-active item
entities: the base-frequency top 250,000 are prediction rows, the other 2,609,835 are lossless
context rows, and items first observed later hash only into those existing context rows. Including
padding, the 24L/H1536 model has 2,859,836 FP32 embedding rows (17,570,832,384 bytes,
16.364 GiB) and 285,571,584 dense parameters.

Two-rank row-sharded training completed one `theta0→theta1` edge. On the fixed held-out
`[544,576)` window with 13,426 positive targets, `theta1` improves NDCG@10 from 0.371468 to
0.380294 and Hit@10 from 0.520259 to 0.547073 while reducing sampled cross entropy from
3.707804 to 3.653369. This is the required positive recommendation signal for mechanism
development, not a replicated quality claim. The edge-specific D1 snapshot fixes 410 exact and
1,638 compiled records, or 20.0195% exact. The corresponding read-only D2 characterizer reports
262,336 mixed lookup tokens versus 1,048,576 for all-exact, and 805,380,096 versus
3,216,408,576 bytes of off-rank FP32 return vectors. Those ratios characterize the fixed work;
they are not runtime speedups.

The complete exact `theta0` old-K/V store has also been physically materialized in ordinary DRAM:
144 GiB total, 72 GiB per rank, with complete coverage. The full GPU0/GPU1 S0 then processed all
2,048 records in nine sequential groups and wrote a complete 144-GiB private `theta1` target, so
old plus target is 288 GiB. It completed exactly once with a 53.497-second makespan. On rank 0,
which determines the makespan, the measured phase sum is 50.017 seconds; pageable→pinned, H2D,
D2H, and ordinary-DRAM publication contribute 26.397 seconds, or 52.8%. This is direct evidence
that sequential grouping exposes a strong DRAM staging/publication bottleneck and justifies S1
double buffering.

S1 needs a smaller resident group to admit two bounded slots, so the same revision also uses a
17-group, group-64 control. Its first 54.577-second execution included per-group Python GC and CUDA
allocator-cache flushing. That run remains evidence that shrinking groups alone does not remove
movement, but it is not the fair control for the current runtime. Removing only that avoidable
between-group maintenance gives a 48.238-second S0 over the identical records, actions, group
cuts, endpoints, and primary timer. It completes exactly once with 20.146-GiB peak allocated HBM.

The strong S1 then overlaps whole-group input staging, D2 execution, and output drain using two
bounded slots and one output credit. It completes in 32.703 seconds, or 1.475x faster than fair S0.
The full mixed wave remains correct, but rank 0/1 still expose 6.795/6.195 seconds of
input-boundary wait. A first D3 probe segments only pageable packing and H2D inside each capacity
group. It lowers those waits to 0.275/0.243 seconds, but the faster producer exposes
4.865/3.706 seconds of output-credit wait, so makespan reaches only 31.096 seconds.

The historical v1 D3 candidate is a bounded hierarchical pipeline rather than another whole-group
buffer.
It alternates two pinned input components so CPU packing for segment \(j+1\) overlaps H2D for
\(j\), and alternates two pinned output components so D2H for segment \(j+1\) overlaps
ordinary-DRAM publication for \(j\). D1 actions, D2 owner membership, capacity cuts, target layout,
and one-drain backpressure are unchanged. At group-64 and globally fixed microbatch-8 it completes
in 28.885 seconds: 1.133x over strong S1 and 1.670x over fair S0. A coupled microbatch-16 check is
slower at 29.337 seconds and raises reserved HBM from 39.42 to 41.54 GiB. Because this result uses
the older v1 runner, it is historical mechanism evidence and cannot be used as an order-only
control for the current planner.

The current development mechanism turns this fixed pipeline into a route-aware `ResidencyPlan`.
Input H2D segmentation, D2 compute/collective batching, and output D2H/publication segmentation are
physically independent for each route, while each complete capacity group is still staged on GPU
before its D2 execution. The planner admits only compiled/exact profiles measured jointly in the
same no-plan run, takes max-rank service times, and scales tail groups by discrete segment counts.
It evaluates the actual one-lookahead/one-drain recurrence, searches small stable-interleaving
spaces exhaustively and larger ones with Pareto-beam dynamic programming, and uses a
global-min-anchored 3% resource tie.

The selected plan keeps `(8,8,8)` for both routes but launches
`[13,0,1,2,3,4,5,6,7,8,9,10,11,14,12,15,16]`. On one exact stack and identity hash, the route-major
control takes 28.514442098 seconds and the selected order takes 28.147194647 seconds. This is a
1.013047x speedup, or a 1.2879% wall-time reduction, attributable to the selected order within
that development pair. The selected run is 1.16186x over strong S1 and 1.71379x over fair S0.
Its analytic prediction is 29.244944224 seconds, 3.90% above observation. Both
77,309,939,712-byte rank targets are byte-identical to S1 with complete, exactly-once coverage.
Profile and plan construction are outside the runtime timer.

The selected triple is symmetric, so the implementation supports route-specific granularity but
this point does not establish a route-asymmetric granularity benefit. Compiled input-16 and
output-4 did not improve their observed development points; they do not reject those choices
generally. An adjacent identity-only revision observed 29.7169→28.0497 (5.61%), but it is retained
only as a variability and mechanism-development diagnostic, not a frozen benefit. These runtime
executions share one trained seed and action/capacity mix.

A same-boundary contribution diagnostic now fills the previously missing grouped paths. Sequential
group-64 all-exact completes in 44.638644214 seconds; its action-oblivious two-slot version
completes in 33.548799294 seconds. An owner-local D1-only path keeps the same 1,638 compiled and
410 scheduled-exact actions but uses the unfused staged semantics
`retained → delta → latest` and a contiguous final cache. It completes in 57.597180375 seconds.
It requests the same 262,336 global embedding tokens as D2, but issues 852 collectives per rank
instead of 387. The current-binary sequential D1+D2 rerun completes in 49.752669533 seconds; the
earlier fair S0 was 48.238327569 seconds, exposing single-run variability that must be resolved by
formal repeats.

The sequential diagnostic is deliberately not monotone: D1-only is 29.03% slower than grouped
all-exact, D2 physical lowering recovers 13.62% relative to D1-only, and full D3 lowers the
current-binary D1+D2 path by 43.43%, for a net 36.94% reduction versus sequential grouped
all-exact. Under the stronger shared two-slot baseline, D1+D2 takes 32.703160657 seconds versus
33.548799294 seconds for all-exact, and full D3 reaches 28.147194647 seconds, 16.10% below strong
all-exact. The selected D3 artifact predates the contribution switches in the runner, so these
cross-row percentages are development guidance rather than a same-binary formal result. The
D1-only diagnostic intentionally keeps target-owner execution; it conservatively
isolates staged contiguous execution from D2 fusion/segmentation and does not measure a
placement-oblivious cross-owner baseline or claim the entire owner-compute contribution.
All four contribution rows pass finite-value, complete-coverage, and exactly-once publication
checks. The D1-only staged output has not yet received a full or sampled tolerance comparison
against its numerical reference, so these timings remain mechanism-development evidence.

The selected D3 point is already predominantly compute-bound. On the critical rank, affine
transform, non-lookup compute, lookup, and assembly total about 26.59 seconds of the 28.147-second
wall time; measured exposed pipeline wait is about 1.49 seconds. Perfectly removing the remaining
I/O bubbles therefore has only about a 5.5% direct ceiling on this fixed D1/D2 work. The stronger
remaining opportunity is interference-aware scheduling: selected-run effective H2D/D2H bandwidth
falls to roughly 4.50/4.86 GB/s from about 23.85/23.94 GB/s in sequential S0, while compute also
lengthens under blind overlap. More buffers or another segment-size sweep is not the priority.
The next D3 exploration should first test phase-aware DMA pacing and rank/collective-arrival-aware
planning, then an independently varied compiled compute batch.

The first large compiled group also exposed an implementation-scale boundary hidden by the small
benchmarks: 128 records per rank at 480 retained tokens create 61,440-token extents whose
layer-23 flattened source offset is 2,170,552,320 elements, beyond signed 32-bit indexing. The
direct-old-K/V Triton kernel now performs pointer arithmetic in 64 bits, and a cold-cache rerun
crossing \(2^{31}\) completed. This is a correctness fix and scale validation, not a separate
design claim. All QK M1 artifacts above remain `scientific_result=false` and
`formal_design3=false`. The S0/S1/D3 sequence establishes a mechanism and its causal profile, but
does not yet freeze a paper protocol or result.

## 4. Supported findings

### 4.1 Motivation and method boundary

The fixed-task 3×3 data/model-capacity motivation and cohort-tiered D1 replication are complete
over KuaiRand, Tenrec QB, and Tenrec QK. The evidence supports:

- stale reuse can lose streaming value, while exact maintenance is expensive;
- age alone is not a calibrated quality state;
- fixed suffix, plain progressive prefix, arbitrary intervals, and recent-token rectangles are
  useful baselines but not the active method;
- plain prefix is never selected in the unified action library;
- all 54 matched recent-token partial actions are slower at the evaluated setting;
- arbitrary intervals add negligible value relative to their planner complexity;
- a shared version-cohort affine repair scales in measured kernel cost and K/V fidelity.

The strict task-quality gate passes 6/9 capacity cells because some exact-maintenance task endpoints
are near zero or negative. This is a boundary, not an invitation to train a task-quality admission
oracle. Full recomputation is the cache-fidelity reference, not a guaranteed upper bound on ranking
quality.

### 4.2 D1 source representation is decisive

The original complete normalized-capsule path processes 17.82 GB of FP16 source state and loses to
exact at all six matched HBM/DRAM × 1/2/4-GPU endpoints. Source processing accounts for
91.35%–96.91% of compiled completion. The semantic compiler and resident operator remain valid;
that physical source representation is a negative systems result.

Stage 4.5 composes the deployed affine into a direct transform over existing exact source-version
old K/V. It adds no per-record `Norm(x)` state and uses three 33.59-MB direct programs. Under the
declared hot-HBM same-boundary setting, complete-cohort compiled repair takes
0.930/0.494/0.255 seconds on 1/2/4 GPUs versus 18.695/9.729/4.766 seconds for paired
raw-history-resident exact, or 20.11×/19.71×/18.72×. This is not a DRAM, SSD, or cold-source claim,
and its equivalence assumes exact source-version input K/V.

### 4.3 Repeated input is bounded by exact renewal

The frozen Stage-4.6 682-record chain recursively advances exact theta0 K/V through 11 updates on
one KuaiRand seed-0 16L/H512 setting. A deterministic 15%–25% exact schedule and maximum migration
depth four cost 0.2134× cumulative all-exact GPU time while preserving minimum cache/score/top-100
fidelity of 0.9632/0.999950/0.9918. The predecessor per-cache threshold route is a negative result
because it synchronized severe refresh waves.

Stage 4.9 corrects the retained-prefix accounting order: migrate the retained old prefix, stop the
paired maintenance timers, and then perform the same target-model append on both branches. The
selected `staggered_renewal_h12` candidate costs 0.100017× paired exact on the 11-edge
host-staged chain, with record-weighted AUC/NDCG@100/Hit@100 recovery
1.000039/0.997463/1.000000. Movement is reported separately, so this is not a full-cohort
HBM-resident or end-to-end movement result.

### 4.4 D2 exposes the logical-to-physical gap

The frozen H12 step contains:

- 548 compiled records;
- 46 scheduled-exact records;
- 88 natural-exact records;
- 682 records total.

Thus `134/682 = 19.6%` is an exact-route record statistic. The complete mixed wave still performs
`347,062/934,917 = 37.1%` of all-exact lookup tokens because exact records and method-common append
have different lengths. Retained-prefix lookup alone falls from 637,954 to 50,099 tokens, but a
compiled record's append is not embedding-free.

On three A40s, the development chain first shows the important negative result: naive owner-mixed
execution loses to one-shot exact because padding, repeated retained-prefix rewrite, and fragmented
phases consume the logical saving. Keeping actions fixed, the candidate D2 lowering adds:

1. segmented suffix-only destination finalization;
2. `(suffix, retained)` shape-aware compiled extents;
3. a merged physical exact pool with separate semantic provenance.

The full-682 B8 development point is 3.633 s for the merged mixed path versus 6.716 s for one-shot
all-exact, with measured one-way off-diagonal vector volume
454.62 MiB versus 1,222.86 MiB. Full-payload development validation covers every record and
approximately 15.32 billion valid K/V/hidden elements. These are mechanism-discovery results marked
`scientific_result=false`; they do not include the formal W4 gate, complete preparation,
publication/commit/reclaim timer, segmented consumer, or paired 1/2/4-GPU protocol.

Synthetic lookup contention is supporting resource characterization only. It is not a production
serving trace, Motivation-2 requirement, or substitute for same-boundary D2 results.

## 5. Working interfaces and revision rules

### 5.1 `ActionPlan`

D1 fixes source/target version, policy, provenance, counts, and content hash at plan level. For each
record, the current v1 artifact fixes:

- semantic action and reason;
- old, retained, delta/latest, target-prefix, and final extents;
- previous-cache presence, last-exact version, and migration depth;
- history and extent-identity hashes.

Compiled-program identity/hash is a separate D1 artifact bound by the D2 adapter. The canonical
H12 v1 action plan contains only `compiled` and `exact`; the successor paper-core system keeps this
two-route domain. Progressive remains a D1-only supporting action; any future full-stack use
requires an explicitly versioned schema, declared auxiliary state, and new D2/D3 baselines.

An `ActionPlan` is immutable within one recorded execution revision. An isolation-track D3
ablation keeps it unchanged. A co-design candidate may create a new plan before execution, but it
must receive a new `stack_revision` and rerun its baselines; it is not compared as if only the
ResidencyPlan changed.

### 5.2 Current D3-facing `WavePlan` view

D2 fixes:

- logical GPU owner and embedding routing;
- operator;
- `(S,R)` compiled-bin or `F`-keyed exact-pool membership;
- collective dependency and ordering constraints;
- segmented target layout;
- coverage, lineage, validation, group-version commit, and reclaim contract.

For a formal isolation experiment, this object should be global and capacity-independent. It does
not freeze D3 micro-wave cuts or launch order. Within that track, D3 only slices current pools and
preserves the recorded collective participation.

It is not yet a standalone repository artifact. `cohortkv_d2_wave_plan_v1` is a Stage-A single-rank
adapter, and W3 `D2IntegratedExtent` objects are capacity-specific resident schedules. The first
two-card benchmark may instead export a minimal `WorkManifest` from the current runtime. A
normalized, validated, stable-hash view is deferred until the candidate mechanism and final layer
boundary are clear.

### 5.3 `ResidencyPlan`

The development `ResidencyPlan` now fixes:

- legal bin/pool slices;
- per-rank admission;
- route-specific input-segment, compute-batch, and output-segment sizes;
- full-capacity-group GPU staging around independently batched D2 compute;
- a stable interleaving of route-internal group sequences;
- one-lookahead/one-drain prefetch/execute/writeback order;
- input/output pinned-buffer credits;
- backpressure;
- embedded same-source route profiles and their hashes;
- physical HBM totals, pinned-memory limits, and a global-min-anchored 3% resource tie;
- compiler/program, relevant source code, Torch/CUDA, GPU UUID/PCI, store-tier, group, checkpoint,
  capacity, and selected-plan identities.

Stage service is the maximum across ranks and tail work is scaled by integer segment counts. Small
stable-interleaving spaces are searched exhaustively; larger spaces use Pareto-beam dynamic
programming. Both ranks repeat capacity and identity preflight before target creation and D2
collectives. Profile and plan construction are recorded but remain outside the current runtime
timer.

Within one isolation-track `stack_revision`, sequential grouping, fixed-FIFO double buffering,
profile-aware generic scheduling, and the proposed scheduler consume the same `WorkManifest`. A co-design track may
regenerate actions, owners, pools, or layout before the run; its corresponding baselines use that
same regenerated stack. The historical M1 implementation still uses a complete private target;
the successor endpoint instead commits and reclaims capacity groups. NUMA/store qualification and
a normalized capacity-independent D2 constraint artifact remain outside the old development
result, but are recorded for the new formal runner.

## 6. Evaluation rules

The non-negotiable invariants are:

- predict item \(t+1\) from hidden state \(t\);
- train only on current stream-date targets and use engaged items as positives;
- fit the vocabulary on the base period only;
- respect sequence lengths and zero padding in all hidden/K/V paths;
- evaluate stale state as an old-version prefix plus the latest token under the current model;
- measure GPU cost rather than substituting an analytical constant;
- treat training seed as the replication unit;
- keep result families with different protocol strings separate;
- separate search, certificate, and final roles;
- compare systems at identical source, destination, dtype, layout, durability, and publication
  endpoints.

For D2, report record-action fraction, lookup-token fraction, physical communication, padding,
rewrite, collective count, wall time, and transaction time separately.

For an isolation-track D3 comparison, all mixed baselines use the same action-required source
bytes:

- compiled reads valid retained old K/V;
- exact reads raw history IDs and no unused old K/V;
- append reads suffix IDs;
- output follows the D2 segmented layout.

Progressive residual repair is evaluated only in its D1 supporting protocol and is not part of the
paper-core D3 `WorkManifest`.

All-exact necessarily has a different action/source multiset. It must share records, target model,
ordinary-host source tier, target dtype/layout/durability, topology, per-rank HBM budget, timer, and
manifest endpoint, while reporting its actual raw-history bytes.

A cross-layer candidate may also have different action/source bytes, but it must report those
differences and rerun baselines under its new `stack_revision`.

The initial M0 development timer covers ordinary-DRAM↔pinned copies, H2D/D2H, GPU compute,
collectives, and historical private-target writes. Per-group sampled finite/metadata checks currently occur
inside the wall timer; global exactly-once coverage is checked after it. Full checksum/numerical
parity, plan, commit, and reclaim are outside this first profile. The later paper-facing timer must
include every group writeback, validation, versioned commit and old-group reclaim through the final
job manifest. A no-I/O chunk sum is characterization only.

Successor mechanism figures use the execution-only timer. E1 end-to-end uses first-wave
update-inclusive cost after model-checkpoint publication, including edge-specific D1 fit/compile,
D2 lowering/plan, D3 profile/plan, and rolling execution; model training is reported separately.
Formal methods are measured in randomized/interleaved blocks. Every uncommitted group writes to a
bounded replacement shadow and preserves its old extent until validation and commit.

## 7. Current execution order

### D2 closure

1. Build the natural-length HET primary manifest and matched HOM control, and freeze the
   paper-core `compiled|exact` ActionPlan domain.
2. Freeze the hardware HBM cap, then qualify the fixed XP embedding/model geometry without
   consulting EvoKV speedup; generate verified 1/2/4-way shards.
3. Generalize the runner to 1/2/4 ranks and export/hash the capacity-independent D2→D3 constraints.
4. Implement same-binary all-exact, strongest placement/exact controls, staged/fused owner-local
   contiguous baselines, segmented consumer, and group validation/commit/reclaim.
5. Record Benchmark Qualification separately; it blocks paper-result promotion, not current
   benchmark design or baseline implementation.
6. Freeze a successor D2 protocol only after the above foundation and independently tuned
   baselines are fixed.

### D3 mechanism entry

The benchmark-first route has reached a real physical M1 mechanism boundary:

1. the two-rank 24L/H1536 `theta0→theta1` edge and positive held-out signal are complete;
2. the 2,048-record D1 snapshot is fixed at 410 exact and 1,638 compiled actions, and its D2
   request/communication characterizer is complete;
3. the 144-GiB old store is materialized; fair group-64 S0 completes the boundary in 48.238
   seconds;
4. same-revision strong S1 completes in 32.703 seconds and identifies residual input-boundary
   stalls;
5. input-only segmentation removes those stalls but exposes serialized output drain;
6. the historical v1 bidirectionally segmented fixed-order candidate completes in 28.885 seconds,
   but its runner differs and it is not the current order-only control;
7. under the current v3 runner and one exact stack/hash, route-major `(8,8,8)` completes in
   28.514442098 seconds and the selected stable order completes in 28.147194647 seconds
   (1.013047x; 1.2879% wall reduction), with full byte parity and complete exactly-once coverage;
8. the selected point retains `(8,8,8)` on both routes; input-16 and output-4 did not improve their
   observed probes, but route-asymmetric granularity benefit and general rejection of alternatives
   are not established;
9. grouped same-boundary all-exact and D1-only contribution diagnostics complete the historical
   development waterfall; a same-binary, independently tuned formal E0→D3 crossover was not
   established;
10. these ten items describe the immutable fixed-512/full-private-target development family; the
    successor benchmark switches to HET, rolling group replacement, XP, and a 1/2/4-rank-capable
    runner rather than promoting this family.

The successor capacity coordinate is

$$
\rho_{\mathrm{KV}}=\max_r
\frac{\text{single-live-version valid K/V bytes}_r}
{\text{usable HBM}_r-\text{fixed model/embedding/program bytes}_r},
$$

and the formal matrix uses manifest-derived 36–720-GiB single-version points rather than an
old+complete-target footprint. Every admitted micro-wave must satisfy, per rank,

```text
fixed(model + embedding shard + program + context)
+ next input
+ current execution transient
+ previous output
<= physical HBM - allocator/safety margin.
```

### Paper evaluation expansion

The complete evaluation portfolio and physical scale budget are now specified in
[`10_paper_experiment_blueprint.md`](10_paper_experiment_blueprint.md). Execution proceeds by:

1. materializing the formal QK role split, HET primary and HOM control manifests;
2. freezing XP and producing edge-specific X1/X2/XP programs, two-route action plans, and verified
   1/2/4-way embedding shards;
3. closing the resident/out-of-core timers, segmented consumer and rolling group lifecycle;
4. applying baseline-first within each layer: D2 foundation, complete D2/frozen stack, D3
   foundation including fixed-FIFO and profile-aware generic schedulers, then D3 mechanisms;
5. executing the separate Benchmark Qualification registry before formal result promotion, then
   de-duplicating the current 70–80-cell envelope into an exact ledger.

The blueprint plans new result families; it does not promote any W3 or D3 development artifact.

## 8. Go/no-go and pivot rules

D2 should not be promoted as a paper design unless fixed-action physical lowering survives the
formal same-boundary timer and strong all-exact baseline. If the W3 benefit disappears, first
attribute the loss to preparation, layout materialization, segmented consumption, communication,
or group-lifecycle overhead. Do not repair the result by changing D1 actions.

The paper-scale D2 foundation proactively uses XP rather than waiting for X2 to fail. XP fixes
2,859,835 base-period semantic rows plus one padding row in a 43.638-GiB physical table and uses
owner-side E4096→H1536 projection; fixed model state already exceeds physical Torch allocatable
bytes. The hardware HBM cap is frozen before XP
qualification, and all methods may project at the owner before returning
H1536 vectors. Cold unused rows or raw-E4096 cross-rank vectors cannot manufacture the claim.
The exact denominator evaluates a bounded joint grid over legal placement/transport,
routing/coalescing, and pipeline combinations rather than pruning on placement alone. X2 remains
the development/quality bridge and 1-rank sanity.

D3 enters the paper only if it:

1. passes full-payload correctness and per-rank bounded-memory admission;
2. reports same-boundary sequential, strong whole-group double-buffer, generic fixed-FIFO
   fine-grained segmented, profile-aware work-conserving generic, proposed, and independently tuned
   all-exact paths;
3. shows an attributable gain over the strongest generic pipeline, not merely over whole-group
   double buffering;
4. has at least one meaningful operating point relative to fastest same-boundary all-exact;
5. reports the positive region and the exact-preferred crossover.

If D3 only beats a sequential loop or whole-group double buffer but not the strongest fixed-FIFO
or profile-aware generic pipeline, it is an implementation path rather than a design. If the unavoidable direct-old-K/V
input lower bound is already slower than same-boundary exact, stop scheduler tuning and revisit
source representation.

## 9. Claims not supported

Do not claim that:

- model age predicts safe reuse;
- per-user drift/JVP/Fisher is the active route;
- fixed suffix, plain prefix, recent-token replay, or arbitrary intervals are active EvoKV designs;
- full recomputation is always the ranking-quality upper bound;
- approximately 20% exact records imply approximately 20% lookup, communication, or time;
- an entire compiled record is embedding-free;
- D2 changes semantic actions according to communication;
- W3 development timings are formal or paper evidence;
- a standalone capacity-independent D2→D3 constraint artifact already exists;
- segmented output is already consumed without hidden materialization;
- the current dense model requires tensor parallelism;
- every recommendation deployment requires full-epoch atomic cache publication;
- synthetic lookup contention is a serving trace or SLO result;
- normalized-capsule DRAM, destination-v4 correctness, or hot-HBM Stage 4.5 is D3 evidence;
- the QK M1 S0/S1/D3 development profiles are a frozen protocol or paper evidence;
- the historical 28.885-second v1 result is an order-only control for the current runner;
- the adjacent-revision 5.61% scheduler reduction is a frozen benefit or independent replication;
- the selected plan demonstrates route-asymmetric granularity benefit;
- the current primary timer includes profile acquisition or plan construction;
- its 52.8% rank-0 movement fraction is entirely PCIe time rather than the declared combination of
  ordinary-memory staging, H2D, D2H, and publication;
- D3 has a final/paper-ready implementation, frozen protocol, replicated performance result, or
  all-exact crossover;
- SSD, database, remote object storage, RDMA, GDS, or host-DRAM oversubscription is evaluated.

## 10. Document and artifact map

- Paper-wide experiment blueprint:
  [10_paper_experiment_blueprint.md](10_paper_experiment_blueprint.md)
- Benchmark qualification registry:
  [11_benchmark_qualification.md](11_benchmark_qualification.md)
- Current D2 design:
  [future_design/DESIGN2_FINAL_PLAN.md](future_design/DESIGN2_FINAL_PLAN.md)
- D2 implementation/evidence status:
  [future_design/DESIGN2_DEVELOPMENT_STATUS.md](future_design/DESIGN2_DEVELOPMENT_STATUS.md)
- D3 design-ready plan:
  [future_design/DESIGN3_FUTURE_DIRECTION.md](future_design/DESIGN3_FUTURE_DIRECTION.md)
- D3 foundation benchmark and exploration plan:
  [future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md](future_design/DESIGN3_FOUNDATION_AND_EXPLORATION_PLAN.md)
- Frozen D1 evidence ledger:
  [09_single_configuration_full_chain_plan.md](09_single_configuration_full_chain_plan.md)
- Result-family semantics:
  [eval_protocol.md](eval_protocol.md)
- Experiment-record index:
  [../experiments/README.md](../experiments/README.md)
- Dataset boundary:
  [dataset_expansion_audit.md](dataset_expansion_audit.md)

Deleted plans and paper-process notes are historical only. Do not recover a current claim or task
from Git history when it conflicts with this roadmap or the evaluation protocol.
