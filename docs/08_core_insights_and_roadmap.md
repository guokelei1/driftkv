# Core insights and roadmap

> Status: authoritative as of 2026-07-30. This document replaces earlier project-wide problem
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
    decide what must be translated, progressively repaired, or exactly recomputed

D2: distributed WavePlan constraints
    decide owner, operator, physical compatibility, communication dependencies, and output layout

D3: capacity-bounded ResidencyPlan
    decide legal capacity cuts, packing, and ordinary-DRAM↔GPU launch order
```

This is the current paper decomposition, not a permanent implementation firewall. For an isolated
D3 ablation, D1 actions and D2 execution stay fixed. During mechanism discovery, capacity and
transport measurements may motivate a new globally planned D1/D2/D3 `stack_revision`; that
revision must rerun its own baselines instead of comparing against an older stack. Existing frozen
D1 and D2 evidence is not retroactively changed.

### 1.1 D1 question: reuse versus recomputation

> Can a version cohort share a migration program that closes a useful portion of the stale-to-fresh
> K/V gap at materially lower GPU cost than exact history replay?

D1 emits a per-record `ActionPlan` that is immutable within one execution/revision. The active
action library is:

- **compiled repair:** fit the shared `fresh - cheap` residual for a source/target version cohort
  and compile it into one affine transform over cached source-version state; the selected hot path
  composes this transform directly over existing old K/V;
- **progressive repair:** replay a current-model prefix and transport the boundary residual to
  deeper current `Norm + Wk/Wv` projections when that declared state is available;
- **exact:** recompute current-model K/V and reset approximation depth.

Every stale cohort receives the declared repair. Version is a compilation, artifact, and batching
key, not a claim that all records at an age are safe to reuse. Recommendation labels and realized
task gain never route caches.

### 1.2 D2 question: logical versus physical sparsity

> In a row-sharded multi-GPU setting, can D1's fixed logical reduction survive embedding lookup,
> padding, suffix append, collectives, destination movement, synchronization, and atomic
> publication?

The target D2 lowering is captured by global `WavePlan` constraints:

- compiled retained-prefix repair executes at the old-K/V owner and performs no embedding lookup;
- scheduled exact, natural exact, and append request only unavoidable row-sharded item vectors;
- compiled work is compatible by `(suffix, retained)` shape;
- semantically distinct but physically identical exact reasons retain provenance while sharing one
  execution pool;
- retained and suffix output remain segmented instead of rewriting the retained prefix;
- collective order, coverage, lineage, private target state, commit, abort, and reclaim are
  explicit.

D2 may choose owner placement, batching, physical pool construction, execution order within its
dependency rules, and segmented layout. The current D2 design does not change
`compiled|progressive|exact`. A later cross-layer revision may regenerate both the D1 plan and D2
lowering before a run; it is then a new stack rather than a D2-only ablation. The mechanisms are
implemented in the current runtime, but their capacity-independent D3-facing constraint view has
not yet been separately serialized and hashed.

### 1.3 D3 question: working set versus HBM capacity

> When complete source plus complete private target K/V exceeds per-rank usable HBM, can the
> current D1/D2 work execute from ordinary host DRAM without losing its gains to
> capacity cuts, CPU staging, PCIe movement, collective stalls, or pipeline bubbles?

D3 derives a separate `ResidencyPlan`:

```text
ordinary host DRAM committed source
  → bounded pinned input staging
  → H2D
  → D2-constrained GPU execution
  → D2H
  → bounded pinned output staging
  → ordinary host DRAM private target
  → validation and atomic target-manifest publication
```

D3 initially owns per-rank admission, legal bin/pool slices, micro-wave packing,
prefetch/execute/writeback order, pinned-buffer credits, backpressure, and NUMA/PCIe staging
placement. This narrow path is the isolation track. If measurement shows that owner placement,
action granularity, pool construction, or target layout must change to exploit out-of-core
execution, the project may explore a global cross-layer revision and assign final ownership only
after the mechanism is understood.

## 2. Scope

The primary object is model-version-stale HSTU prefix K/V produced by streaming training. The
current system boundary begins after target-model checkpoint publication and ends when one complete
target-version cache manifest becomes visible.

In scope:

- HSTU models with pointwise unnormalized attention and first-class per-layer K/V;
- model-version cohorts, compiled programs, progressive state where explicitly declared, and exact
  replay;
- row-sharded embedding lookup and multi-GPU execution;
- committed source state, private target state, validation, atomic publication, abort, and reclaim;
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

The current local model fits on one A40. D2's distributed problem comes from row-sharded embeddings,
large cache state, strict copy-on-write, and communication—not from claiming the dense model cannot
fit on one GPU.

## 3. Current status

| Layer | State | What is closed | What remains |
|---|---|---|---|
| D1 | frozen | method, direct-old-K/V source plan, bounded renewal, Stage-5 guard/fallback/transaction, Stage-6 aggregate | broader replication and optional Stage-4.10 successor |
| D2 | implementation and mechanism discovered; paper evidence open | Stage A, W1/W2, W3 diagnostics, C0 wiring, segmented/shape-aware/merged-exact development path, full-payload development correctness | D3-facing constraint exporter/hash, independent W4, frozen formal protocol, 1/2/4-GPU same-boundary evaluation, segmented consumer, full publication/commit/reclaim |
| D3 | two-card M0 S0 implemented; development only | flexible WorkManifest/grouping, GPU0/GPU1 pageable-DRAM sequential path, real D2 compiled/exact execution, full682 exactly-once private-target writeback and phase profile | same-revision S1, real QK M1, mechanism discovery, then final interfaces/protocol/evidence |

D3 development begins from the current D1 plan and implemented D2 runtime using a minimal
two-rank `WorkManifest`; a normalized exporter is not a prerequisite for the first benchmark.
Within one `stack_revision`, baselines and candidates share the recorded work snapshot. Cross-layer
revisions are allowed during discovery but must rerun their own baselines. All early outputs remain
non-scientific until a D3 protocol is frozen. D2 paper claims remain blocked independently;
starting D3 does not promote W3 evidence.

The current M0 full pass is `scientific_result=false`, `formal_design3=false`, and uses a logical
payload cap rather than physical oversubscription. It partitions H12 into 26 groups and writes
all 682 records (30.64 GB including small metadata/hidden outputs) exactly once. The latest
17.73-second single-pass S0 profile records 7.02–7.79 seconds of D2 execution including embedding
collectives/rank wait and 9.93–10.01 seconds across pageable→pinned, H2D, D2H, and pinned→pageable
phases on the two ranks. Lookup collectives account for 0.72–1.64 seconds; subtracting only that
measured component leaves a 6.15–6.30-second non-collective estimate. Peak allocated HBM is
18.53/13.90 GB and the one-slot pinned footprint is 1.53/1.56 GB. This supports proceeding to
overlap exploration; it is not a speedup, capacity, or paper claim.

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
H12 v1 action plan contains only `compiled` and `exact`; any future progressive action requires an
explicitly versioned schema and declared auxiliary state.

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
- coverage, lineage, validation, and publication contract.

For a formal isolation experiment, this object should be global and capacity-independent. It does
not freeze D3 micro-wave cuts or launch order. Within that track, D3 only slices current pools and
preserves the recorded collective participation.

It is not yet a standalone repository artifact. `cohortkv_d2_wave_plan_v1` is a Stage-A single-rank
adapter, and W3 `D2IntegratedExtent` objects are capacity-specific resident schedules. The first
two-card benchmark may instead export a minimal `WorkManifest` from the current runtime. A
normalized, validated, stable-hash view is deferred until the candidate mechanism and final layer
boundary are clear.

### 5.3 `ResidencyPlan`

D3 will fix:

- legal bin/pool slices;
- per-rank admission;
- micro-wave packing;
- prefetch/execute/writeback order;
- input/output pinned-buffer credits;
- backpressure and NUMA/PCIe staging placement.

Within one isolation-track `stack_revision`, sequential grouping, action-oblivious double
buffering, and the proposed scheduler consume the same `WorkManifest`. A co-design track may
regenerate actions, owners, pools, or layout before the run; its corresponding baselines use that
same regenerated stack.

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
- progressive reads its declared BF16 hidden suffix;
- output follows the D2 segmented layout.

All-exact necessarily has a different action/source multiset. It must share records, target model,
ordinary-host source tier, target dtype/layout/durability, topology, per-rank HBM budget, timer, and
manifest endpoint, while reporting its actual raw-history bytes.

A cross-layer candidate may also have different action/source bytes, but it must report those
differences and rerun baselines under its new `stack_revision`.

The initial M0 development timer covers ordinary-DRAM↔pinned copies, H2D/D2H, GPU compute,
collectives, and private-target writes. Per-group sampled finite/metadata checks currently occur
inside the wall timer; global exactly-once coverage is checked after it. Full checksum/numerical
parity, plan, commit, and reclaim are outside this first profile. The later paper-facing timer must
explicitly close and include its selected complete boundary. A no-I/O chunk sum is characterization
only.

## 7. Current execution order

### D2 closure

1. Export and hash the capacity-independent D2→D3 constraint view, with parity checks against the
   current integrated runtime.
2. Run the independent physical W4 normal and hard-failure cases when all four A40s are safely
   available; do not kill or oversubscribe an external GPU process.
3. Freeze/check the Stage-B summary.
4. Freeze a new formal D2 protocol.
5. Run same-binary all-exact, naive sharded fixed-action mixed, and physical-sparse mixed through
   post-append publication, commit, and reclaim.
6. Complete paired 1/2/4-GPU, segmented-consumer, capacity, failure, and physical-communication
   evidence.

### D3 mechanism entry

The benchmark-first route is active. Steps 1–2 are implemented for development M0; step 3 is next:

1. keep the implemented GPU0/GPU1 H12/W2 `WorkManifest` and S0 as the same-revision reference;
2. retain its ordinary-DRAM source/target, byte-bounded groups, and exactly-once checks;
3. add a basic double buffer, event-based phase timing, and bounded pinned/HBM memory;
4. in parallel audit QK and construct the smallest real two-card physical out-of-core model edge;
5. reproduce sequential/double-buffer/all-exact on that real workload;
6. use the measured profile to explore both an isolated D3 scheduler and, when useful, globally
   replanned D1/D2/D3 co-design revisions;
7. only after a mechanism is clear, normalize interfaces, close transaction semantics, and freeze
   a D3 protocol;
8. defer 3/4-GPU and broader matrices until the GPU0/GPU1 result is understood.

The capacity coordinate is

$$
\rho=\max_r\frac{\text{work bytes}_r}{\text{usable HBM}_r},
$$

The first real benchmark needs only the smallest useful \(\rho>1\) point. A later characterization
may cover \(\rho \in \{0.5,1,2,4\}\). Every admitted micro-wave must satisfy, per rank,

```text
fixed(model + embedding shard + program + context)
+ next input
+ current execution transient
+ previous output
<= physical HBM - allocator/safety margin.
```

## 8. Go/no-go and pivot rules

D2 should not be promoted as a paper design unless fixed-action physical lowering survives the
formal same-boundary timer and strong all-exact baseline. If the W3 benefit disappears, first
attribute the loss to preparation, layout materialization, segmented consumption, communication,
or transaction overhead. Do not repair the result by changing D1 actions.

If the current embedding/cardinality scale cannot expose D2 communication after formal
measurement, construct a larger but semantically valid setting from an accepted dataset: train one
base model and one or two short streaming updates, regenerate exact old K/V, compiled programs,
certificates, and the immutable action plan, then repeat mechanism discovery. Cold unused rows
cannot manufacture the claim.

D3 enters the paper only if it:

1. passes full-payload correctness and per-rank bounded-memory admission;
2. reports same-boundary sequential, strong double-buffer, proposed, and all-exact paths;
3. shows an attributable gain over the action-oblivious double buffer;
4. has at least one meaningful operating point relative to fastest same-boundary all-exact;
5. reports the positive region and the exact-preferred crossover.

If D3 only beats a sequential loop, it is an implementation path rather than a design. If the
unavoidable direct-old-K/V input lower bound is already slower than same-boundary exact, stop
scheduler tuning and revisit source representation.

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
- synthetic lookup contention is a serving trace or SLO result;
- normalized-capsule DRAM, destination-v4 correctness, or hot-HBM Stage 4.5 is D3 evidence;
- D3 has a final/paper-ready mechanism implementation, frozen protocol, or performance result;
- SSD, database, remote object storage, RDMA, GDS, or host-DRAM oversubscription is evaluated.

## 10. Document and artifact map

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
