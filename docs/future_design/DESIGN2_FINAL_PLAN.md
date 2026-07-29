# EvoKV Design 2: Wave-Compiled Segmented Execution

Date: 2026-07-30

Status: **current mechanism fixed; D3-facing constraint exporter, formal Stage-B freeze, paper
protocol, and results remain open**. Historical code and artifact names retain `cohortkv_*` so
existing hashes and result families remain stable.

This document defines D2 only. Current execution state is in
[DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md). D3 begins from the required D2
interface defined here after its exporter closes, and is specified by
[DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md).

## 1. Problem

D1 has already chosen each record's semantic action. In the frozen H12 wave, only 134 of 682
records take an exact route, but that logical sparsity is not automatically a multi-GPU speedup.
Exact and append still access row-sharded embeddings; variable-length incremental work creates
padding; naive phase separation launches many collectives; rewriting a complete target cache can
copy an already repaired retained prefix again; and one target epoch must remain private until all
ranks agree on coverage and lineage.

D2 asks:

> Given one immutable D1 `ActionPlan`, how should its fixed work be lowered onto row-sharded
> multi-GPU execution so that logical sparsity becomes physical reductions in lookup, movement,
> padding, rewrite, collectives, and wall time?

D2 does not decide which cache should be compiled or exact. It solves a distributed execution
problem, not another migration algorithm.

## 2. Position in the three-layer system

```text
D1 ActionPlan
  semantic action, reason, extents, identity, provenance
        ↓
D2 D3-facing WavePlan constraints
  owner, operator, compatible physical pools, collectives, segmented layout, publication contract
        ↓
D3 ResidencyPlan
  capacity cuts, packing, prefetch/execute/writeback order, pinned credits, NUMA placement
```

The responsibility boundary is:

| Decision | Owner |
|---|---|
| `compiled|progressive|exact` and reason | D1 |
| retained/suffix/final semantic extents | D1 |
| old-K/V owner and embedding shard routing | D2 |
| operator and physical compatibility bin/pool | D2 |
| collective dependencies and participation | D2 |
| segmented target layout and lineage/coverage contract | D2 |
| capacity-specific slices and micro-wave packing | D3 |
| DRAM/HBM launch order and pinned-buffer credits | D3 |

The current D2 runtime exposes a resident execution order. That order is not a D3-facing
invariant. The target interface is a separately serialized, capacity-independent constraint
view—owner/operator/compatibility/dependency/layout—which D3 may cut into capacity-safe slices and
legally reorder. That normalized view has not yet been materialized and hashed; producing it is the
first D3-readiness task.

## 3. Immutable input

The canonical development input is the Stage-4.9 H12 theta1→theta2 action plan:

- 548 compiled;
- 46 scheduled exact;
- 88 natural exact;
- 682 total;
- content SHA-256:
  `c4bc383d28f3558fdd11be8788799aaa6f66e80f778a4670f781eb9295f0027e`.

`scheduled_exact` and `natural_exact` execute the same physical exact operator but retain distinct
semantic provenance. The action plan covers record IDs `0..681` exactly once.

The v1 action plan stores source and target version, policy, producer, provenance, counts, and
content hash at plan level. For each record it stores:

```text
record_id
prepared_user_id
requested_action
requested_reason
old_tokens
retained_start
retained_tokens
delta_start
delta_tokens
target_prefix_tokens
latest_tokens
final_tokens
last_exact_version
migration_depth
previous_cache_expected / previous_cache_present
history and extent-identity hashes
```

Owner, program, old-extent, and raw-history bindings are added by the D2 adapter/runtime rather
than stored in the action record. The canonical H12 v1 plan contains only `compiled` and `exact`
actions; it has no progressive-state record field. Any experiment that changes the action-plan
fields or its content hash is not a D2 physical-lowering ablation.

## 4. Required D3-facing `WavePlan` constraints

The D2→D3 contract requires one global, capacity-independent constraint plan containing:

```text
action_plan_hash
model/configuration hashes
record owner
embedding owner rule
operator identity
compiled (suffix, retained) compatibility-bin membership
exact final-length compatibility-pool membership
collective dependency and ordinal constraints
segmented target extent schema
coverage and lineage requirements
validation, commit, abort, and reclaim contract
```

The repository does not yet contain this normalized artifact. The existing
`cohortkv_d2_wave_plan_v1` is a Stage-A single-rank adapter with record bindings, while the W3
integrated path builds capacity-specific `D2IntegratedExtent` schedules at runtime. The latter
sorts owner-local compiled records by `(suffix, retained, final, record_id)` and then cuts them by
fixed `extent_size`; it does not serialize explicit global-bin membership or a standalone
collective-dependency graph.

Before D3 scheduler comparisons, a small D2 exporter must derive, validate, serialize, and hash the
required constraint view from:

- the immutable action plan and owner map;
- operator/program and source bindings;
- the implemented shape-aware ordering and merged-exact membership;
- collective templates;
- segmented layout and transaction requirements.

The exported view must exclude `extent_size`, HBM capacity cuts, resident launch order, pinned
credits, and prefetch/writeback decisions. It does not promise that one compatible pool fits in
HBM or executes as one launch. A D3 `ResidencyPlan` may slice inside a pool but cannot:

- move a record to a different semantic action;
- change the logical old-K/V owner;
- change the embedding shard rule;
- substitute another operator;
- combine incompatible shapes;
- reorder across a collective dependency;
- omit empty-rank collective participation;
- change target layout, coverage, or lineage.

Until exporter parity and content hashing close, D3 may prepare schemas and source-byte ledgers,
but no scheduler comparison may claim that all variants consumed identical frozen D2 constraints.

## 5. Mechanism

### 5.1 Owner-local retained repair

The compiled program is small relative to per-record K/V, so D2 moves the program to the state
owner rather than moving old K/V to a central worker. Retained repair:

- reads local exact source-version old K/V;
- applies the D1 direct affine program;
- writes a private retained target segment;
- performs zero item lookup;
- performs zero embedding collective;
- performs zero old-K/V peer transfer.

This invariant is limited to the retained-prefix phase. New suffix/latest tokens still require
embedding lookup for both mixed and all-exact methods.

### 5.2 Row-sharded exact and append

Exact and append run against a row-sharded embedding table. For each phase:

1. derive requested IDs from the frozen record extents;
2. route IDs to the rank that owns each embedding row;
3. obtain vectors through a deterministic collective order;
4. return only the vectors required by the requesting rank;
5. execute current-model exact or incremental HSTU work;
6. record requested/local/remote IDs and tensor payload bytes.

FP32 vector transport is the mechanical-equivalence baseline. FP16/BF16 transport, dedup, and
topology-specific variants require separate numerical and performance evidence.

### 5.3 Shape-aware ordered compiled extents

For a compiled record, let:

- \(R\): retained-prefix length;
- \(S\): suffix length appended after retained repair;
- \(F=R+S\): final length.

Sorting incremental work by \(F\) is insufficient. Its padding pressure is closer to:

$$
B \times S_{\max} \times (R_{\max}+S_{\max}).
$$

The implemented W3 path therefore orders owner-local compiled records lexicographically by
`(S,R,F,record_id)` before applying a fixed-size extent cut, rather than sorting by final length
alone. This is a capacity-specific development lowering derived from fixed D1 extents; it does not
change fidelity or exact budget. The D3-facing exporter must preserve the shape compatibility
information without freezing that W3 extent cut.

### 5.4 Segmented suffix-only destination

The target cache is represented as:

```text
retained segment | suffix segment
```

Compiled retained K/V is written once. Incremental append emits only new suffix K/V and the final
hidden state; it does not rewrite the retained prefix into a new contiguous cache. A consumer must
read the segmented layout directly. A hidden post-publication concatenation would merely defer the
same cost and invalidates the claimed physical saving.

“Suffix-only destination” does not mean zero temporary work: the current implementation may still
materialize padded transient retained K/V inside an incremental batch. That transient must be
measured.

### 5.5 Merged physical exact pool

`scheduled_exact` and `natural_exact` keep separate action reasons and lineage, but both compute one
full current-model final cache. D2 merges compatible records into a common `F`-keyed physical exact
pool. It thereby avoids collective and launch fragmentation created only by semantic labels.

### 5.6 Private publication

All outputs are private until the target epoch passes:

- complete and duplicate-free record coverage;
- expected source/target lineage;
- shape, dtype, valid-length, and component checks;
- collective completion on every rank;
- target-layout and fragment-set consistency.

Only then may the target manifest become visible. Abort preserves the source manifest and releases
private target references. D2 owns this semantic transaction; D3 later decides when individual
buffers reside in HBM without weakening the transaction.

## 6. Why the mechanisms form one design

The mechanisms are not an unrelated optimization list:

1. D1 creates nonuniform semantic work.
2. owner-local retained repair prevents the large state from becoming communication;
3. row-sharded exact/append exposes the unavoidable lookup traffic;
4. `(S,R)`-aware ordering exposes the true incremental shape;
5. segmented output removes retained-prefix rewrite;
6. merged exact pools remove semantic-only fragmentation;
7. atomic publication turns all physical fragments back into one valid target epoch.

Together they compile a semantic wave into a physical multi-GPU wave. The defining claim is
logical-to-physical sparsity, not owner-compute, batching, dedup, or COW in isolation.

## 7. Work ledger

The frozen H12 development plan has:

| Boundary | All exact | Fixed mixed | Ratio |
|---|---:|---:|---:|
| retained-prefix lookup tokens | 637,954 | 50,099 | 12.73× reduction |
| complete lookup including method-common append | 934,917 | 347,062 | 2.694× reduction |
| exact-route records | 682 | 134 | 19.6% of records |

Natural-exact records have zero retained overlap and contribute 82,612 lookup tokens. They are
known before wave execution from source-cache presence/extent metadata, but they are not selected
by the renewal policy.

All formal reports must keep separate:

- record-action fraction;
- retained and append token counts;
- unique and non-unique requested IDs;
- local/remote IDs;
- collective tensor bytes and, when available, wire bytes;
- H2D/D2H/P2P bytes;
- padded/transient elements;
- retained rewrite bytes;
- launch and collective counts;
- plan, prepare, execute, validate, publish, commit, and reclaim time.

## 8. Baselines

Formal D2 evaluation requires the same SPMD harness for:

1. **one-shot all exact:** fastest independently tuned complete current-model recomputation;
2. **two-stage all exact:** only when it reflects a meaningful execution decomposition;
3. **naive sharded fixed-action mixed:** same D1 actions and row-sharded embeddings without D2
   physical lowering;
4. **owner-local but contiguous mixed:** isolates owner-compute from segmented output;
5. **segmented mixed without shape-aware ordering;**
6. **shape-aware segmented mixed without merged exact pool;**
7. **complete D2.**

The current pre-SPMD record-DP Table-8 numbers cannot be used as a baseline. Every comparator must
be rerun in the same process model, source boundary, target transaction, and timer.

## 9. Timer and correctness boundary

The primary D2 timer begins after immutable source data and the frozen action plan are available,
and includes:

- distributed plan/materialization required for the run;
- item-ID routing and embedding collectives;
- compiled, exact, and append compute;
- transient padding/materialization;
- private target writes;
- validation;
- atomic publication;
- commit and reclaim.

Report a secondary retained-prefix-only boundary to preserve continuity with D1, but do not compare
that number directly to D2's post-append target-epoch time.

Correctness requires:

- FP32 distributed exact/append parity with the full-table reference;
- compiled retained parity with the D1 operator;
- every semantic reason and zero-delta/latest-only branch;
- full valid-element K/V and hidden coverage;
- padding exclusion;
- requested/local/remote ID and byte reconstruction;
- deterministic collective order including empty ranks;
- normal commit, semantic fallback, preflight rejection, mid-job abort, and pre-commit abort;
- segmented consumer or next-wave compatibility.

## 10. Formal paper gate

D2 becomes paper evidence only after:

1. independent four-A40 W4 normal and hard-failure closure;
2. a new frozen D2 protocol in `docs/eval_protocol.md`;
3. paired 1/2/4-GPU runs using the same binary and endpoint;
4. full 682-record post-append publication/commit/reclaim;
5. strong all-exact and naive fixed-action mixed baselines;
6. physical communication and per-rank capacity ledgers;
7. segmented consumer or next-wave support;
8. complete full-payload, transaction, and failure checks.

A positive D2 result must show that the physical-sparse path beats naive fixed-action mixed and has
a meaningful region relative to the fastest same-boundary all-exact path. Synthetic lookup
contention may support resource attribution but is neither a serving claim nor a gate.

## 11. Stop and fallback rules

- If W4 fails, repair collective order, routing, capacity, or termination before formal evaluation.
- If naive mixed remains slower but the complete D2 path wins, retain the logical-to-physical
  motivation.
- If complete D2 loses after all overhead is included, do not change D1 actions to rescue it;
  attribute the loss and demote or remove the mechanism.
- If the current real accessed embedding/cardinality scale is too small to expose the distributed
  problem, construct one larger semantically valid base plus one or two streaming updates from an
  accepted dataset. Rebuild D1 artifacts and the immutable plan. Do not manufacture the claim with
  unused cold rows.
- Do not introduce SSD tiering, serving hotness, or a D3 scheduler to hide an unresolved D2
  bottleneck.

## 12. Code and artifacts

Core implementation:

- `src/hstu_kvcache/migration/design2_plan.py`
- `src/hstu_kvcache/migration/design2_runtime.py`
- `src/hstu_kvcache/migration/design2_metrics.py`
- `src/hstu_kvcache/migration/design2_transaction.py`
- `src/hstu_kvcache/migration/recompute.py`
- `src/hstu_kvcache/models/hstu.py`

Development entry points:

- `scripts/launch_cohortkv_design2_stage_b.py`
- `scripts/freeze_cohortkv_design2_stage_b.py`
- `scripts/benchmark_cohortkv_design2_integrated_w3.py`
- `scripts/validate_cohortkv_design2_integrated_full_payload.py`
- `scripts/benchmark_cohortkv_design2_resource_isolation.py`

Frozen and development artifacts are indexed by
[DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md). Artifact protocol names and hashes
must not be mechanically renamed from CohortKV to EvoKV.

## 13. D3 handoff

The first handoff task is to implement the capacity-independent D2 constraint exporter described
in Section 4 and verify its record coverage, owner/operator bindings, physical membership,
collective templates, layout/transaction contract, and content hash against the current runtime.
After that closure, D3 may assume:

- the D1 action hash is immutable;
- owners, operators, compatible bins/pools, collective dependencies, segmented layout, coverage,
  and lineage are D2 constraints;
- a compatible D2 pool may be cut into smaller slices;
- private publication remains mandatory.

D3 may not assume:

- the resident D2 schedule fits HBM;
- W3 timings are paper evidence;
- a complete segmented consumer already exists;
- direct-old-K/V ordinary-DRAM performance has been measured;
- source preprocessing, pinned staging, or commit/reclaim is free.

Before that closure, non-scientific D3 work is limited to the exporter/schema, ordinary-host
source/target schema, source-byte ledger, and sequential-baseline preparation. Capacity scheduler
comparisons begin only after all variants can bind to the same exported constraint hash.
