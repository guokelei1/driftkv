# EvoKV Design 2: Wave-Compiled Segmented Execution

Last updated: 2026-08-03

Status: **current mechanism fixed; D3-facing constraint exporter, formal Stage-B freeze, paper
protocol, and results remain open**. Historical code and artifact names retain `cohortkv_*` so
existing hashes and result families remain stable.

This document defines D2 only. Current execution state is in
[DESIGN2_DEVELOPMENT_STATUS.md](DESIGN2_DEVELOPMENT_STATUS.md). The interface below remains the
target for a formal fixed-D2 isolation experiment. Historical D3 discovery began from a minimal
two-rank `WorkManifest`; the successor runner is rank-parameterized. A cross-layer candidate that changes D1/D2 is a new
`stack_revision`, not a D2 physical-lowering ablation. D3 is specified by
[DESIGN3_FUTURE_DIRECTION.md](DESIGN3_FUTURE_DIRECTION.md).

## 0. Paper-core successor boundary

The fixed H12/W3 mechanisms and numbers below remain immutable development evidence. The successor
paper workload changes four future-facing contracts without relabeling those artifacts:

- the integrated action domain is `compiled|exact`; progressive residual repair remains a D1-only
  supporting extension;
- natural-length `X-QK-HET` extents are primary from D1 through D3; `X-QK-HOM` reuses the same
  records and valid histories in a masked 512-slot physical layout, rather than selecting a
  different max-context population or a better result after observation;
- XP fixes 2,859,835 base-period semantic rows plus one padding row in a
  2,859,836×4,096 physical FP32 embedding table (43.638 GiB), an
  owner-side E4096→H1536 projection and the 24L/H1536 core; table + dense/projection is about
  44.725 GiB before program/runtime state. Its frozen row-sharded sparse checkpoint builder must
  activate and hash the semantic-request union across both formal edges, all headline manifests,
  all-exact, and every frozen fixed-action exact/append/fallback path, and make optimizer-updated
  embedding bytes plus dense/projection bytes exceed the hardware cap; only then is full
  single-card placement capacity-inadmissible;
- D2 exports group-valid fragments, dependencies, coverage and lineage. D3 chooses capacity groups
  and owns `writeback → validation → group commit → old-group reclaim`; a complete private target
  and global atomic epoch switch are no longer the formal endpoint.

The successor runner is parameterized for 1/2/4 ranks. Benchmark Qualification is recorded
separately and blocks only paper-result promotion, not current benchmark design or baseline
implementation.

Mechanism development now consumes the selected QK theta0--theta4 chain first and the selected QB
theta0--theta3 chain as a secondary stressor. Their manifests are frozen by
`selected_checkpoint_registry_development_v0.json`; D2 never selects or retrains a checkpoint.

## 1. Problem

D1 has already chosen each record's semantic action. In the frozen H12 wave, only 134 of 682
records take an exact route, but that logical sparsity is not automatically a multi-GPU speedup.
Exact and append still access row-sharded embeddings; variable-length incremental work creates
padding; naive phase separation launches many collectives; rewriting a complete target cache can
copy an already repaired retained prefix again; and every replacement group needs explicit
coverage, lineage and versioned commit before its old extent can be reclaimed.

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
  owner, operator, compatible physical pools, collectives, segmented layout, group-valid contract
        ↓
D3 ResidencyPlan
  capacity cuts, prefetch/execute/writeback, group commit/reclaim, pinned credits, NUMA placement
```

The responsibility boundary is:

| Decision | Owner |
|---|---|
| primary `compiled|exact` and reason | D1 |
| progressive residual repair as a supporting-only action | D1-only protocol |
| retained/suffix/final semantic extents | D1 |
| old-K/V owner and embedding shard routing | D2 |
| operator and physical compatibility bin/pool | D2 |
| collective dependencies and participation | D2 |
| segmented target layout and lineage/coverage contract | D2 |
| capacity-specific slices and micro-wave packing | D3 |
| DRAM/HBM launch order, group commit/reclaim and pinned-buffer credits | D3 |

The current D2 runtime exposes a resident execution order. That order is not a D3-facing
invariant. The target interface is a separately serialized, capacity-independent constraint
view—owner/operator/compatibility/dependency/layout—which D3 may cut into capacity-safe slices and
legally reorder. That normalized view has not yet been materialized and hashed; producing it is the
formal-isolation foundation item that must close before protocol freeze. It does not precede
HET/HOM construction, fixed-XP qualification, or rank-runner/baseline foundation, and it does not block
ongoing mechanism exploration.

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
validation, group commit, abort, and reclaim contract
```

The repository does not yet contain this normalized artifact. The existing
`cohortkv_d2_wave_plan_v1` is a Stage-A single-rank adapter with record bindings, while the W3
integrated path builds capacity-specific `D2IntegratedExtent` schedules at runtime. The latter
sorts owner-local compiled records by `(suffix, retained, final, record_id)` and then cuts them by
fixed `extent_size`; it does not serialize explicit global-bin membership or a standalone
collective-dependency graph.

Before a formal fixed-D2 isolation comparison, a small D2 exporter should derive, validate,
serialize, and hash the required constraint view from:

- the immutable action plan and owner map;
- operator/program and source bindings;
- the implemented shape-aware ordering and merged-exact membership;
- collective templates;
- segmented layout and group-lifecycle requirements.

The exported view must exclude `extent_size`, HBM capacity cuts, resident launch order, pinned
credits, and prefetch/writeback decisions. It does not promise that one compatible pool fits in
HBM or executes as one launch. Within the isolation track, a D3 `ResidencyPlan` may slice inside a
pool but cannot:

- move a record to a different semantic action;
- change the logical old-K/V owner;
- change the embedding shard rule;
- substitute another operator;
- combine incompatible shapes;
- reorder across a collective dependency;
- omit empty-rank collective participation;
- change target layout, coverage, or lineage.

Before exporter parity and content hashing close, a minimal two-card benchmark may still compare
variants generated from one recorded `WorkManifest`; these remain development-only. It cannot
claim the formal normalized D2 constraint boundary has closed.

## 5. Mechanism

### 5.1 Owner-local retained repair

The compiled program is small relative to per-record K/V, so D2 moves the program to the state
owner rather than moving old K/V to a central worker. Retained repair:

- reads local exact source-version old K/V;
- applies the D1 direct affine program;
- writes a group-local retained replacement segment;
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

### 5.6 Group-valid publication contract

D2 declares the fragments and checks required before one capacity group can replace its source:

- complete and duplicate-free group coverage;
- expected source/target lineage and explicit group version;
- shape, dtype, valid-length, and component checks;
- collective completion on every participating rank;
- target-layout and fragment-set consistency.

D3 chooses the capacity cut and physical shadow/replacement extent. After the checks pass it
atomically switches the group pointer to the target and reclaims the old group; abort preserves the
old pointer and releases only bounded in-flight state. A wave finishes after every selected group
has committed. This permits an explicitly versioned old/new mixture during maintenance without
requiring a complete private epoch.

## 6. Why the mechanisms form one design

The mechanisms are not an unrelated optimization list:

1. D1 creates nonuniform semantic work.
2. owner-local retained repair prevents the large state from becoming communication;
3. row-sharded exact/append exposes the unavoidable lookup traffic;
4. `(S,R)`-aware ordering exposes the true incremental shape;
5. segmented output removes retained-prefix rewrite;
6. merged exact pools remove semantic-only fragmentation;
7. versioned group commit/reclaim turns each physical group into valid target state without a
   2× whole-epoch footprint.

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
- plan, prepare, execute, validate, group commit, and reclaim time.

## 8. Baselines

Formal D2 evaluation requires the same SPMD harness for:

1. **one-shot all exact:** fastest independently tuned complete current-model recomputation;
2. **two-stage all exact:** only when it reflects a meaningful execution decomposition;
3. **owner-local staged-contiguous fixed-action mixed:** the current executable strong control;
4. **owner-local fused-contiguous fixed-action mixed:** isolates fused finalization before layout
   changes;
5. **segmented mixed without shape-aware ordering and with separate exact pools;**
6. **shape-aware segmented mixed without merged exact pool;**
7. **complete D2 with merged physical exact pools.**

The XP placement screen additionally includes capacity-admitted hot-row replication+cold-row
sharding, wave-scope dedup/coalescing, and BF16/FP16 transport sensitivity. Full replication is
executed only when it fits the same usable-HBM budget; otherwise its result is
`capacity-not-admitted`. All methods may apply the same E4096→H1536 projection at the embedding
owner before returning H1536 vectors; no baseline is forced to communicate raw E4096 vectors.
Exact jointly tunes a predeclared bounded grid of placement/transport, routing/coalescing, and
pipeline combinations on independent profile users; it cannot prune to the top placements before
observing routing/pipeline interactions. Timed paths must remove diagnostic SHA/CPU round trips,
and exact may use the strongest legal fused/compiled implementation.

The headline fixed-action denominator is independently selected between the two legal contiguous
controls. The current W3 `naive` implementation is already owner-local, so formal evaluation does
not relabel it as placement-oblivious or claim a standalone owner-compute speedup. A future
placement-oblivious supporting control is legal only if it materializes and charges all old-K/V
peer traffic; it is not the headline denominator.

The current pre-SPMD record-DP Table-8 numbers cannot be used as a baseline. Every comparator must
be rerun in the same process model, source boundary, group lifecycle, and timer.

## 9. Timer and correctness boundary

The primary D2 timer begins after immutable source data and the frozen action plan are available,
and includes:

- distributed plan/materialization required for the run;
- item-ID routing and embedding collectives;
- compiled, exact, and append compute;
- transient padding/materialization;
- group target writeback;
- group validation, versioned commit and old-group reclaim;
- final job manifest.

Report a secondary retained-prefix-only boundary to preserve continuity with D1, but do not compare
that number directly to D2's post-append rolling-job time.

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

1. XP/HET/HOM identities and primary `compiled|exact` ActionPlan are frozen;
2. a new D2 protocol is frozen in `docs/eval_protocol.md`;
3. the runner supports 1/2/4 ranks, with each formal cell honestly capacity-admitted;
4. rolling post-append group validation/commit/reclaim completes;
5. strong all-exact, placement and owner-local contiguous fixed-action baselines are independently
   tuned;
6. physical communication and per-rank capacity ledgers close;
7. segmented consumer or next-wave support closes;
8. complete full-payload, group-lifecycle and failure checks close.

The historical independent W4/Stage-B gate remains required only to promote that old result
family. It does not block successor benchmark design or baseline implementation. The separate
Benchmark Qualification registry must close before any successor timing is promoted to paper
evidence.

A positive D2 result must show that the physical-sparse path beats the strongest legal
owner-local contiguous fixed-action baseline and has a meaningful region relative to the fastest
same-boundary all-exact path. Synthetic lookup contention may support resource attribution but is
neither a serving claim nor a gate.

## 11. Stop and fallback rules

- If a 1/2/4-rank qualification point fails, repair collective order, routing, capacity, or
  termination before promoting that point.
- If the strong contiguous mixed baseline remains slower but the complete D2 path wins, retain the
  logical-to-physical motivation.
- If complete D2 loses after all overhead is included, do not change D1 actions to rescue it;
  attribute the loss and demote or remove the mechanism.
- XP proactively constructs the larger semantically valid embedding foundation; it is not a
  fallback triggered only after X2 fails. Rebuild its D1 artifacts and immutable plan, and do not
  manufacture the claim with unused cold rows.
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

The historical immediate handoff was a minimal H12/W2 `WorkManifest` and was sufficient for the
GPU0/GPU1 M0/M1 development paths. The successor handoff uses HET valid extents and primary
`compiled|exact` actions, contains record/action/owner/source/target bytes and operator/pool keys,
and is consumable by a 1/2/4-rank-capable runner.

For a later formal isolation track, implement the capacity-independent exporter described in
Section 4 and verify its record coverage, owner/operator bindings, physical membership,
collective templates, layout/group-lifecycle contract, and content hash. That track may assume:

- the D1 action hash is immutable;
- owners, operators, compatible bins/pools, collective dependencies, segmented layout, coverage,
  lineage and group-validation requirements are D2 constraints;
- a compatible D2 pool may be cut into smaller slices;
- each slice must commit a versioned replacement before its old extent is reclaimed.

D3 may not assume:

- the resident D2 schedule fits HBM;
- W3 timings are paper evidence;
- a complete segmented consumer already exists;
- direct-old-K/V ordinary-DRAM performance has been measured;
- source preprocessing, pinned staging, or commit/reclaim is free.

Cross-layer discovery may also change actions, owners, pools, or layout before execution. It must
record a new `stack_revision` and rerun baselines, so it is not presented as a fixed-D2 ablation.
A normalized exporter, group-lifecycle closure and formal GPU matrix remain paper-result tasks,
not prerequisites for continuing two-card mechanism exploration. They cannot be skipped when the
successor protocol is frozen.
