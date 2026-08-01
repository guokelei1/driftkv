# EvoKV Design 2 development status

Date: 2026-07-31

This is the only live D2 implementation/evidence ledger. The mechanism is defined by
[DESIGN2_FINAL_PLAN.md](DESIGN2_FINAL_PLAN.md). Earlier four-stage control and Stage-A/Stage-B
handoff documents were consolidated here and removed.

The PASS rows below are immutable H12/W1/W2/W3 development facts. The paper-scale successor does
not relabel them: it uses HET primary/HOM control extents, a capacity-forced XP embedding,
`compiled|exact` integrated actions, a 1/2/4-rank-capable runner, and rolling group
commit/reclaim. Progressive remains D1-only supporting evidence. The old W4/Stage-C gate blocks
promotion of the old family, not successor benchmark design or baseline implementation.

## 1. Status at a glance

| Item | State | Meaning |
|---|---|---|
| Stage A semantic/mechanical boundary | **FROZEN** | immutable H12 plan, exact frontend, lookup ledger, transaction adapter, topology/capacity characterization |
| W1 single-rank SPMD | **PASS** | process-boundary parity and owner-local compiled invariant |
| W2 two-rank row-sharded path | **PASS** | exact/append parity, communication reconstruction, normal and hard-failure behavior |
| W3 three-rank development diagnostic | **PASS, non-scientific** | asymmetric NCCL composition and hard-failure propagation |
| C0 integrated state machine | **PASS, non-scientific** | fixed 16-record normal/abort wiring only |
| W3 full-cohort mechanism discovery | **PASS, non-scientific** | segmented, shape-aware, merged-exact candidate and full-payload validation |
| D2→D3 constraint exporter | **NOT IMPLEMENTED, DEFERRED FOR D3 M0** | no standalone formal artifact; minimal two-rank WorkManifest is sufficient for initial development |
| W4 four-independent-A40 gate | **BLOCKED/PENDING** | physical normal and hard-failure artifacts do not exist |
| formal D2 protocol | **NOT FROZEN** | no D2 paper-performance family exists |
| formal 1/2/4-GPU evaluation | **NOT STARTED** | W3 timings cannot substitute |

The current double gate is:

```text
stage_c_development_entry = go
stage_c_evaluation_entry  = blocked
```

Development may continue when it addresses a concrete interface risk. No current D2 timing may be
promoted to a paper claim.

## 2. Immutable upstream input

The D2 input is:

- `configs/cohortkv_d2/action_plan_theta1_theta2_staggered_renewal_h12.json`
- 548 compiled, 46 scheduled exact, 88 natural exact, 682 total;
- ActionPlan content SHA-256:
  `c4bc383d28f3558fdd11be8788799aaa6f66e80f778a4670f781eb9295f0027e`;
- ActionPlan file SHA-256:
  `3572a858111b1e9d08e4102512af46ef6a6d2b1fbe7ee7b2828162d28d58518d`;
- theta2 checkpoint SHA-256:
  `37eb8189d36127b735e6b6482f82dc31927d625168e0d67eca9890d5a01f3c18`;
- theta1→theta2 direct program SHA-256:
  `c0a2ff2de64200f482c6eb5097cbad4f7db8d200de385be4238bdcfbf9cf5e7d`.

D2 must not rerun the scheduler or alter semantic actions from its own measurements.

## 3. Frozen Stage A boundary

The checked aggregate is:

- `configs/cohortkv_d2/stage_a_summary.json`.

It freezes:

- deterministic `ActionPlan` reconstruction and record coverage;
- bitwise FP32 HSTU embedded-frontend/exact/append equivalence;
- Stage-5 normal, semantic-fallback, mid-job-abort, and pre-commit-abort adapter behavior;
- phase lookup tokens:
  `0 / 50,099 / 82,612 / 213,669 / 682`;
- retained-prefix ledger:
  `637,954 → 50,099`;
- full mixed lookup ledger:
  `934,917 → 347,062`;
- two NVLink islands `{0,1}` and `{2,3}` on the four local A40s;
- static strict-COW capacity and request/dedup ceilings.

All Stage A characterization is `scientific_result=false`. In particular:

- only compiled retained repair is embedding-free;
- single-rank strict COW is infeasible for the frozen H12 cache state;
- FP32 is the only mechanically equivalent embedding-vector transport;
- topology microbenchmarks are not full-wave results;
- the old record-DP Table-8 numbers cannot serve as a new SPMD baseline.

## 4. Implemented distributed primitives

### W1

World-size 1 reproduces:

- fixed action counts and phase ledger;
- owner-local compiled retained repair;
- exact and append reference outputs;
- private ready/abort behavior;
- forbidden-full-embedding checks.

### W2

Two physical ranks cover:

- natural exact and scheduled exact;
- delta/latest append;
- repeated IDs, valid item ID 0, padding-only and empty-owner cases;
- bitwise FP32 output parity;
- requested/local/remote ID counts and hashes;
- per-peer request/return tensor-payload reconstruction;
- deterministic collective order;
- projected full-cohort per-rank capacity;
- bounded NCCL rank-exit propagation.

The corrected hard-failure artifact is:

- `configs/cohortkv_d2/stage_b_w2_hard_failure.json`
- file SHA-256:
  `49f2cf52550df9f12f6fbb7c5a37945effc91f1ae6d086c128954919f34cc015`.

It binds physical GPU UUIDs and explicit NCCL, observes rank-1 exit code 23, exits nonzero before
launcher timeout, and leaves no live process group.

### W3 development diagnostic

Physical GPU0/GPU1/GPU3 NCCL diagnostics cover asymmetric three-rank composition:

| Artifact | File SHA-256 |
|---|---|
| `configs/cohortkv_d2/dev_w3_sample_inputs.json` | `8e6d44b72efe83ced91309b453899ed6367fd5423060d089eb7eaf04c560bb2d` |
| `configs/cohortkv_d2/dev_w3_primitives.json` | `125b1c0f4b21c1efb1a29a3c2623941125e00a1882b0d90c3a9f28a53a591c72` |
| `configs/cohortkv_d2/dev_w3_hard_failure.json` | `16ecd88fd3b59904dbac5efdaf5b56e4b63bad5699092c0eb6656e0a49115149` |

Normal execution covers nonuniform shards, empty-rank participation, route reversal, ready/abort,
and reconstructed IDs/bytes. Hard failure terminates without a residual process group. Every
artifact states:

```text
scientific_result = false
formal_stage_b_gate = false
substitute_for_w4 = false
stage_c_evaluation_authorized = false
```

W3 cannot prove four independent CUDA contexts, the W4 owner map, four-rank capacity, or formal W4
termination.

## 5. C0 integrated development closure

C0 uses the fixed 16-record `stage_b_sample_inputs.json` fixture. The aggregate is:

- `configs/cohortkv_d2/development/c0_status.json`
- file SHA-256:
  `b193ddacb12be623e3e03e1a9f1cbdbde34cab4f7a3be9ec62d9af0ef27f9507`.

It covers W1/W2/W3 normal plus W3 pre-commit abort. Route, coverage, owner assignment, final token,
compiled-retained embedding bypass, development pointer, source fixture, and reference release all
pass.

It does not cover:

- 682-record capacity or performance;
- formal target-epoch publication;
- real allocator reclaim;
- independent one-shot target-exact numerical reference;
- all final timer components.

The normal path publishes only a development-namespace pointer, and artifacts explicitly record
`target_epoch_published=false`.

## 6. Full-cohort mechanism discovery

All results in this section run on three A40s and are
`scientific_result=false`, `formal_stage_c=false`.

### 6.1 Negative starting point

On a 192-record pilot:

- naive owner-mixed: 3.025 s;
- one-shot all exact: 2.079 s;
- two-stage all exact: 2.275 s.

Thus D1 logical sparsity does not automatically create a systems win. The main exposed costs are
suffix padding, repeated retained-prefix destination work, and collective/phase fragmentation.

### 6.2 Candidate lowering

Keeping action, owner, lookup multiset, and target semantics fixed:

| Version | Physical lowering | 192-record mixed |
|---|---|---:|
| v1 | naive staged owner-mixed | 3.025 s |
| v2 | fused finalization | 2.700 s |
| v3 | segmented suffix-only output | 2.357 s |
| v4 | `(S,R)` shape ordering | 1.542 s |
| v5 | merged physical exact pool | 1.493 s |

The full-682 B8 development point is:

- merged segmented mixed: 3.633 s;
- unmerged segmented mixed: 4.055 s;
- one-shot all exact: 6.716 s;
- mixed lookup tokens: 347,062 versus 934,917;
- one-way off-diagonal vector volume: 454.62 MiB versus 1,222.86 MiB.

Primary development artifact:

- `results/system/cohortkv_design2_integrated_w3_development_v5/full682_shape_append_merged_exact_b8.json`
- file SHA-256:
  `228929768479aa4d5e65e7849428766f4ba8f9b629de7183e3b024828c3e1029`.

These timings exclude complete source/history preparation, final publication/commit/reclaim, and a
segmented consumer. They select a mechanism for formal evaluation; they do not establish the paper
speedup.

### 6.3 Full-payload development correctness

- `results/system/cohortkv_design2_integrated_full_payload_development_v1/full682.json`
- file SHA-256:
  `d17e2d16cf521a249d9d50482fd7ce821ef7fce6053c28767478b58257aa8508`.

The validation covers 682/682 records, all exact reasons, the zero-delta branch, and approximately
15.32 billion valid K/V/last-hidden elements. All coverage and tolerance checks pass. The
segmented and contiguous paths are not claimed bitwise equal because incremental attention changes
floating-point reduction order.

### 6.4 Supporting contention characterization

The synthetic lookup stressor shows that mixed maintenance creates less embedding-tier contention
than all exact under one offered rate, but still causes nonzero tail degradation. It is a
deterministic resource characterization, not a serving workload, SLO result, or D2 gate.

## 7. Known risks

- W4 has not covered all-zero splits, four-rank collective composition, cross-island routes,
  per-rank capacity, or hard-failure termination on four independent GPUs.
- Full-cohort strict-COW capacity is projected, not measured through an actual final D2 target
  transaction.
- Current source old K/V in primitive tests is a correctness fixture, not a measured source
  manifest.
- Collective accounting records tensor payload, not NCCL wire bytes.
- Segmented output does not yet have a serving/next-wave consumer; hidden concatenation could erase
  the gain.
- Plan/history preparation and publication/commit/reclaim are outside W3 performance timing.
- The current single-rank `D2WavePlan` and capacity-specific W3 extents are not a normalized,
  independently hashed D3-facing constraint artifact.
- FP16/BF16 vector transport, dedup, overlap, topology-aware placement, and work stealing are not
  selected mechanisms.
- The current dense model fits one A40; do not claim model-capacity tensor parallelism.

## 8. Pending W4 closure

When four independent A40s are safely available:

```bash
python scripts/launch_cohortkv_design2_stage_b.py \
  --world-sizes 4 \
  --cases normal hard_failure \
  --visible-devices 0 1 2 3

python scripts/freeze_cohortkv_design2_stage_b.py
python scripts/freeze_cohortkv_design2_stage_b.py --check
```

Then rerun the Stage-A freeze check, tests, lint, and diff check. Stage B becomes frozen only when
both W4 artifacts exist and the generated summary passes `--check`.

Forbidden substitutes:

- four ranks sharing fewer than four GPUs;
- Gloo instead of NCCL;
- renaming W2/W3 as W4;
- manually writing the Stage-B summary;
- killing or oversubscribing an external process to obtain a GPU.

If W4 fails, return to distributed routing/order/capacity/termination. If W1 parity or the immutable
ledger fails, return to the D1→D2 adapter before debugging W4.

## 9. Successor formal D2 work

1. Build/freeze the QK HET primary and HOM control manifests.
2. Freeze the hardware HBM cap, then qualify XP at 2,859,835 base-period semantic rows plus one
   padding row in a 2,859,836×4,096 physical FP32 table
   (43.638 GiB), owner-side E4096→H1536 projection, and a 24L/H1536 core without consulting
   EvoKV performance. Count only optimizer-updated active rows toward forced sharding; the
   semantic-request union across both formal edges, all-exact, and every frozen fixed-action
   exact/append/fallback path must be active and hashed, and active embedding plus dense/projection
   bytes must exceed the single-card allocatable budget.
3. Generalize one binary to 1/2/4 ranks and verify canonical 1/2/4-way shards.
4. Rerun strongest all-exact/placement controls, both owner-local staged/fused contiguous
   fixed-action controls, and complete D2; independently freeze the fastest legal denominators.
5. Include plan/history preparation, suffix append, group writeback, validation, versioned commit,
   old-group reclaim and segmented consumer/next-wave compatibility.
6. Complete Benchmark Qualification, then freeze a new protocol and run capacity/failure/physical
   communication matrices.

`stage_c_evaluation_entry=blocked` remains true for the old family. The successor receives its own
promotion state only after its protocol exists.

## 10. Paper-scale foundation

The current 312,145-row H12 setting remains sufficient evidence for mechanism discovery, but it is
not the paper-scale communication premise. XP proactively supplies the larger semantically valid
setting rather than waiting for H12/X2 to fail:

1. select real base-period entities/features from an accepted dataset;
2. train one base model and one or two short streaming updates;
3. regenerate exact old K/V, current endpoints, compiled programs, qualification evidence, and a
   primary `compiled|exact` ActionPlan;
4. build the checkpoint with a frozen 4-rank row-sharded sparse/row-wise optimizer path;
5. report total/active/requested/unique rows, update counts, active bytes, and remote bytes;
6. require the frozen all-comparator request union to be active and active fixed bytes alone to
   force sharding.

## 11. D3 handoff

The first D3 development handoff was a minimal H12/W2 two-rank `WorkManifest`; it remains the
historical M0/M1 identity. The successor WorkManifest carries HET valid extents and is consumable by
a 1/2/4-rank runner. For a later formal fixed-D2 isolation track, derive and hash a
capacity-independent constraint view from:

- the frozen XP-HET `compiled|exact` ActionPlan and its natural valid extents;
- deterministic stable-hash owner and XP embedding-shard rules;
- `(S,R,F,record_id)` shape/pool membership without capacity cuts or fixed-size resident extents;
- merged-exact physical membership;
- operator/program bindings, collective templates, segmented layout, and group-lifecycle
  semantics.

The immutable H12 plan remains a historical parity/shape canary only; it cannot be the source of
the successor formal exporter.

The export must not contain W3 `extent_size`, HBM cuts, resident launch order, or D3 staging
decisions. It must validate record coverage and parity with the current integrated runtime, then
serialize a stable content hash.

Before that formal artifact exists, non-scientific D3 work may continue on GPU0/GPU1, provided one
isolation comparison shares the same recorded WorkManifest. The successor code must nevertheless
be rank-parameterized rather than hard-coded to two cards. Cross-layer
candidates receive a new `stack_revision` and rerun baselines. No such development run may treat
W3 times as formal evidence, claim the normalized D2 boundary has closed, or assume the resident
schedule fits HBM.
