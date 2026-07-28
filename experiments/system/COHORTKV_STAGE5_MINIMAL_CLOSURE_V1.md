# CohortKV Stage 5 minimal implementation closure v1

## Status and amendment scope

This is the completed single-configuration implementation closure. It
supersedes only the unexecuted Stage-5 failure matrix and RQ5 capsule-economics requirements in
`COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md`. Completed Stage-0 through Stage-4.8 evidence is
unchanged. The formal Stage-4.9 11-edge confirmation is complete, and Stage 5 binds
`staggered_renewal_h12`.

Stage 5 is not a fourth research design. It joins the candidate lifecycle policy, operator, and
destination transaction with the minimum correctness and accounting evidence needed by the paper.
Runtime sentinel research, online rework, resume journals, capsule quantization, capture
microbenchmarks, and physical SSD performance are not v1 admission gates.

## A. Minimal executable closure

The implementation must:

1. resolve every record to `migrate` or `exact` before private target extent execution;
2. verify artifact hash, version, shape, capacity, old-K/V presence, and program identity;
3. run one frozen job-level label-free semantic canary before any target extent is produced;
4. route an affected migration cohort directly to exact on program identity, shape, old-K/V
   presence, or semantic-canary failure;
5. reject the whole job before transaction creation on artifact/version or capacity failure;
6. execute migrate and exact records in one target transaction;
7. guard only at `post_retained_prefix_pre_append`;
8. keep retained-prefix state private and commit only `post_append_full_cache`;
9. publish final action, source/target lineage, last-exact version, migration depth, and fallback
   reason for every record.

The semantic canary uses only the frozen program-selection role and one preregistered threshold.
There is no runtime-sentinel family search and no claim of detecting drift after extent execution
starts. Exact is a safe fallback only for a usable, correctly versioned target artifact and a
capacity-admitted transaction; it is not used to hide wrong-artifact or insufficient-capacity
failures.

## B. Failure-safe publication

Failure evidence uses copy-on-write: old extents remain readable until the complete target manifest
commits, after which they become eligible for reclamation. A production cross-job
version-retirement API is deferred. Capacity preflight chooses one feasible representative
failure-safe GPU configuration. The one-GPU normal-path extent-reclamation result remains valid
performance evidence but is not relabeled as abort-safe.

In addition to one normal full-population `theta0 -> theta1` job, exactly three fallback/fault
cases are required:

| Case | Required outcome |
|---|---|
| Integrity-accepted, shape-preserving perturbation of the actual `theta0 -> theta1` direct-old-K/V program | Preflight selects exact for the affected cohort; one complete corrected target commits |
| Mid-job execution exception | Target aborts, staging is reclaimed, and every expected record in the old manifest remains readback-valid |
| Exception immediately before commit | Complete private target remains invisible and every expected record in the old manifest remains readback-valid |

Artifact/version fatal-admission injections and standalone program-shape fallback injections
remain unit and smoke checks. Runtime invalidation of already generated extents, multi-point
failure matrices, rollback journals, and resume are deferred.

## C. Source-state accounting audit

No new source representation is developed. One artifact-derived table reports:

- the primary direct-old-K/V route: zero additional per-record source-state bytes, existing old-K/V
  bytes, direct-program bytes, and measured old/new peak overlap;
- once-per-version-pair direct-program compile/serialization values already recorded by Stage 2
  and Stage 4.5;
- Stage-2 fit/runtime-prepare/certificate cost and the existing resident amortization floor,
  keeping the end-to-end speedup scoped to a prepublished-program data-plane job;
- the rejected normalized FP16 capsule: logical and physical bytes, preload/source time, and its
  completed endpoint outcome;
- the POSIX backend as interface-only, with no SSD performance claim.

The active route has no independent capsule capture or encode step. INT8/FP8 capsules,
capture/D2H/persistence matrices, time-break-even calculations, named SSD benchmarking, GDS,
remote storage, and automatic tier selection are optional post-v1 extensions.

## D. Execution order and completion

Stage-5 interface, schema, unit/static checks, and real-edge smoke completed against the
Stage-4.9 ABI. Formal execution then:

1. ran the Stage-4.9 11-edge same-device confirmation;
2. bind the confirmed scheduler candidate;
3. run one normal integrated job and the three cases above;
4. retain the already generated source-state accounting table from frozen artifacts.

Stage 5 completes when the fixed preflight routes correctly, no tested failure exposes a partial
target, every expected old record passes version, shape, finite-value, and checksum/equivalence
readback after abort, lineage is complete, and the accounting table contains no unmeasured claim.
Its timing is implementation overhead, not a new algorithmic contribution.

## E. Formal result

`results/system/cohortkv_single_config_full_chain_v1/stage5_full_cow_theta0_theta1_seed0.json`
is complete and binds the SHA-checked H12 Stage-4.9 result. Its copy-on-write capacity preflight
passes on two A40s.

| Case | Result |
|---|---|
| Normal 682-record `theta0 -> theta1` | Complete target commits; all 682 records pass manifest, metadata, finite-value, and checksum readback |
| Shape-preserving real-program perturbation | Frozen canary routes the migration cohort to exact before target execution; complete corrected target commits |
| Mid-job fault | Transaction aborts, private staging is reclaimed, no partial target is visible, and all 682 old records pass readback |
| Pre-commit fault | Transaction aborts, the complete private target remains invisible, and all 682 old records pass readback |

Formal Stage-4.9 binding, candidate/action binding, source initialization, capacity, normal,
fallback, both abort cases, JSON Schema, and the Stage-5 cross-field semantic validator all pass.
The source-state accounting table remains derived exclusively from existing Stage-2/4/4.5
artifacts. No runtime-sentinel, rework/resume, INT8/capture, SSD, throughput, or durability claim
is added.
