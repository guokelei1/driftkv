# Open experiment gaps before submission

The current manuscript is deliberately complete in structure but incomplete in admission
evidence. These gaps must remain visible until real results exist.

## Gate 1: freeze and replicate the verified compiler

Current status:

- the simple compiled-projection family has 27 validation runs;
- the attention-weighted full-affine compiler, its semantic contract, and its action selection are
  adaptive seed-0;
- Stage 2 has frozen the seed-0 executable artifact path: all three serialized deployed
  certificates pass, runtime programs and fallback plans are hash-checked, and recovery targets
  50%–80% select compiled while 90% selects exact;
- this closes the implementation path but does not turn seed 0 into confirmatory evidence.

Required experiment:

- freeze fit sample count, attention weighting, rank, ridge, action library, certificate metrics,
  thresholds, bootstrap procedure, and fallback order;
- run on new training seeds or accepted external checkpoints;
- retain disjoint fit, selection, certificate, and final-test users;
- report compiler and certificate wall time, exact-probe work, cohort size, and amortized
  per-record cost;
- report certificate pass rate, selected family, final semantic recovery, task endpoint tracking,
  and failures, not only successful cohorts.

Admission criterion:

- the compiler claim may become confirmatory only after the program is frozen before those model
  replications are inspected.

## Gate 2: repair the complete-cohort source-state Pareto failure — closed for hot HBM

Current status:

- Stage 4 completes the frozen 682-record, 1,087,785-token normal-path matrix for compiled,
  certificate-failed selective, exact, residual-p, and no-transform at HBM/DRAM over 1/2/4 GPUs;
- all 30 independently tuned points pass complete coverage, 17.822B-element transport
  correctness, five capacity preflights, and atomic manifest publication;
- compiled is 2.70–3.49× faster than selective at all six matched endpoints but loses to exact at
  all six;
- the physical FP16 capsule is 17.82 GB versus 89.1 MB of physical raw history, and source
  read/decode/pinning consumes 91.35%–96.91% of compiled completion;
- Stage 4.5 preserves that negative result and freezes a separate direct-old-K/V hot-HBM source
  plan. It composes the certified affine through the source K/V projection, adds zero per-record
  state, and reclaims old extents after replacement staging;
- all three source pairs certify, complete real fused transport covers 17.822B valid elements
  with zero tolerance mismatches, and full-cohort 1/2/4-GPU medians are
  0.930/0.494/0.255 seconds versus paired raw-history-HBM exact at
  18.695/9.729/4.766 seconds;
- every compiled repetition is below every paired exact repetition, all capacity preflights pass,
  and final old-K/V occupancy is zero;
- the result is admitted only for complete existing old K/V already resident in HBM. Cold
  filesystem, SSD, automatic tier selection, and missing/unverified state fall through to exact;
- the frozen aggregate is
  `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json`.

Gate 3 now closes recursive migration of approximate outputs. Remaining work moves to Gate 4:
automatic exact dispatch, semantic degradation detection, transactional rework, and failure
visibility are not implemented.

## Gate 3: repeated-update migrate-or-exact lifecycle

Status: **closed and frozen**.

- The fixed KuaiRand seed-0, 16L/H512, one-A40 chain executes all 11 adjacent updates from exact
  theta0 K/V and consumes each previous actual output.
- The first per-cache norm-sketch threshold beats matched random but is rejected because its
  diagnostic complete chain refreshes 0.15%–65.1% per step.
- The frozen replacement maps label-free fit edge severity to 15%–25% exact budgets, refreshes
  depth-four and then older caches first, and uses stable hash tie-breaking.
- The independent certificate costs 0.2142× cumulatively, stays below 0.2814× per step, and has
  minimum cache/score/top-100 values 0.9613/0.999759/0.9898.
- The complete 682-record chain costs 0.2134×, stays below 0.2543× per step, refreshes
  14.956%–25.073% after rounding, and has minimum 0.9632/0.999950/0.9918.
- All 7,502 complete-chain lineage rows rebuild from the frozen policy; exact resets state and
  migration depth never exceeds four.
- Recommendation labels do not tune or route the policy.
- Frozen evidence is in
  `configs/cohortkv_single_config_v1/stage4_6_lifecycle_summary.json`.

This is one controlled development chain and a bounded heuristic, not global selector optimality,
organic traffic, or cross-seed/cross-dataset lifecycle validation.

## Gate 4: failure and publication tests at realistic scale

Required experiment:

- inject failures before first extent, mid-wave, during publication, and immediately before
  manifest commit;
- republish a structurally valid theta4 degradation with valid integrity metadata so semantic
  preflight/runtime guard, rather than the hash check, must escalate it to exact; report detection
  phase and replaced extents;
- verify no incomplete target version becomes visible;
- verify retry behavior and cleanup cost;
- report manifest and extent metadata overhead for the full cohort.

Boundary:

- this does not establish distributed transactions or coordinator crash recovery unless those
  mechanisms are implemented.

## Gate 5: physical storage backend

Current status:

- POSIX path is a functional filesystem interface;
- remote path uses an in-memory object-store reference.

Required experiment:

- identify the physical local SSD and filesystem;
- report serialized bytes, write amplification, queue depth, fsync policy, bandwidth, CPU cost,
  and completion time;
- compare compiled and exact within that same endpoint;
- include a no-transform K/V placement/movement baseline and, if reproducible artifacts permit,
  an MTServe-style Page–Chunk organization;
- add a real network/object backend only when hardware and failure assumptions are reproducible.

Admission criterion:

- until then, use “POSIX interface” and “remote-object protocol,” never SSD/network performance.

## Gate 6: capsule capacity trade-off

Current status:

- unpadded FP16 `Norm(x)` capsule is 50% of logical FP16 K/V size;
- v2 stores padded capsules totaling 1.507 GiB versus 2.998 GiB logical old K/V for its trace.
- Stage 4 materializes 16.60 GiB logical/17.82 GB physical FP16 capsule state versus 33.20 GiB
  logical old or target K/V for the complete cohort;
- that capsule source path loses to exact at all six endpoints, so a favorable time break-even is
  not established for the Stage-4 path;
- the Stage-4.5 winner retains no capsule and adds zero per-record state. Its three direct programs
  total 100.78 MB per worker;
- the pinned-DRAM normalized-capsule backup retains about 17.86 GB host state, preloads in
  39.5/24.7 seconds on 1/4 GPUs, and has three/six-update time break-even.

Required experiment:

- verify that the frozen direct-old-K/V winner needs no independent capture/encode and report its
  once-per-version-pair compile/publication cost against FP16 capsule and exact raw history;
- if compression remains on the frontier, compare FP16 with symmetric signed INT8 and any smaller
  selected candidate, including scale/offset metadata and timed staging dequantization;
- separately time fresh-K/V only, plus device capture, and plus D2H/encode/POSIX persistence on
  the 60 program-selection histories;
- report total extra capacity, creation cost, read bandwidth, compression error, and the measured
  time crossover `ceil(capture/(exact-compiled-compiler_amortized))`; a nonpositive denominator is
  no break-even, not a missing value;
- evaluate whether capsules replace another retained state or are purely additional.
- keep selective transition and residual hidden-suffix bytes outside the default capsule ratio.

Admission criterion:

- the paper can state that the primary hot-HBM route adds zero per-record source state; it cannot
  generalize this to cold storage or claim that normalized-capsule storage is universally
  favorable;
- INT8 and capture measurements are comparison economics, not an admission gate for the already
  frozen direct-old-K/V route.

## Gate 7: external-validity expansion

Required experiment:

- confirm the compiler and endpoint on at least one non-KuaiRand long-context checkpoint;
- preserve the Tenrec limitation: QB and QK are related tables and use ordinal rather than global
  calendar time;
- avoid Taobao UserBehavior for the primary task because it fails the true-unclicked-impression
  semantics gate;
- document ZhihuRec as a maintenance boundary rather than silently dropping it.

## Gate 8: closest cross-model baseline

Current status:

- the HSTU-compatible old-K/V-reuse reference is implemented and tested independently of the
  incompatible current-projection helper;
- the complete seed-0 resident grid contains 53 intervals per source pair and 177 total points on
  the frozen 60-user selection role, followed by the disjoint 60-user certificate role;
- compiled repair strictly dominates all 53 selective points for theta0/theta4/theta10;
- no selective interval passes the 70% cache/score/top-100 contract. Exact is therefore the
  publishable fallback, while `m12/layers0-11` is frozen only as a certificate-failed diagnostic
  external baseline;
- the common Stage-4 full-cohort destination pipeline is complete and independently runtime-tuned
  at all six HBM/DRAM × 1/2/4-GPU points;
- the diagnostic remains certificate-failed but compiled is 2.70–3.49× faster end to end.

Remaining experiment:

- replicate the frontier only after the primary source-state design is frozen, on predeclared new
  seeds/configurations rather than another adaptive search on this seed;
- retain `certificate_passed=false` in every current Stage-4 diagnostic artifact.

Boundary:

- HCache remains a same-model restoration system and is not a semantic substitute for this
  baseline;
- the single-configuration algorithmic frontier and end-to-end system comparison are complete;
- the seed-0 dominance result cannot be generalized across datasets, model sizes, or training
  seeds until replication.

## Results-to-paper update protocol

When a gate completes:

1. update `03_claim_evidence_matrix.md`;
2. update the exact Results subsection and relevant table/figure;
3. revise Discussion limitations;
4. only then revise Introduction contributions, Abstract, and title;
5. add a new review-log entry with the result family and protocol string.

No projected number should enter the manuscript before a result artifact and protocol record
exist.
