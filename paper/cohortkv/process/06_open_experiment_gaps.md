# Open experiment gaps before submission

The current manuscript is deliberately complete in structure but incomplete in admission
evidence. These gaps must remain visible until real results exist.

## Gate 1: freeze and replicate the verified compiler

Current status:

- the simple compiled-projection family has 27 validation runs;
- the attention-weighted full-affine compiler, its semantic contract, and its action selection are
  adaptive seed-0.

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

## Gate 2: complete-cohort identical-boundary destination benchmark

Current status:

- v2 measures 64 real histories with an assigned 13/19/32 version mix at matched host residency
  and target publication;
- v4 has a destination contract but no complete-cohort performance result;
- verified plans record residual/structural/exact fallbacks, but the destination engine currently
  accepts compiled affine programs and does not automatically execute that escalation chain.

Required experiment:

- use every eligible fixed KuaiRand 4+12 update record;
- add a source manifest/reader that lazily scans capsule shards;
- wire the published selected action and fallback order into the destination coordinator, and
  force at least one cohort through an automatic stronger-action escalation;
- make compiled migration and exact recomputation publish through the same transaction;
- compare separately at fixed HBM and pinned-DRAM destinations;
- independently tune both methods under the same source tier/residency, target dtype/layout,
  destination, and publication semantics, while reporting capsule and raw-history source bytes
  separately;
- report 1/2/4-GPU completion time, tokens/s, physical/logical bytes, program amortization,
  assigned bytes, peak HBM, peak host staging, source residency, target residency, queue wait,
  commit latency, and numerical error.

Admission criterion:

- only this result can support “full-cohort destination update” and v4 end-to-end speedup claims;
- automatic fallback execution may be claimed only after the coordinator, destination transaction,
  and failure tests cover that path.

## Gate 3: failure and publication tests at realistic scale

Required experiment:

- inject failures before first extent, mid-wave, during publication, and immediately before
  manifest commit;
- verify no incomplete target version becomes visible;
- verify retry behavior and cleanup cost;
- report manifest and extent metadata overhead for the full cohort.

Boundary:

- this does not establish distributed transactions or coordinator crash recovery unless those
  mechanisms are implemented.

## Gate 4: physical storage backend

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

## Gate 5: capsule capacity trade-off

Current status:

- unpadded FP16 `Norm(x)` capsule is 50% of logical FP16 K/V size;
- v2 stores padded capsules totaling 1.507 GiB versus 2.998 GiB logical old K/V for its trace.

Required experiment:

- compare source capsule layouts and precision;
- report total extra capacity, creation cost, read bandwidth, compression error, and break-even
  update frequency;
- evaluate whether capsules replace another retained state or are purely additional.

Admission criterion:

- the paper can currently state this as a space-for-update-time trade-off, not a universally
  favorable capacity result.

## Gate 6: external-validity expansion

Required experiment:

- confirm the compiler and endpoint on at least one non-KuaiRand long-context checkpoint;
- preserve the Tenrec limitation: QB and QK are related tables and use ordinal rather than global
  calendar time;
- avoid Taobao UserBehavior for the primary task because it fails the true-unclicked-impression
  semantics gate;
- document ZhihuRec as a maintenance boundary rather than silently dropping it.

## Gate 7: closest cross-model baseline

Current status:

- structural p4/p8 and residual-\(p\) are internal recomputation controls;
- the current evaluation does not implement DroidSpeak's selective-layer sharing and pipelined
  cache-loading system on a compatible model.

Required experiment:

- implement the closest compatible selective-layer recomputation baseline without borrowing
  recommendation labels;
- tune it independently under the same source-residency, semantic-certificate, and
  target-publication boundaries;
- report both semantic recovery and complete-job cost rather than comparing only recomputed layer
  counts.

Boundary:

- HCache remains a same-model restoration system and is not a semantic substitute for this
  baseline;
- until this gate is complete, the paper may distinguish workloads and mechanisms but must not
  claim that cross-model K/V migration itself is novel.

## Results-to-paper update protocol

When a gate completes:

1. update `03_claim_evidence_matrix.md`;
2. update the exact Results subsection and relevant table/figure;
3. revise Discussion limitations;
4. only then revise Introduction contributions, Abstract, and title;
5. add a new review-log entry with the result family and protocol string.

No projected number should enter the manuscript before a result artifact and protocol record
exist.
