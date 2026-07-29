# Destination-oriented out-of-core K/V update system v4

> Historical disposition: superseded normalized-capsule destination prototype. It preserves useful
> transaction and endpoint-interface evidence, but it is not the current EvoKV architecture, not
> the new D3 direct-old-K/V ordinary-DRAM pipeline, and not a source of current next steps.

## Status

`streamkv_destination_out_of_core_v4` was the architecture and implementation contract for this
historical protocol. It replaced an earlier online-lifecycle proposal at that time. It does not
replace algorithm evidence from the verified compiler or measured v2/v3 runtime results.

The implementation is an initial vertical slice, not a destination performance result. DRAM,
filesystem, remote-object, and HBM publication semantics have executable reference paths and
correctness tests. Direct-HBM manifest publication has been validated across all four local A40s,
including automatic trimming of excess workers when a job has fewer extents than declared GPUs.
No SSD device, network transport, GDS, RDMA, or remote GPU has been benchmarked.
The host-staged path bounds transform outputs and publication backlog after capsule
materialization; it does not yet lazily stream the complete source capsule set. The HBM path keeps
the complete target K/V resident and currently executes as one direct job.

## System definition

StreamKV is a model-update-triggered, training- and serving-decoupled K/V update system:

> Given old versioned cache capsules, one or more published migration programs, a fixed update
> cohort, and an explicit destination, transform the complete cohort into target-version K/V
> through bounded transient execution/publication working sets and publish one complete version
> manifest.

Training is outside the boundary and supplies model checkpoints. Online request arrivals,
per-user hotness, request routing, and foreground inference interference are also outside the
current boundary. The update job may run on dedicated GPUs. Its output is a versioned K/V object
set in a caller-selected destination, not an assumption about where an industrial serving stack
must place training or inference.

## Three connected layers

1. **Cohort migration compiler.** A source/target model pair produces a shared, verified migration
   program. The current fast path compiles `fresh - cheap` residual repair into one affine
   projection over old `Norm(x)`.
2. **Capsule-to-K/V operator.** A batch of valid old normalized states is transformed into final
   K/V tensors. The fused implementation can write directly into preallocated destination
   tensors. Cohort-jagged compaction remains a conditional layout mechanism: it helps only when
   the workload has enough short, coalescible fragments, and it is not a positive claim on the
   current long-context KuaiRand trace.
3. **Destination-oriented out-of-core engine.** The engine executes host-staged endpoints in
   bounded transform/publication waves, keeps migration programs resident on its workers, assigns
   byte-weighted extents across GPUs, and overlaps H2D/compute/D2H with host publication. The
   direct-HBM endpoint writes resident target extents without D2H. Both paths commit one
   target-version manifest only after complete record coverage.

The compiler determines what state transformation is valid. The operator determines how one
batch is transformed. The engine determines the execution and publication dataflow for the
declared endpoint.

## Coordinator role

The update coordinator is a thin control-plane wrapper around these three layers:

```text
job specification
  -> resolve published programs and capsule shards
  -> group records by source-version cohort
  -> invoke the out-of-core engine for the declared destination
  -> return the committed manifest and metrics
```

`scripts/run_streamkv_update_coordinator.py` expresses this interface and defaults to plan-only
output. It is integration code, not a fourth contribution. It does not compile or certify a
program, create source capsules, infer whether a version can be reused, choose HBM versus DRAM/
filesystem/remote, schedule online requests, or provide durable job recovery.

## Destination contract

Every backend exposes the same job-level operations:

```text
begin(job, target_version, expected_record_ids)
    -> stage(extent_id, target K/V)
    -> commit complete version manifest
    -> or abort without a visible target version
```

The manifest protocol is `streamkv_destination_manifest_v1`. Each extent records:

- stable extent and record IDs;
- migration-anchor and target K/V versions;
- layer, valid-token, K/V-width, dtype, and byte metadata;
- destination location and device;
- an optional serialized-payload checksum.

The current `payload_bytes` field is the logical tensor footprint. Serialized file/object bytes
and backend write amplification are not yet represented by that field and must be measured
separately in a physical storage experiment.

Producing target K/V does not change the capsule's migration anchor. A committed manifest proves
complete, duplicate-free record coverage for one target version. It does not certify ranking
quality; quality remains the compiler's separate contract.

## Backend-specific execution paradigms

| Destination | Implemented data path | Publication boundary | Current status |
|---|---|---|---|
| GPU HBM | Execute the migration program on the destination GPU and write K/V into preallocated device extents | Manifest points to resident CUDA tensors; no D2H | Functional on one or multiple local GPUs |
| Host DRAM | Pinned-host capsule → H2D → transform → D2H target K/V | In-memory manifest atomically exposes retained CPU extents | Functional reference backend |
| Local filesystem / SSD mount | Same host-staged transform, then serialize immutable extent objects through a bounded publication queue | Same-filesystem directory rename exposes `manifest.json` and all extents together | Functional POSIX backend; no physical SSD performance claim |
| Remote object destination | Same host-staged transform, then upload immutable extent objects | A manifest object written last is the commit marker | Client protocol plus in-memory reference store; no network implementation or performance claim |

HBM is a direct-device endpoint, so the compute worker must currently be the destination GPU.
Cross-GPU P2P publication is not implemented. DRAM, filesystem, and remote destinations use host
staging. Their engine path has a bounded transform wave and a bounded pending-publication queue;
this prevents transformed K/V and pending writes from retaining the entire cohort in transient
buffers. The caller currently materializes the complete CPU capsule-batch sequence before
execution, so the total source side is not yet bounded by this mechanism. Direct HBM currently
allocates the complete target destination and does not use the host-staged wave limit.

The destination is explicit job input. The system does not infer hotness or choose HBM versus SSD
from a fabricated request trace. A future deployment-specific control plane may select a backend,
but this historical protocol compares fixed destination jobs.

## Transaction and failure semantics

1. A backend creates a private transaction namespace.
2. Each transformed extent is staged under that namespace or as an immutable unreferenced object.
3. Duplicate extent IDs, duplicate records, unknown records, wrong target versions, devices
   outside the HBM destination, and incomplete coverage are rejected.
4. A complete manifest is published only after every expected record is staged.
5. Abort removes private filesystem/remote staging objects and leaves no visible target manifest.

For local files, extent objects and the manifest are written through temporary files; the staged
directory is renamed into the version namespace at commit. Optional `fsync` is enabled by default.
For a remote object store, publication assumes individual object puts are atomic and uses the
manifest object as the visibility point. Distributed transactions across several independent
destinations are not implemented.

## Implemented artifacts

- `src/hstu_kvcache/migration/destination.py`
  - destination capabilities, extent metadata, version manifest, transaction contract;
  - DRAM, HBM, filesystem, and remote-object backends;
  - local filesystem checksum/readback and an in-memory remote reference store.
- `src/hstu_kvcache/migration/out_of_core.py`
  - destination-capability dispatch;
  - bounded host-staged waves and publication backpressure;
  - single-/multi-GPU host publication and direct multi-GPU HBM publication;
  - end-to-end job metrics.
- `scripts/validate_streamkv_destination_runtime.py`
  - a small no-training validation command for every backend interface.
- `scripts/run_streamkv_update_coordinator.py`
  - a plan-first job-spec wrapper that connects published artifacts to the existing engine;
  - conceptual control-plane integration only, with no new algorithm or performance result.
- `tests/test_destination_runtime.py`
  - exact CPU DRAM/filesystem/remote readback;
  - incomplete-publication invisibility and remote commit-marker semantics;
  - HBM direct-write numerical validation;
  - two-GPU host-staged publication and excess-HBM-worker trimming.

The reference validation commands are:

```bash
python scripts/validate_streamkv_destination_runtime.py --destination dram
python scripts/validate_streamkv_destination_runtime.py \
  --destination filesystem --root /path/on/a/filesystem
python scripts/validate_streamkv_destination_runtime.py --destination remote
CUDA_VISIBLE_DEVICES=0 python scripts/validate_streamkv_destination_runtime.py \
  --destination hbm --devices cuda:0
```

These commands use tiny synthetic tensors to validate interfaces. They are not system benchmarks
and do not train a model.

The four-GPU direct-HBM interface check is:

```bash
python scripts/validate_streamkv_destination_runtime.py \
  --destination hbm \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --output results/system/streamkv_destination_hbm_4gpu_validation.json
```

It publishes one extent per GPU, commits one complete four-record manifest, and has maximum
absolute K/V error `9.77e-4`. This is correctness evidence only. The real-capsule 1/2/4-GPU
supplement in `FOUR_GPU_SCALING_V1.md` uses the frozen v3 execution boundaries and therefore does
not close the full destination-v4 performance contract.

## Historical follow-up contract

The first real-data result should use the fixed KuaiRand 4+12 checkpoints, published verified
programs, and all eligible fixed update records. It must keep source and destination endpoints
identical between compiled migration and full recomputation.

Before calling that result fully out-of-core, the implementation must add a source manifest/reader
that scans capsule shards without materializing the whole source sequence. The exact-recompute
path must also publish through the same destination transaction rather than being compared across
the older host-only boundary.

For every fixed destination, report:

- total completion time and valid tokens/s;
- logical and physical bytes read and written;
- peak source-resident bytes, working HBM, target-resident bytes, and host staging bytes;
- wave size, publication queue depth, backpressure time, and stage overlap;
- per-device assigned bytes and 1/2/4-GPU scaling;
- complete record coverage and K/V numerical error;
- manifest commit latency and injected-failure visibility.

HBM, DRAM, SSD, and remote completion times answer different endpoint questions and may not be
reported as direct speedups over one another. The primary comparison inside each endpoint is
compiled migration versus an independently tuned full-recompute engine with the same source,
destination, dtype, layout, and publication semantics.

The destination expansion order is:

1. full-cohort pinned-DRAM and local-HBM endpoints;
2. bounded POSIX file publication on an identified local device;
3. a real remote/network backend only when reproducible hardware is available.

## Protocol claim boundary

Supported within this protocol:

- a coherent three-layer architecture with an executable destination contract;
- exact atomic publication behavior for the reference backends;
- local HBM direct-write and host-staged single-/multi-GPU execution paths;
- explicit separation of destination semantics from operator speedup;
- a thin plan/execute coordinator interface that does not alter the three contributions.

Not supported within this protocol:

- physical SSD throughput, GDS, RDMA, remote GPU, or cross-node speedups;
- automatic destination selection;
- an industrial request-arrival, hotness, routing, or serving-SLO result;
- full-cohort out-of-core performance or 1/2/4-GPU scaling under the new protocol;
- source-side lazy streaming, bounded total capsule memory, or a common destination-aware exact
  baseline;
- coordinator crash recovery, resume, or distributed job management;
- crash consistency beyond the stated POSIX and object-manifest assumptions.
