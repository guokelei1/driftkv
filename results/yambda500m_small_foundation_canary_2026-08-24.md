# Yambda-500M Small foundation implementation canary — 2026-08-24

Status: correctness/preflight only; **not scientific evidence and not a formal
Small seed17 run**.

The bounded real-data chain used 32 users, a canary-only Base artifact, and four
training requests per v0/v1/v2/R0 checkpoint. It verified:

- causal manifests score a complete timestamp group before appending any event
  at that timestamp;
- v0→v1→v2 parent hashes and cache-producer hashes are recorded;
- R0 trains only query/readout parameters and preserved the v0 producer hash;
- both natural edges emitted all six aligned paths, including Parent Exact
  Rolling, then sealed 12 raw rows before label join;
- edge2 one-hop and true recursive lineage were distinct on the canary request.

Resource sample on GPU 0, full 4L/H128/item-vocabulary model:

- batch 8, 16 requests, 2 optimizer steps: 1.50 s measured training-loop time;
- peak allocated CUDA memory: 2,038,863,872 bytes (about 1.90 GiB);
- model-only checkpoint: about 384 MiB.

This tiny sample is sufficient for API and memory feasibility, not a stable
runtime forecast. A representative-throughput canary with the final loader,
batching and checkpoint policy is still required before requesting the formal
Small seed17 launch. At roughly the observed 10.7 requests/s, the 652,345-request
v0 pass would be about 17 hours on the current single-GPU implementation; this
is a conservative provisional estimate and motivates the planned batching/FSDP
preflight rather than being treated as a committed runtime.

The subsequent four-rank FSDP canary passed for v0, recursive v1 loading and
the R0 output-only control. Rank-zero full-state checkpoints round-tripped into
an unwrapped HSTU, and R0 preserved the parent cache-producer hash. The formal
queue now checkpoints 25/50/100% progress and uses only the 100% checkpoint as
the next version's parent.

Canary outputs lived under a temporary directory and are intentionally not
retained as evidence. No theta3 data or result was read.
