# Model and state primitives

This package provides HSTU attention/blocks/embeddings, full and incremental execution, candidate-conditioned residual scoring, RMSNorm, and persistent batched K/V state.

The deployed P7-P9 score is a frozen low-capacity Base plus a CC residual; the Base is detached and identical across Recent/Full/Reuse paths. Candidate queries are transient and never write into persistent state.

Keep model primitives independent from experiment orchestration. Diagnostic exact-KV replacement belongs to P9 evaluation code and is not a production migration API. A partial-migration API may be added only after its hidden/KV dependency closure and work-accounting contract are frozen.
