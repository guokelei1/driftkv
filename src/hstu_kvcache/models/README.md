# Model and state primitives

This package provides HSTU attention/blocks/embeddings, full and incremental execution,
RMSNorm, and persistent batched K/V state.

The current HSTU-native score uses the current query/readout head. Candidate queries are
transient and never write into persistent state.

Keep model primitives independent from experiment orchestration. Diagnostic exact-KV replacement belongs to P9 evaluation code and is not a production migration API. A partial-migration API may be added only after its hidden/KV dependency closure and work-accounting contract are frozen.
