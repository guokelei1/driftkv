# Model and state primitives

This package provides HSTU attention/blocks/embeddings, full and incremental execution,
RMSNorm, and persistent batched K/V state.

The current HSTU-native score uses the current query/readout head. Candidate queries are
transient and never write into persistent state.

Keep model primitives independent from experiment orchestration and implement
only the paths needed by the research experiment. Use a small numerical reference
check when changing model/state math; generic validation and production API
hardening are not prerequisites for a prototype.

Diagnostic exact-KV replacement belongs to the Insight 1 code in
`scripts/insight_one_locality/`. An executable migration claim must account for
its hidden/KV dependencies and work; an exploratory implementation can establish
these with a focused comparison before formal evaluation.
