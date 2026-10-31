# Model and state primitives

This package provides HSTU attention/blocks/embeddings, full and incremental execution,
RMSNorm, and persistent batched K/V state.

| Module | Responsibility |
| --- | --- |
| `hstu.py`, `block.py` | Model composition and full/incremental layer execution |
| `attention.py`, `triton_attention.py` | Reference attention and supported CUDA kernels |
| `kv_cache.py`, `state_transition.py` | Persistent state, appends, eviction and dependency-closed transitions |
| `embeddings.py`, `rmsnorm.py` | Shared embedding and normalization primitives |

The [18 selected model endpoints](../../../docs/unified_training_2026_09/model_versions.md)
are used by the completed [Full/Reuse evaluation](../../../results/unified_reuse_2026_09/README.md).
Training and experiment orchestration live in [scripts/](../../../scripts/README.md).

The current HSTU-native score uses the current query/readout head. Candidate queries are
transient and never write into persistent state.

Keep model primitives independent from experiment orchestration and implement
only the paths needed by the research experiment. Use a small numerical reference
check when changing model/state math; generic validation and production API
hardening are not prerequisites for a prototype.

Historical diagnostic exact-KV replacement results remain under
`results/yambda500m_medium_seed17/insight1_locality_v1/`;
their retired launchers are not current model APIs.
An executable migration claim must account for
its hidden/KV dependencies and work; an exploratory implementation can establish
these with a focused comparison before formal evaluation.
