"""HSTU architecture for model-version K/V cache migration research.

The model is intentionally modular so future design tweaks (roadmap U1/U2/U6)
are localised edits:
  * PointwiseAttention  - the defining elu+1 unnormalised attention.
  * HSTUBlock           - one layer (norm + PMA + gating + residual).
  * HSTU                - full transducer with first-class KV output.
  * HSTUKVCache         - batched prefix K/V, the object being migrated.
"""

from .attention import PointwiseAttention, PointwiseAttentionConfig
from .block import HSTUBlock, HSTUBlockConfig
from .cc import (
    FrozenLinearBaseRanker,
    combine_base_and_cc_residual,
    conditional_reranking_loss,
    exact_chunked_listwise_cross_entropy,
    masked_listwise_cross_entropy,
)
from .embeddings import BehaviorEncoder, ItemEmbedding, QueryTokenEncoder, TemporalEncoder
from .hstu import HSTU, HSTUConfig
from .kv_cache import HSTUKVCache
from .rmsnorm import RMSNorm
from .state_transition import (
    append_with_rolling_cap,
    TransitionWork,
    frozen_segment,
    hybrid_tail_refresh,
    project_exact_layer0_segment,
    retain_latest_cache,
    transition_work,
    truncate_cache,
)

__all__ = [
    "append_with_rolling_cap",
    "PointwiseAttention",
    "PointwiseAttentionConfig",
    "HSTUBlock",
    "HSTUBlockConfig",
    "conditional_reranking_loss",
    "FrozenLinearBaseRanker",
    "combine_base_and_cc_residual",
    "exact_chunked_listwise_cross_entropy",
    "masked_listwise_cross_entropy",
    "BehaviorEncoder",
    "ItemEmbedding",
    "QueryTokenEncoder",
    "TemporalEncoder",
    "HSTU",
    "HSTUConfig",
    "HSTUKVCache",
    "RMSNorm",
    "TransitionWork",
    "frozen_segment",
    "hybrid_tail_refresh",
    "project_exact_layer0_segment",
    "retain_latest_cache",
    "transition_work",
    "truncate_cache",
]
