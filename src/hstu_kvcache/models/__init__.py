"""HSTU architecture for streaming KV-cache drift research.

The model is intentionally modular so future design tweaks (roadmap U1/U2/U6)
are localised edits:
  * PointwiseAttention  - the defining elu+1 unnormalised attention.
  * HSTUBlock           - one layer (norm + PMA + gating + residual).
  * HSTU                - full transducer with first-class KV output.
  * HSTUKVCache         - F(theta, x_u), the object whose drift we study.
"""

from .attention import PointwiseAttention, PointwiseAttentionConfig
from .block import HSTUBlock, HSTUBlockConfig
from .embeddings import BehaviorEncoder, ItemEmbedding, TemporalEncoder
from .hstu import HSTU, HSTUConfig
from .kv_cache import HSTUKVCache
from .rmsnorm import RMSNorm

__all__ = [
    "PointwiseAttention",
    "PointwiseAttentionConfig",
    "HSTUBlock",
    "HSTUBlockConfig",
    "BehaviorEncoder",
    "ItemEmbedding",
    "TemporalEncoder",
    "HSTU",
    "HSTUConfig",
    "HSTUKVCache",
    "RMSNorm",
]
