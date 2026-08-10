"""HSTU architecture for model-version K/V cache migration research.

The model is intentionally modular so future design tweaks (roadmap U1/U2/U6)
are localised edits:
  * PointwiseAttention  - the defining elu+1 unnormalised attention.
  * HSTUBlock           - one layer (norm + PMA + gating + residual).
  * HSTU                - full transducer with first-class KV output.
  * HSTUKVCache         - batched prefix K/V, the object being migrated.
"""

from .attention import PointwiseAttention, PointwiseAttentionConfig
from .attention_gauge import (
    apply_attention_coordinate_gauge_,
    apply_attention_coordinate_scale_,
)
from .block import HSTUBlock, HSTUBlockConfig
from .dense_hstu_v2 import DenseHSTUV2, DenseHSTUV2Block, DenseHSTUV2Config
from .embeddings import BehaviorEncoder, ItemEmbedding, TemporalEncoder
from .hstu import HSTU, HSTUConfig
from .kv_cache import HSTUKVCache
from .rmsnorm import RMSNorm
from .target_aware_kv import (
    FeatureCrossKV,
    FeatureCrossKVConfig,
    TargetAwareKV,
    TargetAwareKVConfig,
)

__all__ = [
    "PointwiseAttention",
    "PointwiseAttentionConfig",
    "apply_attention_coordinate_gauge_",
    "apply_attention_coordinate_scale_",
    "HSTUBlock",
    "HSTUBlockConfig",
    "DenseHSTUV2",
    "DenseHSTUV2Block",
    "DenseHSTUV2Config",
    "BehaviorEncoder",
    "ItemEmbedding",
    "TemporalEncoder",
    "HSTU",
    "HSTUConfig",
    "HSTUKVCache",
    "FeatureCrossKV",
    "FeatureCrossKVConfig",
    "TargetAwareKV",
    "TargetAwareKVConfig",
    "RMSNorm",
]
