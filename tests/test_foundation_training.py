from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex, cache_producer_sha256


def test_history_prefix_is_strict_and_bounded() -> None:
    index = FoundationHistoryIndex.from_columns(
        np.asarray([1, 1, 1]), np.asarray([10, 20, 20]),
        np.asarray([2, 3, 4]), np.asarray([1, 1, 2]),
    )
    items, _, timestamps = index.prefix(1, 20, max_history=2)
    assert items.tolist() == [2]
    assert timestamps.tolist() == [10]


def test_r0_output_only_edit_does_not_change_cache_producer_hash() -> None:
    model = HSTU(HSTUConfig(num_items=16, num_behaviors=4, hidden_size=8, num_layers=1, num_heads=1))
    before = cache_producer_sha256(model.state_dict())
    with torch.no_grad():
        model.query_encoder.type_embedding.weight.add_(1.0)
        model.cc_score_head.weight.add_(1.0)
    assert cache_producer_sha256(model.state_dict()) == before
    with torch.no_grad():
        model.blocks[0].attn.k_proj.weight.add_(1.0)
    assert cache_producer_sha256(model.state_dict()) != before
