from __future__ import annotations

import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_query_multiversion import (
    _publish,
    _reconstruct_published,
)


def _model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=16,
            num_behaviors=1,
            hidden_size=8,
            num_layers=1,
            num_heads=1,
            head_dim=8,
            max_seq_len=8,
            input_dropout=0.0,
            gating="none",
        )
    )


def test_reconstruct_and_publish_follow_fixed_alpha() -> None:
    previous = _model()
    source = _model()
    target = _model()
    with torch.no_grad():
        for value in previous.parameters():
            value.fill_(1.0)
        for value in source.parameters():
            value.fill_(2.5)
        for value in target.parameters():
            value.zero_()
    _reconstruct_published(
        target, previous.state_dict(), source.state_dict(), 0.75, 0.5
    )
    assert all(torch.allclose(value, torch.full_like(value, 2.0)) for value in target.parameters())
    raw = _model()
    with torch.no_grad():
        for value in raw.parameters():
            value.fill_(5.0)
    _publish(previous, raw, 0.25)
    assert all(torch.allclose(value, torch.full_like(value, 2.0)) for value in raw.parameters())
