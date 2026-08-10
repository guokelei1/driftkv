from __future__ import annotations

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_kv_strength_screen import _kv_parameters


def test_kv_strength_group_updates_only_k_and_v_projections() -> None:
    model = HSTU(
        HSTUConfig(
            num_items=30,
            num_behaviors=4,
            hidden_size=16,
            num_layers=3,
            num_heads=2,
            head_dim=8,
        )
    )
    parameters, names = _kv_parameters(model)
    assert len(parameters) == 6
    assert len(names) == 6
    assert all(".attn.k_proj." in name or ".attn.v_proj." in name for name in names)
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in model.named_parameters()
    )
