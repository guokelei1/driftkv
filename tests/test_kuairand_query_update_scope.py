from __future__ import annotations

import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_query_update_scope import _configure_scope


def _model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=32,
            num_behaviors=1,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=16,
            input_dropout=0.0,
            gating="none",
        )
    )


def test_core_scope_freezes_only_embedding() -> None:
    model = _model()
    selected = _configure_scope(model, "core_only")
    assert selected
    active = {name for name, value in model.named_parameters() if value.requires_grad}
    assert "item_emb.weight" not in active
    assert "blocks.1.attn.out_proj.weight" in active


def test_full_scope_trains_every_parameter() -> None:
    model = _model()
    selected = _configure_scope(model, "full")
    assert len(selected) == len(list(model.parameters()))
    assert all(value.requires_grad for value in model.parameters())


def test_cache_breaking_scope_excludes_final_safe_path() -> None:
    model = _model()
    selected = _configure_scope(model, "cache_breaking_only")
    assert selected
    active = {name for name, value in model.named_parameters() if value.requires_grad}
    assert "blocks.0.attn.out_proj.weight" in active
    assert "blocks.1.attn.k_proj.weight" in active
    assert "blocks.1.attn.q_proj.weight" not in active
    assert "blocks.1.attn.out_proj.weight" not in active
    assert "final_norm.weight" not in active
    assert all(isinstance(value, torch.nn.Parameter) for value in selected)
