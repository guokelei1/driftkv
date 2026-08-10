import torch

from hstu_kvcache.streaming.kuairand_natural_path_attribution import (
    VARIANTS,
    _dense_state_for_variant,
    load_natural_path_attribution_config,
)

CONFIG = (
    "configs/evokv_root_cause/"
    "kuairand_natural_path_attribution_theta4_theta5_20260810_v0.json"
)


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "core.blocks.0.attn.q_proj.weight": torch.full((2, 2), value),
        "core.blocks.0.attn.k_proj.weight": torch.full((2, 2), value),
        "core.blocks.0.attn.v_proj.weight": torch.full((2, 2), value),
        "core.blocks.0.attn.out_proj.weight": torch.full((2, 2), value),
        "core.blocks.0.ffn.weight": torch.full((2, 2), value),
    }


def test_natural_path_config_binds_adjacent_large_checkpoints_and_control():
    document = load_natural_path_attribution_config(CONFIG)
    assert document["source"]["source_version"] == 4
    assert document["source"]["target_version"] == 5
    assert document["variants"] == list(VARIANTS)


def test_natural_path_variants_replace_only_declared_parameter_groups():
    source = _state(1.0)
    target = _state(2.0)
    q_only = _dense_state_for_variant(
        source, target, "embedding_projection_plus_q"
    )
    assert q_only["core.blocks.0.attn.q_proj.weight"].eq(2.0).all()
    assert q_only["core.blocks.0.attn.k_proj.weight"].eq(1.0).all()
    assert q_only["core.blocks.0.ffn.weight"].eq(1.0).all()
    without_qkvo = _dense_state_for_variant(source, target, "full_without_qkvo")
    assert without_qkvo["core.blocks.0.attn.out_proj.weight"].eq(1.0).all()
    assert without_qkvo["core.blocks.0.ffn.weight"].eq(2.0).all()
