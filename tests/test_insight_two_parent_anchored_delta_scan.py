from __future__ import annotations

import torch
from insight_two.parent_anchored_delta_scan import (
    joint_kv_decoder,
    medium_parent_anchored_delta_scan_floor,
    recover_parent_preblock_hidden,
    recover_parent_rms_output,
)


def test_joint_kv_and_rms_scalar_recover_parent_checkpoint() -> None:
    generator = torch.Generator().manual_seed(17)
    batch, length, hidden = 2, 7, 8
    state = torch.randn(batch, length, hidden, generator=generator, dtype=torch.float64)
    norm_weight = torch.rand(hidden, generator=generator, dtype=torch.float64) + 0.5
    rms = torch.sqrt(state.square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized = state / rms * norm_weight
    key_weight = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64)
    value_weight = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64)
    key = normalized @ key_weight.transpose(0, 1)
    value = normalized @ value_weight.transpose(0, 1)

    decoder = joint_kv_decoder(key_weight, value_weight)
    recovered_norm = recover_parent_rms_output(key, value, decoder)
    recovered_state = recover_parent_preblock_hidden(
        key, value, decoder, rms, norm_weight
    )

    torch.testing.assert_close(recovered_norm, normalized, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(recovered_state, state, rtol=1e-10, atol=1e-10)


def test_joint_decoder_rejects_noninjective_projection() -> None:
    weight = torch.zeros(4, 4)
    try:
        joint_kv_decoder(weight, weight)
    except ValueError as error:
        assert "not injective" in str(error)
    else:
        raise AssertionError("non-injective joint projection was accepted")


def test_medium_floor_fails_before_attention_delta_work() -> None:
    floor = medium_parent_anchored_delta_scan_floor()
    assert floor.rms_metadata_scalars == 6_144
    assert floor.parent_kv_scalars == 2_359_296
    assert floor.rms_metadata_scalars / floor.parent_kv_scalars == 1 / 384
    assert floor.k_only_checkpoint_decode_flops == 452_984_832
    assert floor.stable_joint_checkpoint_decode_flops == 905_969_664
    assert floor.historical_query_and_gate_flops == 754_974_720
    assert floor.k_only_subtotal_flops == 1_207_959_552
    assert floor.stable_joint_subtotal_flops == 1_660_944_384
    assert floor.k_only_subtotal_over_exact > 0.25
    assert floor.stable_joint_subtotal_over_exact > 0.34

