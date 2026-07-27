import pytest
import torch

from hstu_kvcache.migration import (
    ResidualHiddenSuffixState,
    capture_layerwise_state,
    capture_residual_hidden_suffix,
    capture_selective_contiguous_state,
    migrate_prefix_residual_cache,
    migrate_prefix_residual_from_hidden_suffix,
    migrate_selective_contiguous_cache,
    selective_contiguous_intervals,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def make_inputs():
    torch.manual_seed(41)
    model = HSTU(
        HSTUConfig(
            num_items=100,
            num_behaviors=8,
            hidden_size=32,
            num_layers=4,
            num_heads=2,
            head_dim=16,
            max_seq_len=8,
            input_dropout=0.0,
        )
    ).eval()
    item_ids = torch.randint(1, 101, (2, 8))
    behaviors = torch.randint(1, 9, (2, 8))
    time_deltas = torch.rand(2, 8) * 100
    lengths = torch.tensor([8, 5])
    item_ids[1, 5:] = 0
    behaviors[1, 5:] = 0
    time_deltas[1, 5:] = 0
    return model, item_ids, behaviors, time_deltas, lengths


def make_version_pair():
    old, item_ids, behaviors, time_deltas, lengths = make_inputs()
    old_state = capture_layerwise_state(
        old,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    current, _, _, _, _ = make_inputs()
    current.load_state_dict(old.state_dict())
    with torch.no_grad():
        for parameter in current.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    current.eval()
    return (
        current,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )


def test_frozen_interval_grid_contains_53_unique_candidates():
    intervals = selective_contiguous_intervals(16, (2, 4, 6, 8, 12))

    assert len(intervals) == 53
    assert len(set(intervals)) == 53
    assert intervals[:2] == ((0, 1), (1, 2))
    assert intervals[-1] == (4, 15)


def test_selective_contiguous_reuses_source_kv_outside_interval():
    current, old_state, item_ids, behaviors, time_deltas, _ = make_version_pair()
    state = capture_selective_contiguous_state(old_state, start_layer=1)
    migrated = migrate_selective_contiguous_cache(
        current,
        state,
        item_ids,
        behaviors,
        time_deltas,
        end_layer=2,
    )

    assert torch.equal(migrated.k[0], old_state.kv.k[0])
    assert torch.equal(migrated.v[0], old_state.kv.v[0])
    assert torch.equal(migrated.k[3], old_state.kv.k[3])
    assert torch.equal(migrated.v[3], old_state.kv.v[3])


def test_selective_contiguous_inside_interval_matches_minimal_replay():
    current, old_state, item_ids, behaviors, time_deltas, lengths = (
        make_version_pair()
    )
    state = capture_selective_contiguous_state(old_state, start_layer=1)
    migrated = migrate_selective_contiguous_cache(
        current,
        state,
        item_ids,
        behaviors,
        time_deltas,
        end_layer=2,
    )
    valid = torch.arange(item_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    x, first = current.blocks[1](
        old_state.hidden_states[1],
        return_kv=True,
    )
    x = x * valid.unsqueeze(-1)
    terminal_norm = current.blocks[2].norm(x)
    terminal = (
        current.blocks[2].attn.k_proj(terminal_norm),
        current.blocks[2].attn.v_proj(terminal_norm),
    )

    assert torch.allclose(migrated.k[1], first[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v[1], first[1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.k[2], terminal[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v[2], terminal[1], atol=1e-5, rtol=1e-5)


def test_full_selective_interval_equals_exact_current_cache():
    current, old_state, item_ids, behaviors, time_deltas, lengths = (
        make_version_pair()
    )
    state = capture_selective_contiguous_state(old_state, start_layer=0)
    migrated = migrate_selective_contiguous_cache(
        current,
        state,
        item_ids,
        behaviors,
        time_deltas,
        end_layer=len(current.blocks) - 1,
    )
    exact = current.compute_kv(
        item_ids,
        behaviors,
        time_deltas,
        lengths=lengths,
    )

    assert torch.allclose(migrated.k, exact.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v, exact.v, atol=1e-5, rtol=1e-5)


def test_selective_state_requires_the_matching_transition():
    current, old_state, item_ids, behaviors, time_deltas, _ = make_version_pair()
    state = capture_selective_contiguous_state(old_state, start_layer=1)
    invalid = type(state)(
        source_kv=state.source_kv,
        transition_hidden=None,
        lengths=state.lengths,
        start_layer=1,
    )

    with pytest.raises(ValueError, match="transition hidden"):
        migrate_selective_contiguous_cache(
            current,
            invalid,
            item_ids,
            behaviors,
            time_deltas,
            end_layer=2,
        )


def test_residual_hidden_suffix_is_sufficient_for_each_depth():
    current, old_state, item_ids, behaviors, time_deltas, _ = make_version_pair()

    for depth in range(1, len(current.blocks) + 1):
        suffix = capture_residual_hidden_suffix(old_state, depth)
        migrated = migrate_prefix_residual_from_hidden_suffix(
            current,
            suffix,
            item_ids,
            behaviors,
            time_deltas,
        )
        expected = migrate_prefix_residual_cache(
            current,
            old_state,
            item_ids,
            behaviors,
            time_deltas,
            depth,
        )
        assert torch.allclose(migrated.k, expected.k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(migrated.v, expected.v, atol=1e-5, rtol=1e-5)


def test_incomplete_residual_hidden_suffix_rejects_before_execution():
    current, old_state, item_ids, behaviors, time_deltas, _ = make_version_pair()
    suffix = capture_residual_hidden_suffix(old_state, 2)
    invalid = ResidualHiddenSuffixState(
        hidden_states=suffix.hidden_states[:-1],
        lengths=suffix.lengths,
        start_layer=suffix.start_layer,
        num_layers=suffix.num_layers,
    )

    with pytest.raises(ValueError, match="incomplete"):
        migrate_prefix_residual_from_hidden_suffix(
            current,
            invalid,
            item_ids,
            behaviors,
            time_deltas,
        )


def test_storage_dtype_changes_declared_auxiliary_state_bytes():
    _, old_state, _, _, _, _ = make_version_pair()
    selective_fp32 = capture_selective_contiguous_state(old_state, 2)
    selective_fp16 = capture_selective_contiguous_state(
        old_state,
        2,
        storage_dtype=torch.float16,
    )
    residual_fp32 = capture_residual_hidden_suffix(old_state, 2)
    residual_bf16 = capture_residual_hidden_suffix(
        old_state,
        2,
        storage_dtype=torch.bfloat16,
    )
    lengths_bytes = old_state.lengths.numel() * old_state.lengths.element_size()

    assert selective_fp16.nbytes - lengths_bytes == (
        selective_fp32.nbytes - lengths_bytes
    ) // 2
    assert residual_bf16.nbytes - lengths_bytes == (
        residual_fp32.nbytes - lengths_bytes
    ) // 2
    assert all(
        value.dtype == torch.bfloat16
        and bool(torch.isfinite(value).all())
        for value in residual_bf16.hidden_states
    )
