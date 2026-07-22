import torch

from hstu_kvcache.migration import (
    capture_layerwise_state,
    contiguous_intervals,
    extra_state_numel,
    interval_extra_state_numel,
    migrate_contiguous_cache,
    migrate_legacy_suffix_cache,
    migrate_suffix_cache,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def make_inputs():
    torch.manual_seed(7)
    model = HSTU(
        HSTUConfig(
            num_items=100,
            num_behaviors=8,
            hidden_size=32,
            num_layers=3,
            num_heads=2,
            head_dim=16,
            max_seq_len=8,
            input_dropout=0.0,
        )
    )
    model.eval()
    item_ids = torch.randint(1, 101, (2, 8))
    behaviors = torch.randint(1, 9, (2, 8))
    time_deltas = torch.rand(2, 8) * 100
    lengths = torch.tensor([8, 5])
    item_ids[1, 5:] = 0
    behaviors[1, 5:] = 0
    time_deltas[1, 5:] = 0
    return model, item_ids, behaviors, time_deltas, lengths


def test_same_model_migration_reproduces_original_cache():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)

    for top_n in range(4):
        migrated = migrate_legacy_suffix_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            top_n,
        )
        assert torch.allclose(migrated.k, state.kv.k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(migrated.v, state.kv.v, atol=1e-5, rtol=1e-5)


def test_all_full_layers_equal_current_model_recompute():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    migrated = migrate_legacy_suffix_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        len(model.blocks),
    )
    fresh = model.compute_kv(item_ids, behaviors, time_deltas, lengths=lengths)

    assert torch.allclose(migrated.k, fresh.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v, fresh.v, atol=1e-5, rtol=1e-5)


def test_hybrid_uses_current_projections_and_full_suffix_blocks():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    migrated = migrate_legacy_suffix_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        top_n_full=2,
    )
    expected = []
    expected.append(
        (
            model.blocks[0].attn.k_proj(state.normed_states[0]),
            model.blocks[0].attn.v_proj(state.normed_states[0]),
        )
    )
    x = state.hidden_states[1]
    for block in model.blocks[1:]:
        x, kv = block(x, return_kv=True)
        expected.append(kv)

    for layer, (expected_k, expected_v) in enumerate(expected):
        assert torch.allclose(migrated.k[layer], expected_k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(migrated.v[layer], expected_v, atol=1e-5, rtol=1e-5)


def test_extra_state_decreases_with_full_suffix_depth():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    sizes = [extra_state_numel(state, top_n) for top_n in range(4)]

    assert sizes[0] == sizes[1]
    assert sizes[1] > sizes[2] > sizes[3]
    assert sizes[3] == 0


def test_optimized_suffix_matches_legacy_cache_at_every_depth():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    for top_n in range(len(model.blocks) + 1):
        legacy = migrate_legacy_suffix_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            top_n,
        )
        optimized = migrate_suffix_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            top_n,
        )
        assert torch.allclose(optimized.k, legacy.k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(optimized.v, legacy.v, atol=1e-5, rtol=1e-5)


def test_full_optimized_interval_equals_current_model_recompute():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    migrated = migrate_contiguous_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        0,
        len(model.blocks) - 1,
    )
    fresh = model.compute_kv(item_ids, behaviors, time_deltas, lengths=lengths)

    assert torch.allclose(migrated.k, fresh.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v, fresh.v, atol=1e-5, rtol=1e-5)


def test_non_suffix_interval_propagates_only_inside_interval():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    migrated = migrate_contiguous_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        0,
        1,
    )
    valid = torch.arange(item_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    x = model.embed_inputs(item_ids, behaviors, time_deltas) * valid.unsqueeze(-1)
    x, first = model.blocks[0](x, return_kv=True)
    x = x * valid.unsqueeze(-1)
    second_norm = model.blocks[1].norm(x)
    expected = [
        first,
        (
            model.blocks[1].attn.k_proj(second_norm),
            model.blocks[1].attn.v_proj(second_norm),
        ),
        (
            model.blocks[2].attn.k_proj(state.normed_states[2]),
            model.blocks[2].attn.v_proj(state.normed_states[2]),
        ),
    ]

    for layer, (expected_k, expected_v) in enumerate(expected):
        assert torch.allclose(migrated.k[layer], expected_k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(migrated.v[layer], expected_v, atol=1e-5, rtol=1e-5)


def test_interval_space_and_state_costs():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)

    assert contiguous_intervals(3) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]
    cheap = interval_extra_state_numel(state, None, None)
    suffix_one = interval_extra_state_numel(state, 2, 2)
    suffix_two = interval_extra_state_numel(state, 1, 2)
    full = interval_extra_state_numel(state, 0, 2)

    assert cheap == suffix_one
    assert cheap > suffix_two > full
    assert full == 0
