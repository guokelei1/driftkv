import torch

from hstu_kvcache.migration import (
    capture_layerwise_state,
    compile_low_rank_cache_adapter,
    compile_projection_cache_adapter,
    contiguous_intervals,
    extra_state_numel,
    fit_low_rank_cache_adapter,
    interval_extra_state_numel,
    migrate_compiled_low_rank_cache,
    migrate_contiguous_cache,
    migrate_current_norm_cache,
    migrate_embedding_delta_cache,
    migrate_fused_projection_cache,
    migrate_legacy_suffix_cache,
    migrate_low_rank_cache,
    migrate_prefix_residual_cache,
    migrate_recent_suffix_cache,
    migrate_suffix_cache,
    prefix_residual_extra_state_numel,
    recent_suffix_extra_state_numel,
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


def test_same_model_current_norm_and_embedding_delta_reproduce_cache():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)

    renorm = migrate_current_norm_cache(model, state)
    carried = migrate_embedding_delta_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
    )

    assert torch.allclose(renorm.k, state.kv.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(renorm.v, state.kv.v, atol=1e-5, rtol=1e-5)
    assert torch.allclose(carried.k, state.kv.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(carried.v, state.kv.v, atol=1e-5, rtol=1e-5)


def test_embedding_delta_zero_scale_matches_current_norm():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    renorm = migrate_current_norm_cache(model, state)
    carried = migrate_embedding_delta_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        torch.zeros(len(model.blocks)),
    )

    assert torch.allclose(renorm.k, carried.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(renorm.v, carried.v, atol=1e-5, rtol=1e-5)


def test_low_rank_adapter_recovers_synthetic_cache_residual():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    feature_layers = []
    residual_layers = []
    generator = torch.Generator().manual_seed(11)
    transforms = []
    offsets = []
    for normed in state.normed_states:
        valid = torch.arange(normed.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        features = normed[valid]
        transform = torch.randn(
            features.shape[1],
            2 * features.shape[1],
            generator=generator,
        )
        offset = torch.randn(2 * features.shape[1], generator=generator)
        feature_layers.append(features)
        residual_layers.append(features @ transform + offset)
        transforms.append(transform)
        offsets.append(offset)

    adapter = fit_low_rank_cache_adapter(
        feature_layers,
        residual_layers,
        rank=model.cfg.hidden_size,
        ridge=0.0,
    )
    base = migrate_current_norm_cache(model, state)
    migrated = migrate_low_rank_cache(model, state, adapter)
    valid = torch.arange(item_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    for layer, normed in enumerate(state.normed_states):
        expected = normed @ transforms[layer] + offsets[layer]
        expected = expected * valid.unsqueeze(-1)
        expected_k, expected_v = expected.split(model.cfg.hidden_size, dim=-1)
        assert torch.allclose(
            migrated.k[layer],
            base.k[layer] + expected_k,
            atol=2e-3,
            rtol=2e-3,
        )
        assert torch.allclose(
            migrated.v[layer],
            base.v[layer] + expected_v,
            atol=2e-3,
            rtol=2e-3,
        )


def test_compiled_low_rank_adapter_matches_online_execution():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    features = [
        value.reshape(-1, value.shape[-1])
        for value in state.normed_states
    ]
    residuals = [
        torch.randn(value.shape[0], 2 * value.shape[1])
        for value in features
    ]
    adapter = fit_low_rank_cache_adapter(features, residuals, rank=8)
    compiled = compile_low_rank_cache_adapter(model, adapter)

    online = migrate_low_rank_cache(model, state, adapter)
    migrated = migrate_compiled_low_rank_cache(state, compiled)

    assert torch.allclose(migrated.k, online.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v, online.v, atol=1e-5, rtol=1e-5)


def test_fused_projection_matches_existing_cheap_operator():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    expected = migrate_contiguous_cache(
        model,
        state,
        item_ids,
        behaviors,
        time_deltas,
        None,
        None,
    )
    fused = migrate_fused_projection_cache(model, state)

    assert torch.allclose(fused.k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused.v, expected.v, atol=1e-5, rtol=1e-5)


def test_precompiled_projection_matches_fused_projection():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    compiled = compile_projection_cache_adapter(model)

    expected = migrate_fused_projection_cache(model, state)
    migrated = migrate_compiled_low_rank_cache(state, compiled)

    assert torch.allclose(migrated.k, expected.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(migrated.v, expected.v, atol=1e-5, rtol=1e-5)


def test_low_rank_adapter_truncation_changes_rank_and_size():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(model, item_ids, behaviors, time_deltas, lengths)
    features = [
        value.reshape(-1, value.shape[-1])
        for value in state.normed_states
    ]
    residuals = [
        torch.cat((value, value), dim=-1)
        for value in features
    ]
    adapter = fit_low_rank_cache_adapter(features, residuals, rank=8)
    truncated = adapter.truncate(3)

    assert adapter.rank == 8
    assert truncated.rank == 3
    assert truncated.numel < adapter.numel


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


def test_recent_suffix_replay_matches_same_model_cache():
    model, item_ids, behaviors, time_deltas, _ = make_inputs()
    lengths = torch.full((item_ids.shape[0],), item_ids.shape[1])
    state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )

    for top_n in range(len(model.blocks) + 1):
        for replay_tokens in (0, 2, 5, item_ids.shape[1]):
            migrated = migrate_recent_suffix_cache(
                model,
                state,
                item_ids,
                behaviors,
                time_deltas,
                top_n,
                replay_tokens,
            )
            assert torch.allclose(migrated.k, state.kv.k, atol=1e-5, rtol=1e-5)
            assert torch.allclose(migrated.v, state.kv.v, atol=1e-5, rtol=1e-5)


def test_recent_suffix_boundaries_match_cheap_suffix_and_full():
    model, item_ids, behaviors, time_deltas, _ = make_inputs()
    lengths = torch.full((item_ids.shape[0],), item_ids.shape[1])
    old_state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    cheap = migrate_recent_suffix_cache(
        model,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        top_n_full=0,
        replay_tokens=item_ids.shape[1],
    )
    expected_cheap = migrate_suffix_cache(
        model,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        top_n_full=0,
    )
    assert torch.allclose(cheap.k, expected_cheap.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cheap.v, expected_cheap.v, atol=1e-5, rtol=1e-5)

    for top_n in range(1, len(model.blocks) + 1):
        replayed = migrate_recent_suffix_cache(
            model,
            old_state,
            item_ids,
            behaviors,
            time_deltas,
            top_n,
            item_ids.shape[1],
        )
        expected = migrate_suffix_cache(
            model,
            old_state,
            item_ids,
            behaviors,
            time_deltas,
            top_n,
        )
        assert torch.allclose(replayed.k, expected.k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(replayed.v, expected.v, atol=1e-5, rtol=1e-5)

    full = migrate_recent_suffix_cache(
        model,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        top_n_full=len(model.blocks),
        replay_tokens=item_ids.shape[1],
    )
    fresh = model.compute_kv(
        item_ids,
        behaviors,
        time_deltas,
        lengths=lengths,
    )
    assert torch.allclose(full.k, fresh.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full.v, fresh.v, atol=1e-5, rtol=1e-5)


def test_recent_suffix_state_cost_has_exact_endpoints():
    model, item_ids, behaviors, time_deltas, _ = make_inputs()
    lengths = torch.full((item_ids.shape[0],), item_ids.shape[1])
    state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    cheap = sum(value.numel() for value in state.normed_states)

    assert recent_suffix_extra_state_numel(state, 0, item_ids.shape[1]) == cheap
    assert recent_suffix_extra_state_numel(state, len(model.blocks), 0) == cheap
    assert (
        recent_suffix_extra_state_numel(
            state,
            len(model.blocks),
            item_ids.shape[1],
        )
        == 0
    )
    assert (
        recent_suffix_extra_state_numel(state, 2, 4)
        < recent_suffix_extra_state_numel(state, 1, 4)
    )


def test_prefix_residual_reproduces_same_model_cache():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )

    for depth in range(len(model.blocks) + 1):
        migrated = migrate_prefix_residual_cache(
            model,
            state,
            item_ids,
            behaviors,
            time_deltas,
            depth,
        )
        assert torch.allclose(migrated.k, state.kv.k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(migrated.v, state.kv.v, atol=1e-5, rtol=1e-5)


def test_prefix_residual_endpoints_match_cheap_and_full():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    old_state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)

    cheap = migrate_prefix_residual_cache(
        model,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        prefix_depth=0,
    )
    expected_cheap = migrate_suffix_cache(
        model,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        top_n_full=0,
    )
    full = migrate_prefix_residual_cache(
        model,
        old_state,
        item_ids,
        behaviors,
        time_deltas,
        prefix_depth=len(model.blocks),
    )
    expected_full = model.compute_kv(
        item_ids,
        behaviors,
        time_deltas,
        lengths=lengths,
    )

    assert torch.allclose(cheap.k, expected_cheap.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cheap.v, expected_cheap.v, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full.k, expected_full.k, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full.v, expected_full.v, atol=1e-5, rtol=1e-5)


def test_prefix_residual_state_cost_decreases_with_depth():
    model, item_ids, behaviors, time_deltas, lengths = make_inputs()
    state = capture_layerwise_state(
        model,
        item_ids,
        behaviors,
        time_deltas,
        lengths,
    )
    sizes = [
        prefix_residual_extra_state_numel(state, depth)
        for depth in range(len(model.blocks) + 1)
    ]

    assert sizes[0] > sizes[1] > sizes[2] > sizes[3]
    assert sizes[3] == 0
