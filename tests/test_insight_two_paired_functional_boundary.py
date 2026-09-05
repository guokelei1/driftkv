from __future__ import annotations

import torch
from insight_two.attention_cone_moments import (
    build_positive_affine_moments,
    scaled_qk_logits,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    paired_release_replay,
)
from insight_two.paired_functional_boundary import (
    PRIMARY_PROBES,
    build_factorized_positive_moments,
    build_full_exact_response_memory,
    build_paired_factorized_response_memory,
    factorized_majority_positive_cone,
    medium_functional_boundary_cost,
)
from insight_two.paired_region_delta import trace_history_item_region_queries

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed: int, *, length: int = 32) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=128,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=length,
            temporal_num_freqs=2,
            input_dropout=0.0,
            activation="elu_plus1",
            block_variant="legacy",
            relative_position_bias=False,
        )
    ).eval()


def _history(length: int = 32) -> tuple[torch.Tensor, ...]:
    items = (torch.arange(length).reshape(1, length) % 127 + 1).long()
    behaviors = (torch.arange(length).reshape(1, length) % 2 + 1).long()
    deltas = torch.arange(length).reshape(1, length).float()
    query_delta = torch.tensor([float(length + 3)])
    return items, behaviors, deltas, query_delta


def _materialized_heads(
    model: HSTU,
    factors: FactorizedCacheLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    key, value = factors.materialize()
    heads = model.cfg.num_heads
    width = model.cfg.head_dim
    key = key.view(1, key.shape[1], heads, width).transpose(1, 2)
    value = value.view(1, value.shape[1], heads, width).transpose(1, 2)
    return key, value


def test_factorized_mask_and_moments_equal_materialized_dense_algebra() -> None:
    torch.manual_seed(1201)
    model = _model(1203)
    attention = model.blocks[0].attn
    layer = FactorizedCacheLayer(
        left=torch.randn(1, 32, 4),
        key_core=torch.randn(1, 4, 16),
        value_core=torch.randn(1, 4, 16),
    )
    queries = torch.randn(PRIMARY_PROBES, 2, 1, 8)
    observed_mask = factorized_majority_positive_cone(attention, queries, layer)
    key, value = _materialized_heads(model, layer)
    expected_mask = (
        2
        * (
            scaled_qk_logits(
                queries,
                key.expand(PRIMARY_PROBES, -1, -1, -1),
                scale=attention.scale,
            )
            >= 0
        )
        .sum(dim=0)
        .squeeze(1)
        >= PRIMARY_PROBES
    ).unsqueeze(0)
    assert torch.equal(observed_mask, expected_mask)

    observed = build_factorized_positive_moments(attention, layer, observed_mask)
    expected = build_positive_affine_moments(key, value, expected_mask)
    assert torch.allclose(observed.base, expected.base, atol=2e-5, rtol=2e-5)
    assert torch.allclose(observed.linear, expected.linear, atol=8e-5, rtol=8e-5)


def test_full_rank_paired_compiler_matches_full_exact_moment_oracle() -> None:
    parent = _model(1211)
    current = _model(1213)
    items, behaviors, deltas, query_delta = _history()
    exact_parent = parent.compute_kv(items, behaviors, deltas)
    exact_current = current.compute_kv(items, behaviors, deltas)
    probes = trace_history_item_region_queries(
        current,
        exact_parent,
        items,
        query_delta,
        probe_count=PRIMARY_PROBES,
    )
    replay = paired_release_replay(
        parent,
        current,
        parent.embed_inputs(items, behaviors, deltas),
        current.embed_inputs(items, behaviors, deltas),
        rank=16,
        compression="exact_svd",
    )
    paired = build_paired_factorized_response_memory(
        current,
        replay,
        probes,
        source_kv_scalars=exact_parent.k.numel() + exact_parent.v.numel(),
    )
    exact = build_full_exact_response_memory(current, exact_current, exact_parent, probes)
    for observed, expected in zip(paired.layers, exact.layers, strict=True):
        assert torch.equal(observed.current_positive_mask, expected.current_positive_mask)
        assert torch.equal(observed.parent_positive_mask, expected.parent_positive_mask)
        assert torch.allclose(observed.base, expected.base, atol=2e-4, rtol=2e-4)
        assert torch.allclose(observed.linear, expected.linear, atol=4e-4, rtol=4e-4)
    assert paired.stored_scalars == exact.stored_scalars


def test_medium_cost_ledger_counts_sidecar_reader_and_rejects_budget() -> None:
    paired = medium_functional_boundary_cost("paired_r4_functional_moments")
    single = medium_functional_boundary_cost("single_current_r8_functional_moments")
    oracle = medium_functional_boundary_cost("full_exact_functional_moment_oracle")
    paired_matrix_free = medium_functional_boundary_cost(
        "paired_r4_functional_moments", matrix_free_initial=True
    )
    single_matrix_free = medium_functional_boundary_cost(
        "single_current_r8_functional_moments", matrix_free_initial=True
    )
    assert paired.sidecar_scalars == 38_016
    assert paired.sidecar_fp32_bytes == 152_064
    assert paired.incremental_reader_flops_per_query == 77_184
    assert paired.mask_sign_comparisons == 589_824
    assert paired_matrix_free.initial_sin_cos_evaluations == 65_536
    assert paired_matrix_free.initial_gaussian_draws == 3_072
    assert paired.mask_and_moment_flops > 0
    assert paired.anchor_probe_flops > 0
    assert not paired.within_twenty_percent
    assert not single.within_twenty_percent
    assert paired_matrix_free.total_constructor_flops == 938_047_624
    assert paired_matrix_free.within_twenty_percent
    assert single_matrix_free.total_constructor_flops == 966_314_752
    assert not single_matrix_free.within_twenty_percent
    assert oracle.constructor_fraction > 1.0
