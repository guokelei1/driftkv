from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from hstu_kvcache.models import HSTU, HSTUConfig
from one_release_refinement import (
    _cast_slice,
    _cast_values_at,
    build_evidence_measure_basis_cache,
    parameter_cast_maps,
)


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=64,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=32,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def test_evidence_measure_basis_has_matched_layout_and_signed_pair_values() -> None:
    parent, current = _model(7), _model(11)
    items = torch.arange(1, 13).view(1, 12)
    behaviors = (torch.arange(12).view(1, 12) % 2) + 1
    deltas = torch.arange(12).view(1, 12).float()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    maps = parameter_cast_maps(parent, current)

    basis, layout = build_evidence_measure_basis_cache(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        cast_maps=maps,
        repair_width=8,
    )
    assert layout.nominal_positions == 12
    assert layout.cast_positions == 4
    assert layout.repair_evidence == 8
    assert layout.carriers == 4
    assert layout.padding_positions == 4
    assert basis.k.shape == parent_cache.k.shape
    assert basis.v.shape == parent_cache.v.shape
    assert torch.count_nonzero(basis.k[:, :, :4]) == 0
    assert torch.count_nonzero(basis.v[:, :, :4]) == 0

    # Four value-only maps cost the same as the two old-prefix joint maps that
    # were reallocated.  The other two old positions retain Parent values.
    translated_old = _cast_slice(parent_cache, maps, 2, 4)
    prefix_k = torch.cat([parent_cache.k[:, :, :2], translated_old.k], dim=2)
    prefix_v = torch.cat([parent_cache.v[:, :, :2], translated_old.v], dim=2)
    repair_endpoints = torch.tensor([1, 3, 5, 7])
    embedded = current.embed_inputs(
        items[:, -8:].index_select(1, repair_endpoints),
        behaviors[:, -8:].index_select(1, repair_endpoints),
        deltas[:, -8:].index_select(1, repair_endpoints),
    )
    from hstu_kvcache.models import HSTUKVCache

    prefix = HSTUKVCache(k=prefix_k, v=prefix_v, seq_len=4)
    _, compact = current.forward_with_cache_embedded(prefix, embedded)
    earlier_positions = torch.tensor([4, 6, 8, 10])
    expected_shared = _cast_values_at(parent_cache, maps, earlier_positions)
    expected_values = compact.v[:, :, 4:] + expected_shared

    assert torch.allclose(basis.k[:, :, 4:8], prefix_k)
    assert torch.allclose(basis.v[:, :, 4:8], prefix_v)
    assert torch.allclose(basis.k[:, :, 8:], compact.k[:, :, 4:], atol=1e-6)
    assert torch.allclose(basis.v[:, :, 8:], expected_values, atol=1e-6)


def test_singleton_group_does_not_double_its_value_mass() -> None:
    parent, current = _model(13), _model(17)
    items = torch.arange(1, 8).view(1, 7)
    behaviors = torch.ones_like(items)
    deltas = torch.arange(7).view(1, 7).float()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    maps = parameter_cast_maps(parent, current)
    basis, layout = build_evidence_measure_basis_cache(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        cast_maps=maps,
        repair_width=5,
    )
    assert layout.carriers == 3
    # The final committed carrier is a singleton; its value equals the Current
    # dependency-closed carrier rather than Current plus a duplicate base.
    old = 2
    prefix = _cast_slice(parent_cache, maps, 1, old)
    from hstu_kvcache.models import HSTUKVCache

    mixed = HSTUKVCache(
        k=torch.cat([parent_cache.k[:, :, :1], prefix.k], dim=2),
        v=torch.cat([parent_cache.v[:, :, :1], prefix.v], dim=2),
        seq_len=old,
    )
    endpoints = torch.tensor([1, 3, 4])
    embedded = current.embed_inputs(
        items[:, -5:].index_select(1, endpoints),
        behaviors[:, -5:].index_select(1, endpoints),
        deltas[:, -5:].index_select(1, endpoints),
    )
    _, compact = current.forward_with_cache_embedded(mixed, embedded)
    assert torch.allclose(basis.v[:, :, -1], compact.v[:, :, -1], atol=1e-6)
