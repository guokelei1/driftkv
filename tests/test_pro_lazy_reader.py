from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from broadcast_residual import generate_av_broadcast_residual
from candidate_shared_causal import _cached_prefix_heads
from evaluate_yambda500m_foundation_raw import evaluate_full_cache_cohort
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from one_release_refinement import cast_prefix, parameter_cast_maps
from pro_lazy_cost import (
    exact_lazy_carrier_cost,
    full_recompute_flops,
    pro_cost,
    progressive_pro_cost,
)
from pro_lazy_reader import (
    PRO_PATH,
    build_parent_conditioned_carriers,
    fused_joint_map_prefix_heads,
    generate_lazy_pro_probe_components,
    generate_lazy_pro_sidecar,
)
from progressive_pro import (
    build_progressive_parent_conditioned_carriers,
    combine_two_probe_components,
    progressive_corrections,
    segment_coverage,
)


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=96,
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


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    items = torch.arange(1, 33).reshape(2, 16)
    behaviors = (torch.arange(32).reshape(2, 16) % 2 + 1).long()
    deltas = torch.arange(16).float().repeat(2, 1)
    return items, behaviors, deltas


def test_fused_joint_map_read_matches_materialized_prefix() -> None:
    parent, current = _model(3), _model(5)
    items, behaviors, deltas = _inputs()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    maps = parameter_cast_maps(parent, current)
    mapped = cast_prefix(parent_cache, maps, length=8)
    query = current.embed_query_tokens(torch.tensor([[40], [41]]), torch.zeros(2))
    x_norm = current.blocks[0].norm(query)
    q, _, _ = current.blocks[0].attn._project(x_norm)
    fused = fused_joint_map_prefix_heads(
        current.blocks[0].attn,
        q,
        parent_cache,
        maps[0],
        layer=0,
        length=8,
    )
    reference = _cached_prefix_heads(
        current.blocks[0].attn, q, mapped, layer=0, candidate_count=1
    ).reshape_as(fused)
    assert torch.allclose(fused, reference, rtol=2e-5, atol=2e-5)


def test_lazy_pro_matches_materialized_reference_without_returning_prefix() -> None:
    parent, current = _model(7), _model(11)
    items, behaviors, deltas = _inputs()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    maps = parameter_cast_maps(parent, current)
    carriers, layout = build_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=8,
        carrier_count=2,
    )
    assert carriers.seq_len == 2
    assert carriers.k.shape[2] == 2
    assert layout.old_positions == 8
    assert layout.represented_mass == 4

    mapped_old = cast_prefix(parent_cache, maps, length=layout.old_positions)
    padding = parent_cache.seq_len - layout.old_positions - carriers.seq_len
    zero_k = mapped_old.k.new_zeros(
        mapped_old.k.shape[0], mapped_old.k.shape[1], padding, mapped_old.k.shape[-1]
    )
    zero_v = mapped_old.v.new_zeros(
        mapped_old.v.shape[0], mapped_old.v.shape[1], padding, mapped_old.v.shape[-1]
    )
    materialized_reference = HSTUKVCache(
        k=torch.cat([zero_k, mapped_old.k, carriers.k], dim=2),
        v=torch.cat([zero_v, mapped_old.v, carriers.v], dim=2),
        seq_len=parent_cache.seq_len,
    )
    probe_items = items[:, -1]
    reference = generate_av_broadcast_residual(
        current, materialized_reference, parent_cache, probe_items
    )
    lazy = generate_lazy_pro_sidecar(
        current,
        parent_cache,
        carriers,
        maps,
        probe_items,
        old_positions=layout.old_positions,
    )
    assert lazy.replay_max_abs_error < 1e-6
    assert len(lazy.corrections) == len(reference.corrections) == 2
    for actual, expected in zip(lazy.corrections, reference.corrections, strict=True):
        assert torch.allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_probe_components_exactly_split_old_and_recent_correction() -> None:
    parent, current = _model(23), _model(29)
    items, behaviors, deltas = _inputs()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    maps = parameter_cast_maps(parent, current)
    carriers, layout = build_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=8,
        carrier_count=2,
    )
    probe_items = items[:, -1]
    components = generate_lazy_pro_probe_components(
        current,
        parent_cache,
        carriers,
        maps,
        probe_items,
        old_positions=layout.old_positions,
    )
    legacy = generate_lazy_pro_sidecar(
        current,
        parent_cache,
        carriers,
        maps,
        probe_items,
        old_positions=layout.old_positions,
    )
    for total, old, recent, original in zip(
        components.corrections,
        components.old_corrections,
        components.recent_corrections,
        legacy.corrections,
        strict=True,
    ):
        assert torch.allclose(total, old + recent, rtol=1e-6, atol=1e-6)
        assert torch.allclose(total, original, rtol=1e-6, atol=1e-6)
    assert components.replay_max_abs_error < 1e-6


def test_progressive_carriers_support_unequal_c48_partition() -> None:
    torch.manual_seed(31)
    current = HSTU(
        HSTUConfig(
            num_items=256,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=128,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()
    items = torch.arange(1, 129).reshape(1, 128)
    behaviors = torch.ones_like(items)
    deltas = torch.arange(128).float().reshape(1, 128)
    parent_cache = current.compute_kv(items, behaviors, deltas)
    carriers, layout = build_progressive_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=128,
        carrier_count=48,
    )
    assert carriers.seq_len == 48
    assert layout.carriers == 48
    assert sum(layout.represented_masses) == 128
    assert set(layout.represented_masses) == {2, 3}


def test_two_probe_sidecar_and_segment_coverage_only_update_scalars() -> None:
    parent, current = _model(37), _model(41)
    items, behaviors, deltas = _inputs()
    parent_cache = parent.compute_kv(items, behaviors, deltas)
    maps = parameter_cast_maps(parent, current)
    carriers, layout = build_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=8,
        carrier_count=2,
    )
    components = generate_lazy_pro_probe_components(
        current,
        parent_cache,
        carriers,
        maps,
        items[:, -1],
        old_positions=layout.old_positions,
    )
    sidecar = combine_two_probe_components(components, components)
    assert torch.allclose(sidecar.probe_direction_cosines, torch.ones_like(sidecar.probe_direction_cosines))
    assert torch.allclose(sidecar.probe_norm_ratios, torch.ones_like(sidecar.probe_norm_ratios))
    at_cutover = progressive_corrections(
        sidecar, torch.zeros(items.shape[0]), old_positions=8, recent_positions=8
    )
    for actual, expected in zip(at_cutover, components.corrections, strict=True):
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)

    old, recent = segment_coverage(
        torch.tensor([0, 8, 12, 16]), old_positions=8, recent_positions=8
    )
    assert torch.allclose(old, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(recent, torch.tensor([1.0, 1.0, 0.5, 0.0]))


def test_frozen_lightweight_cost_axis_is_below_twenty_percent() -> None:
    full = full_recompute_flops()
    assert full == 625_475_584
    sixteen = pro_cost(16)
    thirty_two = pro_cost(32)
    assert sixteen["materialized_version_translated_prefix_scalars"] == 0
    assert thirty_two["materialized_version_translated_prefix_scalars"] == 0
    assert sixteen["total_flops_per_user"] < thirty_two["total_flops_per_user"]
    assert sixteen["over_full_fraction"] < 0.10
    assert thirty_two["over_full_fraction"] < 0.10
    assert sixteen["conservative_parent_state_stream_bytes_fp32"] == 5 * 2**20
    assert thirty_two["post_release_coverage_scale_flops_per_user_request"] == 512
    assert thirty_two["post_release_injection_adds_per_candidate"] == 512
    assert exact_lazy_carrier_cost(16)["over_full_fraction"] > 0.20
    assert exact_lazy_carrier_cost(32)["over_full_fraction"] > 0.39


def test_progressive_two_probe_cost_axis_is_ordered_and_bounded() -> None:
    costs = [progressive_pro_cost(carriers) for carriers in (32, 48, 64)]
    fractions = [float(cost["over_full_fraction"]) for cost in costs]
    assert fractions == sorted(fractions)
    assert 0.10 < fractions[0] < 0.15
    assert 0.14 < fractions[1] < 0.16
    assert 0.18 < fractions[2] < 0.20
    assert all(cost["materialized_version_translated_prefix_scalars"] == 0 for cost in costs)
    assert all(cost["sidecar_write_scalars"] == 520 for cost in costs)


def test_full_cache_rolling_cohort_emits_coverage_scaled_pro_path() -> None:
    parent, current = _model(17), _model(19)
    timestamps = np.arange(1, 515, dtype=np.int64)
    items = np.asarray([(index % 90) + 1 for index in range(514)], dtype=np.int64)
    behaviors = np.asarray([(index % 2) + 1 for index in range(514)], dtype=np.int64)
    history = SimpleNamespace(rows={7: (timestamps, items, behaviors)})
    request = {
        "request_id": "pro-rolling-1",
        "uid": 7,
        "query_timestamp": 514,
        "item_idx": 92,
    }
    rows = evaluate_full_cache_cohort(
        uids=[7],
        by_user={7: [request]},
        history=history,
        parent=parent,
        current=current,
        parent_name="v0",
        current_name="v1",
        edge="v0_to_v1",
        checkpoint_hash="current",
        parent_hash="parent",
        manifest_hash="manifest",
        cutover=513,
        lineage_models=[("v0", parent)],
        event_end_exclusive=515,
        include_request_local=False,
        include_parent_exact=True,
        pro_lazy_maps=parameter_cast_maps(parent, current),
        pro_lazy_carriers=32,
        query_chunk_size=16,
    )
    by_path = {row["path"]: row for row in rows}
    assert set(by_path) == {
        "parent_exact_rolling",
        "current_exact_rolling",
        "one_hop_reuse_rolling",
        PRO_PATH,
    }
    assert by_path[PRO_PATH]["rolling_evictions"] == 1
    assert by_path[PRO_PATH]["hstu_logit"] != by_path["one_hop_reuse_rolling"]["hstu_logit"]
