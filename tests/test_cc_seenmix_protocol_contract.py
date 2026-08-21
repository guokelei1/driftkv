from __future__ import annotations

import random

import pytest

from hstu_kvcache.data import (
    SeenMixQuotas,
    build_history_matched_panel,
    build_seenmix_panel,
    match_key,
    seenmix_quota_grid,
    split_seen_pools,
)


def _history() -> list[tuple[int, int, int]]:
    # 40 old-only items, then 8 recent items that do not overlap them.
    old = [(item, 1000 + item, 1) for item in range(10, 50)]
    recent = [(item, 5000 + item, 1) for item in range(50, 58)]
    return old + recent


def test_split_seen_pools_are_disjoint_and_exclude_target() -> None:
    pools = split_seen_pools(_history(), target_item=50, recent_window=8, max_history=48)
    assert pools.target_stratum == "recent_seen"
    assert 50 not in pools.recent_items
    assert 50 not in pools.old_items
    assert set(pools.recent_items).isdisjoint(pools.old_items)
    assert 10 in pools.old_items
    assert 57 in pools.recent_items


def test_old_only_target_stratum_and_unseen_target() -> None:
    old_only = split_seen_pools(_history(), target_item=11, recent_window=8, max_history=48)
    unseen = split_seen_pools(_history(), target_item=999, recent_window=8, max_history=48)
    assert old_only.target_stratum == "old_only"
    assert 11 not in old_only.old_items
    assert unseen.target_stratum == "unseen"


def test_seenmix_panel_uses_fixed_quotas_without_cross_stratum_backfill() -> None:
    quotas = SeenMixQuotas(m_recent=4, m_old=8, m_discovery=87)
    pools = split_seen_pools(_history(), target_item=999, recent_window=8, max_history=48)
    discovery = list(range(1000, 1200))
    panel = build_seenmix_panel(
        pools, discovery, quotas, rng=random.Random(1), target_item=999, inject_target=True
    )
    assert panel.complete
    assert panel.item_ids[0] == 999
    assert panel.roles[0] == "target"
    assert panel.roles.count("recent_seen") == 4
    assert panel.roles.count("old_seen") == 8
    assert panel.roles.count("discovery") == 87
    assert set(panel.item_ids[1:]).isdisjoint({999})
    assert set(item for item, role in zip(panel.item_ids, panel.roles) if role == "discovery").isdisjoint(
        set(pools.recent_items) | set(pools.old_items)
    )


def test_short_old_pool_is_incomplete_and_not_backfilled() -> None:
    quotas = SeenMixQuotas(m_recent=4, m_old=24, m_discovery=71)
    pools = split_seen_pools(_history()[-10:], target_item=999, recent_window=8, max_history=48)
    panel = build_seenmix_panel(
        pools,
        list(range(2000, 2200)),
        quotas,
        rng=random.Random(2),
        target_item=999,
        inject_target=True,
    )
    assert not panel.complete
    assert panel.missing["old"] > 0
    assert panel.roles.count("old_seen") < 24
    assert panel.roles.count("discovery") <= 71


def test_fidelity_panel_does_not_inject_target() -> None:
    quotas = SeenMixQuotas(m_recent=4, m_old=8, m_discovery=87)
    pools = split_seen_pools(_history(), target_item=12, recent_window=8, max_history=48)
    panel = build_seenmix_panel(
        pools, list(range(3000, 3200)), quotas, rng=random.Random(3), target_item=12, inject_target=False
    )
    assert panel.complete
    assert 12 not in panel.item_ids
    assert "target" not in panel.roles
    assert len(panel.item_ids) == 99


def test_quota_grid_is_coverage_ordered_and_keeps_discovery_floor() -> None:
    grid = seenmix_quota_grid()
    assert grid[0].m_old >= grid[-1].m_old
    assert all(option.m_discovery >= 32 for option in grid)
    assert all(option.competitors == 99 for option in grid)


def test_history_matched_panel_keeps_same_key_only() -> None:
    target_key = match_key(
        stratum="old_only",
        item_count=3,
        recency_seconds=100_000,
        familiarity="old_artist",
        global_count=80,
    )
    other_key = match_key(
        stratum="unseen",
        item_count=0,
        recency_seconds=None,
        familiarity="unseen_artist",
        global_count=5,
    )
    items = [11, 12, 13, 14, 15]
    keys = [target_key, other_key, target_key, target_key, other_key]
    panel = build_history_matched_panel(
        99, target_key, items, keys, competitor_slots=3, rng=random.Random(4)
    )
    assert panel.complete
    assert panel.item_ids[0] == 99
    assert set(panel.item_ids[1:]) == {11, 13, 14}


def test_invalid_quota_sum_is_rejected() -> None:
    with pytest.raises(ValueError):
        SeenMixQuotas(m_recent=10, m_old=10, m_discovery=10)
