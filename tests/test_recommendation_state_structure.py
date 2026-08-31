from __future__ import annotations

import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from probe_recommendation_state_structure import (  # noqa: E402
    candidate_panel,
    pairing_record,
    pairings,
    spectral_metrics,
)


def test_semantic_pairings_are_partitions_and_increase_item_matches() -> None:
    # Equal items are deliberately 64 positions apart, so positional pairs do
    # not receive an accidental same-item advantage.
    items = np.concatenate([np.arange(1, 65), np.arange(1, 65)]).astype(np.int64)
    actions = np.tile(np.asarray([1, 2], dtype=np.int64), 64)
    observed = pairings(items, actions, action_slots=5)
    for pairs in observed.values():
        assert pairs.shape == (64, 2)
        assert sorted(pairs.reshape(-1).tolist()) == list(range(128))
    positional = pairing_record("edge", 1, "pos", observed["positional_pairs"], items, actions)
    semantic = pairing_record("edge", 1, "item", observed["same_item_pairs"], items, actions)
    assert positional["same_item_pair_fraction"] == 0.0
    assert semantic["same_item_pair_fraction"] == 1.0


def test_spectral_metrics_identify_candidate_shared_rank_one() -> None:
    base = torch.linspace(0.1, 1.0, 32)
    scales = torch.linspace(0.5, 2.0, 64)
    matrix = (scales[:, None] * base[None, :]).repeat(3, 1, 1)
    metrics = spectral_metrics(matrix)
    assert np.all(metrics["rank90"] == 1)
    assert np.allclose(metrics["top_direction_energy_fraction"], 1.0, atol=1e-5)
    assert np.allclose(metrics["effective_rank"], 1.0, atol=1e-5)


def test_candidate_panel_keeps_fixed_width_without_negative_semantics() -> None:
    first = np.arange(1, 513, dtype=np.int64)
    second = np.arange(10_001, 10_513, dtype=np.int64)
    panels, modes, audit = candidate_panel(np.stack([first, second]))
    assert panels.shape == (2, 64)
    assert modes.shape == (2, 64)
    for row in modes:
        values, counts = np.unique(row, return_counts=True)
        observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
        assert observed == {
            "novel_to_prefix": 32,
            "old_only_repeat": 16,
            "recent_repeat": 16,
        }
    assert audit["minimum_selected_recent"] == 16
    assert audit["minimum_selected_old"] == 16
    assert audit["maximum_selected_novel"] == 32
