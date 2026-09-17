import json

import numpy as np
import pytest
from design.competitor_data import history_arrays, make_candidate_panel, read_uid_split
from insight_one_locality import common

from hstu_kvcache.training.foundation import FoundationHistoryIndex


def test_uid_split_keeps_explicit_order_without_old_population_restriction(tmp_path):
    split = {"evaluation": [30_002, 30_001], "fit": [9], "selection": [8]}
    path = tmp_path / "uids.json"
    path.write_text(json.dumps(split))
    assert read_uid_split(path) == split


@pytest.mark.parametrize("split", [
    {"evaluation": [1, 1]},
    {"evaluation": [1], "fit": [1]},
    {"evaluation": [1], "fit": [2], "selection": [2]},
])
def test_uid_split_rejects_duplicates_and_cross_group_leakage(tmp_path, split):
    path = tmp_path / "uids.json"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="duplicate|disjoint"):
        read_uid_split(path)


def test_history_uses_strict_cutover_static_start_and_real_adjacent_deltas():
    timestamps = np.array([1, 3, 6, 8, 10, 10, 12])
    history = FoundationHistoryIndex({7: (timestamps, np.arange(1, 8), np.ones(7, dtype=np.int64))})
    times, items, behaviors, deltas, query_deltas = history_arrays(history, np.array([7]), 10, 3)
    np.testing.assert_array_equal(times, [[3, 6, 8]])
    np.testing.assert_array_equal(items, [[2, 3, 4]])
    np.testing.assert_array_equal(behaviors, [[1, 1, 1]])
    np.testing.assert_array_equal(deltas, [[0, 3, 2]])
    np.testing.assert_array_equal(query_deltas, [2])
    # Both complete target-timestamp rows and the later row are excluded.
    with pytest.raises(ValueError, match="lacks a full"):
        history_arrays(history, np.array([7]), 10, 5)


def test_history_layout_matches_existing_static_reference():
    length = common.HISTORY
    timestamps = np.cumsum(np.arange(length + 3) % 5 + 1)
    history = FoundationHistoryIndex({11: (timestamps, np.arange(1, length + 4),
                                         np.ones(length + 3, dtype=np.int64))})
    cutover = int(timestamps[-2])
    actual = history_arrays(history, np.array([11]), cutover, length)
    expected = common.histories_at_cutover(history, np.array([11]), cutover)
    for values, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(values, reference)


def _panel_histories(offset=0):
    # Independent pre-cutover histories supply enough novel bank items even
    # though only a short 320-event window is needed to exercise old/recent.
    rows = np.stack([np.arange(1, 321), np.arange(401, 721), np.arange(801, 1121)]) + offset
    # Shared frequent items test bank popularity ordering, then item-ID ties.
    rows[:, :8] = np.arange(1501, 1509) + offset
    return rows


def test_candidate_panel_matches_old_policy_without_mutating_its_globals():
    items = _panel_histories()
    before = common.KNOWN_ITEMS
    expected_panels, expected_modes, _ = common.candidate_panel(items)
    panels, modes = make_candidate_panel(items, known_items=before)
    np.testing.assert_array_equal(panels, expected_panels)
    np.testing.assert_array_equal(modes, expected_modes)
    assert common.KNOWN_ITEMS == before


def test_large_vocabulary_ids_are_candidates_and_oov_is_excluded():
    offset = common.KNOWN_ITEMS + 100
    known_items = offset + 2000
    items = _panel_histories(offset)
    items[:, -1] = known_items + 3  # An OOV bucket remains history, never candidate.
    panels, modes = make_candidate_panel(items, known_items=known_items)
    assert panels.shape == modes.shape == (3, 64)
    assert (panels > common.KNOWN_ITEMS).all()
    assert (panels < known_items).all()
    for row, panel, mode in zip(items, panels, modes, strict=True):
        assert len(set(panel.tolist())) == 64
        assert set(panel[mode == 2]).isdisjoint(row.tolist())
    # The popularity bank must span evaluation users, not individual batches.
    with pytest.raises(ValueError, match="cannot be filled"):
        make_candidate_panel(items[:1], known_items=known_items)
