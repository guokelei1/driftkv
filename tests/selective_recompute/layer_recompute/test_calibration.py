from types import SimpleNamespace

import numpy as np
import torch

from scripts.selective_recompute_2026_09.calibrate import profile_batch, snapshot


def test_cutover_snapshot_excludes_release_and_orders_mapped_ties():
    history = SimpleNamespace(rows={7: (
        np.array([1, 2, 2, 4, 5]), np.array([1, 9, 3, 8, 2]), np.array([1, 1, 2, 1, 1]),
    )})
    times, items, behaviors = snapshot(history, 7, cutover=4, max_length=2)
    assert times.tolist() == [2, 2]
    assert items.tolist() == [3, 9]
    assert behaviors.tolist() == [2, 1]


def test_profile_executes_all_intervals_with_full_teacher_endpoint(model_factory, raw_events):
    parent, current = model_factory(17), model_factory(29)
    histories = {}
    for index, uid in enumerate((7, 8)):
        histories[uid] = (np.cumsum(raw_events[2][index].numpy()).astype(np.int64),
                          raw_events[0][index].numpy(), raw_events[1][index].numpy())
    actual = profile_batch(parent, current, histories, [7, 8], cutover=30,
                           known_items=40, query_count=3, device=torch.device("cpu"))
    assert set(actual) == {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    assert actual[(0, 2)] < 1e-10
    assert any(error > 1e-8 for error in actual.values())
