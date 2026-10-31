import numpy as np

from experiments.recflow_request_ranking.code.index_data import INDEX_DTYPE
from experiments.recflow_request_ranking.code.sampling import stratified_indices


def test_nested_budgets_cover_user_days_without_looking_at_labels():
    rows = np.zeros(120, INDEX_DTYPE)
    rows["day"] = np.repeat([24, 25, 26], 40)
    rows["uid"] = np.tile(np.repeat([1, 2, 3, 4], 10), 3)
    rows["rid"] = np.arange(120)
    small, large = stratified_indices(rows, 24), stratified_indices(rows, 60)
    assert len(small) == 24 and len(large) == 60
    assert set(small) < set(large)
    assert len({(int(r["uid"]), int(r["day"])) for r in rows[small]}) == 12
    rows["positive"] = np.arange(120)
    assert np.array_equal(small, stratified_indices(rows, 24))


def test_deterministic_after_input_order_changes():
    rows = np.zeros(100, INDEX_DTYPE)
    rows["uid"], rows["day"], rows["rid"] = np.arange(100) % 7, 24, np.arange(100)
    selected = set(rows["rid"][stratified_indices(rows, 41)])
    reversed_rows = rows[::-1].copy()
    assert selected == set(reversed_rows["rid"][stratified_indices(reversed_rows, 41)])
