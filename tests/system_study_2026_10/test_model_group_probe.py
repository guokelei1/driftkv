import numpy as np

from system_study_2026_10.model_group_probe import select_extremes


def test_selection_ignores_future_rows_and_release_timestamp_ties():
    def events(times):
        return np.array(times), np.arange(len(times)), np.ones(len(times))
    histories = {7: events([1, 2, 10, 11, 12, 13]),
                 9: events([1, 2, 3]), 5: events([1, 10, 12])}
    selected, profiles = select_extremes(histories, 10, 1024)
    assert [(row["uid"], row["length"]) for row in selected] == [(5, 1), (9, 3)]
    assert {row["uid"]: row["length"] for row in profiles}[7] == 2


def test_full_length_ties_keep_two_distinct_users_without_truncating_input():
    histories = {uid: (np.arange(20), np.arange(20), np.ones(20)) for uid in (9, 3, 7)}
    selected, _ = select_extremes(histories, 20, 8)
    assert [(row["uid"], row["length"]) for row in selected] == [(3, 8), (7, 8)]
