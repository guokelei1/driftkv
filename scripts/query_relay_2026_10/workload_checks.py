"""CPU checks for the new candidate workload's scientific boundaries."""
import numpy as np

from workload import candidate_rows, trigger_indices


def main():
    releases = (("v3", 10), ("v4", 20))
    events = (np.array([0, 5, 10, 10, 11, 12, 13, 20, 21, 25]), np.arange(1, 11))
    args = dict(stride=2, candidates=4, known_vocab_size=12, releases=releases, end=30)
    indices = trigger_indices(events, stride=2, known_vocab_size=12, releases=releases, end=30)
    assert indices["v3"].tolist() == [2, 4, 6]
    assert indices["v4"].tolist() == [7, 9]
    rows = candidate_rows(99, events, **args)
    assert rows == candidate_rows(99, events, **args)
    assert len(rows) == 20
    for offset in range(0, len(rows), 4):
        group = rows[offset:offset + 4]
        assert sum(row["label"] for row in group) == 1
        assert len({row["item_idx"] for row in group}) == 4
        assert len({row["round_id"] for row in group}) == 1
        assert all(np.searchsorted(events[0], row["query_timestamp"], side="left") > 0 for row in group)
    # Timestamp ties are excluded together, not disclosed through row order.
    first = rows[0]
    assert first["query_timestamp"] == 10
    assert np.searchsorted(events[0], 10, side="left") == 2
    # Unobserved future extension cannot alter present negatives or triggers.
    extended = (np.r_[events[0], 31, 32], np.r_[events[1], 1, 2])
    assert rows == candidate_rows(99, extended, **args)
    print("workload deterministic candidates/ties/future-extension checks passed")


if __name__ == "__main__":
    main()
