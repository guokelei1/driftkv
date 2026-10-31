import copy
import sys
from pathlib import Path

import pytest

from experiments.ch3_motivation.code.analyze_rule_bridge import paired_lifetimes


def fixture():
    rows = []
    for uid in range(512):
        for label in (0, 1):
            for method in ("reuse", "rebuild", "layer_1layers"):
                rows.append(dict(uid=uid, query_timestamp=1, raw_item_id=label,
                    item_idx=label, label=label, method=method, score=float(label)))
    return rows


def test_same_uid_draw_exactly_cancels_identical_lifetimes():
    rows = fixture()
    result = paired_lifetimes(rows, copy.deepcopy(rows), list(range(512)), replicates=20)
    assert all(row["request_local_minus_persistent_auc"] == 0 for row in result["results"])
    assert all(row["request_local_minus_persistent_auc_ci95"] == [0, 0] for row in result["results"])
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["method"] == "layer_1layers":
            row["score"] *= -1
    result = paired_lifetimes(rows, changed, list(range(512)), replicates=20)
    layer = next(row for row in result["results"] if row["method"] == "layer_1layers")
    assert layer["request_local_minus_persistent_auc_ci95"] == [-1, -1]


def test_bridge_rejects_unpaired_feedback_or_small_quality_population():
    rows = fixture()
    with pytest.raises(ValueError, match="same feedback"):
        paired_lifetimes(rows, rows[:-1], list(range(512)), replicates=2)
    with pytest.raises(ValueError):
        paired_lifetimes(rows, rows, list(range(8)), replicates=2)
