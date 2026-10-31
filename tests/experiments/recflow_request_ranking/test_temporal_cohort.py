import json

import numpy as np
import pytest

from experiments.recflow_request_ranking.code.evaluate_temporal_cohort import compare_parity
from experiments.recflow_request_ranking.code.temporal_cohort import (
    TemporalCohortBatches,
    evaluation_mask,
    select_users,
    validate_checkpoint_pair,
)


def population():
    return {"uids": np.arange(20), "role": np.r_[np.zeros(16), np.ones(4)],
            "initial_history_count": np.full(20, 1024), "cache_fit_uids": np.array([2, 4]),
            "cache_validation_uids": np.array([6])}


def test_fixed_selection_excludes_calibration_and_is_order_independent():
    original = population()
    users = select_users(original, 8)
    reversed_pop = {k: (v[::-1] if k in ["uids", "role", "initial_history_count"] else v)
                    for k, v in original.items()}
    assert np.array_equal(users, select_users(reversed_pop, 8))
    assert len(users) == 8 and len(np.intersect1d(users, [2, 4, 6, 16, 17, 18, 19])) == 0


def records():
    value = np.zeros(5, dtype=[("uid", "i4"), ("role", "i1"), ("resident", "?"),
                               ("valid_request", "?"), ("observed", "i4"), ("positive", "i4"),
                               ("history_length", "i2")])
    value["uid"] = [10, 10, 11, 12, 10]
    value["resident"], value["valid_request"], value["history_length"] = True, True, 1024
    value["observed"] = [0, 2, 1, 1, 1]
    value["positive"] = [0, 1, 0, 1, 1]
    value["valid_request"][-1] = False
    return value


def test_request_population_never_depends_on_exposure_or_positive_feedback():
    value = records()
    keep = evaluation_mask(value, [10, 11])
    assert keep.tolist() == [True, True, True, False, False]
    value["observed"], value["positive"] = 0, 0
    assert np.array_equal(keep, evaluation_mask(value, [10, 11]))


def test_actual_reader_keeps_no_feedback_requests_and_rejects_fake_long_history(tmp_path):
    stage = tmp_path / "stage/d28"
    stage.mkdir(parents=True)
    (tmp_path / "stage_index_manifest.json").write_text(json.dumps({"stage_dir": "stage"}))
    np.savez(tmp_path / "population.npz", uids=np.array([10, 11]))
    value = records()
    np.save(stage / "index.npy", value)
    data = TemporalCohortBatches(tmp_path, [28], tmp_path / "population.npz")
    assert len(data) == 3 and data.records["observed"].tolist() == [0, 2, 1]
    value["history_length"][0] = 100
    np.save(stage / "index.npy", value)
    with pytest.raises(ValueError, match="1024"):
        TemporalCohortBatches(tmp_path, [28], tmp_path / "population.npz")


@pytest.mark.parametrize("fault", [None, "future", "parent", "diagnostic"])
def test_real_parent_and_future_cutoff_are_required(fault):
    current = {"parent_checkpoint_sha256": "old", "days": [27], "scale": "max", "quality_use_permitted": True}
    parent = {"days": [24, 25, 26], "scale": "max", "quality_use_permitted": True}
    case = {"day": 28, "checkpoint_sha256": "new", "parent_sha256": "old"}
    if fault == "future":
        current["days"] = [28]
    elif fault == "parent":
        current["parent_checkpoint_sha256"] = "unrelated"
    elif fault == "diagnostic":
        parent["quality_use_permitted"] = False
    if fault:
        with pytest.raises(ValueError):
            validate_checkpoint_pair(current, parent, "new", "old", case)
    else:
        validate_checkpoint_pair(current, parent, "new", "old", case)


@pytest.mark.parametrize("fault", [None, "candidates", "scores"])
def test_parity_rejects_changed_inputs_and_model_scores(tmp_path, fault):
    fields = {"uid": np.arange(16), "rid": np.arange(16), "day": np.full(16, 28),
              "counts": np.full(16, 2), "raw_video_ids": np.tile([1, 2], (16, 1)),
              "labels": np.tile([-1, 1], (16, 1)),
              **{name: np.ones((16, 2), np.float32) for name in ["current", "parent", "reuse"]}}
    for name in ["old", "new"]:
        destination = tmp_path / name / "rank_0_scores"
        destination.mkdir(parents=True)
        if name == "new" and fault == "candidates":
            fields["raw_video_ids"][0, 0] = 99
        if name == "new" and fault == "scores":
            fields["reuse"][0, 0] += 1e-3
        np.savez(destination / "batch_000000.npz", **fields)
        (destination.parent / "rank_0.csv").write_text("uid,current_ndcg@10\n0,1.0\n")
    if fault:
        with pytest.raises(ValueError):
            compare_parity(tmp_path / "new", tmp_path / "old", 1)
    else:
        result = compare_parity(tmp_path / "new", tmp_path / "old", 1)
        assert result["score_max_abs"] == {"current": 0., "parent": 0., "reuse": 0.}
        assert result["quality_use_permitted"] is False
