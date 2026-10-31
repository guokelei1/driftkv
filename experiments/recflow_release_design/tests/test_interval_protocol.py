import json
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments.recflow_release_design.code.interval_protocol import (
    SampledRequestBatches,
    ancestry,
    prepare_selection,
)


def checkpoint(root, name, day, parent=None, **metadata):
    path = root / f"{name}.pt"
    path.touch()
    content = {"days": [day], "scale": "max", "proposal": {"history": 1024},
               "quality_use_permitted": True,
               "arguments": {"parent": str(parent) if parent else None}, **metadata}
    path.with_suffix(".json").write_text(json.dumps({"status": "committed", "metadata": content}))
    return path


@pytest.fixture
def population(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    np.savez(root / "population.npz", train=np.arange(10, dtype=np.int32), validation=[10])
    dtype = [("uid", "i4"), ("ts", "i8"), ("rid", "i8"), ("start", "i8"),
             ("stop", "i8"), ("history_stop", "i8"), ("history_length", "i2"), ("day", "u1")]
    rows = np.zeros(12, dtype=dtype)
    rows["uid"] = [0, 1, 2, 3, 0, 2, 3, 1, 2, 0, 3, 1]
    rows["rid"] = np.arange(len(rows)) + 100
    rows["day"] = 33
    rows["ts"] = 10_000
    rows["start"], rows["stop"] = 2 * np.arange(len(rows)), 2 * np.arange(len(rows)) + 2
    rows["history_stop"], rows["history_length"] = 3, np.arange(len(rows)) % 3
    np.save(root / "full.npy", rows)
    (root / "selection_manifest.json").write_text(json.dumps({
        "datasets": {"2:33": {"file": "full.npy", "requests": len(rows)}}}))
    history = root / "history"
    history.mkdir()
    events = np.zeros(3, dtype=[("features", "i4", (5,)), ("behavior", "i1"), ("ts", "i8")])
    events["features"][:, 0] = [1, 2, 3]
    events["ts"] = [1000, 2000, 3000]
    np.save(history / "events.npy", events)
    stage = root / "stage_v2" / "d33"
    stage.mkdir(parents=True)
    features = np.zeros((2 * len(rows), 5), dtype=np.int32)
    features[:, 0] = np.arange(len(features)) + 1
    np.save(stage / "features.npy", features)
    labels = np.tile([1, -1], len(rows))
    labels[:4] = 0  # Zero-positive requests must survive selection too.
    np.save(stage / "labels.npy", labels)
    return root, rows


def test_ancestry_keeps_unpublished_intermediate_and_actual_parent(tmp_path):
    m3 = checkpoint(tmp_path, "m3", 30)
    m4 = checkpoint(tmp_path, "m4", 31, m3)
    m5 = checkpoint(tmp_path, "m5", 32, m4)
    result = ancestry(m5, m3)
    assert [r["checkpoint"] for r in result] == list(map(str, [m3, m4, m5]))
    assert [r["train_days"] for r in result] == [[30], [31], [32]]
    assert result[-1]["actual_parent"] == str(m4)
    wrong = checkpoint(tmp_path, "unrelated", 29)
    with pytest.raises(ValueError, match="not an actual training ancestor"):
        ancestry(m5, wrong)
    with pytest.raises(ValueError, match="earlier ancestor"):
        ancestry(m5, m5)


def test_ancestry_rejects_future_parent_and_diagnostic_checkpoint(tmp_path):
    parent = checkpoint(tmp_path, "parent", 33)
    child = checkpoint(tmp_path, "child", 32, parent)
    with pytest.raises(ValueError, match="cutoff is later"):
        ancestry(child, parent)
    parent = checkpoint(tmp_path, "parent", 31, quality_use_permitted=False)
    with pytest.raises(ValueError, match="Diagnostic"):
        ancestry(child, parent)


def test_selection_preserves_all_requests_and_inactive_chosen_users(population, tmp_path):
    root, rows = population
    result = prepare_selection(root, 33, tmp_path / "sample", user_count=4)
    permutation = np.load(result["uid_permutation_path"])
    chosen = np.load(result["chosen_uids_path"])
    selected = np.load(result["selection_path"])
    assert np.array_equal(np.sort(permutation), np.arange(10))
    assert np.array_equal(chosen, permutation[:4])
    assert np.array_equal(selected, rows[np.isin(rows["uid"], chosen)])
    assert not np.isin(selected["uid"], np.setdiff1d(np.arange(10), chosen)).any()
    assert result["chosen_users"] == 4
    assert result["active_users"] < result["chosen_users"]
    assert result["activity_or_feedback_selection"] is False
    assert json.loads(json.dumps(result)) == result
    larger = prepare_selection(root, 33, tmp_path / "larger", user_count=8)
    assert np.array_equal(chosen, np.load(larger["chosen_uids_path"])[:4])
    other = prepare_selection(root, 33, tmp_path / "other", user_count=4, user_offset=4)
    assert not np.intersect1d(chosen, np.load(other["chosen_uids_path"])).size
    with pytest.raises(FileExistsError):
        prepare_selection(root, 33, tmp_path / "sample", user_count=4)


def worker_read(dataset):
    batch = dataset[list(range(len(dataset)))]
    return (str(dataset._selection_path), dataset.records["rid"].tolist(),
            batch["counts"].tolist(), batch["labels"].tolist())


def test_sample_pickle_and_spawn_reload_sample_not_full_population(population, tmp_path):
    root, _ = population
    result = prepare_selection(root, 33, tmp_path / "sample", user_count=4)
    dataset = SampledRequestBatches(root, 33, result["selection_path"])
    expected = worker_read(dataset)
    restored = pickle.loads(pickle.dumps(dataset))
    assert isinstance(restored.records, np.memmap)
    assert Path(restored.records.filename) == Path(result["selection_path"])
    assert worker_read(restored) == expected
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(worker_read, (dataset,)) == expected


def test_partial_user_day_is_rejected(population, tmp_path):
    root, rows = population
    partial = tmp_path / "partial.npy"
    np.save(partial, rows[:1])
    with pytest.raises(ValueError, match="every chosen user's"):
        SampledRequestBatches(root, 33, partial)
