"""Synthetic CPU entry-point checks: real six-layer execution, mocked I/O."""

import csv
import json
from dataclasses import asdict

import numpy as np
import pyarrow as pa
import torch
from design import run_insight1_competitors as entry
from insight_one_locality.adjudicate import sigmoid

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(seed):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(
        num_items=96, num_behaviors=3, hidden_size=12, num_layers=6,
        num_heads=3, max_seq_len=8, input_dropout=0.2, attn_dropout=0.1,
    )).eval()


def _arrays(uids):
    uids = np.asarray(uids, dtype=np.int64)
    positions = np.arange(8)
    items = (uids[:, None] * 3 + positions[None, :] * 5) % 90 + 1
    behaviors = np.broadcast_to(positions % 2 + 1, items.shape).copy()
    deltas = (uids[:, None] % 7 + positions[None, :]).astype(np.float32)
    deltas[:, 0] = 0
    return (uids, items, behaviors, deltas, (uids % 11 + 1).astype(np.float32))


def test_complete_run_uses_separate_calibration_and_aggregates_batches(tmp_path, monkeypatch):
    parent, current = _model(17), _model(23)
    population = np.arange(10_000, 10_005)
    candidates = np.broadcast_to(np.arange(16, 80), (len(population), 64)).copy()
    modes = np.zeros_like(candidates)
    pair = {
        "scale": "medium", "edge": "theta0_to_theta1", "cutover": 1000,
        "config": asdict(current.cfg),
        "dataset": {"manifest": "synthetic/dataset.json", "users": "synthetic/users.parquet",
                    "known_items": 97, "oov_buckets": 0},
        "parent": {"checkpoint": "synthetic/v0.pt"},
        "current": {"checkpoint": "synthetic/v1.pt"},
        "admission": {"reuse_eligible": True},
    }
    pair_calls = []

    def load_model_pair(scale, edge_index, *, verify_hashes):
        pair_calls.append((scale, edge_index, verify_hashes))
        return pair

    monkeypatch.setattr(entry, "load_model_pair", load_model_pair)

    def read_table(path, *, columns):
        assert str(path).endswith("synthetic/users.parquet")
        assert columns == ["uid"]
        return pa.table({"uid": population.tolist() + [1, 2, 3, 4]})

    monkeypatch.setattr(entry.pq, "read_table", read_table)

    loaded_uids, history_groups = [], []

    def load_histories(uids, **kwargs):
        loaded_uids.extend(uids)
        return object()

    def history_arrays(history, uids, cutover, history_length):
        assert cutover == pair["cutover"]
        assert history_length == current.cfg.max_seq_len
        history_groups.append(np.asarray(uids).tolist())
        return _arrays(uids)

    def load_model(path, device):
        assert device.type == "cpu"
        model = parent if path.name == "v0.pt" else current
        return model, {"config": asdict(model.cfg)}

    panel_inputs = []

    def make_candidate_panel(items, known_items):
        assert known_items == pair["dataset"]["known_items"]
        panel_inputs.append(items.copy())
        return candidates, modes

    monkeypatch.setattr(entry, "load_histories", load_histories)
    monkeypatch.setattr(entry, "history_arrays", history_arrays)
    monkeypatch.setattr(entry, "make_candidate_panel", make_candidate_panel)
    monkeypatch.setattr(entry, "load_model", load_model)
    real_fit, fitted = entry.fit, []

    def fit(source, target, **kwargs):
        for actual, model, uids in (
            (source, parent, [1, 2]), (target, current, [1, 2]),
            (kwargs["selection_source"], parent, [3, 4]),
            (kwargs["selection_target"], current, [3, 4]),
        ):
            expected = model.compute_kv(*entry.history_tensors(_arrays(uids), 0, len(uids), "cpu"))
            torch.testing.assert_close(actual.k, expected.k)
            torch.testing.assert_close(actual.v, expected.v)
        mapper = real_fit(source, target, **kwargs)
        fitted.append(mapper)
        return mapper

    monkeypatch.setattr(entry, "fit", fit)
    real_evaluate, batches = entry.evaluate_batch, []

    def evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        batches.append({name: value.cpu().numpy().copy() for name, value in result.items()})
        return result

    monkeypatch.setattr(entry, "evaluate_batch", evaluate)
    uid_path = tmp_path / "uids.json"
    uid_path.write_text(json.dumps({
        "evaluation": population.tolist(), "fit": [1, 2], "selection": [3, 4],
    }))
    output = tmp_path / "new_diagnostic"
    args = entry.build_parser().parse_args([
        "--scale", "medium", "--edge-index", "0", "--device", "cpu",
        "--max-users", "3", "--eval-offset", "2",
        "--batch-size", "2", "--candidate-chunk", "16", "--torch-threads", "2",
        "--history-threads", "1", "--layer-intervals", "2:3", "--tail-lengths", "2",
        "--map-ks", "1", "--uids", str(uid_path), "--output", str(output),
    ])
    summary = entry.run(args)
    assert pair_calls == [("medium", 0, False), ("medium", 0, True)]
    assert sorted(loaded_uids) == sorted(population.tolist() + [1, 2, 3, 4])
    assert sorted(history_groups) == sorted([[1, 2], [3, 4], population.tolist()])
    assert len(panel_inputs) == 1
    # Offset/limit affect scoring, never the pre-cutover candidate-bank population.
    np.testing.assert_array_equal(panel_inputs[0], _arrays(population)[1])
    assert len(fitted) == 1 and fitted[0].selection_evaluation == "provided_validation"
    assert [len(batch["reuse"]) for batch in batches] == [2, 1]
    assert summary["calibration"]["teacher_users"] == 4
    assert summary["calibration"]["teacher_tokens"] == 32
    assert summary["calibration"]["excludes_entire_evaluation_population"]
    assert summary["candidate_panel"]["population_users"] == len(population)

    with np.load(output / "scores.npz", allow_pickle=False) as saved:
        paths = saved["path_ids"].tolist()
        assert paths == ["reuse", "current_exact", "lr_2_3", "tr_2", "kt_k1"]
        np.testing.assert_array_equal(saved["uids"], population[2:5])
        np.testing.assert_array_equal(saved["panel_population_uids"], population)
        np.testing.assert_array_equal(saved["candidates"], candidates[2:5])
        assert saved["scores"].shape == (3, 5, 64)
        for index, path in enumerate(paths):
            np.testing.assert_array_equal(saved["scores"][:, index], np.concatenate([batch[path] for batch in batches]))
        scores = saved["scores"].copy()

    # Metrics must come from all concatenated users, not equal-weight batch means
    # (the final batch has only one user).
    exact_probability = sigmoid(scores[:, 1].astype(np.float64))
    reuse_gap = np.abs(sigmoid(scores[:, 0].astype(np.float64)) - exact_probability).mean()
    assert reuse_gap > 1e-12
    for index, row in enumerate(summary["metrics"]):
        gap = np.abs(sigmoid(scores[:, index].astype(np.float64)) - exact_probability).mean()
        np.testing.assert_allclose(row["mean_abs_probability_gap"], gap, rtol=0, atol=1e-14)
        np.testing.assert_allclose(row["probability_gap_recovery"], 1 - gap / reuse_gap, rtol=0, atol=1e-14)
        assert row["users"] == 3 and row["candidates_per_user"] == 64
        assert "cost" not in row
    on_disk = json.loads((output / "summary.json").read_text())
    assert on_disk == summary
    with (output / "metrics.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [row["config_id"] for row in csv_rows] == paths
    assert len(csv_rows) == 5


def test_describe_uses_pair_metadata_without_model_or_user_io(monkeypatch):
    pair = {
        "scale": "large", "edge": "theta0_to_theta1", "cutover": 1000,
        "config": {"num_layers": 10, "hidden_size": 320, "num_heads": 4, "max_seq_len": 1024},
        "dataset": {"manifest": "unused/dataset.json", "users": "unused/users.parquet",
                    "known_items": 100, "oov_buckets": 8},
        "parent": {"checkpoint": "unused/theta0.pt"},
        "current": {"checkpoint": "unused/theta1.pt"},
        "admission": {"reuse_eligible": False},
    }
    calls = []

    def load_model_pair(scale, edge_index, *, verify_hashes):
        calls.append((scale, edge_index, verify_hashes))
        return pair

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must not load a model, UID population, or history")

    monkeypatch.setattr(entry, "load_model_pair", load_model_pair)
    for name in ("load_model", "load_histories", "read_uid_split", "history_arrays", "make_candidate_panel"):
        monkeypatch.setattr(entry, name, forbidden)
    monkeypatch.setattr(entry.pq, "read_table", forbidden)
    args = entry.build_parser().parse_args(["--scale", "large", "--edge-index", "0", "--describe"])
    assert entry.run(args) == pair
    assert calls == [("large", 0, False)]
