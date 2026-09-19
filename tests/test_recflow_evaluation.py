"""A small reference check for full and sampled evaluation denominators."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hstu_kvcache.recflow.metrics import random_expected_metrics


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / "recflow" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daily_update_rejects_future_trained_parent(monkeypatch):
    import pytest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/recflow"))
    chain = _script("window_chain")
    settings = {"windows": {"B": {"fit": [19, 19], "compare_parent_and_current": [20, 20]}}}
    assert chain.phase_windows(settings, "B")[:2] == ("update_19_19", "20_20")
    chain.check_parent_days([1, 18], [19, 19], False)
    chain.check_parent_days([19, 19], [20, 20], False)
    # A former three-day B must never initialize a new day20 update.
    with pytest.raises(ValueError):
        chain.check_parent_days([19, 21], [20, 20], False)
    with pytest.raises(ValueError):
        chain.check_parent_days([19, 19], [20, 20], True)
    settings["windows"]["B"]["compare_parent_and_current"] = [19, 20]
    with pytest.raises(ValueError):
        chain.phase_windows(settings, "B")


def test_sampled_subpanel_and_legacy_random_null_use_their_own_requests(tmp_path, monkeypatch):
    probe, baseline = _script("development_probe"), _script("random_baseline")

    class Data:
        raw_ids = np.arange(1, 5)
        oov = 5
        catalog_set = {1, 2, 3, 4}
        popularity = np.ones(4)
        popular_ranking = [1, 2, 3, 4]
        labels = ([1, 99], [98], [2], [3])
        prepared = SimpleNamespace(requests=[
            dict(uid=7, day=19, request_id=100), dict(uid=7, day=19, request_id=101),
            dict(uid=8, day=20, request_id=102), dict(uid=9, day=21, request_id=103),
        ])

        def batch(self, indices, device):
            return {}

        def targets(self, index):
            labels = np.asarray(self.labels[index])
            return labels, labels[labels < self.oov]

    class Model:
        def eval(self):
            return self

        def generate_topk(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]]), None

        def score_items(self, candidate_ids, **kwargs):
            return -candidate_ids.float()

    args = SimpleNamespace(device="cpu", decoder="beam", eval_precision="fp32", beam_width=4,
                           sampled_distractors=[1], candidate_seed=17)
    for subset in (np.array([2, 0]), None):
        # The supplied subset order differs from free evaluation order; saved
        # rows and hashes must describe the actual, consistently ordered panel.
        run = tmp_path / ("subset" if subset is not None else "legacy")
        run.mkdir()
        result = probe.evaluate(Model(), Data(), np.arange(4), args, "A_days19_21", run,
                                sampled_indices=subset)
        sample_order = [0, 2] if subset is not None else [0, 1, 2, 3]
        expected_recall = .75 if subset is not None else .625
        assert result["full_catalog"]["requests"] == 4
        assert result["full_catalog"]["recall@20"] == .625
        assert result["sampled_candidate_diagnostics"]["uniform_1"]["requests"] == len(sample_order)
        assert result["sampled_candidate_diagnostics"]["uniform_1"]["recall@20"] == expected_recall
        assert result["per_day"]["19"]["full_catalog"]["recall@20"] == .25
        assert result["per_day"]["19"]["sampled_candidate_diagnostics"]["uniform_1"]["recall@20"] == (
            .5 if subset is not None else .25)
        sampled_path = run / "A_days19_21_sampled_request_metrics.npz"
        with np.load(sampled_path) as panel:
            np.testing.assert_array_equal(panel["indices"], sample_order)
            assert len(panel["uids"]) == len(sample_order)
            assert panel["uniform_1__recall@20"].mean() == expected_recall
        if subset is None:
            # Historical results had neither the sampled NPZ nor metadata.
            sampled_path.unlink()
            result.pop("sampled_request_panel")
        probe.save_json(run / "summary.json", dict(
            configuration=dict(catalog_items=4, seed=17), A_days19_21=result))
        output = run / "random"
        monkeypatch.setattr("sys.argv", ["random_baseline.py", "--runs", str(run),
                                         "--output", str(output), "--draws", "100"])
        baseline.main()
        comparison = json.loads((output / "summary.json").read_text())["results"][0]["comparisons"]
        assert comparison["full_catalog"]["requests"] == 4
        for pool in ("uniform_1", "popularity_1"):
            actual = comparison[pool]
            assert actual["requests"] == len(sample_order)
            assert actual["metrics"]["recall@20"]["analytic_expectation"] == expected_recall
            expected_ndcg = np.mean([
                random_expected_metrics(len(Data.labels[i]), len(Data().targets(i)[1]),
                                        min(4, len(Data().targets(i)[1]) + 1))["ndcg@20"]
                for i in sample_order
            ])
            np.testing.assert_allclose(actual["metrics"]["ndcg@20"]["analytic_expectation"], expected_ndcg)
            assert actual["null_trial_file"] == comparison["uniform_1"]["null_trial_file"]


def test_parallel_merge_preserves_global_users_oov_and_sampled_subset(tmp_path):
    parallel = _script("parallel_evaluate")

    class Data:
        raw_ids = np.arange(1, 5)
        oov = 5
        catalog_set = {1, 2, 3, 4}
        popularity = np.ones(4)
        popular_ranking = [4, 3, 2, 1]
        labels = ([1, 99], [98], [2], [4], [3], [])
        prepared = SimpleNamespace(requests=[
            dict(uid=7, day=19, request_id=100), dict(uid=7, day=19, request_id=101),
            dict(uid=7, day=20, request_id=102), dict(uid=8, day=20, request_id=103),
            dict(uid=7, day=21, request_id=104), dict(uid=9, day=21, request_id=105),
        ])

        def batch(self, indices, device):
            return {}

        def targets(self, index):
            labels = np.asarray(self.labels[index], dtype=np.int64)
            return labels, labels[labels < self.oov]

    class Model:
        def eval(self):
            return self

        def generate_topk(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]]), None

        def score_items(self, candidate_ids, **kwargs):
            return -candidate_ids.float()

    args = SimpleNamespace(device="cpu", decoder="beam", eval_precision="fp32", beam_width=4,
                           sampled_distractors=[1], candidate_seed=17)
    indices, sampled = np.array([5, 1, 4, 0, 3, 2]), np.array([3, 5])
    serial_dir, merged_dir = tmp_path / "serial", tmp_path / "merged"
    serial_dir.mkdir()
    merged_dir.mkdir()
    serial = parallel.evaluate(Model(), Data(), indices, args, "A", serial_dir, sampled_indices=sampled)
    workers = []
    for rank in range(2):
        directory = tmp_path / f"worker{rank}"
        directory.mkdir()
        selected = indices[rank::2]
        parallel.evaluate(Model(), Data(), selected, args, "A", directory,
                          sampled_indices=sampled[np.isin(sampled, selected)])
        workers.append(directory)
    merged = parallel.merge_evaluations(Data(), indices, sampled, args, "A", workers, merged_dir, 1.0)
    for key in ("full_catalog", "initial_popularity", "sampled_candidate_diagnostics",
                "random_expected", "per_day", "full_request_panel", "sampled_request_panel"):
        assert merged[key] == serial[key]
    # User7 spans shards and has four requests, versus one for each other user.
    # Averaging shard user means would silently change the population metric.
    assert merged["full_catalog"]["user_mean"]["recall@20"] == .8125
    assert merged["full_catalog"]["all_oov_positive_requests"] == 1
    assert merged["full_catalog"]["zero_positive_requests"] == 1
    for name in ("request_metrics", "sampled_request_metrics"):
        with np.load(serial_dir / f"A_{name}.npz") as expected, np.load(merged_dir / f"A_{name}.npz") as actual:
            assert set(actual) == set(expected)
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
    assert json.loads((serial_dir / "A_rankings.json").read_text()) == json.loads(
        (merged_dir / "A_rankings.json").read_text())
