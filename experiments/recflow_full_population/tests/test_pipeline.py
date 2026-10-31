import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.recflow_full_population.code.batches import GlobalBatchSampler, RequestBatches
from experiments.recflow_full_population.code.configuration import load_proposal
from experiments.recflow_full_population.code.gates import choose_foundation, early_learning, native_learning
from experiments.recflow_full_population.code.model import RequestRanker
from experiments.recflow_full_population.code.protocol import version_schedule

DATA = Path(__file__).resolve().parents[1] / "data"


def test_all_rank_targets_and_resume_cursor():
    for world in [1, 2, 4]:
        seen, suffix = [], []
        for rank in range(world):
            common = dict(global_batch=64, micro_batch=4, world_size=world, rank=rank)
            seen.extend(i for b in GlobalBatchSampler(131, **common) for i in b if i >= 0)
            suffix.extend(i for b in GlobalBatchSampler(131, start_step=1, **common) for i in b if i >= 0)
        order = np.arange(131)
        np.random.default_rng(17).shuffle(order)
        assert sorted(seen) == list(range(131))
        assert sorted(suffix) == sorted(order[64:])


def test_population_and_window_separation():
    groups = np.load(DATA / "population.npz")
    assert [len(groups[k]) for k in ["train", "validation", "cache_fit", "cache_validation"]] == [30000, 2000, 384, 128]
    combined = np.concatenate([groups[k] for k in groups.files])
    assert len(combined) == len(np.unique(combined))
    manifest = json.loads((DATA / "selection_manifest.json").read_text())
    for key, entry in manifest["datasets"].items():
        records = np.load(DATA / entry["file"], mmap_mode="r")
        expected = groups["validation" if key.startswith("1:") else "train"]
        assert np.isin(records["uid"], expected).all()
    schedule = version_schedule()
    assert len(schedule) == 10 and schedule[-1]["training_days"] == [36, 36]
    assert load_proposal()["history"]["max_events"] == 1024


def test_actual_mixed_history_gather_and_spawn_is_mmap():
    dataset = RequestBatches(DATA, list(range(1, 24)))
    indices = [int(np.flatnonzero(dataset.records["history_length"] == length)[0]) for length in [0, 1, 127, 511, 1023, 1024]]
    batch = dataset[indices + [-1]]
    assert batch["lengths"][:6].tolist() == [0, 1, 127, 511, 1023, 1024]
    for row, length in enumerate(batch["lengths"].tolist()):
        assert (batch["history_features"][row, length:] == -1).all()
        assert (batch["behaviors"][row, length:] == 0).all()
    assert batch["query_deltas"][0] == 0 and not batch["candidate_mask"][-1].any()
    dumped = pickle.dumps(dataset)
    assert len(dumped) < 4096  # no four-million-row copy per spawned worker
    restored = pickle.loads(dumped)
    assert isinstance(restored.records, np.memmap)
    for key, value in batch.items():
        torch.testing.assert_close(value, restored[indices + [-1]][key], rtol=0, atol=0)


def test_no_accidental_old_filter_or_dynamic_cap():
    with pytest.raises(ValueError):
        RequestBatches(DATA, [28], history_limit=128)
    with pytest.raises(ValueError):
        RequestBatches(DATA, [28], limit=4096)
    with pytest.raises(ValueError):
        RequestBatches(DATA, [27], role=2)


def test_feature_empty_and_short_prefix_reference_scores_gradients(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    raw = np.arange(100, dtype=np.int64)
    item_map = (17 + ((raw ^ (raw >> 16)) * 2654435761 + 17) % 32).astype(np.int32)
    item_map[:16] = np.arange(1, 17)
    np.save(tmp_path / "item_map.npy", item_map)
    (tmp_path / "manifest.json").write_text(json.dumps({"explicit_ids": 16, "hash_buckets": 32}))
    model = RequestRanker(tmp_path).eval()
    lengths = torch.tensor([0, 1, 17, 33])
    features = torch.randint(0, 30, (4, 33, 5), dtype=torch.int32)
    for row, length in enumerate(lengths):
        features[row, length:] = -1
    batch = dict(history_features=features, behaviors=torch.ones((4, 33), dtype=torch.long),
                 deltas=torch.zeros(4, 33), lengths=lengths,
                 candidate_features=torch.randint(0, 300, (4, 7, 5), dtype=torch.int32),
                 query_deltas=torch.tensor([0., 1., 2., 3.]))
    fast, ref = model.encode_history(batch), model.encode_history(batch, reference=True)
    torch.testing.assert_close(fast.k, ref.k, rtol=0, atol=0)
    torch.testing.assert_close(fast.v, ref.v, rtol=0, atol=0)
    score, oracle = model.read(batch, fast), model.read(batch, ref)
    torch.testing.assert_close(score, oracle, rtol=0, atol=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    left = torch.autograd.grad(score.square().mean(), parameters)
    right = torch.autograd.grad(oracle.square().mean(), parameters)
    for actual, expected in zip(left, right, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_learning_gates_do_not_select_cache_gap(tmp_path):
    for name, loss in [("initial", .7), ("learned", .5), ("bad", .9)]:
        path = tmp_path / name
        path.mkdir()
        with (path / "rank_0.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["uid", "day", "rid", "current_observed_bce"])
            writer.writeheader()
            writer.writerows(dict(uid=i, day=24, rid=i, current_observed_bce=loss) for i in range(40))
    assert early_learning(tmp_path / "initial", tmp_path / "learned")["passed"]
    assert not early_learning(tmp_path / "initial", tmp_path / "bad")["passed"]
    summary = dict(paired_current_minus={k: dict(ci95=[.001, .01]) for k in ["random", "popularity"]})
    summary["paired_current_minus"]["reuse"] = dict(ci95=[-.1, -.01])
    assert native_learning(summary)["passed"]
    candidates = [dict(epoch=i, summary=dict(metrics=dict(current={"ndcg@10": value}))) for i, value in [(1, .05), (2, .04)]]
    assert choose_foundation(candidates)["epoch"] == 1
