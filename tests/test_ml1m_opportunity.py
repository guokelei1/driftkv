import numpy as np
import torch

from hstu_kvcache.streaming.ml1m_opportunity import (
    METRICS,
    _candidate_ids,
    _comparison,
    _sample_candidates,
    _user_order,
    load_config,
)
from hstu_kvcache.streaming.ml1m_query_objective import _base_examples as _query_base_examples


def test_ml1m_opportunity_config_is_frozen() -> None:
    document = load_config(
        "configs/evokv_root_cause/ml1m_opportunity_factor_screen_20260807_v0.json"
    )
    assert document["data"]["user_limit"] == 1024
    assert document["training"]["negative_samples"] == 128


def test_ml1m_opportunity_user_order_is_deterministic() -> None:
    users = [f"u{index}" for index in range(20)]
    assert _user_order(users, 11) == _user_order(list(reversed(users)), 11)
    assert _user_order(users, 11) != _user_order(users, 12)


def test_ml1m_negative_sampling_excludes_target() -> None:
    targets = torch.tensor([1, 5, 10])
    generator = torch.Generator().manual_seed(7)
    candidates = _sample_candidates(targets, 10, 99, generator)
    assert torch.equal(candidates[:, 0], targets)
    assert not torch.any(candidates[:, 1:] == targets.unsqueeze(1))
    assert int(candidates.min()) >= 1
    assert int(candidates.max()) <= 10


def test_ml1m_candidate_expansion_is_unique_and_deterministic() -> None:
    record = {
        "positive_items": np.asarray([7]),
        "candidates": np.asarray([1, 7, 3, 4]),
    }
    first = _candidate_ids(record, "u1", 20, 10, 17)
    second = _candidate_ids(record, "u1", 20, 10, 17)
    assert np.array_equal(first, second)
    assert first[0] == 7
    assert len(np.unique(first)) == 10
    assert {1, 3, 4}.issubset(set(first))
    assert np.array_equal(_candidate_ids(record, "u1", 20, 20, 17), np.arange(1, 21))


def test_ml1m_seen_filtered_candidate_expansion_excludes_history() -> None:
    record = {
        "positive_items": np.asarray([7]),
        "candidates": np.asarray([1, 7, 3, 4]),
        "history": np.asarray([2, 5, 6, 8, 9]),
    }
    candidates = _candidate_ids(record, "u1", 20, 10, 17, filter_seen=True)
    assert not set(candidates).intersection({2, 5, 6, 8, 9})


def test_ml1m_comparison_orients_lower_ce_and_higher_rank() -> None:
    records = []
    for index in range(20):
        metrics = {}
        for method in ("better", "worse"):
            metrics[method] = {metric: 0.0 for metric in METRICS}
        metrics["better"]["candidate_cross_entropy"] = 1.0
        metrics["worse"]["candidate_cross_entropy"] = 1.2
        metrics["better"]["mrr"] = 0.4
        metrics["worse"]["mrr"] = 0.3
        records.append({"user_id": str(index), "metrics": metrics})
    result = _comparison(records, "better", "worse", 200, 13)
    assert np.isclose(result["candidate_cross_entropy"]["absolute"], 0.2)
    assert np.isclose(result["mrr"]["absolute"], 0.1)
    assert result["candidate_cross_entropy"]["positive_direction_with_ci"]
    assert result["mrr"]["positive_direction_with_ci"]


def test_ml1m_query_example_never_contains_target_in_query_slot() -> None:
    records = {
        "u1": {
            "history": np.asarray([2, 3, 4, 5]),
            "positive_items": np.asarray([7]),
        }
    }
    _, inputs, targets = _query_base_examples(records, ["u1"], 8)[0]
    assert inputs[-1] == 0
    assert targets[-1] == 7
    assert 7 not in inputs
