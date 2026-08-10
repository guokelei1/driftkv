from copy import deepcopy

import numpy as np
import torch

from hstu_kvcache.streaming.kuairand_query_transition import (
    PREDICTION_QUERY_OBJECTIVE,
    TRUE_NEXT_ITEM_OBJECTIVE,
    _collate,
    _collate_true_next_item,
    _evaluate,
    _frequency_matched_candidates,
    _future_targets,
    _positions,
    _relative_rows,
    _user_order,
    make_model,
)


def test_kuairand_query_positions_are_nested_and_deterministic() -> None:
    values = np.arange(10, 30, dtype=np.int64)
    selected = _positions(values, 4)
    assert np.array_equal(selected, np.asarray([10, 16, 22, 29]))
    assert set(selected).issubset(set(values))


def test_kuairand_query_user_order_is_deterministic() -> None:
    users = list(range(100))
    assert _user_order(users, 7) == _user_order(list(reversed(users)), 7)
    assert _user_order(users, 7) != _user_order(users, 8)


def test_kuairand_frequency_matched_candidates_are_clean() -> None:
    pool = np.arange(1, 100001, dtype=np.int64)
    ranks = np.arange(-1, 100000, dtype=np.int64)
    first = _frequency_matched_candidates(10000, {10000, 10001}, pool, ranks, 49, 17)
    second = _frequency_matched_candidates(10000, {10000, 10001}, pool, ranks, 49, 17)
    assert first == second
    assert len(first) == len(set(first)) == 49
    assert not set(first).intersection({10000, 10001})
    assert min(first) >= 8192
    assert max(first) < 16384


def test_kuairand_layerwise_relative_rows_preserve_layer_and_record_axes() -> None:
    reference = torch.ones(3, 2, 4, 5)
    value = reference.clone()
    value[1, 0] *= 2
    value[2, 1] *= 3
    relative = _relative_rows(value, reference)
    assert relative.shape == (3, 2)
    assert torch.allclose(
        relative,
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
    )


def test_true_next_item_collate_uses_last_real_history_item() -> None:
    examples = [
        (7, np.asarray([3, 4, 5], dtype=np.int64), 6),
        (8, np.asarray([9, 10], dtype=np.int64), 11),
    ]
    items, behaviors, _, lengths, targets = _collate_true_next_item(
        examples, torch.device("cpu")
    )
    assert lengths.tolist() == [3, 2]
    assert items.tolist() == [[3, 4, 5], [9, 10, 0]]
    assert behaviors.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert targets.tolist() == [6, 11]


def test_multi_horizon_query_collate_reuses_one_history() -> None:
    examples = [
        (7, np.asarray([3, 4, 5], dtype=np.int64), np.asarray([6, 7, 8])),
        (8, np.asarray([9, 10], dtype=np.int64), np.asarray([11, 12, 13])),
    ]
    items, behaviors, _, lengths, targets = _collate(
        examples, torch.device("cpu")
    )
    assert lengths.tolist() == [4, 3]
    assert items.tolist() == [[3, 4, 5, 0], [9, 10, 0, 0]]
    assert behaviors.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert targets.tolist() == [[6, 7, 8], [11, 12, 13]]


def test_future_targets_do_not_cross_stream_dates() -> None:
    items = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)
    dates = np.asarray(["1", "1", "1", "2", "2"])
    eligible = np.asarray([True, True, False, True, True])
    assert np.array_equal(
        _future_targets(items, dates, eligible, 0, 2),
        np.asarray([2, 3], dtype=np.int64),
    )
    assert _future_targets(items, dates, eligible, 1, 2) is None
    assert _future_targets(items, dates, eligible, 3, 1) == 5


def test_true_next_item_evaluation_caches_only_prefix_before_latest_item() -> None:
    document = {
        "model": {
            "hidden_size": 16,
            "num_layers": 2,
            "num_heads": 2,
            "max_seq_len": 8,
            "input_dropout": 0.0,
            "activation": "relu",
            "qk_scale": 1.0,
            "gating": "silu_gate",
            "query_mode": "last_history_item",
        },
        "training": {
            "objective": TRUE_NEXT_ITEM_OBJECTIVE,
            "temperature": 0.1,
        },
        "evaluation": {"batch_size": 2},
    }
    previous = make_model(document, 20, torch.device("cpu"))
    current = deepcopy(previous)
    previous.eval()
    current.eval()
    workload = {
        "selected_users": [1, 2],
        "evaluation_keys": [1, 2],
        "evaluation": {
            1: {
                "user_id": 1,
                "query_ordinal": 0,
                "history": np.asarray([2, 3, 4, 5], dtype=np.int64),
                "target": 6,
            },
            2: {
                "user_id": 2,
                "query_ordinal": 0,
                "history": np.asarray([7, 8, 9, 10], dtype=np.int64),
                "target": 11,
            },
        },
        "candidate_maps": {
            1: np.asarray([6, 12, 13, 14], dtype=np.int64),
            2: np.asarray([11, 15, 16, 17], dtype=np.int64),
        },
        "author_by_item": np.zeros(21, dtype=np.int64),
    }
    result = _evaluate(previous, current, workload, document)
    assert result["sanity"]["passed"]
    assert [record["cache_prefix_length"] for record in result["records"]] == [3, 3]
    assert all(record["cache_k_relative_error"] == 0.0 for record in result["records"])
    assert all(record["cache_v_relative_error"] == 0.0 for record in result["records"])
    for record in result["records"]:
        for metric in record["metrics"]["recompute"]:
            assert np.isclose(
                record["metrics"]["recompute"][metric],
                record["metrics"]["reuse"][metric],
                atol=1e-6,
                rtol=1e-6,
            )


def test_latest_item_query_recomputes_latest_outside_the_stale_cache() -> None:
    document = {
        "model": {
            "hidden_size": 16,
            "num_layers": 2,
            "num_heads": 2,
            "max_seq_len": 8,
            "input_dropout": 0.0,
            "activation": "silu",
            "qk_scale": 1.0,
            "gating": "silu_gate",
            "block_variant": "hstu_reference",
            "relative_position_bias": True,
            "causal_diagonal": "exclusive",
            "query_mode": "latest_item_query",
        },
        "training": {
            "objective": PREDICTION_QUERY_OBJECTIVE,
            "temperature": 0.1,
        },
        "evaluation": {"batch_size": 2},
    }
    previous = make_model(document, 20, torch.device("cpu"))
    current = deepcopy(previous)
    previous.eval()
    current.eval()
    workload = {
        "selected_users": [1, 2],
        "evaluation_keys": [1, 2],
        "evaluation": {
            1: {
                "user_id": 1,
                "query_ordinal": 0,
                "history": np.asarray([2, 3, 4, 5], dtype=np.int64),
                "target": 6,
            },
            2: {
                "user_id": 2,
                "query_ordinal": 0,
                "history": np.asarray([7, 8, 9, 10], dtype=np.int64),
                "target": 11,
            },
        },
        "candidate_maps": {
            1: np.asarray([6, 12, 13, 14], dtype=np.int64),
            2: np.asarray([11, 15, 16, 17], dtype=np.int64),
        },
        "author_by_item": np.zeros(21, dtype=np.int64),
    }
    result = _evaluate(previous, current, workload, document)
    assert result["sanity"]["passed"]
    assert [record["cache_prefix_length"] for record in result["records"]] == [3, 3]
    for record in result["records"]:
        for metric in record["metrics"]["recompute"]:
            assert np.isclose(
                record["metrics"]["recompute"][metric],
                record["metrics"]["reuse"][metric],
                atol=1e-6,
                rtol=1e-6,
            )
