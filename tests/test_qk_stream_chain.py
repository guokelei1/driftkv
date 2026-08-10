from __future__ import annotations

import numpy as np
import torch

from hstu_kvcache.data.qk_stream_chain import (
    ROLE_CODES,
    QKStreamChainCorpus,
    chain_boundaries,
    stable_user_order,
)
from hstu_kvcache.streaming.qk_stream_version import (
    build_training_batch,
    evaluation_suffix,
    prequential_evaluation_role_audit,
)
from hstu_kvcache.streaming.trainer import build_next_item_targets


def _prequential_corpus() -> QKStreamChainCorpus:
    lengths = np.asarray([16, 16, 16], dtype=np.uint16)
    labels = np.zeros(48, dtype=np.uint8)
    labels[[5, 7, 9, 11]] = 1
    labels[16 + 9] = 1
    labels[32 + 5] = 1
    return QKStreamChainCorpus(
        path=None,
        arrays={
            "record_user_ids": np.asarray([101, 102, 201]),
            "record_role": np.asarray(
                [
                    ROLE_CODES["stream_train"],
                    ROLE_CODES["stream_train"],
                    ROLE_CODES["fit_tuning"],
                ],
                dtype=np.uint8,
            ),
            "record_offsets": np.asarray([0, 16, 32, 48]),
            "edge_last_ordinals": chain_boundaries(
                lengths,
                base_prefix=4,
                update_count=2,
            ),
            "item_idx": np.tile(np.arange(1, 17), 3),
            "behavior": np.ones(48, dtype=np.uint8),
            "label": labels,
        },
        metadata={},
        file_sha256="test",
        content_sha256="test",
    )


def test_chain_boundaries_are_variable_and_prequential() -> None:
    lengths = np.asarray([96, 160, 512], dtype=np.uint16)
    values = chain_boundaries(lengths, base_prefix=64, update_count=7)
    assert values.shape == (3, 9)
    assert np.array_equal(values[:, 0], np.asarray([63, 63, 63]))
    assert np.array_equal(values[:, -1], lengths - 1)
    assert np.all(values[:, 1:] > values[:, :-1])
    assert len({tuple(row) for row in values}) == 3
    edge = 1
    theta1_train = (values[:, edge - 1] + 1, values[:, edge] + 1)
    theta1_eval = (values[:, edge] + 1, values[:, edge + 1] + 1)
    theta2_train = (values[:, edge] + 1, values[:, edge + 1] + 1)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(theta1_eval, theta2_train, strict=True)
    )
    assert np.all(theta1_train[1] <= theta1_eval[0])


def test_stable_user_order_is_seeded_and_complete() -> None:
    users = np.arange(100, dtype=np.int64)
    first = stable_user_order(users, "first")
    repeated = stable_user_order(users, "first")
    second = stable_user_order(users, "second")
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)
    assert np.array_equal(np.sort(first), users)


def test_same_user_next_window_is_unseen_then_becomes_next_training() -> None:
    corpus = _prequential_corpus()
    theta1 = build_training_batch(
        corpus,
        np.asarray([0]),
        batch_size=1,
        edge=1,
    )
    theta1_targets, theta1_valid = build_next_item_targets(
        theta1["item_ids"],
        theta1["lengths"],
        theta1["labels"],
        theta1["train_mask"],
    )
    _, _, _, window2_targets, window2_labels = evaluation_suffix(
        corpus,
        0,
        edge=1,
    )
    theta2 = build_training_batch(
        corpus,
        np.asarray([0]),
        batch_size=1,
        edge=2,
    )
    theta2_targets, theta2_valid = build_next_item_targets(
        theta2["item_ids"],
        theta2["lengths"],
        theta2["labels"],
        theta2["train_mask"],
    )
    trained_by_theta1 = theta1_targets[theta1_valid]
    evaluated_by_theta1 = window2_targets[window2_labels]
    trained_by_theta2 = theta2_targets[theta2_valid]
    assert torch.equal(evaluated_by_theta1, trained_by_theta2)
    assert not torch.any(torch.isin(evaluated_by_theta1, trained_by_theta1))


def test_update_local_role_is_primary_and_disjoint_role_is_supplemental() -> None:
    audit = prequential_evaluation_role_audit(
        _prequential_corpus(),
        edge=1,
    )
    assert audit["primary_role"] == "stream_train"
    assert audit["primary_role_users"] == 2
    assert audit["optimizer_participant_users"] == 1
    assert audit["supplemental_role"] == "fit_tuning"
    assert audit["supplemental_users"] == 1
    assert audit["user_overlap"] == 0
    assert audit["training_window"] == 1
    assert audit["evaluation_window"] == 2
    assert audit["evaluation_targets_used_for_current_training"] is False
