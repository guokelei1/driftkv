from __future__ import annotations

from unittest.mock import patch

import torch

from hstu_kvcache.data import (
    build_explicit_feedback_query,
    build_return_to_familiar_request,
    feedback_history_stratum_v2,
)
from hstu_kvcache.models import (
    HSTU,
    FrozenLinearBaseRanker,
    HSTUConfig,
    combine_base_and_cc_residual,
    exact_chunked_listwise_cross_entropy,
    masked_listwise_cross_entropy,
)


def test_return_candidate_universe_is_target_free_and_deterministic() -> None:
    history = [
        (2, 10, 1),
        (1, 20, 1),
        (2, 30, 2),
        (3, 40, 1),
    ]
    kwargs = {
        "history": history,
        "query_timestamp": 40 + 259_200,
        "artist_by_item": {1: 7, 2: 7, 3: 8},
        "global_popularity": {1: 11, 2: 13, 3: 17},
    }
    first = build_return_to_familiar_request(**kwargs)
    second = build_return_to_familiar_request(**kwargs)
    assert first == second
    assert first.item_ids == (2, 3, 1)
    assert first.quality_target_index(3) == 1
    assert first.quality_target_index(99) is None
    assert first.candidates[0].artist_count == 3
    assert len(first.candidates[0].base_features()) == 7


def test_base_score_is_identical_when_only_cc_history_path_changes() -> None:
    ranker = FrozenLinearBaseRanker(
        torch.tensor([0.5, -0.25]),
        intercept=0.1,
        feature_mean=torch.tensor([1.0, 2.0]),
        feature_scale=torch.tensor([2.0, 4.0]),
    )
    materialized_features = torch.tensor([[[3.0, 6.0], [5.0, 10.0]]])
    full_base = ranker(materialized_features)
    recent_base = ranker(materialized_features)
    full_residual = torch.tensor([[0.2, -0.1]])
    recent_residual = torch.tensor([[0.0, 0.3]])
    assert torch.equal(full_base, recent_base)
    assert torch.allclose(
        combine_base_and_cc_residual(full_base, full_residual) - full_residual,
        full_base,
    )
    assert torch.allclose(
        combine_base_and_cc_residual(recent_base, recent_residual) - recent_residual,
        recent_base,
    )
    assert list(ranker.parameters()) == []


def test_feedback_query_excludes_coincident_listen_and_can_strip_label() -> None:
    listens = [
        (4, 10, 1),
        (5, 20, 1),
        (4, 30, 2),
        (4, 40, 1),  # Coincident with feedback: must not enter the prefix.
        (6, 50, 1),  # Future: must not enter the prefix.
    ]
    quality = build_explicit_feedback_query(listens, 4, 40, label=1)
    fidelity = build_explicit_feedback_query(listens, 4, 40, label=None)
    assert quality.label == 1
    assert fidelity.label is None
    assert quality.causal_prefix == ((4, 10, 1), (5, 20, 1), (4, 30, 2))
    assert quality.candidate_history_position == "recent_seen"
    assert quality.coincident_target_listens_excluded == 1
    assert all(timestamp < 40 for _, timestamp, _ in fidelity.causal_prefix)


def test_feedback_history_strata_v2_distinguishes_lifetime_from_capped_history() -> None:
    listens = [(9, index + 1, 1) for index in range(600)]
    listens[0] = (7, 1, 1)
    listens[-10] = (8, 591, 1)
    assert feedback_history_stratum_v2(listens, 8, 700) == "recent_seen"
    assert feedback_history_stratum_v2(listens, 9, 700) == "recent_seen"
    assert feedback_history_stratum_v2(listens, 7, 700) == "seen_only_before_512"
    assert feedback_history_stratum_v2(listens, 6, 700) == "never_seen"


def test_exact_chunked_scores_and_loss_do_not_depend_on_chunk_size() -> None:
    torch.manual_seed(71)
    model = HSTU(
        HSTUConfig(
            num_items=32,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            max_seq_len=16,
            input_dropout=0.0,
            num_query_types=3,
            num_query_actions=2,
            query_type_id=1,
            query_action_id=1,
        )
    ).eval()
    items = torch.tensor([[1, 2, 3, 4]])
    behaviors = torch.tensor([[1, 2, 1, 2]])
    deltas = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    candidates = torch.tensor([[5, 6, 7, 8, 9]])
    query_delta = torch.tensor([259_200.0])
    lengths = torch.tensor([4])
    with torch.inference_mode():
        serial = model.score_cc_full(
            items, behaviors, deltas, candidates, query_delta, lengths
        )
        chunks_1 = model.score_cc_full_chunked(
            items,
            behaviors,
            deltas,
            candidates,
            query_delta,
            chunk_size=1,
            lengths=lengths,
        )
        chunks_3 = model.score_cc_full_chunked(
            items,
            behaviors,
            deltas,
            candidates,
            query_delta,
            chunk_size=3,
            lengths=lengths,
        )
        chunks_full = model.score_cc_full_chunked(
            items,
            behaviors,
            deltas,
            candidates,
            query_delta,
            chunk_size=candidates.shape[1],
            lengths=lengths,
        )
        permutation = torch.tensor([4, 0, 3, 1, 2])
        permuted_chunks = model.score_cc_full_chunked(
            items,
            behaviors,
            deltas,
            candidates[:, permutation],
            query_delta,
            chunk_size=2,
            lengths=lengths,
        )
    assert torch.allclose(torch.cat(chunks_1, dim=1), serial, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.cat(chunks_3, dim=1), serial, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.cat(chunks_full, dim=1), serial, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        torch.cat(permuted_chunks, dim=1), serial[:, permutation], atol=1e-6, rtol=1e-6
    )
    expected = torch.nn.functional.cross_entropy(serial, torch.tensor([3]))
    actual_1 = exact_chunked_listwise_cross_entropy(chunks_1, positive_index=3)
    actual_3 = exact_chunked_listwise_cross_entropy(chunks_3, positive_index=3)
    assert torch.allclose(actual_1, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_3, expected, atol=1e-6, rtol=1e-6)
    permuted_target = int(torch.nonzero(permutation == 3, as_tuple=False).item())
    permuted_loss = exact_chunked_listwise_cross_entropy(
        permuted_chunks, positive_index=permuted_target
    )
    assert torch.allclose(permuted_loss, expected, atol=1e-6, rtol=1e-6)

    with patch.object(model, "forward", wraps=model.forward) as prefix_forward:
        with torch.inference_mode():
            model.score_cc_full_chunked(
                items,
                behaviors,
                deltas,
                candidates,
                query_delta,
                chunk_size=2,
                lengths=lengths,
            )
        assert prefix_forward.call_count == 1


def test_masked_listwise_loss_is_request_mean_and_ignores_padding() -> None:
    scores = torch.tensor([[2.0, 1.0, -99.0], [0.0, 1.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    targets = torch.tensor([0, 2])
    actual = masked_listwise_cross_entropy(scores, mask, targets)
    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(scores[:1, :2], targets[:1]),
            torch.nn.functional.cross_entropy(scores[1:], targets[1:] - 0),
        ]
    ).mean()
    assert torch.allclose(actual, expected)
    actual.backward()
    assert scores.grad is not None
    assert float(scores.grad[0, 2]) == 0.0
