"""Small P0 canaries for candidate-conditioned CC scoring."""

from __future__ import annotations

import torch

from hstu_kvcache.models import HSTU, HSTUConfig


def _model(*, relative_position_bias: bool = False) -> HSTU:
    torch.manual_seed(23)
    return HSTU(
        HSTUConfig(
            num_items=32,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
            input_dropout=0.0,
            relative_position_bias=relative_position_bias,
            num_query_types=2,
            num_query_actions=3,
            query_type_id=1,
            query_action_id=2,
        )
    ).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    items = torch.tensor([[1, 4, 7, 0], [2, 5, 8, 11]])
    behaviors = torch.tensor([[1, 2, 1, 0], [2, 1, 2, 1]])
    deltas = torch.tensor([[0.0, 13.0, 41.0, 0.0], [0.0, 17.0, 29.0, 53.0]])
    lengths = torch.tensor([3, 4])
    candidates = torch.tensor([[3, 6, 9], [12, 13, 14]])
    query_delta = torch.tensor([67.0, 71.0])
    return items, behaviors, deltas, lengths, candidates, query_delta


def test_query_action_is_independent_from_behavior_padding() -> None:
    model = _model()
    assert model.query_encoder.action_embedding.padding_idx is None
    assert model.query_encoder.type_embedding is not model.behavior_emb.embed
    candidates = torch.tensor([[3, 6]])
    tokens = model.embed_query_tokens(candidates, torch.tensor([17.0]))
    changed_action = model.embed_query_tokens(
        candidates,
        torch.tensor([17.0]),
        query_action_ids=torch.tensor([1]),
    )
    assert not torch.allclose(tokens, changed_action)


def test_full_reuse_serial_and_flattened_scoring_are_equivalent() -> None:
    model = _model(relative_position_bias=True)
    items, behaviors, deltas, lengths, candidates, query_delta = _inputs()
    with torch.inference_mode():
        full = model.score_cc_full(
            items, behaviors, deltas, candidates, query_delta, lengths=lengths
        )
        parent = model.compute_kv(items, behaviors, deltas, lengths)
        parent_before = (parent.k.clone(), parent.v.clone())
        reuse = model.score_cc_reuse(parent, candidates, query_delta, prefix_lengths=lengths)
        serial = torch.cat(
            [
                model.score_cc_full(
                    items,
                    behaviors,
                    deltas,
                    candidates[:, index : index + 1],
                    query_delta,
                    lengths=lengths,
                )
                for index in range(candidates.shape[1])
            ],
            dim=1,
        )

    assert torch.allclose(full, reuse, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full, serial, atol=1e-6, rtol=1e-6)
    assert torch.equal(parent.k, parent_before[0])
    assert torch.equal(parent.v, parent_before[1])


def test_candidate_order_and_history_isolation() -> None:
    model = _model()
    items, behaviors, deltas, lengths, candidates, query_delta = _inputs()
    with torch.inference_mode():
        scores = model.score_cc_full(items, behaviors, deltas, candidates, query_delta, lengths)
        permuted = candidates[:, [2, 0, 1]]
        permuted_scores = model.score_cc_full(
            items, behaviors, deltas, permuted, query_delta, lengths
        )
        one = model.score_cc_full(
            items[:1], behaviors[:1], deltas[:1], candidates[:1, :1], query_delta[:1], lengths[:1]
        )
        two = model.score_cc_full(
            items[1:], behaviors[1:], deltas[1:], candidates[1:, :1], query_delta[1:], lengths[1:]
        )
        same_candidates = candidates[:1].expand(2, -1)
        different_history = model.score_cc_full(
            items,
            behaviors,
            deltas,
            same_candidates,
            query_delta,
            lengths,
        )

    assert torch.allclose(permuted_scores, scores[:, [2, 0, 1]], atol=1e-6, rtol=1e-6)
    assert torch.allclose(scores[:, :1], torch.cat([one, two]), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(different_history[0], different_history[1])


def test_candidate_item_embedding_is_on_the_query_path() -> None:
    model = _model()
    items, behaviors, deltas, lengths, candidates, query_delta = _inputs()
    with torch.inference_mode():
        before = model.score_cc_full(items, behaviors, deltas, candidates, query_delta, lengths)
        saved = model.item_emb.weight.detach().clone()
        model.item_emb.weight.zero_()
        after = model.score_cc_full(items, behaviors, deltas, candidates, query_delta, lengths)
        model.item_emb.weight.copy_(saved)
    before_candidate_std = before.std(dim=1).mean()
    after_candidate_std = after.std(dim=1).mean()
    assert float(before_candidate_std) > 1e-8
    assert float(after_candidate_std) < float(before_candidate_std) * 0.05


def test_query_time_is_shared_across_candidates_and_lengths_are_exact() -> None:
    model = _model(relative_position_bias=True)
    items, behaviors, deltas, lengths, candidates, query_delta = _inputs()
    with torch.inference_mode():
        shared = model.score_cc_full(items, behaviors, deltas, candidates, query_delta, lengths)
        explicit = model.score_cc_full(
            items,
            behaviors,
            deltas,
            candidates,
            query_delta[:, None].expand_as(candidates),
            lengths,
        )
        row0 = model.score_cc_full(
            items[:1, :3],
            behaviors[:1, :3],
            deltas[:1, :3],
            candidates[:1],
            query_delta[:1],
            torch.tensor([3]),
        )
    assert torch.allclose(shared, explicit, atol=1e-6, rtol=1e-6)
    assert torch.allclose(shared[:1], row0, atol=1e-6, rtol=1e-6)


def test_full_cc_path_is_trainable_for_theta0() -> None:
    model = _model()
    model.train()
    items, behaviors, deltas, lengths, candidates, query_delta = _inputs()
    scores = model.score_cc_full(items, behaviors, deltas, candidates, query_delta, lengths)
    scores[:, 0].mean().backward()
    assert model.item_emb.weight.grad is not None
    assert model.query_encoder.action_embedding.weight.grad is not None
