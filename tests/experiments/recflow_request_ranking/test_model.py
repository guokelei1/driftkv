import copy

import pytest
import torch
import torch.nn.functional as F

from experiments.recflow_request_ranking.code.objective import ranking_loss
from experiments.recflow_request_ranking.code.shared_reader import score_full, score_shared_cache
from hstu_kvcache.models.hstu import HSTU, HSTUConfig


def example(position_bias=True):
    torch.manual_seed(17)
    model = HSTU(HSTUConfig(
        num_items=30, num_behaviors=4, hidden_size=8, num_layers=2,
        num_heads=2, head_dim=4, max_seq_len=9,
        temporal_num_freqs=2, input_dropout=0, block_variant="hstu_reference",
        activation="silu", relative_position_bias=position_bias,
    )).double().eval()
    items = torch.randint(1, 31, (2, 7))
    behaviors = torch.randint(1, 5, (2, 7))
    deltas = torch.rand(2, 7, dtype=torch.float64)
    lengths = torch.tensor([3, 7])
    candidates = torch.tensor([[8, 19, 4], [3, 7, 15]])
    query_times = torch.tensor([2.0, 5.0], dtype=torch.float64)
    return model, (items, behaviors, deltas, lengths, candidates, query_times)


@pytest.mark.parametrize("position_bias", [False, True])
def test_shared_prefix_scores_and_gradients_match_native_independent_queries(position_bias):
    model, args = example(position_bias)
    oracle = copy.deepcopy(model)
    shared = score_full(model, *args, chunk_size=2)
    items, behaviors, deltas, lengths, candidates, times = args
    reference = oracle.score_cc_full(items, behaviors, deltas, candidates, times, lengths=lengths)
    torch.testing.assert_close(shared, reference, rtol=1e-9, atol=1e-10)
    shared.square().sum().backward()
    reference.square().sum().backward()
    for (name, p), (other_name, other) in zip(model.named_parameters(), oracle.named_parameters(), strict=True):
        assert name == other_name
        if p.grad is None or other.grad is None:
            assert p.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(p.grad, other.grad, rtol=1e-8, atol=1e-9, msg=name)


def test_candidate_chunks_permutations_padding_and_native_cache_identity():
    model, args = example()
    items, behaviors, deltas, lengths, candidates, times = args
    before = score_full(model, *args, chunk_size=1)
    torch.testing.assert_close(before, score_full(model, *args, chunk_size=3))
    permutation = torch.tensor([2, 0, 1])
    permuted = score_full(model, items, behaviors, deltas, lengths, candidates[:, permutation], times)
    torch.testing.assert_close(permuted, before[:, permutation])
    mutated_items = items.clone()
    mutated_items[0, 3:] = (mutated_items[0, 3:] + 3) % 30 + 1
    mutated = score_full(model, mutated_items, behaviors, deltas, lengths, candidates, times)
    torch.testing.assert_close(mutated, before)
    cache = model.compute_kv(items, behaviors, deltas, lengths=lengths)
    original_k, original_v = cache.k.clone(), cache.v.clone()
    reuse = score_shared_cache(model, cache, candidates, times, lengths)
    torch.testing.assert_close(reuse, before)
    torch.testing.assert_close(cache.k, original_k, rtol=0, atol=0)
    torch.testing.assert_close(cache.v, original_v, rtol=0, atol=0)


def test_unknown_candidates_are_not_bce_negatives():
    logits = torch.tensor([[1., -2., 4.]], requires_grad=True)
    labels = torch.tensor([[1, 0, -1]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss, _ = ranking_loss(logits, labels, mask, listwise_weight=0)
    expected = F.binary_cross_entropy_with_logits(logits[:, :2], labels[:, :2].float())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0


def test_listwise_normalizer_spans_all_candidates_and_padding_has_finite_gradients():
    logits = torch.tensor([[1., -2., 4., -torch.inf], [0., 0., 0., 0.]], requires_grad=True)
    labels = torch.tensor([[1, 0, -1, -1], [-1, -1, -1, -1]])
    mask = torch.tensor([[True, True, True, False], [False, False, False, False]])
    loss, parts = ranking_loss(logits, labels, mask, listwise_weight=.1)
    torch.testing.assert_close(parts["listwise"], torch.logsumexp(logits[0, :3], dim=0) - logits[0, 0])
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] > 0  # explicitly weak contrastive signal, not BCE
    assert logits.grad[0, 3] == 0 and not logits.grad[1].any()


def test_all_negative_exposures_keep_bce_but_have_no_listwise_target():
    logits = torch.zeros((1, 3), requires_grad=True)
    loss, parts = ranking_loss(logits, torch.tensor([[0, 0, -1]]), torch.ones((1, 3), dtype=torch.bool))
    assert parts["listwise"] == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0


def test_unequal_ddp_ranks_and_accumulated_microbatches_match_global_request_mean():
    features = torch.tensor([[1., .5], [2., -1.], [3., 2.], [0., 0.]])
    labels = torch.tensor([[1, 0], [0, 1], [1, -1], [-1, -1]])
    mask = labels != -1
    weight = torch.tensor([.2, .7], requires_grad=True)
    complete, _ = ranking_loss(features * weight, labels, mask)
    complete.backward()
    expected = weight.grad.clone()
    # Two ranks, two microbatches each; one microbatch has zero valid requests.
    gradients = []
    for indices in [[0, 1], [2, 3]]:
        local = weight.detach().clone().requires_grad_()
        for index in indices:
            loss, _ = ranking_loss(features[index:index+1] * local, labels[index:index+1],
                                   mask[index:index+1], global_active_requests=3, ddp_world_size=2)
            loss.backward()
        gradients.append(local.grad)
    torch.testing.assert_close(torch.stack(gradients).mean(dim=0), expected)
    empty = torch.zeros((1, 2), requires_grad=True)
    loss, parts = ranking_loss(empty, torch.full((1, 2), -1), torch.zeros((1, 2), dtype=torch.bool))
    loss.backward()
    assert parts["active_requests"] == 0 and loss == 0
    assert empty.grad is not None and not empty.grad.any()


def test_distributed_loss_requires_global_count_and_rejects_invalid_count():
    args = (torch.zeros((1, 2)), torch.tensor([[1, 0]]), torch.ones((1, 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="requires the global"):
        ranking_loss(*args, ddp_world_size=2)
    for count in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="Global request count"):
            ranking_loss(*args, global_active_requests=count)


def test_checkpointed_queries_and_external_features_preserve_scores_and_gradients():
    model, args = example()
    oracle = copy.deepcopy(model)
    items, behaviors, deltas, lengths, candidates, times = args
    direct = score_full(model, *args)
    checkpointed = score_full(
        oracle, *args, chunk_size=2, checkpoint_queries=True, validate=False,
        prefix_item_vectors=oracle.lookup_item_embeddings(items),
        candidate_item_vectors=oracle.lookup_item_embeddings(candidates),
    )
    torch.testing.assert_close(direct, checkpointed)
    direct.square().sum().backward()
    checkpointed.square().sum().backward()
    for p, q in zip(model.parameters(), oracle.parameters(), strict=True):
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-9)


def test_max_depth_and_heads_match_native_candidate_queries():
    # Keep the diagnostic catalog/prefix tiny while testing the actual16-layer,
    # 10-head/320-wide algebra and production positional normalization. Full
    # catalog memory/restore and1024 execution are separate actual GPU audits.
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = HSTU(HSTUConfig(
        num_items=30, num_behaviors=4, hidden_size=320, num_layers=16,
        num_heads=10, head_dim=32, max_seq_len=1025, input_dropout=0,
        block_variant="hstu_reference", activation="silu", relative_position_bias=True,
    )).double().eval()
    oracle = copy.deepcopy(model)
    _, args = example()
    actual = score_full(model, *args, chunk_size=2, checkpoint_queries=True)
    items, behaviors, deltas, lengths, candidates, times = args
    expected = oracle.score_cc_full(items, behaviors, deltas, candidates, times, lengths=lengths)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, parameter), (_, reference) in zip(model.named_parameters(), oracle.named_parameters(), strict=True):
        if parameter.grad is None or reference.grad is None:
            assert parameter.grad is None and reference.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-7, atol=1e-8, msg=name)
