import copy

import pytest
import torch

from hstu_kvcache.models.hstu import HSTUConfig
from hstu_kvcache.recflow.model import RecFlowGenerator


def _fixture(history_categories=False):
    torch.manual_seed(17)
    # The second category 200 deliberately has two different parents.
    paths = torch.tensor([
        [-1, -1], [10, 100], [10, 100], [10, 200],
        [20, 200], [20, 300], [20, 300], [-1, -1],
    ])
    cfg = HSTUConfig(
        num_items=7, num_behaviors=4, hidden_size=16, num_layers=2,
        num_heads=2, max_seq_len=16, input_dropout=0.0,
        block_variant="hstu_reference", activation="silu", relative_position_bias=True,
    )
    model = RecFlowGenerator(cfg, paths, history_categories=history_categories).eval()
    items = torch.tensor([[1, 2, 4, 3, 0], [5, 7, 0, 0, 0]])
    behaviors = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 0, 0, 0]])
    times = torch.tensor([[0., 1., 30., 10., 0.], [0., 100., 0., 0., 0.]])
    lengths = torch.tensor([4, 2])
    return model, (items, behaviors, times, lengths)


@pytest.mark.parametrize("history_categories", [False, True])
def test_teacher_paths_equal_free_candidate_scores_and_normalize(history_categories):
    model, history = _fixture(history_categories)
    all_ids = torch.arange(1, 7)[None].expand(2, -1)
    scored = model.score_items(*history, all_ids)
    for item in range(1, 7):
        teacher = model.teacher_log_probs(*history, torch.full((2,), item)).sum(-1)
        torch.testing.assert_close(teacher, scored[:, item - 1], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(scored.exp().sum(-1), torch.ones(2))
    # A sampled candidate set retains full-tree normalization.
    subset = torch.tensor([[2, 4, 0, 7], [2, 4, 0, 7]])
    sampled = model.score_items(*history, subset)
    torch.testing.assert_close(sampled[:, :2], scored[:, [1, 3]])
    assert bool(torch.isneginf(sampled[:, 2:]).all())


@pytest.mark.parametrize("history_categories", [False, True])
def test_wide_beam_matches_exhaustive_ranking_and_preserves_prefix(history_categories):
    model, history = _fixture(history_categories)
    candidates = torch.arange(1, 7)[None].expand(2, -1)
    exact = model.score_items(*history, candidates)
    expected_scores, order = exact.sort(descending=True)
    before = model._prefix_cache(*history)
    ids, scores = model.generate_topk(*history, k=6, beam_width=16)
    after = model._prefix_cache(*history)
    torch.testing.assert_close(ids, candidates.gather(1, order))
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(before.k, after.k, rtol=0, atol=0)
    torch.testing.assert_close(before.v, after.v, rtol=0, atol=0)
    assert all(len(torch.unique(row)) == len(row) for row in ids)


@pytest.mark.parametrize("history_categories", [False, True])
def test_no_target_leakage_and_padding_has_no_effect(history_categories):
    model, history = _fixture(history_categories)
    unknown = torch.tensor([[0, model.num_known + 1]])
    torch.testing.assert_close(
        model._history_item_vectors(unknown), model.backbone.lookup_item_embeddings(unknown),
        rtol=0, atol=0,
    )
    first = model._teacher_states(*history, torch.tensor([1, 1]))
    other_leaf = model._teacher_states(*history, torch.tensor([2, 2]))
    other_c2 = model._teacher_states(*history, torch.tensor([3, 3]))
    other_c1 = model._teacher_states(*history, torch.tensor([5, 5]))
    # Video identity is never a decoder input; c2 is unavailable to the c2
    # predictor, and neither target category is available at the REC state.
    torch.testing.assert_close(first, other_leaf, rtol=0, atol=0)
    torch.testing.assert_close(first[:, :2], other_c2[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(first[:, :1], other_c1[:, :1], rtol=0, atol=0)
    items, behaviors, times, lengths = history
    mutated = items.clone()
    for row, length in enumerate(lengths):
        mutated[row, length:] = 6
    changed = model._teacher_states(mutated, behaviors, times, lengths, torch.tensor([1, 1]))
    torch.testing.assert_close(first, changed, rtol=0, atol=0)
    # A short row has identical positions in a padded batch and by itself.
    alone = model._teacher_states(
        items[1:2, :2], behaviors[1:2, :2], times[1:2, :2], lengths[1:2], torch.tensor([1]),
    )
    torch.testing.assert_close(first[1:2], alone, atol=2e-6, rtol=2e-6)


def test_loss_backpropagates_through_history_and_three_heads():
    model, history = _fixture()
    model.train()
    loss = model.loss_per_example(*history, torch.tensor([2, 6]))
    assert loss.shape == (2,) and bool(torch.isfinite(loss).all())
    loss.mean().backward()
    for parameter in (
        model.rec_token, model.c1_head.weight, model.c2_head.weight,
        model.backbone.item_emb.weight, model.backbone.blocks[0].attn.q_proj.weight,
    ):
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert parameter.grad.abs().sum() > 0


def test_request_time_gap_matches_teacher_and_free_paths():
    model, history = _fixture()
    gaps = torch.tensor([9., 3600.])
    candidates = torch.arange(1, 7)[None].expand(2, -1)
    scored = model.score_items(*history, candidates, query_time_deltas=gaps)
    teacher = model.teacher_log_probs(*history, torch.tensor([2, 6]), query_time_deltas=gaps)
    torch.testing.assert_close(teacher.sum(-1), scored[[0, 1], [1, 5]], atol=2e-6, rtol=2e-6)
    ids, scores = model.generate_topk(*history, k=6, beam_width=16, query_time_deltas=gaps)
    expected_scores, order = scored.sort(descending=True)
    torch.testing.assert_close(ids, candidates.gather(1, order))
    torch.testing.assert_close(scores, expected_scores)
    without_gap = model.score_items(*history, candidates)
    assert not torch.allclose(scored, without_gap)


def test_batch_leaf_gather_preserves_reference_loss_and_all_gradients():
    model, history = _fixture()
    reference = copy.deepcopy(model)
    # Include repeated and distinct leaf groups plus untargeted catalog rows.
    history = tuple(torch.cat([value, value], dim=0) for value in history)
    targets = torch.tensor([1, 2, 3, 6])
    gaps = torch.tensor([9., 3600., 0., 30.])
    actual = model.loss_per_example(*history, targets, query_time_deltas=gaps)

    hidden = reference._teacher_states(*history, targets, query_time_deltas=gaps)
    paths = reference.item_paths[targets]
    rows = torch.arange(len(targets))
    first = reference.c1_head(hidden[:, 0]).float().log_softmax(-1)[rows, paths[:, 0]]
    second = reference._c2_log_probs(hidden[:, 1], paths[:, 0])[rows, paths[:, 1]]
    leaf_scores = torch.zeros(len(targets))
    keys = paths[:, 0] * reference.num_c2 + paths[:, 1]
    for key in torch.unique(keys).tolist():
        selected = torch.nonzero(keys == key, as_tuple=True)[0]
        leaves = reference._leaves(key)
        # Original implementation: independent full-table gather per group.
        log_probs = reference._leaf_log_probs(hidden[selected, 2], leaves)
        local = torch.searchsorted(leaves, targets[selected])
        leaf_scores = leaf_scores.index_copy(0, selected, log_probs.gather(1, local[:, None])[:, 0])
    expected = -(first + second + leaf_scores)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.mean().backward()
    expected.mean().backward()
    for name, parameter in model.named_parameters():
        old_gradient = reference.get_parameter(name).grad
        if old_gradient is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, old_gradient, rtol=1e-5, atol=1e-6, msg=name)


def test_exact_bound_search_matches_full_catalog_and_can_stop_early():
    model, history = _fixture()
    candidates = torch.arange(1, 7)[None].expand(2, -1)
    gaps = torch.tensor([9., 3600.])
    exact = model.score_items(*history, candidates, query_time_deltas=gaps)
    expected_scores, order = exact.sort(descending=True)
    ids, scores, stats = model.generate_exact_topk(
        *history, k=6, branch_batch_size=2, query_time_deltas=gaps, return_stats=True,
    )
    torch.testing.assert_close(ids, candidates.gather(1, order))
    torch.testing.assert_close(scores, expected_scores)
    assert stats["expanded_pairs"] == [stats["total_pairs"]] * 2
    # A dominant pair makes the admissible bound prune the other pairs. This
    # tests the stopping rule, beyond simply reproducing exhaustive traversal.
    with torch.no_grad():
        model.c1_head.weight.zero_()
        model.c1_head.bias.copy_(torch.tensor([20., -20.]))
        model.c2_head.weight.zero_()
        model.c2_head.bias.copy_(torch.tensor([20., -20., -20.]))
    exact = model.score_items(*history, candidates)
    ids, scores, stats = model.generate_exact_topk(*history, k=1, branch_batch_size=1, return_stats=True)
    torch.testing.assert_close(ids, candidates.gather(1, exact.argmax(-1, keepdim=True)))
    torch.testing.assert_close(scores, exact.max(-1, keepdim=True).values)
    assert stats["expanded_pairs"] == [1, 1]
