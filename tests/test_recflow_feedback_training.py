import copy

import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.recflow.feedback import grouped_scores


def fixture():
    torch.manual_seed(17)
    model = HSTU(HSTUConfig(num_items=50, num_behaviors=4, hidden_size=32, num_layers=2,
        num_heads=2, max_seq_len=8, input_dropout=0))
    items = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]])
    behaviors = (items > 0).long()
    times = torch.zeros_like(items, dtype=torch.float32)
    lengths = torch.tensor([4, 2])
    candidates = torch.tensor([[8, 9, 10], [11, 12, 13]])
    gap = torch.ones(2)
    return model, (items, behaviors, times, lengths, candidates, gap)


def test_grouped_forward_and_gradients_match_native_independent_queries():
    model, args = fixture()
    other = copy.deepcopy(model)
    actual = grouped_scores(model, *args)
    items, behavior, times, lengths, candidates, gap = args
    expected = other.score_cc_full(items, behavior, times, candidates, gap, lengths=lengths)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, parameter), (other_name, reference) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=4e-6, rtol=5e-4, msg=name)


def test_queries_cannot_observe_other_candidates_and_padding_is_inert():
    model, args = fixture()
    baseline = grouped_scores(model, *args)
    altered = list(args)
    altered[4] = args[4].clone()
    altered[4][:, 1:] = 40
    torch.testing.assert_close(grouped_scores(model, *altered)[:, 0], baseline[:, 0])
    extended = list(args)
    extended[0] = torch.nn.functional.pad(args[0], (0, 3))
    extended[1] = torch.nn.functional.pad(args[1], (0, 3))
    extended[2] = torch.nn.functional.pad(args[2], (0, 3))
    torch.testing.assert_close(grouped_scores(model, *extended), baseline, atol=2e-6, rtol=2e-5)


def test_direct_lookup_accumulates_the_same_dense_embedding_gradient():
    model, args = fixture()
    other = copy.deepcopy(model)
    other.direct_embedding_gradient = True
    grouped_scores(model, *args).square().sum().backward()
    grouped_scores(other, *args).square().sum().backward()
    for (name, value), (_, actual) in zip(model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(value.grad, actual.grad, atol=2e-6, rtol=2e-5, msg=name)
