"""CPU fallbacks run by default; CUDA capture requires an explicit opt-in.

Coordinator-only CUDA test command:
RECFLOW_GRAPH_GPU_TEST=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python -m pytest \
    -q experiments/figure9recflow/test_graph_ops.py
"""
import os

import pytest
import torch

from experiments.figure9recflow.cache_ops import append_behaviors, encode_cache, score_candidates
from experiments.figure9recflow.graph_ops import GraphExecutor
from experiments.figure9recflow.test_cache_ops import TinyRanker, close_cache, raw
from hstu_kvcache.models.kv_cache import HSTUKVCache


def fixture_state(device="cpu", n=4):
    torch.set_num_threads(1)
    torch.manual_seed(173)
    model = TinyRanker().eval().to(device)
    events = tuple(value.to(device) for value in raw(n))
    cache = encode_cache(model, *events)
    second_k, second_v = cache.k + .1, cache.v - .08
    batch = HSTUKVCache(torch.cat((cache.k, second_k), dim=1),
                       torch.cat((cache.v, second_v), dim=1), n)
    return model, batch


def test_cpu_fallback_complete_candidates_and_strict_fifo_are_functional():
    model, cache = fixture_state()
    executor = GraphExecutor(model, method_count=2, history_limit=4, chunk_size=3)
    features = raw(7)[0]
    delta = torch.arange(7).float()[None]
    expected = score_candidates(model, cache, features, delta, chunk_size=3)
    torch.testing.assert_close(executor.score(cache, features, delta), expected)
    old_k, old_v = cache.k.clone(), cache.v.clone()
    append = raw(3)
    result = executor.append(cache, *append)
    close_cache(result, append_behaviors(model, cache, *append, window_size=4))
    assert torch.equal(cache.k, old_k) and torch.equal(cache.v, old_v)
    assert executor.stats["eager_reads"] == executor.stats["eager_appends"] == 1
    assert executor.read_graph is None and executor.append_graph is None
    saved = executor.snapshot(result)
    result.k.add_(10)
    assert not torch.equal(result.k, saved.k)


def test_short_history_and_zero_events_eager_without_cuda_initialization():
    model, cache = fixture_state(n=2)
    executor = GraphExecutor(model, method_count=2, history_limit=4, chunk_size=3, borrow_state=True)
    events = raw(0)
    result = executor.append(cache, *events)
    close_cache(result, cache)
    output = executor.score(cache, events[0], torch.tensor([0.]))
    assert output.shape == (2, 0)
    assert executor.resident is None
    with pytest.raises(ValueError, match="No resident"):
        executor.snapshot()
    executor.clear()
    assert executor.read_graph is executor.append_graph is None


def test_constructor_rejects_invalid_graph_geometry_and_training():
    model, _ = fixture_state()
    for kwargs in [dict(method_count=0), dict(history_limit=17), dict(chunk_size=0), dict(warmup=0)]:
        with pytest.raises(ValueError):
            GraphExecutor(model, history_limit=kwargs.pop("history_limit", 4), **kwargs)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        GraphExecutor(model, history_limit=4)


@pytest.mark.skipif(os.environ.get("RECFLOW_GRAPH_GPU_TEST") != "1",
                    reason="GPU capture runs only with coordinator opt-in")
@pytest.mark.parametrize("borrow", [False, True])
def test_opt_in_cuda_graph_replay_padding_fifo_and_result_lifetimes(borrow):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cache = fixture_state("cuda")
    executor = GraphExecutor(model, method_count=2, history_limit=4, chunk_size=3, borrow_state=borrow)
    candidates = raw(7)[0].cuda()
    times = torch.arange(7, device="cuda").float()[None]
    initial = executor.score(cache, candidates, times)
    torch.testing.assert_close(initial, score_candidates(model, cache, candidates, times), rtol=0, atol=2e-5)
    saved_scores = initial.clone()
    initial_cache = executor.snapshot(cache)
    events = tuple(value.cuda() for value in raw(3))
    actual = executor.append(cache, *events)
    expected = append_behaviors(model, initial_cache, *events, window_size=4)
    close_cache(actual, expected, atol=2e-5)
    after = executor.score(actual, candidates[:, :2], torch.tensor([3.], device="cuda"))
    torch.testing.assert_close(after, score_candidates(model, expected, candidates[:, :2],
                                                      torch.tensor([3.], device="cuda")), rtol=0, atol=2e-5)
    owned = executor.snapshot(actual)
    # Reloading another user's cache cannot clobber saved scores or snapshots.
    executor.score(initial_cache, candidates[:, :1], torch.tensor([1.], device="cuda"))
    close_cache(owned, expected, atol=2e-5)
    torch.testing.assert_close(initial, saved_scores, rtol=0, atol=0)
    if not borrow:
        close_cache(actual, expected, atol=2e-5)
    assert executor.stats["read_captures"] == executor.stats["append_captures"] == 1
    assert executor.stats["read_replays"] == 5
    assert executor.stats["append_replays"] == 3
    executor.clear()
    assert executor.resident is None and executor.read_graph is None and executor.append_graph is None
