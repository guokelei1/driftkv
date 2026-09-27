import torch

from hstu_kvcache.baselines.query_recompute import query_attention_scores, recompute_query
from hstu_kvcache.baselines.sparse_recompute import recompute_positions, select_top_positions


def test_query_selector_matches_actual_first_layer_attention(model_factory, raw_events):
    parent, current = model_factory(17), model_factory(29)
    cache = parent.compute_kv(*raw_events)
    candidates = torch.tensor([[7], [9]])
    delta = torch.tensor([2., 4.])
    # Observe Q on the actual native transient-query path; compare QK weights.
    captured = {}
    handle = current.blocks[0].attn.q_proj.register_forward_hook(
        lambda _module, _args, output: captured.update(q=output.detach().clone())
    )
    try:
        current.score_cc_reuse(cache, candidates, delta)
    finally:
        handle.remove()
    attn = current.blocks[0].attn
    q = captured["q"].reshape(2, 1, 3, 4).transpose(1, 2)
    k = cache.k[0].reshape(2, cache.seq_len, 3, 4).permute(0, 2, 3, 1)
    weights = torch.matmul(q, k) * attn.scale
    weights += attn._relative_position_bias(torch.tensor([cache.seq_len]),
                                            torch.arange(cache.seq_len), weights.dtype)
    expected = attn._activate(weights).abs().mean((1, 2))
    scores = query_attention_scores(current, cache, candidates, delta)
    torch.testing.assert_close(scores, expected)
    selected = recompute_query(current, cache, *raw_events, candidates, delta, n=2)
    reference = recompute_positions(current, cache, *raw_events, select_top_positions(scores, 2))
    torch.testing.assert_close(selected.k, reference.k)
    torch.testing.assert_close(selected.v, reference.v)


def test_query_zero_and_full_skip_selection(model_factory, raw_events, monkeypatch):
    parent, current = model_factory(17), model_factory(29)
    cache = parent.compute_kv(*raw_events)
    candidates, delta = torch.tensor([[7], [9]]), torch.tensor([2., 4.])
    def forbidden(*args, **kwargs):
        raise AssertionError("endpoint incurred token-selection overhead")
    monkeypatch.setattr("hstu_kvcache.baselines.query_recompute.core.query_attention_scores", forbidden)
    assert recompute_query(current, cache, *raw_events, candidates, delta, n=0) is cache
    actual = recompute_query(current, cache, *raw_events, candidates, delta, n=cache.seq_len)
    full = current.compute_kv(*raw_events)
    torch.testing.assert_close(actual.k, full.k)
    torch.testing.assert_close(actual.v, full.v)
