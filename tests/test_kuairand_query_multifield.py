import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_query_transition import _forward, _input_vectors


def test_multifield_learned_query_incremental_matches_full() -> None:
    torch.manual_seed(7)
    model = HSTU(
        HSTUConfig(
            num_items=200,
            num_prediction_items=100,
            num_behaviors=1,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
            input_dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        model.item_emb.weight[0].fill_(0.25)
    author_by_item = torch.zeros(201, dtype=torch.long)
    author_by_item[1:101] = torch.arange(101, 201)
    prefix = torch.randint(1, 100, (4, 16))
    query = torch.zeros(4, 1, dtype=torch.long)
    prefix_behaviors = torch.ones_like(prefix)
    prefix_deltas = torch.zeros_like(prefix, dtype=torch.float32)
    cache = model.compute_kv_from_item_embeddings(
        _input_vectors(model, prefix, author_by_item),
        prefix_behaviors,
        prefix_deltas,
    )
    incremental, _ = model.forward_with_cache_from_item_embeddings(
        cache,
        _input_vectors(model, query, author_by_item),
        torch.ones_like(query),
        torch.zeros_like(query, dtype=torch.float32),
    )
    full = torch.cat((prefix, query), dim=1)
    lengths = torch.full((4,), full.shape[1], dtype=torch.long)
    full_hidden = _forward(
        model,
        full,
        torch.ones_like(full),
        torch.zeros_like(full, dtype=torch.float32),
        lengths,
        author_by_item,
    )
    assert torch.allclose(incremental[:, -1], full_hidden[:, -1], atol=1e-5, rtol=1e-5)


def test_history_only_query_incremental_matches_full_and_uses_prefix() -> None:
    torch.manual_seed(11)
    model = HSTU(
        HSTUConfig(
            num_items=200,
            num_prediction_items=100,
            num_behaviors=1,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
            input_dropout=0.0,
            gating="none",
        )
    ).eval()
    model.query_mode = "history_only_zero"
    author_by_item = torch.zeros(201, dtype=torch.long)
    author_by_item[1:101] = torch.arange(101, 201)
    prefix = torch.randint(1, 100, (4, 16))
    prefix_behaviors = torch.ones_like(prefix)
    prefix_deltas = torch.zeros_like(prefix, dtype=torch.float32)
    cache = model.compute_kv_from_item_embeddings(
        _input_vectors(model, prefix, author_by_item),
        prefix_behaviors,
        prefix_deltas,
    )
    query_embedded = torch.zeros(4, 1, 32)
    incremental, _ = model.forward_with_cache_embedded(cache, query_embedded)
    full = torch.cat((prefix, torch.zeros(4, 1, dtype=torch.long)), dim=1)
    lengths = torch.full((4,), full.shape[1], dtype=torch.long)
    full_hidden = _forward(
        model,
        full,
        torch.ones_like(full),
        torch.zeros_like(full, dtype=torch.float32),
        lengths,
        author_by_item,
    )
    empty, _ = model.forward_with_cache_embedded(
        type(cache)(k=cache.k[:, :, :0], v=cache.v[:, :, :0], seq_len=0),
        query_embedded,
    )
    assert torch.allclose(incremental[:, -1], full_hidden[:, -1], atol=1e-5, rtol=1e-5)
    assert torch.linalg.vector_norm(incremental[:, -1]).item() > 0
    assert torch.equal(empty, torch.zeros_like(empty))
