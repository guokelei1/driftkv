import torch

from hstu_kvcache.migration import (
    exact_hidden_and_kv,
    exact_hidden_and_kv_from_item_embeddings,
    exact_kv,
)
from hstu_kvcache.models import HSTU, HSTUConfig


def _model() -> HSTU:
    return HSTU(
        HSTUConfig(
            num_items=80,
            num_behaviors=9,
            hidden_size=32,
            num_layers=3,
            num_heads=2,
            head_dim=16,
            max_seq_len=16,
            input_dropout=0.2,
            attn_dropout=0.1,
        )
    )


def _batch(
    batch_size: int = 3,
    seq_len: int = 7,
) -> dict[str, torch.Tensor]:
    return {
        "item_ids": torch.randint(
            1,
            81,
            (batch_size, seq_len),
        ),
        "behaviors": torch.randint(
            0,
            10,
            (batch_size, seq_len),
        ),
        "time_deltas": torch.rand(batch_size, seq_len) * 100.0,
        "lengths": torch.tensor([seq_len, seq_len - 3, 0])[
            :batch_size
        ],
    }


def test_split_exact_frontend_is_bitwise_equivalent() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    batch = _batch()
    state_keys = tuple(model.state_dict())
    hidden, cache = model(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
        lengths=batch["lengths"],
    )
    vectors = model.lookup_item_embeddings(batch["item_ids"])
    split_hidden, split_cache = model.forward_embedded(
        model.combine_input_features(
            vectors,
            batch["behaviors"],
            batch["time_deltas"],
        ),
        return_kv=True,
        lengths=batch["lengths"],
    )
    assert cache is not None
    assert split_cache is not None
    assert torch.equal(hidden, split_hidden)
    assert torch.equal(cache.k, split_cache.k)
    assert torch.equal(cache.v, split_cache.v)
    assert cache.seq_len == split_cache.seq_len
    assert tuple(model.state_dict()) == state_keys


def test_compute_kv_from_item_embeddings_bypasses_lookup_and_restores_mode() -> None:
    torch.manual_seed(5)
    model = _model()
    batch = _batch()
    vectors = model.lookup_item_embeddings(batch["item_ids"]).detach()
    calls = []
    handle = model.item_emb.register_forward_hook(
        lambda module, inputs, output: calls.append(inputs[0].shape)
    )
    embedded = model.compute_kv_from_item_embeddings(
        vectors,
        batch["behaviors"],
        batch["time_deltas"],
        lengths=batch["lengths"],
    )
    assert model.training
    assert calls == []
    direct = model.compute_kv(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        lengths=batch["lengths"],
    )
    handle.remove()
    assert model.training
    assert calls == [batch["item_ids"].shape]
    assert torch.equal(direct.k, embedded.k)
    assert torch.equal(direct.v, embedded.v)


def test_training_forward_keeps_dropout_and_gradients() -> None:
    torch.manual_seed(7)
    model = _model().train()
    batch = _batch(batch_size=2)
    rng = torch.random.get_rng_state()
    direct_hidden, _ = model(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        lengths=batch["lengths"],
    )
    torch.random.set_rng_state(rng)
    vectors = model.lookup_item_embeddings(batch["item_ids"])
    split_hidden, _ = model.forward_embedded(
        model.combine_input_features(
            vectors,
            batch["behaviors"],
            batch["time_deltas"],
        ),
        lengths=batch["lengths"],
    )
    assert torch.equal(direct_hidden, split_hidden)
    direct_hidden.square().mean().backward()
    assert model.item_emb.weight.grad is not None


def test_library_exact_helpers_match_public_path_and_scores() -> None:
    torch.manual_seed(11)
    model = _model().train()
    batch = _batch()
    hidden, cache = exact_hidden_and_kv(
        model,
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        batch["lengths"],
    )
    vectors = model.lookup_item_embeddings(batch["item_ids"])
    split_hidden, split_cache = (
        exact_hidden_and_kv_from_item_embeddings(
            model,
            vectors,
            batch["behaviors"],
            batch["time_deltas"],
            batch["lengths"],
        )
    )
    kv_only = exact_kv(
        model,
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        batch["lengths"],
    )
    candidates = torch.arange(1, 81).repeat(3, 1)
    direct_scores = model.score_candidates(
        hidden,
        candidates,
        batch["lengths"],
    )
    split_scores = model.score_candidates(
        split_hidden,
        candidates,
        batch["lengths"],
    )
    assert model.training
    assert torch.equal(hidden, split_hidden)
    assert torch.equal(cache.k, split_cache.k)
    assert torch.equal(cache.v, split_cache.v)
    assert torch.equal(cache.k, kv_only.k)
    assert torch.equal(cache.v, kv_only.v)
    assert torch.equal(direct_scores, split_scores)
    assert torch.equal(
        direct_scores.topk(20).indices,
        split_scores.topk(20).indices,
    )


def test_incremental_embedded_frontend_is_bitwise_equivalent() -> None:
    torch.manual_seed(13)
    model = _model().train()
    prefix = _batch(batch_size=2, seq_len=5)
    prefix_cache = model.compute_kv(
        prefix["item_ids"],
        prefix["behaviors"],
        prefix["time_deltas"],
        lengths=prefix["lengths"],
    )
    item_ids = torch.randint(1, 81, (2, 3))
    behaviors = torch.randint(0, 10, (2, 3))
    time_deltas = torch.rand(2, 3) * 50.0
    direct_hidden, direct_cache = model.forward_with_cache(
        prefix_cache,
        item_ids,
        behaviors,
        time_deltas,
    )
    vectors = model.lookup_item_embeddings(item_ids)
    split_hidden, split_cache = (
        model.forward_with_cache_from_item_embeddings(
            prefix_cache,
            vectors,
            behaviors,
            time_deltas,
        )
    )
    assert model.training
    assert torch.equal(direct_hidden, split_hidden)
    assert torch.equal(direct_cache.k, split_cache.k)
    assert torch.equal(direct_cache.v, split_cache.v)
    assert direct_cache.seq_len == split_cache.seq_len
