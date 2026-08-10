import torch
import torch.nn.functional as F

from hstu_kvcache.models import DenseHSTUV2, DenseHSTUV2Config


def _model() -> DenseHSTUV2:
    torch.manual_seed(7)
    model = DenseHSTUV2(
        DenseHSTUV2Config(
            num_items=100,
            num_behaviors=2,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
            input_dropout=0.0,
        )
    )
    model.eval()
    return model


def _inputs(batch: int = 3, sequence: int = 12):
    torch.manual_seed(11)
    return (
        torch.randint(1, 101, (batch, sequence)),
        torch.ones(batch, sequence, dtype=torch.long),
        torch.zeros(batch, sequence),
    )


def test_dense_hstu_v2_incremental_matches_full() -> None:
    model = _model()
    item_ids, behaviors, deltas = _inputs()
    full, _ = model(item_ids, behaviors, deltas)
    cache = model.compute_kv(item_ids[:, :8], behaviors[:, :8], deltas[:, :8])
    incremental, _ = model.forward_with_cache(
        cache,
        item_ids[:, 8:],
        behaviors[:, 8:],
        deltas[:, 8:],
    )
    assert torch.allclose(incremental, full[:, 8:], atol=2e-5, rtol=2e-5)


def test_dense_hstu_v2_padding_does_not_change_valid_hidden() -> None:
    model = _model()
    item_ids, behaviors, deltas = _inputs(batch=1, sequence=7)
    short, _ = model(item_ids, behaviors, deltas)
    padded_items = F.pad(item_ids, (0, 5))
    padded_behaviors = F.pad(behaviors, (0, 5))
    padded_deltas = F.pad(deltas, (0, 5))
    padded, _ = model(
        padded_items,
        padded_behaviors,
        padded_deltas,
        lengths=torch.tensor([7]),
    )
    assert torch.allclose(short, padded[:, :7], atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(padded[:, 7:]) == 0


def test_dense_hstu_v2_stale_cache_changes_hidden() -> None:
    model = _model()
    item_ids, behaviors, deltas = _inputs(batch=2, sequence=10)
    stale = model.compute_kv(item_ids[:, :8], behaviors[:, :8], deltas[:, :8])
    with torch.no_grad():
        for block in model.blocks:
            block.uvqk.weight.add_(0.05 * torch.randn_like(block.uvqk.weight))
    fresh = model.compute_kv(item_ids[:, :8], behaviors[:, :8], deltas[:, :8])
    stale_hidden, _ = model.forward_with_cache(
        stale,
        item_ids[:, 8:],
        behaviors[:, 8:],
        deltas[:, 8:],
    )
    fresh_hidden, _ = model.forward_with_cache(
        fresh,
        item_ids[:, 8:],
        behaviors[:, 8:],
        deltas[:, 8:],
    )
    assert not torch.allclose(stale_hidden, fresh_hidden, atol=1e-4, rtol=1e-4)


def test_dense_hstu_v2_normalized_scoring() -> None:
    model = _model()
    item_ids, behaviors, deltas = _inputs(batch=2, sequence=5)
    hidden, _ = model(item_ids, behaviors, deltas)
    candidates = torch.tensor([[1, 2, 3], [4, 5, 6]])
    actual = model.score_candidates(
        hidden,
        candidates,
        normalize=True,
        temperature=0.05,
    )
    expected = torch.einsum(
        "bh,bch->bc",
        F.normalize(hidden[:, -1], dim=-1),
        F.normalize(model.item_emb.weight[candidates], dim=-1),
    ) / 0.05
    assert torch.allclose(actual, expected)
