"""Test KV cache reuse: incremental forward must equal full forward (same model).

This is the correctness foundation: if theta_old == theta_new, prefix-reuse
inference must produce IDENTICAL hidden states to a full recomputation.
Only then can we trust the stale-KV loss measurement (when theta_old != theta_new).
"""

import torch

from hstu_kvcache.models import HSTU, HSTUConfig


def _make(batch_size=2, seq_len=16, **kw):
    cfg = dict(
        num_items=100, num_behaviors=8, hidden_size=64, num_layers=3,
        num_heads=4, head_dim=16, max_seq_len=64,
    )
    cfg.update(kw)
    model = HSTU(HSTUConfig(**cfg))
    model.eval()
    torch.manual_seed(42)
    item_ids = torch.randint(1, 101, (batch_size, seq_len))
    behaviors = torch.randint(0, 9, (batch_size, seq_len))
    time_deltas = torch.rand(batch_size, seq_len) * 100
    return model, item_ids, behaviors, time_deltas


def test_incremental_equals_full_same_model():
    """Cache prefix + compute suffix == full forward, when same model."""
    model, item_ids, behaviors, time_deltas = _make(seq_len=16)
    n_prefix = 10

    # full forward
    full_hidden, _ = model(item_ids, behaviors, time_deltas, return_kv=False)
    full_last = full_hidden[:, -1, :]  # [B, H]

    # incremental: cache prefix, then add suffix
    prefix_kv = model.compute_kv(item_ids[:, :n_prefix], behaviors[:, :n_prefix], time_deltas[:, :n_prefix])
    inc_hidden, _ = model.forward_with_cache(
        prefix_kv,
        item_ids[:, n_prefix:],
        behaviors[:, n_prefix:],
        time_deltas[:, n_prefix:],
    )
    inc_last = inc_hidden[:, -1, :]  # [B, H]

    diff = (full_last - inc_last).abs().max().item()
    assert diff < 1e-4, f"incremental forward differs from full forward: max diff={diff}"


def test_incremental_single_token():
    """Adding one token at a time via cache == full forward."""
    model, item_ids, behaviors, time_deltas = _make(seq_len=8)
    full_hidden, _ = model(item_ids, behaviors, time_deltas, return_kv=False)

    # build up one token at a time
    kv = model.compute_kv(item_ids[:, :1], behaviors[:, :1], time_deltas[:, :1])
    hidden = None
    for t in range(1, 8):
        hidden, kv = model.forward_with_cache(
            kv,
            item_ids[:, t : t + 1],
            behaviors[:, t : t + 1],
            time_deltas[:, t : t + 1],
        )
    inc_last = hidden[:, -1, :]
    full_last = full_hidden[:, -1, :]
    diff = (full_last - inc_last).abs().max().item()
    assert diff < 1e-4, f"single-token incremental differs: max diff={diff}"


def test_stale_kv_differs_from_fresh():
    """When theta changes, stale-KV inference must DIFFER from full recompute."""
    model, item_ids, behaviors, time_deltas = _make(seq_len=12)
    n_prefix = 8

    # cache with original model
    prefix_kv = model.compute_kv(item_ids[:, :n_prefix], behaviors[:, :n_prefix], time_deltas[:, :n_prefix])

    # perturb model (simulate theta update)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    # full recompute with new model
    full_hidden, _ = model(item_ids, behaviors, time_deltas, return_kv=False)
    full_last = full_hidden[:, -1, :]

    # stale KV reuse: old prefix KV + new model for suffix
    inc_hidden, _ = model.forward_with_cache(
        prefix_kv,
        item_ids[:, n_prefix:],
        behaviors[:, n_prefix:],
        time_deltas[:, n_prefix:],
    )
    inc_last = inc_hidden[:, -1, :]

    diff = (full_last - inc_last).abs().max().item()
    assert diff > 1e-3, f"stale KV should differ from fresh recompute: diff={diff}"


def test_updated_kv_cache_length():
    """The returned KV cache from forward_with_cache has correct length."""
    model, item_ids, behaviors, time_deltas = _make(seq_len=10)
    n_prefix = 7
    n_suffix = 3

    prefix_kv = model.compute_kv(item_ids[:, :n_prefix], behaviors[:, :n_prefix], time_deltas[:, :n_prefix])
    assert prefix_kv.k.shape[2] == n_prefix

    _, updated_kv = model.forward_with_cache(
        prefix_kv,
        item_ids[:, n_prefix:],
        behaviors[:, n_prefix:],
        time_deltas[:, n_prefix:],
    )
    assert updated_kv.k.shape[2] == n_prefix + n_suffix
    assert updated_kv.seq_len == n_prefix + n_suffix
