import torch

from hstu_kvcache.models import HSTU, HSTUConfig


def _make_model(**kw):
    cfg = dict(
        num_items=50, num_behaviors=8, hidden_size=32, num_layers=2,
        num_heads=2, head_dim=16, max_seq_len=32,
    )
    cfg.update(kw)
    return HSTU(HSTUConfig(**cfg))


def _make_batch(B=2, L=8, device="cpu"):
    return {
        "item_ids": torch.randint(1, 51, (B, L), device=device),
        "behaviors": torch.randint(0, 9, (B, L), device=device),
        "time_deltas": torch.rand(B, L, device=device) * 100,
    }


def test_forward_shapes():
    model = _make_model()
    model.eval()
    batch = _make_batch()
    hidden, kv = model(batch["item_ids"], batch["behaviors"], batch["time_deltas"], return_kv=True)
    assert hidden.shape == (2, 8, 32)
    assert kv.k.shape[0] == 2  # num_layers
    assert kv.k.shape[1] == 2  # batch


def test_kv_deterministic():
    model = _make_model()
    batch = _make_batch()
    kv1 = model.compute_kv(**batch)
    kv2 = model.compute_kv(**batch)
    d = kv1.drift_norm(kv2)
    assert d["k_l2"] < 1e-5 and d["v_l2"] < 1e-5


def test_drift_nonzero_after_param_change():
    model = _make_model()
    batch = _make_batch()
    kv0 = model.compute_kv(**batch)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    kv1 = model.compute_kv(**batch)
    d = kv0.drift_norm(kv1)
    assert d["k_l2"] > 0.1 and d["v_l2"] > 0.1


def test_candidate_scoring():
    model = _make_model()
    model.eval()
    batch = _make_batch()
    hidden, _ = model(batch["item_ids"], batch["behaviors"], batch["time_deltas"])
    cands = torch.randint(1, 51, (2, 10))
    scores = model.score_candidates(hidden, cands)
    assert scores.shape == (2, 10)


def test_pointwise_attention_no_softmax():
    from hstu_kvcache.models.attention import PointwiseAttention, PointwiseAttentionConfig

    cfg = PointwiseAttentionConfig(hidden_size=16, num_heads=1, head_dim=16)
    attn = PointwiseAttention(cfg)
    attn.eval()
    x = torch.randn(1, 4, 16)
    out, (k, v) = attn(x, return_kv=True)
    assert out.shape == (1, 4, 16)
    assert k.shape == (1, 4, 16)
