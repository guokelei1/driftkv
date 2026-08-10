import copy

import pytest
import torch

from hstu_kvcache.models import (
    HSTU,
    HSTUConfig,
    apply_attention_coordinate_gauge_,
    apply_attention_coordinate_scale_,
)


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


def test_prediction_catalog_can_be_smaller_than_context_vocabulary():
    cfg = HSTUConfig(
        num_items=100,
        num_prediction_items=40,
        num_behaviors=8,
    )

    assert cfg.num_prediction_items == 40

    with pytest.raises(ValueError):
        HSTUConfig(
            num_items=40,
            num_prediction_items=41,
            num_behaviors=8,
        )


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
    d = kv1.difference_metrics(kv2)
    assert d["k_l2"] < 1e-5 and d["v_l2"] < 1e-5


def test_reference_hstu_cache_matches_full_forward():
    torch.manual_seed(59)
    model = _make_model(
        input_dropout=0.0,
        activation="silu",
        block_variant="hstu_reference",
        relative_position_bias=True,
        causal_diagonal="exclusive",
    )
    model.eval()
    batch = _make_batch(B=2, L=8)
    full, _ = model(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
    )
    prefix = 6
    cache = model.compute_kv(
        batch["item_ids"][:, :prefix],
        batch["behaviors"][:, :prefix],
        batch["time_deltas"][:, :prefix],
    )
    incremental, _ = model.forward_with_cache(
        cache,
        batch["item_ids"][:, prefix:],
        batch["behaviors"][:, prefix:],
        batch["time_deltas"][:, prefix:],
    )
    assert torch.allclose(incremental, full[:, prefix:], atol=2e-5, rtol=2e-5)


def test_cache_changes_after_param_change():
    model = _make_model()
    batch = _make_batch()
    kv0 = model.compute_kv(**batch)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    kv1 = model.compute_kv(**batch)
    d = kv0.difference_metrics(kv1)
    assert d["k_l2"] > 0.1 and d["v_l2"] > 0.1


def test_candidate_scoring():
    model = _make_model()
    model.eval()
    batch = _make_batch()
    hidden, _ = model(batch["item_ids"], batch["behaviors"], batch["time_deltas"])
    cands = torch.randint(1, 51, (2, 10))
    scores = model.score_candidates(hidden, cands)
    assert scores.shape == (2, 10)


def test_untied_prediction_embedding_is_independent():
    model = _make_model(num_prediction_items=40, tie_item_embeddings=False)
    assert model.output_emb is not None
    assert model.item_emb.weight.shape == (51, 32)
    assert model.prediction_item_weight.shape == (41, 32)
    before = model.prediction_item_weight.detach().clone()
    with torch.no_grad():
        model.item_emb.weight.add_(1.0)
    assert torch.equal(model.prediction_item_weight, before)
    hidden = torch.randn(2, 32)
    candidates = torch.randint(1, 41, (2, 5))
    expected = model.output_emb.score(hidden, candidates)
    assert torch.equal(model.score_hidden(hidden, candidates), expected)


def test_pointwise_attention_no_softmax():
    from hstu_kvcache.models.attention import PointwiseAttention, PointwiseAttentionConfig

    cfg = PointwiseAttentionConfig(hidden_size=16, num_heads=1, head_dim=16)
    attn = PointwiseAttention(cfg)
    attn.eval()
    x = torch.randn(1, 4, 16)
    out, (k, v) = attn(x, return_kv=True)
    assert out.shape == (1, 4, 16)
    assert k.shape == (1, 4, 16)


def test_attention_coordinate_gauge_preserves_fresh_and_changes_stale_cache():
    torch.manual_seed(71)
    source = _make_model(input_dropout=0.0)
    source.eval()
    transformed = copy.deepcopy(source)
    batch = _make_batch(B=2, L=8)
    source_hidden, source_cache = source(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
    )
    certificate = apply_attention_coordinate_gauge_(transformed, 0.3)
    transformed.eval()
    transformed_hidden, transformed_cache = transformed(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
    )
    assert certificate["maximum_orthogonality_error"] < 1e-6
    assert torch.allclose(source_hidden, transformed_hidden, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(source_cache.k, transformed_cache.k)
    assert not torch.allclose(source_cache.v, transformed_cache.v)

    prefix = 6
    transformed_prefix = transformed.compute_kv(
        batch["item_ids"][:, :prefix],
        batch["behaviors"][:, :prefix],
        batch["time_deltas"][:, :prefix],
    )
    source_prefix = source.compute_kv(
        batch["item_ids"][:, :prefix],
        batch["behaviors"][:, :prefix],
        batch["time_deltas"][:, :prefix],
    )
    suffix = transformed.embed_inputs(
        batch["item_ids"][:, prefix:],
        batch["behaviors"][:, prefix:],
        batch["time_deltas"][:, prefix:],
    )
    fresh_suffix, _ = transformed.forward_with_cache_embedded(
        transformed_prefix,
        suffix,
    )
    stale_suffix, _ = transformed.forward_with_cache_embedded(source_prefix, suffix)
    assert torch.allclose(
        fresh_suffix,
        transformed_hidden[:, prefix:],
        atol=2e-5,
        rtol=2e-5,
    )
    assert float((fresh_suffix - stale_suffix).abs().max().item()) > 1e-3


def test_attention_coordinate_scale_preserves_fresh_and_attenuates_stale_cache():
    torch.manual_seed(73)
    source = _make_model(input_dropout=0.0)
    source.eval()
    transformed = copy.deepcopy(source)
    batch = _make_batch(B=2, L=8)
    source_hidden, _ = source(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
    )
    certificate = apply_attention_coordinate_scale_(
        transformed,
        key_log_scale=0.2,
        value_log_scale=0.2,
    )
    transformed.eval()
    transformed_hidden, _ = transformed(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
    )
    assert certificate["key_factor"] == pytest.approx(torch.exp(torch.tensor(0.2)).item())
    assert torch.allclose(source_hidden, transformed_hidden, atol=2e-5, rtol=2e-5)

    prefix = 6
    source_prefix = source.compute_kv(
        batch["item_ids"][:, :prefix],
        batch["behaviors"][:, :prefix],
        batch["time_deltas"][:, :prefix],
    )
    transformed_prefix = transformed.compute_kv(
        batch["item_ids"][:, :prefix],
        batch["behaviors"][:, :prefix],
        batch["time_deltas"][:, :prefix],
    )
    suffix = transformed.embed_inputs(
        batch["item_ids"][:, prefix:],
        batch["behaviors"][:, prefix:],
        batch["time_deltas"][:, prefix:],
    )
    fresh_suffix, _ = transformed.forward_with_cache_embedded(transformed_prefix, suffix)
    stale_suffix, _ = transformed.forward_with_cache_embedded(source_prefix, suffix)
    assert torch.allclose(
        fresh_suffix,
        transformed_hidden[:, prefix:],
        atol=2e-5,
        rtol=2e-5,
    )
    assert float((fresh_suffix - stale_suffix).abs().max().item()) > 1e-3
