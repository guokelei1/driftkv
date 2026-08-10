import torch

from hstu_kvcache.models import (
    FeatureCrossKV,
    FeatureCrossKVConfig,
    TargetAwareKV,
    TargetAwareKVConfig,
)


def _model() -> TargetAwareKV:
    torch.manual_seed(7)
    model = TargetAwareKV(
        TargetAwareKVConfig(num_items=20, hidden_size=16, input_dropout=0.0)
    )
    model.eval()
    return model


def test_target_aware_incremental_matches_full() -> None:
    model = _model()
    items = torch.tensor([[1, 2, 1, 3, 1, 4]])
    labels = torch.tensor([[1, 0, 1, 0, 1, 0]])
    full, _ = model(items, labels)
    cache = model.compute_kv(items[:, :3], labels[:, :3])
    suffix, _ = model.forward_with_cache(cache, items[:, 3:], labels[:, 3:])
    assert torch.allclose(full[:, 3:], suffix, atol=1e-6)


def test_target_aware_does_not_read_current_or_future_label() -> None:
    model = _model()
    items = torch.tensor([[1, 2, 1, 3]])
    first_labels = torch.tensor([[1, 0, 1, 0]])
    second_labels = torch.tensor([[1, 1, 0, 1]])
    first, _ = model(items, first_labels)
    second, _ = model(items, second_labels)
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6)
    assert not torch.allclose(first[:, 2:], second[:, 2:])


def test_target_aware_stale_cache_changes_score() -> None:
    model = _model()
    prefix_items = torch.tensor([[1, 2, 1]])
    prefix_labels = torch.tensor([[1, 0, 1]])
    suffix_items = torch.tensor([[1, 3]])
    suffix_labels = torch.tensor([[0, 1]])
    stale = model.compute_kv(prefix_items, prefix_labels)
    with torch.no_grad():
        model.value_proj.weight.add_(0.1 * torch.randn_like(model.value_proj.weight))
    fresh = model.compute_kv(prefix_items, prefix_labels)
    stale_logits, _ = model.forward_with_cache(stale, suffix_items, suffix_labels)
    fresh_logits, _ = model.forward_with_cache(fresh, suffix_items, suffix_labels)
    assert not torch.allclose(stale_logits, fresh_logits)


def test_feature_cross_incremental_matches_full_and_stale_changes_score() -> None:
    torch.manual_seed(23)
    model = FeatureCrossKV(FeatureCrossKVConfig(num_items=20, hidden_size=8))
    model.eval()
    items = torch.tensor([[1, 2, 1, 3, 1, 4]])
    labels = torch.tensor([[1, 0, 1, 0, 1, 0]])
    full, _ = model(items, labels)
    stale = model.compute_kv(items[:, :3], labels[:, :3])
    incremental, _ = model.forward_with_cache(stale, items[:, 3:], labels[:, 3:])
    assert torch.allclose(full[:, 3:], incremental, atol=1e-6)
    with torch.no_grad():
        model.outcome_emb.weight.add_(0.1 * torch.randn_like(model.outcome_emb.weight))
    fresh = model.compute_kv(items[:, :3], labels[:, :3])
    stale_logits, _ = model.forward_with_cache(stale, items[:, 3:], labels[:, 3:])
    fresh_logits, _ = model.forward_with_cache(fresh, items[:, 3:], labels[:, 3:])
    assert not torch.allclose(stale_logits, fresh_logits)
