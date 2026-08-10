import numpy as np
import pandas as pd
import torch

from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming.kuairand_history_residual import (
    HistoryResidualEngagementModel,
    load_experiment_plan,
)


def test_single_query_has_zero_history_residual() -> None:
    torch.manual_seed(11)
    model = HistoryResidualEngagementModel(
        HSTUConfig(
            num_items=20,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            input_dropout=0.0,
        )
    )
    model.eval()
    item = torch.tensor([[7]])
    behavior = torch.tensor([[3]])
    delta = torch.tensor([[2.0]])
    hidden, _ = model(item, behavior, delta)
    logits = model.residual_logits(hidden, item, delta)
    assert torch.allclose(logits, model.engagement_head.bias.view(1, 1), atol=1e-6)


def test_history_changes_residual_score_and_stale_cache() -> None:
    torch.manual_seed(19)
    model = HistoryResidualEngagementModel(
        HSTUConfig(
            num_items=20,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            input_dropout=0.0,
        )
    )
    model.eval()
    prefix_items = torch.tensor([[2, 4, 6]])
    prefix_behaviors = torch.tensor([[1, 2, 1]])
    prefix_deltas = torch.tensor([[0.0, 1.0, 1.0]])
    query_item = torch.tensor([[7]])
    query_behavior = torch.tensor([[3]])
    query_delta = torch.tensor([[2.0]])
    stale = model.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    history_hidden, _ = model.forward_with_cache(
        stale,
        query_item,
        query_behavior,
        query_delta,
    )
    history_score = model.residual_logits(history_hidden, query_item, query_delta)
    with torch.no_grad():
        model.backbone.blocks[0].attn.q_proj.weight.add_(0.1)
    fresh = model.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    fresh_hidden, _ = model.forward_with_cache(
        fresh,
        query_item,
        query_behavior,
        query_delta,
    )
    stale_hidden, _ = model.forward_with_cache(
        stale,
        query_item,
        query_behavior,
        query_delta,
    )
    fresh_score = model.residual_logits(fresh_hidden, query_item, query_delta)
    stale_score = model.residual_logits(stale_hidden, query_item, query_delta)
    assert not torch.allclose(history_score, model.engagement_head.bias.view(1, 1))
    assert not torch.allclose(fresh_score, stale_score)


def test_author_plan_uses_fixed_semantic_hash(monkeypatch) -> None:
    class Trace:
        def __init__(self) -> None:
            self.interactions = pd.DataFrame(
                {
                    "video_id": [10, 11, 12],
                    "date": ["d0", "d0", "d1"],
                    "time_ms": [1, 2, 3],
                }
            )
            self.num_items = 9
            self.num_prediction_items = 9
            self.context_hash_buckets = 0
            self.item_map = {10: 1}

    class Plan:
        def __init__(self) -> None:
            self.trace = Trace()
            self.daily_segments = {}

        @property
        def num_items(self):
            return self.trace.num_items

        @property
        def num_prediction_items(self):
            return self.trace.num_prediction_items

    monkeypatch.setattr(
        "hstu_kvcache.streaming.kuairand_history_residual.load_plan",
        lambda document: (Plan(), {}),
    )
    monkeypatch.setattr(
        "hstu_kvcache.streaming.kuairand_history_residual.pd.read_csv",
        lambda *args, **kwargs: pd.DataFrame(
            {"video_id": [10, 11], "author_id": [100, 100]}
        ),
    )
    monkeypatch.setattr(
        "hstu_kvcache.streaming.kuairand_history_residual.file_sha256",
        lambda path: "bound",
    )
    plan, metadata = load_experiment_plan(
        {
            "data": {
                "semantic_token": "author_hash",
                "semantic_hash_buckets": 65536,
                "feature_source": {"path": "features.csv"},
            }
        }
    )
    tokens = plan.trace.interactions["item_idx"].to_numpy()
    assert tokens[0] == tokens[1]
    assert tokens[2] == 1
    assert np.all((tokens >= 1) & (tokens <= 65536))
    assert metadata["missing_feature_rows"] == 1
