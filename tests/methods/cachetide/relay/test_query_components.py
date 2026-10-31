"""The query-component probe follows native construction and pre-cut causality."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
from hstu_kvcache.models import HSTU, HSTUConfig
from experiments.archive.query_relay.diagnose_query_components import (
    first_query_from_raw, normalize, pre_cut_gaps, query_components, vector_metrics,
)


@torch.no_grad()
def test_components_exact_native_query_and_pad_contrast(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(17)
    model = HSTU(HSTUConfig(num_items=20, num_behaviors=2, hidden_size=24, num_layers=1,
        num_heads=6, max_seq_len=8, input_dropout=0, attn_dropout=0)).double().eval()
    ids, delta = torch.tensor([[0, 3, 4, 9]]), torch.tensor([321.], dtype=torch.float64)
    components = query_components(model, ids, delta)
    token = model.embed_query_tokens(ids, delta)
    torch.testing.assert_close(model.in_proj(sum(components.values())), token, rtol=0, atol=0)
    q = first_query_from_raw(model, sum(components.values()))
    wanted = model.blocks[0].attn._project(model.blocks[0].norm(token))[0]
    torch.testing.assert_close(q, wanted, rtol=0, atol=0)
    pad_q = first_query_from_raw(model, sum(query_components(model, torch.zeros_like(ids), delta).values()))
    assert not torch.count_nonzero(q[:, :, :1] - pad_q[:, :, :1])
    metric = vector_metrics(normalize(q - pad_q))
    assert metric["candidate_std_rms"] > 0


def test_gap_distribution_excludes_future_and_respects_timestamp_ties():
    times = np.array([0, 2, 2, 4, 10, 30])
    result = pre_cut_gaps(times, 10)
    assert result["cut_gap"] == 6
    assert result["retained_events"] == 4
    np.testing.assert_array_equal(result["adjacent"], [2, 0, 2])
    np.testing.assert_array_equal(result["strict"], [2, 2, 2])
    assert result["representative"] == 2
