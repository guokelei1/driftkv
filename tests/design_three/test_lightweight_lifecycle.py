"""Incremental Value codes preserve the native multi-release trajectory."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design_system.evaluate_lightweight import LightweightSession
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.lightweight import LightweightValueAdapter
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig


@torch.inference_mode()
def test_incremental_codes_match_full_reencode_across_releases(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(20261004)
    config = HSTUConfig(num_items=64, num_behaviors=2, hidden_size=8,
        num_layers=3, num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    models = [HSTU(config).eval().requires_grad_(False) for _ in range(3)]
    adapters = [LightweightValueAdapter(3, 2, 4, rank=2, response_hidden_width=0,
        max_length=4).eval().requires_grad_(False) for _ in range(2)]
    for adapter in adapters:
        for layer in adapter.layers:
            layer.decoder.normal_(std=.02)
    events = (np.arange(24), np.arange(1, 25), np.zeros(24, dtype=np.int64))
    plain = NativeTrajectory(models[0], 2, events, 4, max_length=4, append_band_size=4)
    trajectory = NativeTrajectory(models[0], 2, events, 4, max_length=4, append_band_size=4)
    session = LightweightSession(trajectory, kernel="torch", exit_when_clear=False)
    for index, (start, end) in enumerate(((4, 9), (9, 18))):
        model, adapter = models[index+1], adapters[index]
        session.publish(model, index+3, start, adapter)
        plain.switch(model, index+3, start)
        for at in (start, start+1, end):
            session.advance(at)
            plain.advance(at)
            torch.testing.assert_close(session.cache.k, plain.cache.k, atol=0, rtol=0)
            torch.testing.assert_close(session.cache.v, plain.cache.v, atol=0, rtol=0)
            ids = torch.as_tensor(plain.retained_events[1])[None]
            full_state = adapter.encode_cache(plain.cache, model.lookup_item_embeddings(ids))
            torch.testing.assert_close(session.state.codes, full_state.codes, atol=2e-6, rtol=2e-5)
            candidates = torch.tensor([[30, 31]])
            delta = torch.tensor([at-plain.last_timestamp], dtype=torch.float32)
            full_logits = score(model, plain.cache, candidates, delta,
                history_read_fn=adapter.make_history_read(full_state))[0]
            torch.testing.assert_close(session.score(at, candidates), full_logits, atol=2e-6, rtol=2e-5)
            assert session.state.seq_len == session.cache.seq_len == 4
    assert session.costs["initial_encode_flops"] == 2*adapters[0].estimate_flops(tokens=4)["token_transform"]
    assert session.memory["compact_peak_bytes"] == 3*4*2*2*4
