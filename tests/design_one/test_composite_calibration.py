"""Frozen Item mapping followed by response fitting at the actual serving queries."""
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from design_one import calibrate_nonlinear as calibration
from hstu_kvcache.design_one import composite as core
from hstu_kvcache.models import HSTU, HSTUConfig


def test_frozen_item_once_then_response_actual_queries_and_teacher(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND', 'torch')
    torch.set_num_threads(2)
    torch.manual_seed(172022)
    cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=2,
        num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
    rows, native = {}, {}
    item = calibration.make_adapter(current, 'kv_item')
    with torch.no_grad():
        for layer in item.layers:
            layer.output.weight.normal_(std=.01)
        for uid, length in enumerate((8, 5, 7, 6)):
            ids = torch.arange(1+uid*10, 1+uid*10+length)[None]
            dt = torch.ones_like(ids).float()
            dt[:, 0] = 0
            old = parent.compute_kv(ids, torch.ones_like(ids), dt)
            native[uid] = old
            rows[uid] = dict(parent=old, teacher=current.compute_kv(ids, torch.ones_like(ids), dt),
                item_features=current.lookup_item_embeddings(ids).detach(), candidates=torch.arange(1, 17), query_delta=2.)
    state = {name: value.clone() for name, value in item.state_dict().items()}
    foundation = {name: value.clone() for name, value in current.state_dict().items()}
    original = native[0].k.clone()
    costs = defaultdict(int)
    calibration.prepare_item_read_view(item, rows, list(rows), batch_size=2, device=torch.device('cpu'), costs=costs)
    expected = sum(module.token_forward_flops(length=sum(cache.seq_len for cache in native.values())) for module in item.layers)
    assert costs['frozen_item_view_flops'] == expected
    observed = {}
    collect = calibration.collect

    def capture(*args, **kwargs):
        result = collect(*args, **kwargs)
        observed[args[4]] = result
        return result

    monkeypatch.setattr(calibration, 'collect', capture)
    response, _ = calibration.fit(current, rows, [0, 1, 2], [3], 'nonlinear_response',
        batch_size=2, device=torch.device('cpu'), epochs=3, costs=costs)
    combined = core.FrozenItemResponseAdapter(item, response)
    with torch.no_grad():
        row, count = rows[0], torch.tensor([8.])
        mapped = combined.map_cache(native[0], row['item_features'])
        fitted, fitted_trace = calibration.score(current, mapped, row['candidates'][None], torch.tensor([2.]),
            history_override=response.make_history_override(count), trace=True)
        served, served_trace = calibration.score(current, native[0], row['candidates'][None], torch.tensor([2.]),
            history_override=combined.make_history_override(current, mapped), trace=True)
        torch.testing.assert_close(served, fitted)
        torch.testing.assert_close(served_trace.queries[1], fitted_trace.queries[1])
        torch.testing.assert_close(served_trace.queries[1][0], observed[1][0][0])
        q = observed[1][0][0:1]
        full_read = calibration.history_read(current.blocks[1].attn, q, row['teacher'].k[1], row['teacher'].v[1])
        mapped_read = calibration.history_read(current.blocks[1].attn, q, mapped.k[1], mapped.v[1])
        torch.testing.assert_close((full_read-mapped_read)[0]/8, observed[1][2][0])
    for name, value in item.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in item.parameters())
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, foundation[name], atol=0, rtol=0)
    torch.testing.assert_close(native[0].k, original, atol=0, rtol=0)
