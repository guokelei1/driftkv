"""Frozen item-feature calibration: alignment, fitting statistics and actual queries."""
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from design_one import calibrate_nonlinear as calibration
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTU, HSTUConfig


def test_item_fit_has_aligned_frozen_features_fit_only_normalization_and_actual_prefix(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND', 'torch')
    torch.set_num_threads(2)
    torch.manual_seed(172014)
    cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=2,
                    num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
    rows = {}
    with torch.no_grad():
        for uid, count in enumerate((8, 5, 7, 6)):
            ids = torch.arange(1+10*uid, 1+10*uid+count)[None]
            dt = torch.ones_like(ids).float()
            dt[:, 0] = 0
            rows[uid] = dict(parent=parent.compute_kv(ids, torch.ones_like(ids), dt),
                teacher=current.compute_kv(ids, torch.ones_like(ids), dt),
                candidates=torch.arange(1, 17), query_delta=2.,
                item_features=current.lookup_item_embeddings(ids).detach().cpu())
    foundation = {name: value.clone() for name, value in current.state_dict().items()}
    native = rows[0]['parent'].k.clone()
    observed = {}
    collect = calibration.collect

    def capture(*args, **kwargs):
        result = collect(*args, **kwargs)
        observed[args[4]] = result
        return result

    monkeypatch.setattr(calibration, 'collect', capture)
    adapter, diagnostics = calibration.fit(current, rows, [0, 1, 2], [3], 'kv_item',
        batch_size=2, device=torch.device('cpu'), epochs=3, costs=defaultdict(int))
    assert adapter.get_config()['hidden_width'] == 64
    assert all(record['fitting_queries'] == 16 for record in diagnostics['layers'])
    for layer, module in enumerate(adapter.layers):
        expected = torch.cat([torch.cat((rows[uid]['parent'].k[layer, 0],
            rows[uid]['parent'].v[layer, 0], rows[uid]['item_features'][0]), -1) for uid in (0, 1, 2)]).double()
        torch.testing.assert_close(module.input_center, expected.mean(0).float())
        torch.testing.assert_close(module.output_scale, expected.std(0, correction=0)[:64].clamp_min(1e-4).float())
    with torch.no_grad():
        override = calibration.prepared_override(current, adapter, rows[0]['parent'],
            torch.tensor([8.]), 'kv_item', 1, item_features=rows[0]['item_features'])
        _, trace = score(current, rows[0]['parent'], rows[0]['candidates'][None],
                         torch.tensor([2.]), history_override=override, trace=True)
        torch.testing.assert_close(trace.queries[1][0], observed[1][0][0])
    torch.testing.assert_close(rows[0]['parent'].k, native, atol=0, rtol=0)
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, foundation[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in current.parameters())

    # Both refinements start from identical learned weights and units. Refitting
    # statistics, even on the same history, would violate this comparison.
    inherited = adapter.export_state()
    units = [record['residual_rate_rms'] for record in diagnostics['layers']]
    buffers = {name: value.clone() for name, value in adapter.named_buffers()}

    def reject_normalization(*args, **kwargs):
        raise AssertionError('warm refinement must preserve inherited normalization')

    monkeypatch.setattr(calibration, 'normalize_layer', reject_normalization)
    for objective in ('response', 'logit'):
        warm = type(adapter).from_state_dict(inherited)
        initial_weights = [{name: value.clone() for name, value in layer.named_parameters()}
                           for layer in warm.layers]
        costs = defaultdict(int, inherited_calibration_flops=123456)
        if objective == 'response':
            warm, result = calibration.fit(current, rows, [0, 1, 2], [3], 'kv_item',
                adapter=warm, residual_rate_rms=units, learning_rate=.0001,
                batch_size=2, device=torch.device('cpu'), epochs=2, costs=costs)
            assert [record['residual_rate_rms'] for record in result['layers']] == units
            with torch.no_grad():
                override = calibration.prepared_override(current, warm, rows[0]['parent'],
                    torch.tensor([8.]), 'kv_item', 1, item_features=rows[0]['item_features'])
                _, trace = score(current, rows[0]['parent'], rows[0]['candidates'][None],
                    torch.tensor([2.]), history_override=override, trace=True)
                torch.testing.assert_close(trace.queries[1][0], observed[1][0][0])
        else:
            warm, result = calibration.refine_logits(current, warm, rows, [0, 1, 2], [3],
                method='kv_item', batch_size=2, device=torch.device('cpu'), epochs=2, costs=costs)
            assert all(value > 0 for value in result['joint_logit']['first_step_layer_gradient_norms'])
        for name, value in warm.named_buffers():
            torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
        for layer, before in zip(warm.layers, initial_weights, strict=True):
            assert any(not torch.equal(value, before[name]) for name, value in layer.named_parameters())
        assert costs['inherited_calibration_flops'] == 123456
        assert costs['normalization_statistics_flops'] == 0
        assert result['validation']['full_logit_mse'] >= 0
        torch.testing.assert_close(rows[0]['parent'].k, native, atol=0, rtol=0)
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, foundation[name], atol=0, rtol=0)
        assert all(parameter.grad is None for parameter in current.parameters())
