"""Small frozen-backbone integration for the two response-distilled candidates."""

from collections import defaultdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from design_one import calibrate_nonlinear as calibration
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.nonlinear import append_context
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.state_transition import append_with_rolling_band


@pytest.mark.parametrize("method,hidden_width,queries", (
    ("nonlinear_response", None, 16), ("kv_view", None, 16), ("kv_view", 64, 64)))
def test_fit_uses_actual_queries_fit_only_coordinates_and_frozen_native_state(monkeypatch, method, hidden_width, queries):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(172010)
        cfg = HSTUConfig(num_items=max(64, queries+1), num_behaviors=3, hidden_size=32, num_layers=2,
                         num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
        parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
        rows = {}
        with torch.no_grad():
            for uid, count in enumerate((8, 5, 7, 6)):
                items = torch.arange(1+uid*10, 1+uid*10+count)[None]
                behaviors = torch.ones_like(items)
                dt = torch.ones_like(items).float()
                dt[:, 0] = 0
                rows[uid] = dict(parent=parent.compute_kv(items, behaviors, dt),
                    teacher=current.compute_kv(items, behaviors, dt),
                    candidates=torch.arange(1, queries+1), query_delta=2.)
        native_before = rows[0]["parent"].k.clone()
        foundation_before = {name: value.clone() for name, value in current.state_dict().items()}
        captured = {}
        original = calibration.collect

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            captured[args[4]] = result
            return result

        monkeypatch.setattr(calibration, "collect", observe)
        costs = defaultdict(int)
        adapter, diagnostics = calibration.fit(current, rows, [0, 1, 2], [3], method,
            batch_size=2, device=torch.device("cpu"), epochs=3, costs=costs, hidden_width=hidden_width)
        assert adapter.get_config()["hidden_width"] == (calibration.WIDTHS[method] if hidden_width is None else hidden_width)
        assert all(record["fitting_queries"] == queries for record in diagnostics["layers"])
        assert all(record["final_train"]["objective"] < record["initial_train"]["objective"]
                   for record in diagnostics["layers"])
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in current.parameters())
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, foundation_before[name], atol=0, rtol=0)
        torch.testing.assert_close(rows[0]["parent"].k, native_before, atol=0, rtol=0)
        with torch.no_grad():
            count = torch.tensor([8.])
            override = calibration.prepared_override(current, adapter, rows[0]["parent"], count, method, 1)
            _, trace = score(current, rows[0]["parent"], rows[0]["candidates"][None],
                torch.tensor([2.]), history_override=override, trace=True)
            torch.testing.assert_close(trace.queries[1][0], captured[1][0][0])
        if method == "nonlinear_response":
            q, native, _ = captured[0]
            features = adapter.layers[0].features(q, native, torch.tensor([8., 5., 7.]))
            expected_center = features.double().mean((0, 1)).float()
        else:
            joined = torch.cat([torch.cat((rows[uid]["parent"].k[0, 0],
                                           rows[uid]["parent"].v[0, 0]), -1) for uid in (0, 1, 2)])
            expected_center = joined.double().mean(0).float()
        torch.testing.assert_close(adapter.layers[0].input_center, expected_center)
        assert costs["teacher_same_query_flops"] > 0 and costs["teacher_full_query_flops"] > 0
        assert costs["network_and_read_training_flops_estimate"] > 0
        assert diagnostics["validation"]["full_logit_mse"] >= 0
        prediction, target = torch.randn(3, 1, 16, 32), torch.randn(3, 1, 16, 32)
        counts = torch.tensor([8., 5., 7.])
        actual = calibration.aggregate_loss(prediction, target, counts, torch.tensor(2.), 8)
        expected = ((prediction-target)*counts[:, None, None, None]).square().mean()/(8*2)**2
        torch.testing.assert_close(actual, expected)
    finally:
        torch.set_num_threads(previous_threads)


def test_joint_logit_refinement_reaches_all_layers_with_fixed_teacher_and_backbone(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(172011)
        cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=6,
                         num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
        parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
        rows = {}
        with torch.no_grad():
            for uid, count in enumerate((8, 8, 5, 6)):
                items = torch.arange(1+uid*10, 1+uid*10+count)[None]
                behavior = torch.ones_like(items)
                dt = torch.ones_like(items).float()
                dt[:, 0] = 0
                rows[uid] = dict(parent=parent.compute_kv(items, behavior, dt),
                    teacher=current.compute_kv(items, behavior, dt),
                    candidates=torch.arange(1, 17), query_delta=2.)
        adapter = calibration.make_adapter(current, "kv_view")
        with torch.no_grad():
            for layer in adapter.layers:
                layer.output.weight.normal_(std=.01)
        adapter = type(adapter).from_state_dict(adapter.export_state()).eval().requires_grad_(False)
        foundation = {name: value.clone() for name, value in current.state_dict().items()}
        parameters = [{name: value.clone() for name, value in layer.named_parameters()} for layer in adapter.layers]
        buffers = {name: value.clone() for name, value in adapter.named_buffers()}
        source = rows[0]["parent"].k.clone()
        observed_targets = {}
        original = calibration.collect_logits

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            observed_targets.update(result)
            return result

        monkeypatch.setattr(calibration, "collect_logits", observe)
        costs = defaultdict(int, inherited_calibration_flops=123456)
        adapter, diagnostics = calibration.refine_logits(current, adapter, rows, [0, 1, 2], [3],
            batch_size=2, device=torch.device("cpu"), epochs=2, costs=costs)
        joint = diagnostics["joint_logit"]
        assert len(joint["first_step_layer_gradient_norms"]) == 6
        assert all(value > 0 for value in joint["first_step_layer_gradient_norms"])
        for layer, initial in zip(adapter.layers, parameters, strict=True):
            assert any(not torch.equal(value, initial[name]) for name, value in layer.named_parameters())
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, foundation[name], atol=0, rtol=0)
        for name, value in adapter.named_buffers():
            torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in current.parameters())
        torch.testing.assert_close(rows[0]["parent"].k, source, atol=0, rtol=0)
        with torch.no_grad():
            full, _ = score(current, rows[3]["teacher"], rows[3]["candidates"][None], torch.tensor([2.]))
        torch.testing.assert_close(observed_targets[3][0], full[0])
        assert all(not value.requires_grad for pair in observed_targets.values() for value in pair)
        expected_unit = torch.stack([observed_targets[uid][0]-observed_targets[uid][1]
                                     for uid in (0, 1, 2)]).double().square().mean().sqrt()
        assert joint["fit_only_full_minus_native_logit_rms"] == pytest.approx(float(expected_unit), rel=1e-6)
        assert costs["inherited_calibration_flops"] == 123456
        assert costs["joint_logit_training_flops_estimate"] > 0
        model = calibration.CostModel(32, 6, 1, "torch")
        expected_teacher_reads = sum(calibration.eager_read(model, rows[batch[0]]["parent"].seq_len,
            batch=len(batch), queries=16) for batch in calibration.groups(list(rows), rows, 2))
        assert costs["teacher_full_query_flops"] == expected_teacher_reads
        assert diagnostics["validation"]["full_logit_mse"] >= 0
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("method", ("kv_view", "kv_context"))
def test_rolling_scenes_keep_user_roles_query_budget_and_actual_prefix(monkeypatch, method):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(172012)
        cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=2,
                         num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
        parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
        rows = {}
        with torch.no_grad():
            for uid in range(3):
                items = torch.arange(1+uid*20, 21+uid*20)[None]
                behavior = torch.ones_like(items)
                dt = torch.ones_like(items).float()
                fresh_dt = dt[:, -8:].clone()
                fresh_dt[:, 0] = 0
                teacher = current.compute_kv(items[:, -8:], behavior[:, -8:], fresh_dt)
                for scene, writes in enumerate((0, 2, 4, 6)):
                    end = 20-writes
                    cache = parent.compute_kv(items[:, end-8:end], behavior[:, end-8:end], fresh_dt)
                    context = torch.ones(1, 8, 2)
                    if writes:
                        cache = append_with_rolling_band(current, cache, items[:, end:],
                            behavior[:, end:], dt[:, end:], 8)
                        context = append_context(context, old_length=8, width=writes, max_length=8)
                    rows[f"{uid}:{scene}"] = dict(uid=uid, scene_kind=f"native{writes}",
                        parent=cache, teacher=teacher, context=context,
                        candidates=torch.arange(1, 17)[scene::4], query_delta=1.)
        train = [f"{uid}:{scene}" for uid in (0, 1) for scene in range(4)]
        validation = [f"2:{scene}" for scene in range(4)]
        for uid in range(3):
            candidates = torch.cat([rows[f"{uid}:{scene}"]["candidates"] for scene in range(4)])
            assert len(candidates) == len(candidates.unique()) == 16
        captured = {}
        original = calibration.collect

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            captured[args[4]] = result
            return result

        monkeypatch.setattr(calibration, "collect", observe)
        adapter, diagnostics = calibration.fit(current, rows, train, validation, method,
            batch_size=2, device=torch.device("cpu"), epochs=2)
        assert all(record["fitting_users"] == 2 and record["fitting_scenes"] == 8
                   and record["fitting_queries"] == 4 for record in diagnostics["layers"])
        assert all(group["scenes"] == 1 for group in diagnostics["validation"]["by_scene"].values())
        features = []
        for key in train:
            row = rows[key]
            parts = [row["parent"].k[0, 0], row["parent"].v[0, 0]]
            if method == "kv_context":
                parts.append(row["context"][0])
            features.append(torch.cat(parts, -1))
        torch.testing.assert_close(adapter.layers[0].input_center,
                                   torch.cat(features).double().mean(0).float())
        mixed = rows["0:3"]
        with torch.no_grad():
            override = calibration.prepared_override(current, adapter, mixed["parent"], torch.tensor([8.]),
                method, 1, context=mixed["context"] if method == "kv_context" else None)
            _, trace = score(current, mixed["parent"], mixed["candidates"][None],
                             torch.tensor([1.]), history_override=override, trace=True)
        torch.testing.assert_close(trace.queries[1][0], captured[1][0][3])
        with pytest.raises(ValueError, match="disjoint by UID"):
            calibration.fit(current, rows, ["0:0"], ["0:1"], method,
                            batch_size=1, device=torch.device("cpu"), epochs=1)
    finally:
        torch.set_num_threads(previous_threads)
