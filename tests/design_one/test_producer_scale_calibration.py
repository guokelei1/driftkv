"""Only six producer scalars train; the original pure-parent path stays exact."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from design_one import calibrate_nonlinear as calibration
from design_one.rolling_calibration import build_rolling_scenes
from hstu_kvcache.design_one.producer_scaled_item import ProducerScaledItemKVAdapter
from hstu_kvcache.models import HSTU, HSTUConfig


def test_only_six_scales_train_on_mixed_and_pure_logits_stay_bitwise(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.set_num_threads(2)
    torch.manual_seed(172018)
    cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=6,
        num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
    histories, expanded, rows = {}, {}, {}
    with torch.no_grad():
        for uid in range(3):
            expanded[uid] = (np.arange(10), np.arange(1+uid*16, 11+uid*16), np.ones(10, dtype=np.int64))
            histories[uid] = tuple(values[-8:] for values in expanded[uid])
            ids = torch.as_tensor(histories[uid][1])[None]
            dt = torch.ones_like(ids).float()
            dt[:, 0] = 0
            rows[uid] = dict(parent=parent.compute_kv(ids, torch.ones_like(ids), dt),
                teacher=current.compute_kv(ids, torch.ones_like(ids), dt),
                candidates=torch.arange(1, 17), query_delta=1.)
        scenes, _ = build_rolling_scenes(parent, current, rows, histories, expanded, list(rows),
            cutover=10, batch_size=2, device=torch.device("cpu"), max_length=8,
            attention_backend="torch", costs=defaultdict(int), append_targets=(0, 2), full_queries=True)
        calibration.attach_scene_item_features(current, scenes, torch.device("cpu"))
        for row in scenes.values():
            row["producer_mask"] = torch.tensor(row.get("producer", [4]*row["parent"].seq_len))[None] == 5
        base = calibration.make_adapter(current, "kv_item")
        for layer in base.layers:
            layer.output.weight.normal_(std=.01)
        source = scenes["0:a0"]
        before_view = base.map_cache(source["parent"], source["item_features"])
        before_logits, _ = calibration.score(current, source["parent"], source["candidates"][None], torch.tensor([1.]),
            history_override=base.make_history_override(current, before_view))
    adapter = ProducerScaledItemKVAdapter.from_item_adapter(base)
    assert [name for name, p in adapter.named_parameters() if p.requires_grad] == [f"layers.{i}.native_scale" for i in range(6)]
    frozen = {name: value.clone() for name, value in base.state_dict().items()}
    backbone = {name: value.clone() for name, value in current.state_dict().items()}
    costs = defaultdict(int, inherited_calibration_flops=123)
    adapter, stats = calibration.refine_logits(current, adapter, scenes, ["0:a0", "1:a0"], ["2:a0"],
        mixed_train=["0:a2", "1:a2"], mixed_validation=["2:a2"], producer_scale=True, method="kv_item",
        epochs=2, batch_size=2, device=torch.device("cpu"), costs=costs)
    assert stats["joint_logit"]["trainable_parameters"] == 6
    assert [record["scene_kind"] for record in stats["joint_logit"]["epochs"]] == ["a2", "a2"]
    assert any(float(layer.native_scale) != 1. for layer in adapter.layers)
    assert all(p.grad is None for name, p in adapter.named_parameters() if not name.endswith("native_scale"))
    for name, value in frozen.items():
        torch.testing.assert_close(adapter.state_dict()[name], value, atol=0, rtol=0)
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, backbone[name], atol=0, rtol=0)
    with torch.no_grad():
        after_view = adapter.map_cache(source["parent"], source["item_features"], source["producer_mask"])
        after_logits, _ = calibration.score(current, source["parent"], source["candidates"][None], torch.tensor([1.]),
            history_override=adapter.make_history_override(current, after_view))
    assert torch.equal(before_view.k, after_view.k) and torch.equal(before_view.v, after_view.v)
    assert torch.equal(before_logits, after_logits)
    assert costs["optimizer_and_clip_flops_estimate"] == 2*18*6
    calibration.save_fixed_layer0_control(tmp_path/"fixed", (base, {}, dict(calibration_flops=123, teacher_flops=45)), {})
    saved = torch.load(tmp_path/"fixed/calibration.pt", weights_only=False)
    fixed = ProducerScaledItemKVAdapter.from_state_dict(saved["adapter"])
    assert [float(layer.native_scale.detach()) for layer in fixed.layers] == [0., 1., 1., 1., 1., 1.]
    assert saved["metadata"]["cost"]["calibration_flops"] == 123
