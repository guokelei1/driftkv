"""Centered score alignment updates only gates, including corrected writes."""
from collections import Counter
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.models import HSTU, HSTUConfig
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.calibrate import attach_rollout_tail, prepare
from query_relay_2026_10.calibration_cost import OperationCounts
from query_relay_2026_10.refit_gates import centered_mse, cumulative_cost, fit_centered


def test_centered_objective_is_invariant_to_per_set_offsets():
    student = torch.tensor([[1., 3., 2.], [4., 7., 5.]], dtype=torch.float64,
                            requires_grad=True)
    teacher = torch.tensor([[2., 2., 4.], [6., 5., 3.]], dtype=torch.float64)
    original = centered_mse(student, teacher)
    shifted = centered_mse(student + torch.tensor([[20.], [-30.]]),
                           teacher + torch.tensor([[-10.], [15.]]))
    torch.testing.assert_close(original, shifted, rtol=0, atol=1e-13)
    original.backward()
    torch.testing.assert_close(student.grad.sum(-1), torch.zeros(2, dtype=torch.float64),
                               rtol=0, atol=1e-13)


def test_gate_only_centered_refit_keeps_encoder_decoder_frozen_and_unrolls_writer(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(17)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        cfg = HSTUConfig(num_items=32, num_behaviors=2, hidden_size=192, num_layers=2,
            num_heads=6, max_seq_len=6, input_dropout=0., attn_dropout=0.)
        old = HSTU(cfg).eval().requires_grad_(False)
        current = copy.deepcopy(old)
        with torch.no_grad():
            current.item_emb.weight[1:18].add_(torch.randn(17, 192) * .08)
        source = StructuredKVAdapter(2, 6, 32, max_length=6, layout="shared12_private6").eval()
        with torch.no_grad():
            for layer in source.layers:
                layer.input_weight.normal_(std=.05)
                layer.input_bias.normal_(std=.02)
                layer.decoder.normal_(std=.01)
                layer.key_decoder.normal_(std=.01)
        adapter = UnifiedRelayAdapter.from_structured(source, mode="B", read_shared=3,
            read_private=3, gate_normalization="rms").eval()
        captured, histories, records = {}, {}, []
        for uid, length in enumerate((2, 4, 3, 5, 4, 2)):
            items = torch.arange(1 + uid, 1 + uid + length)[None]
            behaviors = torch.zeros_like(items)
            delta = torch.ones_like(items, dtype=torch.float32)
            delta[:, 0] = 0
            captured[uid] = dict(parent=old.compute_kv(items, behaviors, delta),
                teacher=current.compute_kv(items, behaviors, delta),
                item_features=current.lookup_item_embeddings(items),
                candidates=torch.tensor([20, 22, 25]), query_delta=2.)
            histories[uid] = (np.r_[np.arange(length) + 100, 106, 107],
                np.r_[items.numpy()[0], 30, 31], np.zeros(length + 2, dtype=np.int64))
            records.append(dict(uid=uid, history_length=length, last_history_timestamp=99 + length))
        attach_rollout_tail(captured, SimpleNamespace(rows=histories), 106, 4, 6, records)
        operations = OperationCounts()
        rows = prepare(current, captured, adapter, torch.device("cpu"), operations)
        frozen = {name: parameter.detach().clone() for name, parameter in adapter.named_parameters()
                  if "gate_weight" not in name}
        args = SimpleNamespace(epochs=1, learning_rate=.002, batch_size=2, seed=17, rollout_steps=4)
        result = fit_centered(current, adapter, rows, [0, 1, 2, 3], [4, 5], args, operations)
        assert len(result["history"]) == 2
        assert result["history"][result["selected_epoch"]]["validation_centered_logit_mse"] <= \
            result["history"][0]["validation_centered_logit_mse"]
        assert result["operation_counts"]["refit_backward_calls"] == 4
        assert all(value is not None and value > 0 for value in result["first_gradients"].values())
        for name, parameter in adapter.named_parameters():
            if "gate_weight" not in name:
                assert parameter.grad is None and not parameter.requires_grad
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in current.parameters())
        assert operations.components["joint_rollout_write_frozen_backbone_backward_flops_estimate"] > 0
        assert not any(key.endswith("encoder_backward_flops_estimate") and value
                       for key, value in operations.components.items())
        assert operations.calls["refit_static_prepare_rows"] == sum((2, 4, 3, 5, 4, 2))
    finally:
        torch.set_num_threads(previous_threads)


def test_refit_cost_retains_parent_local_joint_and_origin_expense():
    parent = dict(calibration_flops=100, calibration_with_initial_origin_flops=160,
                  source_reconstruction_sensitivity=dict(additional_flops=20),
                  warmstart_origin=dict(calibration_flops=60))
    stage = dict(calibration_flops=30, stage_calibration_flops=30,
                 source_reconstruction_sensitivity=dict(additional_flops=7))
    result = cumulative_cost(parent, stage, Path("parent.pt"), "parenthash")
    assert result["calibration_flops"] == 130
    assert result["calibration_with_initial_origin_flops"] == 190
    assert result["source_reconstruction_sensitivity"]["additional_flops"] == 27
    assert result["stage_calibration_flops"] == 30
