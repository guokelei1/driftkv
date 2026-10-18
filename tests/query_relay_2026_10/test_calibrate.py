"""Small end-to-end calibration using real frozen HSTU prefix and query paths."""
from collections import Counter
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.calibrate import attach_rollout_tail, fit_joint, fit_local, gate_statistics, prepare, rollout_score
from query_relay_2026_10.calibration_cost import OperationCounts
from query_relay_2026_10.fixed_features import build_bank
from query_relay_2026_10.session import append_corrected_event


@pytest.mark.parametrize("mode,read_shared,read_private,rollout_steps,feature_mode,gate_source",
                         [("legacy", 12, 6, 0, "native", "layer"), ("B", 6, 6, 0, "native", "layer"),
                          ("B", 3, 3, 4, "native", "layer"), ("B", 3, 3, 4, "origin_encoder", "layer"),
                          ("B", 3, 3, 4, "origin_encoder", "first"),
                          ("B", 3, 3, 4, "origin_encoder", "item")])
def test_one_epoch_local_and_joint_use_real_frozen_teacher(monkeypatch, mode, read_shared,
                                                         read_private, rollout_steps,
                                                         feature_mode, gate_source, tmp_path):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(20261014)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = HSTUConfig(num_items=32, num_behaviors=2, hidden_size=192,
                            num_heads=6, num_layers=2, max_seq_len=6,
                            input_dropout=0., attn_dropout=0.)
        old = HSTU(config).eval().requires_grad_(False)
        current = copy.deepcopy(old)
        with torch.no_grad():
            current.item_emb.weight[1:18].add_(torch.randn(17, 192) * .08)
        source = StructuredKVAdapter(2, 6, 32, max_length=6,
                                     layout="shared12_private6").eval()
        with torch.no_grad():
            for layer in source.layers:
                layer.input_weight.normal_(std=.08)
                layer.input_bias.normal_(std=.02)
                layer.decoder.normal_(std=.008)
                layer.key_decoder.normal_(std=.008)
        feature_settings = dict(feature_mode=feature_mode)
        if feature_mode == "origin_encoder":
            path = tmp_path / "origin_bank.pt"
            record = build_bank(source, current.item_emb.weight.detach(), path,
                                source_bindings={"test": "causal_origin_fit"})
            feature_settings.update(feature_bank_path=str(path),
                                    feature_bank_sha256=record["sha256"])
        adapter = UnifiedRelayAdapter.from_structured(source, mode=mode,
            read_shared=read_shared, read_private=read_private,
            gate_normalization="rms" if rollout_steps else "raw", gate_source=gate_source,
            **feature_settings).eval()
        captured, histories, records = {}, {}, []
        for uid, length in enumerate((2, 4, 3, 5, 4, 2)):
            items = torch.arange(1 + uid, 1 + uid + length)[None]
            behavior = torch.zeros_like(items)
            delta = torch.ones_like(items, dtype=torch.float32)
            delta[:, 0] = 0
            captured[uid] = dict(parent=old.compute_kv(items, behavior, delta),
                                 teacher=current.compute_kv(items, behavior, delta),
                                 candidates=torch.tensor([20, 22, 25]), query_delta=2.,
                                 item_features=current.lookup_item_embeddings(items))
            # An event exactly at cutover and one after it must not enter fit.
            histories[uid] = (np.r_[np.arange(length) + 100, 106, 107],
                np.r_[items.numpy()[0], 30, 31], np.zeros(length + 2, dtype=np.int64))
            records.append(dict(uid=uid, history_length=length,
                                last_history_timestamp=100 + length - 1))
        tail_records = attach_rollout_tail(captured, SimpleNamespace(rows=histories), 106,
                                            rollout_steps, 6, records)
        assert all(row["last_timestamp"] < 106 for row in tail_records)
        assert all(not np.isin(row["rollout"]["items"], [30, 31]).any() for row in captured.values())
        operations = OperationCounts()
        rows = prepare(current, captured, adapter, torch.device("cpu"), operations)
        args = SimpleNamespace(seed=17, learning_rate=.002,
                               local_epochs=1, joint_epochs=1, batch_size=2,
                               rollout_steps=rollout_steps)
        train, validation = [0, 1, 2, 3], [4, 5]
        costs = Counter()
        local = fit_local(current, adapter, rows, train, validation, args, costs, operations)
        assert len(local) == 2
        assert all(report["best_mse"] <= report["initial_mse"] for report in local)
        assert any(report["best_mse"] < report["initial_mse"] for report in local)
        for report in local:
            assert report["first_gradients"]["key_gate_weight"] > 0
            assert report["first_gradients"]["value_gate_weight"] > 0
        assert costs["local_backward_calls"] == 4

        if rollout_steps:
            # The retained raw-gap teacher target is causally reproducible from
            # its trimmed prefix; no future tail is hidden in prefix KV rows.
            row = rows[3]
            prefix_length = row["rollout"]["prefix_length"]
            teacher = row["teacher"]
            reference = HSTUKVCache(teacher.k[:, :, :prefix_length],
                                    teacher.v[:, :, :prefix_length], prefix_length)
            zeros = LightweightValueState(reference.k.new_zeros(2, 1, prefix_length, 6, 8),
                                            prefix_length)
            with torch.no_grad():
                tail = row["rollout"]
                for index in range(len(tail["items"])):
                    reference, zeros, _ = append_corrected_event(current, reference, zeros, adapter,
                        tail["items"][index:index + 1][None], tail["behaviors"][index:index + 1][None],
                        tail["deltas"][index:index + 1][None], 6, code_count=0)
            torch.testing.assert_close(reference.k, teacher.k, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(reference.v, teacher.v, atol=2e-6, rtol=2e-5)
            adapter.zero_grad(set_to_none=True)
            rolled, written, state = rollout_score(current, adapter, row)
            written.k.retain_grad()
            written.v.retain_grad()
            rolled.square().mean().backward()
            assert written.k.grad[1:, :, prefix_length:].abs().sum() > 0
            assert written.v.grad[1:, :, prefix_length:].abs().sum() > 0
            assert adapter.layers[0].key_gate_weight.grad.abs().sum() > 0
            assert not torch.count_nonzero(state.codes[:, :, prefix_length:])
        joint = fit_joint(current, adapter, rows, train, validation, args, costs, operations)
        history = joint["history"]
        assert len(history) == 2
        assert all(torch.isfinite(torch.tensor(row["validation_teacher_logit_mse"]))
                   for row in history)
        assert history[joint["selected_epoch"]]["validation_teacher_logit_mse"] <= history[0]["validation_teacher_logit_mse"]
        assert costs["joint_backward_calls"] == len(train)
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in adapter.parameters() if parameter.requires_grad)
        assert all(parameter.grad is None for parameter in current.parameters())
        assert all(not parameter.requires_grad for parameter in current.parameters())
        if rollout_steps:
            assert history[0]["validation_rollout_teacher_logit_mse"] is not None
            assert operations.components["joint_rollout_write_frozen_backbone_backward_flops_estimate"] > 0
            assert operations.calls["joint_rollout_write_behavior_queries"] == 18

        means, standard_deviation = gate_statistics(current, adapter, rows, train, operations)
        assert means.shape == (2, 2, 6, read_shared + read_private)
        assert standard_deviation.shape == means.shape
        assert torch.isfinite(means).all() and torch.isfinite(standard_deviation).all()
        assert standard_deviation.max() > 0
        row = rows[validation[0]]
        state = adapter.advance_codes(row["cache"], row["previous"], row["item"],
                                       item_ids=row.get("item_ids"))
        constants = adapter.make_history_read(state, gate_constants=means)
        logits, _ = score(current, row["cache"], row["ids"], row["delta"],
                           history_read_fn=constants)
        assert logits.shape == row["wanted_logits"].shape
        assert torch.isfinite(logits).all()
        report = operations.report()
        assert report["calibration_flops"] > 0
        assert report["components"]["teacher_score_prepare_native_forward_flops"] > 0
        assert report["operation_counts"]["local_steps"] == 4
        assert report["operation_counts"]["joint_steps"] == 2
    finally:
        torch.set_num_threads(previous_threads)
