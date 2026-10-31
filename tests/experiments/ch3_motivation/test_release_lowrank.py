"""Joint release-request matrix: causality, full restoration, FIFO and coverage."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.fg_trajectory_control import TRAINED_FG_METHOD, H_DISABLED_DIAGNOSTIC
from experiments.ch3_motivation.code.probe_variants import build_variant
from experiments.ch3_motivation.code.probe_replay import replay_probe_user
from experiments.ch3_motivation.code.release_lowrank import (
    capture_release_user, plan_request_panels, evaluate_release_user,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache


def requests():
    return {86402: [dict(uid=91, query_timestamp=86402, raw_item_id=item, item_idx=item, label=item%2)
                   for item in (20, 21, 22)],
            86412: [dict(uid=91, query_timestamp=86412, raw_item_id=item, item_idx=item, label=item%2)
                    for item in (20, 23)]}


def test_230_instance_panels_preserve_all_real_reads_and_separate_repeated_items():
    real = {100+index: [dict(uid=91, query_timestamp=100+index, raw_item_id=20,
                            item_idx=20, label=index%2)] for index in range(465)}
    panels = plan_request_panels(real, uid=91, edge=1, known_items=1000)
    again = plan_request_panels(real, uid=91, edge=1, known_items=1000)
    assert len(panels) == 3 and all(len(panel.instances) == 230 for panel in panels)
    assert [panel.instances for panel in panels] == [panel.instances for panel in again]
    actual = [instance for panel in panels for instance in panel.instances if instance["actual"] is not None]
    assert len(actual) == 465
    assert len({(row["timestamp"], row["actual"]["raw_item_id"]) for row in actual}) == 465
    for panel in panels:
        real_times = {row["timestamp"] for row in panel.instances if row["actual"] is not None}
        assert all(row["timestamp"] in real_times for row in panel.instances)
        assert len({(row["timestamp"], row["item_idx"]) for row in panel.instances}) == 230


def models(variant):
    torch.manual_seed(128)
    cfg = HSTUConfig(num_items=64, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.03)
    fg = build_variant(variant, current, stage="fg", hidden_width=8, depth=2,
        query_chunk=3, row_chunk=2).eval().requires_grad_(False)
    for parameter in fg.parameters():
        parameter.normal_(std=.2)
    full = fg.expanded("fgh").eval().requires_grad_(False)
    for module in full.h:
        for parameter in module.parameters():
            parameter.normal_(std=.3)
        module.biases[-1].fill_(.4)
    return parent, current, fg, full


@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
@pytest.mark.parametrize("initial_rows,future_rows", [(8, 3), (8, 10), (0, 3)])
@torch.no_grad()
def test_joint_matrix_fullrank_matches_actual_snapshots_without_fifo_mask_in_svd(monkeypatch, variant, initial_rows, future_rows):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full = models(variant)
    events = (np.r_[np.arange(initial_rows)*100+100, 86401+np.arange(future_rows)],
              np.arange(1, initial_rows+future_rows+1), np.arange(initial_rows+future_rows)%2)
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4,
        query_batch=2, stop_after_last_query=True)
    recorded, saved_snapshots = [], []
    def restore(panel, h, context):
        assert h.shape == (2, 2, 8, initial_rows, 8)
        report, output = dict(panel=panel.index, max_error=0.), []
        for timestamp in sorted({instance["timestamp"] for instance in panel.instances}):
            positions = panel.positions_at(timestamp)
            snapshot = context.snapshots[timestamp]
            offset, inherited = snapshot.source_offset, snapshot.source_rows
            source = HSTUKVCache(context.initial_source.k[:, :, offset:],
                context.initial_source.v[:, :, offset:], inherited)
            features = tuple(rows[:, :, offset:] for rows in context.initial_features)
            ids = torch.tensor([[panel.instances[index]["item_idx"] for index in positions]])
            indices = torch.tensor(positions)
            def fixed(layer, qc, rows, query_start, row_start):
                selected = indices[query_start:query_start+qc.shape[2]]
                return h[layer][:, selected, offset+row_start:offset+row_start+rows.shape[2]][None]
            scores, _ = full.score(current, source, context.initial_ids[:, offset:], ids,
                torch.tensor([snapshot.query_gap], dtype=torch.float32), prepared_cache=snapshot.cache,
                read_features=features, interaction_override=fixed)
            torch.testing.assert_close(scores[0], panel.full_scores[positions], atol=1e-6, rtol=1e-6)
            report["max_error"] = max(report["max_error"], float((scores[0]-panel.full_scores[positions]).abs().max()))
            if inherited < initial_rows:
                # Dormant old columns keep their actual offline h values.
                assert h[:, :, positions, :offset].abs().max() > 0
            saved_snapshots.append((snapshot, snapshot.cache.k.clone(), snapshot.cache.v.clone()))
            for index, score in zip(positions, scores[0].tolist(), strict=True):
                row = panel.instances[index]["actual"]
                if row is not None:
                    output.append(dict(row, method="full", score=score))
        recorded.append((panel, h.clone()))
        return output, report
    scores, report = capture_release_user(parent, current, full, fg, events, requests(), 91, cfg,
        known_items=64, candidate_size=8, on_panel_ready=restore)
    assert report["actual_feedback_instances"] == 5 and report["panel_instances"] == 8
    assert len([row for row in scores if row["method"] == TRAINED_FG_METHOD]) == 5
    assert len([row for row in scores if row["method"] == "full"]) == 5
    assert report["snapshot_count"] == 2 and report["initial_prefill_calls"] == 1
    assert report["cleared_feedback_instances"] == (5 if initial_rows == 0 else 2 if future_rows == 10 else 0)
    for snapshot, k, v in saved_snapshots:
        assert torch.equal(snapshot.cache.k, k) and torch.equal(snapshot.cache.v, v)
    panel, _ = recorded[0]
    repeated = [index for index, row in enumerate(panel.instances)
                if row["actual"] is not None and row["item_idx"] == 20]
    assert len(repeated) == 2 and panel.instances[repeated[0]]["timestamp"] != panel.instances[repeated[1]]["timestamp"]
    assert not torch.equal(panel.full_queries[-1, :, repeated[0]], panel.full_queries[-1, :, repeated[1]])


@torch.no_grad()
def test_panels_stream_in_order_with_one_h_buffer_and_no_feedback_drop(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full = models("full_item")
    events = (np.r_[np.arange(8)*100+100, [86401, 86403]], np.arange(1, 11), np.arange(10)%2)
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4,
        query_batch=2, stop_after_last_query=True)
    order, actual_keys = [], []
    def finished(panel, h, context):
        order.append(panel.index)
        actual_keys.extend((row["timestamp"], row["actual"]["raw_item_id"])
            for row in panel.instances if row["actual"] is not None)
        return [], dict(panel=panel.index)
    _, report = capture_release_user(parent, current, full, fg, events, requests(), 91, cfg,
        known_items=64, candidate_size=2, on_panel_ready=finished)
    assert order == [0, 1, 2] and len(actual_keys) == len(set(actual_keys)) == 5
    assert report["maximum_resident_h_panel_bytes"] == 2*2*2*8*8*4


@pytest.mark.parametrize("variant", ["head", "full_item", "layerwise_kv"])
@pytest.mark.parametrize("future_rows", [3, 10])
@torch.no_grad()
def test_joint_svd_replays_all_real_times_and_integer_full_column_rank_restores_full(monkeypatch, variant, future_rows):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full = models(variant)
    events = (np.r_[np.arange(8)*100+100, 86401+np.arange(future_rows)],
              np.arange(1, 9+future_rows), np.arange(8+future_rows)%2)
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4,
        query_batch=2, stop_after_last_query=True)
    actual = requests()
    actual[86402].append(dict(actual[86402][0], label=1-actual[86402][0]["label"]))
    options = dict(ranks=(0, 2, 8), variants=("both_norm", "both_raw", "key_norm", "value_norm"),
        method="exact", verify_exact_heads=0, batch_rank_scores=True, compare_rank_execution=True)
    rows, report = evaluate_release_user(parent, current, full, fg, events, actual, 91, cfg,
        known_items=64, candidate_size=8, panel_options=options)
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    assert {TRAINED_FG_METHOD, H_DISABLED_DIAGNOSTIC, "full"} <= set(groups)
    for scores in groups.values():
        assert len(scores) == 6
        assert len({(row["query_timestamp"], row["raw_item_id"], row["request_instance_ordinal"]) for row in scores}) == 6
    full_scores = [row["score"] for row in groups["full"]]
    for variant_name in options["variants"]:
        torch.testing.assert_close(torch.tensor([row["score"] for row in groups[f"{variant_name}_r8"]]),
            torch.tensor(full_scores), atol=1e-6, rtol=1e-6)
    assert report["actual_feedback_instances"] == 6
    assert all(panel["matrix_columns"] == 8 and panel["full_rank_max_error"] < 1e-5 for panel in report["panels"])
    checks = [check for panel in report["panels"] for frame in panel["frames"]
        for check in frame["rank_execution_checks"]]
    assert checks and all(check["cache_batch"] == 1 and check["rank_query_batch"] == 3 for check in checks)
    assert max(check["logit_max_error"] for check in checks) < 1e-5
    assert {check["variant"] for check in checks} == set(options["variants"])
    if future_rows == 10:
        for name, scores in groups.items():
            if name.startswith(("both_", "key_", "value_")):
                actual_cleared = [row["score"] for row in scores if row["query_timestamp"] == 86412]
                full_cleared = [row["score"] for row in groups["full"] if row["query_timestamp"] == 86412]
                assert actual_cleared == full_cleared


@torch.no_grad()
def test_streamed_decomposition_callbacks_leave_the_original_full_h_writer_unchanged(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full = models("full_item")
    events = (np.r_[np.arange(8)*100+100, [86401, 86403, 86405]], np.arange(1, 12), np.arange(11)%2)
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4,
        query_batch=2, stop_after_last_query=True)
    actual = requests()
    values, report = evaluate_release_user(parent, current, full, fg, events, actual, 91, cfg,
        known_items=64, candidate_size=2,
        panel_options=dict(ranks=(0, 2, 8), method="exact", verify_exact_heads=0, batch_rank_scores=True))
    reference, _ = replay_probe_user(parent, current, full, events, actual, 91, cfg)
    key = lambda row: (row["query_timestamp"], row["raw_item_id"])
    wanted = [row["score"] for row in sorted(reference, key=key)]
    got = [row["score"] for row in sorted((row for row in values if row["method"] == "full"), key=key)]
    torch.testing.assert_close(torch.tensor(got), torch.tensor(wanted), atol=1e-6, rtol=1e-6)
    assert len(report["panels"]) == 3 and report["initial_prefill_calls"] == 1


@torch.no_grad()
def test_randomized_oracle_checks_are_per_variant_and_can_be_repeated_per_user(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    parent, current, fg, full = models("full_item")
    events = (np.arange(8)*100+100, np.arange(1, 9), np.arange(8)%2)
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=8, append_band_size=4,
        query_batch=2, stop_after_last_query=True)
    options = dict(ranks=(0, 2, 8), method="randomized", oversample=8,
        variants=("both_norm", "both_raw", "key_norm", "value_norm"), verify_exact_heads=2)
    shared = set()
    def run(state):
        _, report = evaluate_release_user(parent, current, full, fg, events, requests(), 91, cfg,
            known_items=64, candidate_size=8, panel_options=options, verification_state=state)
        return [check for panel in report["panels"] for check in panel["svd_checks"]]
    checks = run(shared)
    assert len(checks) == 8 and len(shared) == 4
    assert {(check["layer"], check["head"]) for check in checks} == {(0, 0), (1, 1)}
    assert min(point["captured_energy_ratio_to_exact"] for check in checks for point in check["points"]) > .999
    assert max(point["normalized_interaction_reconstruction_disagreement"]
        for check in checks for point in check["points"]) < 1e-6
    assert run(shared) == []
    assert len(run(None)) == 8
