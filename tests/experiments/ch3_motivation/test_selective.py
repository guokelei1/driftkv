"""Numerical and causal checks for the new Chapter 3 selective experiment."""
from collections import Counter
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import selective
from hstu_kvcache.evaluation.baselines import event_shapes, raw_tensors
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.baselines import layer_recompute as lr
from hstu_kvcache.methods.baselines.layer_recompute.core import _capture_boundaries
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.evaluation.cost import CostModel, eager_read


def setup_case(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(71)
    model_cfg = HSTUConfig(num_items=40, num_behaviors=2, hidden_size=32,
        num_heads=1, num_layers=3, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent = HSTU(model_cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    with torch.no_grad():
        current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight) * .07)
        current.blocks[1].attn.k_proj.weight.add_(torch.randn_like(current.blocks[1].attn.k_proj.weight) * .02)
    events = (np.arange(10) * 10, np.arange(1, 11), np.zeros(10, dtype=np.int64))
    histories = SimpleNamespace(rows={7: events})
    rounds = {7: {65: [dict(uid=7, query_timestamp=65, raw_item_id=22, item_idx=22, label=0)],
                  70: [dict(uid=7, query_timestamp=70, raw_item_id=i, item_idx=i, label=i % 2) for i in (23, 24)],
                  85: [dict(uid=7, query_timestamp=85, raw_item_id=25, item_idx=25, label=1)]}}
    cfg = dict(edge=1, release_day=60 / 86400, end_day=100 / 86400,
        history_length=4, append_band_size=2, query_batch=16, repair_query_batch=16,
        attention_backend="torch", partial_points=dict(layer=[1, 2, 3],
            tail=[.25, .5], deviation=[.25, .5], query=[.25, .5]))
    return parent, current, histories, rounds, cfg


@torch.no_grad()
def captures(parent, current, histories, cfg):
    raw = raw_tensors(tuple(column[2:6] for column in histories.rows[7]), "cpu")
    with _capture_boundaries(parent) as boundaries:
        old = parent.compute_kv(*raw)
    return {7: dict(parent=old, teacher=current.compute_kv(*raw), boundaries=boundaries,
        candidates=torch.tensor([21, 22, 23]), query_delta=10., raw=raw)}


@torch.no_grad()
def test_direct_profile_checks_every_interval_and_uses_tail_window(monkeypatch):
    parent, current, histories, _, cfg = setup_case(monkeypatch)
    rows = captures(parent, current, histories, cfg)
    before = rows[7]["parent"].k.clone()
    profiles, flops = selective.profile_intervals(current, rows, histories, [7], cfg, "cpu")
    assert len(profiles["intervals"]) == 6
    expected = {}
    state = lr.LayerRecomputeState(rows[7]["parent"], rows[7]["boundaries"])
    query, delta = rows[7]["candidates"][None], torch.tensor([10.])
    teacher = score(current, rows[7]["teacher"], query, delta)[0]
    for start in range(3):
        for end in range(start, 3):
            repaired = lr.recompute_interval(current, state, *rows[7]["raw"], (start, end))
            expected[start, end] = float((score(current, repaired.cache, query, delta)[0] - teacher).square().mean())
    for record in profiles["intervals"]:
        assert record["mse"] == expected[tuple(record["interval"])]
    assert profiles["selected"]["3"] == [0, 2]
    assert expected[0, 2] < 1e-12
    for width in (1, 2, 3):
        assert profiles["selected"][str(width)] == list(min(
            (interval for interval in expected if interval[1] - interval[0] + 1 == width),
            key=lambda interval: (expected[interval], interval[0])))
    assert torch.equal(before, rows[7]["parent"].k)
    cm = CostModel(32, 3, 1, "torch")
    wanted = (2 * cm.full_cache(4) + 2 * eager_read(cm, 4, queries=3) + 9
              + sum(cm.layer_rebuild(4, interval) + eager_read(cm, 4, queries=3) + 9
                    for interval in expected))
    assert flops == wanted == sum(profiles["flop_components"].values())
    # Same tensor length is insufficient: the first window is a different input.
    rows[7]["raw"] = raw_tensors(tuple(column[:4] for column in histories.rows[7]), "cpu")
    with pytest.raises(ValueError, match="cutover window"):
        selective.profile_intervals(current, rows, histories, [7], cfg, "cpu")


def test_policy_costs_use_actual_first_consumer_and_charge_selection(monkeypatch):
    parent, current, histories, by_user, cfg = setup_case(monkeypatch)
    rows = captures(parent, current, histories, cfg)
    profiles, _ = selective.profile_intervals(current, rows, histories, [7], cfg, "cpu")
    cm = CostModel(32, 3, 1, "torch")
    shapes = event_shapes(histories, by_user, [7], cfg)
    assert shapes["first"] == Counter({(4, "behavior", 1): 1})
    policies, denominator = selective.make_policies(cm, shapes, profiles, cfg)
    assert denominator == cm.full_cache(4)
    assert policies[0]["cost_percent"] == 0
    assert policies[1]["cost_percent"] == 100
    layer = next(policy for policy in policies if policy["tag"] == "layer_2layers")
    selection_cost = profiles["standalone_calibration_flops_by_width"]["2"]
    assert selection_cost < profiles["calibration_flops"]
    assert selection_cost == sum(profiles["common_flop_components"].values()) + sum(profiles["parts_by_width"]["2"].values())
    assert layer["extra_flops"] == selection_cost + cm.layer_rebuild(4, layer["interval"])
    query = next(policy for policy in policies if policy["tag"] == "query_25pct")
    expected = cm.operation_cost("query", 4, selected_tokens=1, queries=1)["total_flops"]
    expected += cm.embedding(1) - cm.embedding(1, query=True)
    assert query["extra_flops"] == expected
    shapes["requests"] = Counter({(4, "query", 1): 2, (4, "query", 2): 1})
    assert selective.repair_cost(dict(method="rebuild"), shapes, cm, cfg, schedule="request_local")[0] == 3 * denominator


@torch.no_grad()
def test_persistent_reference_prepares_before_behavior_and_keeps_policy_state(monkeypatch):
    parent, current, histories, by_user, cfg = setup_case(monkeypatch)
    policies = [dict(tag="reuse", method="reuse"), dict(tag="rebuild", method="rebuild"),
                dict(tag="layer", method="layer", interval=(1, 2))]
    actual, metadata = selective.evaluate_reference(parent, current, histories, by_user,
        [7], policies, cfg, "cpu")
    expected = NativeTrajectory(parent, 0, histories.rows[7], 60, max_length=4, append_band_size=2)
    expected.switch(current, 1, 60, rebuild=True)
    for timestamp, requests in sorted(by_user[7].items()):
        expected.advance(timestamp)
        ids = torch.tensor([[row["item_idx"] for row in requests]])
        values = score(current, expected.cache, ids, torch.tensor([timestamp - expected.last_timestamp]))[0][0]
        got = [row["score"] for row in actual if row["method"] == "rebuild" and row["query_timestamp"] == timestamp]
        torch.testing.assert_close(torch.tensor(got), values, rtol=0, atol=0)
    assert metadata["diagnostic"] is True
    assert all(value["prepare_calls"] == 1 and value["prepare_rows"] == 4
               and value["appended_rows"] == 4 for value in metadata["observed"].values())
    # Policy order cannot pollute another method's state.
    reverse, _ = selective.evaluate_reference(parent, current, histories, by_user,
        [7], policies[::-1], cfg, "cpu")
    key = lambda row: (row["method"], row["query_timestamp"], row["raw_item_id"])
    assert {key(row): row["score"] for row in actual} == {key(row): row["score"] for row in reverse}


@torch.no_grad()
def test_request_local_diagnostic_discards_repair_and_preserves_real_boundaries(monkeypatch):
    parent, current, histories, by_user, cfg = setup_case(monkeypatch)
    policies = [dict(tag="reuse", method="reuse"), dict(tag="layer", method="layer", interval=(1, 2))]
    actual, metadata = selective.evaluate_reference(parent, current, histories, by_user,
        [7], policies, cfg, "cpu", semantics="request_local")
    assert metadata["diagnostic"]
    assert all(value["prepare_calls"] == 3 for value in metadata["observed"].values())
    # Removing the repair policy must leave every later Reuse score unchanged.
    control, _ = selective.evaluate_reference(parent, current, histories, by_user,
        [7], policies[:1], cfg, "cpu", semantics="request_local")
    assert [row for row in actual if row["method"] == "reuse"] == control
    # Explicit old-input maintenance gives the same non-prefix repair after FIFO.
    raw = raw_tensors(tuple(column[2:6] for column in histories.rows[7]), "cpu")
    state = lr.capture_state(parent, *raw)
    cursor, last = 6, 50
    for timestamp, requests in sorted(by_user[7].items()):
        stop = int(np.searchsorted(histories.rows[7][0], timestamp, side="left"))
        while cursor < stop:
            width = 1 << (min(2, stop - cursor).bit_length() - 1)
            items = torch.tensor(histories.rows[7][1][cursor:cursor + width])[None]
            behavior = torch.zeros_like(items)
            times = histories.rows[7][0][cursor:cursor + width]
            delta = torch.tensor(np.diff(np.r_[last, times]), dtype=torch.float32)[None]
            state = lr.append_band(current, state, items, behavior, delta, 4)
            cursor += width
            last = int(times[-1])
        aligned = raw_tensors(tuple(column[cursor - 4:cursor] for column in histories.rows[7]), "cpu")
        view = lr.recompute_interval(current, state, *aligned, (1, 2))
        ids = torch.tensor([[row["item_idx"] for row in requests]])
        expected = score(current, view.cache, ids, torch.tensor([timestamp - last]))[0][0]
        got = [row["score"] for row in actual if row["method"] == "layer" and row["query_timestamp"] == timestamp]
        torch.testing.assert_close(torch.tensor(got), expected, atol=0, rtol=0)


@torch.no_grad()
def test_batched_persistent_path_matches_independent_reference(monkeypatch):
    """Exercise the real batch/preparation control flow with a CPU native reader.

    This does not validate CUDA kernel/graph arithmetic; a later allocated-GPU
    diagnostic must compare graph replay with these reference scores too.
    """
    from hstu_kvcache.evaluation import fast_baselines as fast
    parent, current, histories, by_user, cfg = setup_case(monkeypatch)
    policies = [dict(tag="reuse", method="reuse"), dict(tag="rebuild", method="rebuild"),
        dict(tag="layer1", method="layer", interval=(0, 0)),
        dict(tag="layer2", method="layer", interval=(0, 1)),
        *[dict(tag=method, method=method, fraction=.5) for method in ("tail", "deviation", "query")]]

    class CpuGraphs:
        def run(self, kind, model, cache, ids, delta, *, behavior=None, **kwargs):
            if kind == "append":
                return append_with_rolling_band(model, cache, ids, behavior, delta, cfg["history_length"])
            return score(model, cache, ids, delta)[0]

        def describe(self):
            return dict(cpu_reference=True)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    expected, _ = selective.evaluate_reference(parent, current, histories, by_user,
        [7], policies, cfg, "cpu")
    key = lambda row: (row["method"], row["query_timestamp"], row["raw_item_id"])
    expected = {key(row): row["score"] for row in expected}
    for shared in (False, True):
        actual, _ = fast.evaluate_batched(parent, current, histories, by_user,
            [7], policies, cfg, "cpu", graphs=CpuGraphs(), shared_preparation=shared)
        assert {key(row) for row in actual} == set(expected)
        torch.testing.assert_close(torch.tensor([row["score"] for row in actual]),
            torch.tensor([expected[key(row)] for row in actual]), atol=3e-6, rtol=3e-6)


def test_driver_230_ledger_adds_a_consumer_for_otherwise_idle_users(monkeypatch):
    from experiments.ch3_motivation.code.run_selective import cost_ledgers
    parent, current, histories, by_user, cfg = setup_case(monkeypatch)
    histories.rows[8] = tuple(column[:6] for column in histories.rows[7])
    cfg["candidate_reads_per_user"] = 230
    profiles, _ = selective.profile_intervals(current,
        captures(parent, current, histories, cfg), histories, [7], cfg, "cpu")
    cm = CostModel(32, 3, 1, "torch")
    shapes = event_shapes(histories, by_user, [7, 8], cfg)
    policies, denominator = selective.make_policies(cm, shapes, profiles, cfg)
    result = cost_ledgers(histories, by_user, [7, 8], policies, shapes, cm, cfg)
    actual, normalized = result["actual_event_ledger"], result["candidate_reads_230"]
    assert actual["once_rebuild_flops"] == denominator == cm.full_cache(4)
    assert normalized["once_rebuild_flops"] == 2 * denominator
    assert normalized["common_native_append_flops"] == actual["common_native_append_flops"]
    assert all(row["extra_read_flops"] == 0 for row in normalized["results"])
    layer = next(row for row in normalized["results"] if row["method"] == "layer_2layers")
    point = next(policy for policy in policies if policy["tag"] == layer["method"])
    assert layer["added_flops"] == point["calibration_flops"] + 2 * cm.layer_rebuild(4, point["interval"])


def test_driver_coverage_comparison_and_no_tiny_population_auc(tmp_path):
    from experiments.ch3_motivation.code.run_selective import compare_scores, summarize
    reference = [dict(method="reuse", uid=7, query_timestamp=1, raw_item_id=3, label=0, score=.2)]
    actual = [dict(reference[0], score=.20001)]
    assert compare_scores(reference, actual)["passed"]
    with pytest.raises(AssertionError, match="coverage"):
        compare_scores(reference, [])
    with pytest.raises(AssertionError, match="maximum error"):
        compare_scores(reference, [dict(reference[0], score=.4)])
    with pytest.raises(ValueError, match="512"):
        summarize(reference, [7], [], {}, {}, tmp_path, {}, {})


def test_population_summary_exports_paired_intervals_and_fresh_endpoints(tmp_path):
    from experiments.ch3_motivation.code.run_selective import summarize
    users = list(range(512))
    policies = [dict(tag="reuse", method="reuse"), dict(tag="rebuild", method="rebuild")]
    scores = [dict(method=policy["tag"], uid=uid, query_timestamp=1, raw_item_id=uid,
        label=uid % 2, score=.4 if policy["method"] == "reuse" else float(uid % 2))
        for policy in policies for uid in users]
    costs = dict(actual_event_ledger=dict(results=[dict(method=policy["tag"], added_flops=i * 100,
        calibration_flops=0, selection_and_repair_flops=i * 100, percent_of_once_rebuild=i * 100.,
        extra_read_flops=0) for i, policy in enumerate(policies)]),
        candidate_reads_230=dict(results=[dict(method=policy["tag"], percent_of_once_rebuild=i * 100.)
                                          for i, policy in enumerate(policies)]))
    result = summarize(scores, users, policies, costs,
        dict(edge=1, confidence_replicates=16), tmp_path, {}, {})
    assert result["historical_scores_reused"] is False
    assert [row["auc"] for row in result["results"]] == [.5, 1.]
    assert [row["recovery_percent"] for row in result["results"]] == [0., 100.]
    assert result["confidence"]["eligible_users"] == 512
    assert (tmp_path / "summary.json").exists()
    assert len((tmp_path / "metrics_by_point.csv").read_text().splitlines()) == 3
