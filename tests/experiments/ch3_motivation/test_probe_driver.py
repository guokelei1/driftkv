"""E5 timeline/ledger checks against the new selective-study definition."""
from collections import Counter
import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.evaluate_probes import (
    cached_preparation_flops, cached_read_extra_flops, correction_cost, merge_execution,
    scalar_behavior_flops,
)
from experiments.ch3_motivation.code.probe_replay import evaluate_probe, ProbeTrajectory
from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder, native_query_flops
from experiments.ch3_motivation.code.selective import evaluate_reference
from hstu_kvcache.evaluation.baselines import event_shapes
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.evaluation.cost import CostModel


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case(stage="fgh"):
    torch.manual_seed(183)
    config = HSTUConfig(num_items=40, num_behaviors=2, hidden_size=32, num_heads=1,
        num_layers=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent = HSTU(config).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    with torch.no_grad():
        current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight) * .03)
    cfg = dict(edge=1, release_day=1, end_day=2, history_length=4, append_band_size=2,
               query_batch=16, repair_query_batch=16)
    times = 86400 + np.arange(-4, 6) * 10
    events = (times, np.arange(1, 11), np.zeros(10, dtype=np.int64))
    histories = SimpleNamespace(rows={7: events, 8: events, 9: events})
    by_user = {}
    for uid in (7, 9):
        by_user[uid] = {86400 + timestamp: [dict(uid=uid, query_timestamp=86400 + timestamp,
            item_idx=item, raw_item_id=item, label=item % 2)]
            for timestamp, item in ((5, 21), (45, 22), (55, 23))}
    probe = ProbeLadder(ProbeConfig.from_model(current, hidden_width=7, depth=1,
                         row_chunk=2, query_chunk=2), stage)
    return parent, current, probe, histories, by_user, cfg


@pytest.mark.parametrize("stage", ["f", "fg", "fgh"])
@torch.no_grad()
def test_zero_probe_timeline_matches_e3_reuse_with_behavior_first_and_fifo(stage):
    parent, current, probe, histories, by_user, cfg = case(stage)
    expected, _ = evaluate_reference(parent, current, histories, by_user, [7, 8, 9],
        [dict(tag=stage, method="reuse")], cfg, "cpu")
    actual, execution = evaluate_probe(parent, current, probe, histories, by_user,
        [7, 8, 9], cfg, "cpu")
    key = lambda row: (row["uid"], row["query_timestamp"], row["raw_item_id"])
    expected = {key(row): row["score"] for row in expected}
    assert {key(row) for row in actual} == set(expected)
    torch.testing.assert_close(torch.tensor([row["score"] for row in actual]),
        torch.tensor([expected[key(row)] for row in actual]), atol=2e-6, rtol=2e-6)
    # User 8 has no scoring event but remains in the complete event ledger.
    assert sum(row["calls"] for row in execution["initial_shapes"]) == 3
    assert execution["counts"]["prepare_calls"] == 3
    assert execution["counts"]["appended_rows"] == 18
    if stage != "f":
        # The final inherited row is evicted inside a corrected scalar call;
        # afterwards the remaining behavior writes switch to native bands.
        assert execution["counts"]["corrected_append_calls"] == 12
        assert any(row["kind"] == "behavior" and row["inherited_rows"] == 0
                   for row in execution["read_shapes"])


@torch.no_grad()
def test_merged_shards_preserve_idle_users_and_every_operation():
    parent, current, probe, histories, by_user, cfg = case()
    whole_scores, whole = evaluate_probe(parent, current, probe, histories, by_user,
        [7, 8, 9], cfg, "cpu")
    scores, parts = [], []
    for users in ([7, 8], [9]):
        rows, execution = evaluate_probe(parent, current, probe, histories, by_user, users, cfg, "cpu")
        scores.extend(rows)
        parts.append(execution)
    assert scores == whole_scores
    merged = merge_execution(parts)
    assert merged["counts"] == whole["counts"]
    for name in ("initial_shapes", "prepare_shapes", "read_shapes", "append_shapes"):
        canonical = lambda values: sorted(tuple(sorted(row.items())) for row in values)
        assert canonical(merged[name]) == canonical(whole[name])


def test_resumed_graph_metadata_retains_distinct_pool_generations():
    # First two shards share one cumulative pool; a resumed process builds a
    # fresh pool for the third. Summing those reports would double charge it.
    parts=[]
    for generation,replays in (("first",8),("first",15),("resumed",3)):
        parts.append(dict(counts={},initial_shapes=[],prepare_shapes=[],read_shapes=[],append_shapes=[],
            graph_pool_generation=generation,graph_pool=[dict(operation="score",queries=2,
                warmup_forward_calls=2,capture_forward_calls=1,replays=replays)],
            quality_shortcut=dict(stop_after_last_query=True,skipped_no_feedback_users=1)))
    merged=merge_execution(parts)
    snapshots=merged["block_graph_pool_snapshots"]
    assert [row["generation"] for row in snapshots]==["first","first","resumed"]
    assert [row["graphs"][0]["replays"] for row in snapshots]==[8,15,3]
    assert len(merged["block_quality_shortcuts"])==3


def test_cached_features_are_normalized_once_in_preparation_not_per_read():
    _, current, normalized, _, _, _ = case()
    unnormalized = ProbeLadder(replace(normalized.cfg, normalize=False), "fgh")
    n, q = 4, 3
    c = normalized.cfg
    lh = c.num_layers * c.num_heads
    row_chunks = (n+c.row_chunk-1)//c.row_chunk
    norm_read = cached_read_extra_flops(normalized, n, n, q)
    plain_read = cached_read_extra_flops(unnormalized, n, n, q)
    assert norm_read-plain_read == q*lh*c.head_dim*(1+row_chunks)
    assert (cached_preparation_flops(normalized, n)-cached_preparation_flops(unnormalized, n)
            == n*lh*2*c.row_dim)
    # The last scalar write still evaluates g even after it evicts source row 0.
    assert cached_read_extra_flops(normalized, 3, 0, 1, kind="behavior") > 0
    assert cached_read_extra_flops(normalized, 3, 0, 1, kind="query") == 0


@torch.no_grad()
def test_scalar_probe_append_executes_behavior_embedding_and_final_norm_without_score_head():
    parent, current, probe, histories, _, cfg = case()
    trajectory = ProbeTrajectory(parent, current, probe, histories.rows[7], cfg)
    calls = Counter()
    hooks = [module.register_forward_hook(lambda _module, _args, _result, name=name: calls.update([name]))
             for name, module in (("head", current.cc_score_head), ("final_norm", current.final_norm),
                                  ("query_input", current.query_encoder), ("behavior_input", current.behavior_emb))]
    try:
        trajectory.advance(86401)
        assert calls == Counter(final_norm=1, behavior_input=1)
        source = trajectory.source.k.clone()
        cache = trajectory.cache.k.clone()
        trajectory.score(torch.tensor([[21]]), torch.tensor([1.]))
        trajectory.score(torch.tensor([[22]]), torch.tensor([1.]))
        assert calls == Counter(final_norm=3, behavior_input=1, head=2, query_input=2)
        torch.testing.assert_close(trajectory.cache.k, cache, atol=0, rtol=0)
        torch.testing.assert_close(trajectory.source.k, source, atol=0, rtol=0)
    finally:
        for handle in hooks:
            handle.remove()


@torch.no_grad()
def test_corrected_dense_writers_and_native_bands_have_distinct_costs():
    parent, current, probe, histories, by_user, cfg = case()
    _, execution = evaluate_probe(parent, current, probe, histories, by_user, [7], cfg, "cpu")
    cm = CostModel(32, 2, 1, "torch")
    shapes = event_shapes(histories, by_user, [7], cfg)
    result = correction_cost(probe, execution, 12345, cm, shapes, cfg, 1, model=current)
    # Each of four corrected writes reads cap-1 rows, computes final norm,
    # embeds a behavior and never calls the candidate score head.
    expected_scalar = native_query_flops(current, history_length=3, queries=1)-32-(2*32+1)
    assert scalar_behavior_flops(current, 3) == expected_scalar
    assert result["corrected_scalar_native_writer_flops"] == 4*expected_scalar
    assert result["native_band_writer_flops"] == 2*cm.band_append(4, 1, window_size=4)
    expected_baseline = sum(count*cm.band_append(n, width, window_size=4)
                            for (n, width, _), count in shapes["appends"].items())
    assert result["baseline_native_band_writer_flops"] == expected_baseline
    assert result["extra_native_writer_flops"] == 4*expected_scalar + result["native_band_writer_flops"]-expected_baseline
    assert result["complete_added_flops"] == (result["calibration_flops"]+result["release_preparation_flops"]
        +result["extra_read_flops"]+result["extra_native_writer_flops"])
    normalized = result["candidate_reads_230"]
    assert normalized["extra_candidate_read_flops"] == cached_read_extra_flops(probe, 4, 4, 230)
    assert normalized["complete_added_flops"] == (normalized["calibration_flops"]
        +normalized["release_preparation_flops"]+normalized["extra_candidate_read_flops"]
        +normalized["actual_behavior_read_extra_flops"]+normalized["extra_native_writer_flops"])


@torch.no_grad()
def test_f_mapping_uses_the_identical_native_writer_ledger():
    parent, current, probe, histories, by_user, cfg = case("f")
    _, execution = evaluate_probe(parent, current, probe, histories, by_user, [7, 8], cfg, "cpu")
    cm = CostModel(32, 2, 1, "torch")
    result = correction_cost(probe, execution, 123, cm,
        event_shapes(histories, by_user, [7, 8], cfg), cfg, 2, model=current)
    assert result["extra_native_writer_flops"] == 0
    assert result["extra_read_flops"] == 0
    assert result["corrected_scalar_native_writer_flops"] == 0
