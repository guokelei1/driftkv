from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from hstu_kvcache.models import HSTU, HSTUConfig
from insight.reader_compatibility_correction import (
    STAGES,
    intervene_reader_correction,
    trace_reader_correction,
)
from insight_two.common import (
    ANCHOR_INDICES,
    HELDOUT_INDICES,
    STAGE_PRESENTATION,
    score_metrics,
)
from insight_two.low_rank_correction import (
    fit_predict_low_rank,
    low_rank_final_representation,
    low_rank_layered_correction,
)
from insight_two.temporal_persistence import (
    append_bucket,
    correction_drift,
    correction_sha256,
    remaining_parent_fraction,
    scale_correction,
    time_bucket,
)


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=96,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=16,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def test_frozen_candidate_split_is_disjoint_and_complete() -> None:
    assert len(ANCHOR_INDICES) == len(HELDOUT_INDICES) == 32
    assert set(ANCHOR_INDICES).isdisjoint(HELDOUT_INDICES)
    assert sorted(ANCHOR_INDICES + HELDOUT_INDICES) == list(range(64))
    assert set(STAGE_PRESENTATION) == set(STAGES)


def test_score_metrics_has_exact_and_reuse_endpoints() -> None:
    exact = torch.tensor([[2.0, -1.0, 0.5]])
    reuse = torch.tensor([[0.0, 1.0, -0.5]])
    exact_metrics = score_metrics(exact, reuse, exact)
    reuse_metrics = score_metrics(exact, reuse, reuse)
    assert torch.allclose(exact_metrics["probability_gap_recovery"], torch.ones(1))
    assert torch.allclose(exact_metrics["logit_gap_recovery"], torch.ones(1))
    assert torch.allclose(reuse_metrics["probability_gap_recovery"], torch.zeros(1))
    assert torch.allclose(reuse_metrics["logit_gap_recovery"], torch.zeros(1))


def test_anchor_correction_replays_on_heldout_without_mutating_cache() -> None:
    parent, current = _model(7), _model(11)
    items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
    deltas = torch.arange(8).float().unsqueeze(0)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    original_k = reuse_cache.k.clone()
    original_v = reuse_cache.v.clone()
    anchors = torch.tensor([[17, 19, 21, 23]])
    heldout = torch.tensor([[18, 20, 22, 24]])
    query_delta = torch.tensor([10.0])
    trace = trace_reader_correction(
        current, exact_cache, reuse_cache, anchors, query_delta
    )
    exact_scores = current.score_cc_reuse(exact_cache, heldout, query_delta)
    reuse_scores = current.score_cc_reuse(reuse_cache, heldout, query_delta)
    for stage in STAGES:
        scores, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage=stage,
            corrections=trace.corrections[stage],
        )
        metrics = score_metrics(exact_scores, reuse_scores, scores)
        assert all(torch.isfinite(value).all() for value in metrics.values())
    assert torch.equal(reuse_cache.k, original_k)
    assert torch.equal(reuse_cache.v, original_v)


def test_rank_zero_reduced_rank_fit_is_anchor_mean() -> None:
    features = torch.randn(2, 4, 6)
    targets = torch.randn(2, 4, 8)
    heldout = torch.randn(2, 3, 6)
    anchor_prediction, heldout_prediction, diagnostics, storage = fit_predict_low_rank(
        features, targets, heldout, rank=0
    )
    expected = targets.mean(dim=1, keepdim=True)
    assert torch.allclose(anchor_prediction, expected.expand_as(targets))
    assert torch.allclose(heldout_prediction, expected.expand(2, 3, 8))
    assert len(diagnostics) == 2
    assert storage == 8


def test_rank_zero_new_adapter_matches_frozen_stage_intervention() -> None:
    parent, current = _model(29), _model(31)
    items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
    deltas = torch.arange(8).float().unsqueeze(0)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    anchors = torch.tensor([[17, 19, 21, 23]])
    heldout = torch.tensor([[18, 20, 22, 24]])
    query_delta = torch.tensor([10.0])
    trace = trace_reader_correction(current, exact_cache, reuse_cache, anchors, query_delta)
    for stage in ("av_aggregation", "u_gated_update", "layer_hidden"):
        expected, _ = intervene_reader_correction(
            current,
            reuse_cache,
            heldout,
            query_delta,
            stage=stage,
            corrections=trace.corrections[stage],
        )
        observed = low_rank_layered_correction(
            current,
            exact_cache,
            reuse_cache,
            anchors,
            heldout,
            query_delta,
            stage=stage,
            rank=0,
        )
        assert torch.allclose(observed.scores, expected, atol=1e-6)
    expected, _ = intervene_reader_correction(
        current,
        reuse_cache,
        heldout,
        query_delta,
        stage="final_readout",
        corrections=trace.corrections["final_readout"],
    )
    observed = low_rank_final_representation(
        current,
        exact_cache,
        reuse_cache,
        anchors,
        heldout,
        query_delta,
        rank=0,
    )
    assert torch.allclose(observed.scores, expected, atol=1e-6)


def test_temporal_persistence_buckets_and_coverage_are_frozen() -> None:
    assert time_bucket(0) == "[0d,1d)"
    assert time_bucket(86_400) == "[1d,3d)"
    assert time_bucket(7 * 86_400) == "[7d,14d)"
    assert append_bucket(0) == "0"
    assert append_bucket(8) == "[1,8]"
    assert append_bucket(513) == ">512"
    assert remaining_parent_fraction(0, 1024) == 1.0
    assert remaining_parent_fraction(128, 1024) == 0.875
    assert remaining_parent_fraction(2048, 1024) == 0.0


def test_temporal_correction_drift_and_hash() -> None:
    frozen = (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]]))
    current = scale_correction(frozen, 2.0)
    metrics = correction_drift(current, frozen)
    assert torch.allclose(metrics["direction_cosine"], torch.ones(1))
    assert torch.allclose(
        metrics["current_to_frozen_norm_ratio"], torch.full((1,), 2.0)
    )
    original_hash = correction_sha256(frozen)
    assert original_hash == correction_sha256(tuple(value.clone() for value in frozen))
    assert original_hash != correction_sha256(current)
