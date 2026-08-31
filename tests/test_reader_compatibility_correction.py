from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from hstu_kvcache.models import HSTU, HSTUConfig
from reader_compatibility_correction import (
    STAGES,
    correction_cosine,
    correction_norm,
    intervene_reader_correction,
    scale_correction,
    trace_reader_correction,
)
from evaluate_reader_correction_persistence_raw import evaluate_group_batch


def _model(seed: int) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=64,
            num_behaviors=3,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            head_dim=8,
            max_seq_len=32,
            temporal_num_freqs=2,
            input_dropout=0.0,
        )
    ).eval()


def _inputs():
    items = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
    )
    behaviors = torch.tensor(
        [[1, 2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1, 2, 1]]
    )
    deltas = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6, 8, 10, 12, 14]]
    ).float()
    return items, behaviors, deltas


def test_stage_trace_matches_native_reader_and_has_all_frozen_stages() -> None:
    parent, current = _model(3), _model(5)
    items, behaviors, deltas = _inputs()
    exact = current.compute_kv(items, behaviors, deltas)
    reuse = parent.compute_kv(items, behaviors, deltas)
    candidates = torch.tensor([[17, 18, 19, 20], [21, 22, 23, 24]])
    query_deltas = torch.tensor([10.0, 20.0])
    trace = trace_reader_correction(
        current, exact, reuse, candidates, query_deltas
    )
    assert max(trace.correctness.values()) < 1e-6
    assert set(trace.stage_scores) == set(STAGES)
    assert set(trace.corrections) == set(STAGES)
    assert len(trace.corrections["av_aggregation"]) == 2
    assert trace.corrections["av_aggregation"][0].shape == (2, 2, 8)
    assert trace.corrections["u_gated_update"][0].shape == (2, 16)
    assert trace.corrections["final_readout"][0].shape == (2, 16)
    assert len(trace.energy_metrics) == 2 * 4 + 1
    for metrics in trace.energy_metrics:
        assert torch.max(metrics["orthogonality_error"]) < 1e-5
        assert torch.allclose(
            metrics["shared_energy"] + metrics["residual_energy"],
            metrics["total_energy"],
            rtol=1e-4,
            atol=1e-5,
        )


def test_one_candidate_final_broadcast_correction_reconstructs_exact() -> None:
    parent, current = _model(7), _model(11)
    items, behaviors, deltas = _inputs()
    exact = current.compute_kv(items, behaviors, deltas)
    reuse = parent.compute_kv(items, behaviors, deltas)
    candidates = torch.tensor([[17], [21]])
    query_deltas = torch.tensor([10.0, 20.0])
    trace = trace_reader_correction(
        current, exact, reuse, candidates, query_deltas
    )
    assert torch.allclose(
        trace.stage_scores["final_readout"], trace.exact_scores, atol=1e-6
    )
    scores, _ = intervene_reader_correction(
        current,
        reuse,
        candidates,
        query_deltas,
        stage="final_readout",
        corrections=trace.corrections["final_readout"],
    )
    assert torch.allclose(scores, trace.exact_scores, atol=1e-6)


def test_frozen_same_request_corrections_replay_dynamic_stage_paths() -> None:
    parent, current = _model(13), _model(17)
    items, behaviors, deltas = _inputs()
    exact = current.compute_kv(items, behaviors, deltas)
    reuse = parent.compute_kv(items, behaviors, deltas)
    candidates = torch.tensor([[17, 18, 19, 20], [21, 22, 23, 24]])
    query_deltas = torch.tensor([10.0, 20.0])
    trace = trace_reader_correction(current, exact, reuse, candidates, query_deltas)
    for stage in STAGES:
        scores, _ = intervene_reader_correction(
            current,
            reuse,
            candidates,
            query_deltas,
            stage=stage,
            corrections=trace.corrections[stage],
        )
        assert torch.allclose(scores, trace.stage_scores[stage], atol=1e-6)


def test_correction_signature_cosine_norm_and_scaling() -> None:
    current = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]),)
    previous = (torch.tensor([[2.0, 4.0], [6.0, 8.0]]),)
    assert torch.allclose(correction_cosine(current, previous), torch.ones(2))
    assert torch.allclose(correction_norm(previous), 2.0 * correction_norm(current))
    scaled = scale_correction(previous, torch.tensor([0.5, 0.25]))
    assert torch.allclose(scaled[0], torch.tensor([[1.0, 2.0], [1.5, 2.0]]))


def test_real_group_batch_emits_adjacent_request_persistence() -> None:
    parent, current = _model(19), _model(23)
    items, behaviors, deltas = _inputs()
    exact = current.compute_kv(items[:1], behaviors[:1], deltas[:1])
    reuse = parent.compute_kv(items[:1], behaviors[:1], deltas[:1])
    prior = {}
    score_records, energy_records, persistence_records, correctness_records = [], [], [], []
    common = dict(
        current=current,
        exact_cache=exact,
        reuse_cache=reuse,
        owners=[0],
        edge="v0_to_v1",
        append_counts=[0],
        evictions=[0],
        cutover=100,
        prior=prior,
        verify_full_delta=True,
        score_records=score_records,
        energy_records=energy_records,
        persistence_records=persistence_records,
        correctness_records=correctness_records,
    )
    evaluate_group_batch(
        **common,
        groups=[
            {
                "uid": 7,
                "query_timestamp": 110,
                "max_width": 4,
                "items": torch.tensor([17, 18, 19, 20]).numpy(),
            }
        ],
        query_deltas=torch.tensor([10.0]),
    )
    assert not persistence_records
    evaluate_group_batch(
        **common,
        groups=[
            {
                "uid": 7,
                "query_timestamp": 130,
                "max_width": 4,
                "items": torch.tensor([21, 22, 23, 24]).numpy(),
            }
        ],
        query_deltas=torch.tensor([30.0]),
    )
    assert len(persistence_records) == 4
    assert {row["stage"] for row in persistence_records} == {
        "av_aggregation",
        "u_gated_update",
        "layer_hidden",
        "final_readout",
    }
    assert all(row["seconds_between_requests"] == 20 for row in persistence_records)
