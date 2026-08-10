from __future__ import annotations

from runpy import run_path

import torch

from hstu_kvcache.streaming.qk_stream_runner import _parameter_probe
from hstu_kvcache.streaming.sharded_edge import ExternalEmbeddingHSTU
from hstu_kvcache.streaming.xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
)

MATERIALIZER = run_path("scripts/materialize_evokv_qk_theta2_core_heavy.py")
SUMMARIZER = run_path("scripts/summarize_evokv_qk_theta2_core_heavy.py")
CANDIDATES = MATERIALIZER["CANDIDATES"]
PRIMARY_GATE = SUMMARIZER["_primary_gate"]


def test_core_heavy_bindings_match_frozen_matrix() -> None:
    entries = (
        "scripts/train_evokv_qk_theta2_core_d150_p100_e025_n32.py",
        "scripts/train_evokv_qk_theta2_core_d200_p100_e025_n32.py",
        "scripts/train_evokv_qk_theta2_core_d150_p150_e025_n32.py",
        "scripts/train_evokv_qk_theta2_core_d150_p100_e050_n32.py",
    )
    bindings = tuple(run_path(path)["BINDING"] for path in entries)
    observed = {
        value.candidate_name: (
            value.epochs,
            value.dense_learning_rate,
            value.projection_learning_rate,
            value.embedding_learning_rate,
            value.train_negative_count,
        )
        for value in bindings
    }
    assert observed == CANDIDATES
    assert {value.training_seed for value in bindings} == {2026080611}
    assert {value.negative_seed for value in bindings} == {2026080623}


def test_parameter_probe_separates_direct_kv_parameters() -> None:
    spec = XPProjectedModelSpec(
        num_embeddings=11,
        embedding_width=5,
        hidden_size=4,
        num_prediction_items=6,
        num_behaviors=3,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        max_seq_len=8,
    )
    dense = ExternalEmbeddingHSTU(spec.hstu_config())
    embedding = TrainableProjectedModuloEmbedding(
        local_weight=torch.zeros(spec.num_embeddings, spec.embedding_width),
        projection_weight=torch.zeros(spec.hidden_size, spec.embedding_width),
        num_embeddings=spec.num_embeddings,
        rank=0,
        world_size=1,
    )
    probe = _parameter_probe(dense, embedding)
    assert set(probe) == {"dense", "direct_kv", "non_kv_dense", "projection"}
    assert probe["direct_kv"].numel() > 0
    assert probe["non_kv_dense"].numel() > 0


def test_core_heavy_gate_requires_gap_intervals_and_quality_floor() -> None:
    row = {
        "positive_targets": 77479,
        "recompute": {"ndcg_at_10": 0.0053, "mrr": 0.0058},
        "gaps": {
            "ndcg_at_10": {
                "relative_percent": 6.0,
                "positive_direction_with_ci": True,
            },
            "mrr": {
                "relative_percent": 5.0,
                "positive_direction_with_ci": True,
            },
        },
    }
    arguments = ([5.0, 10.0], 5000, 0.0053, 0.0058, 0.98)
    assert PRIMARY_GATE(row, *arguments)["status"] == "pass"
    row["recompute"]["ndcg_at_10"] = 0.0050
    assert PRIMARY_GATE(row, *arguments)["status"] == "fail"
    row["recompute"]["ndcg_at_10"] = 0.0053
    row["gaps"]["mrr"]["positive_direction_with_ci"] = False
    assert PRIMARY_GATE(row, *arguments)["status"] == "fail"
