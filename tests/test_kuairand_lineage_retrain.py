import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hstu_kvcache.streaming import kuairand_projected_persistent as persistent
from hstu_kvcache.streaming.kuairand_lineage_retrain import (
    _lineage_gate,
    _lineage_workload_limit,
    _temperature_document,
    load_lineage_retrain_config,
)
from hstu_kvcache.streaming.xp_projected_edge import OptimizerActiveRowTracker

CONFIG = (
    "configs/evokv_root_cause/kuairand_projected_lineage_retrain_theta7_theta8_20260808_v0.json"
)
CONFIG_V1 = (
    "configs/evokv_root_cause/kuairand_projected_lineage_retrain_theta7_theta8_20260808_v1.json"
)
CONFIG_V2 = (
    "configs/evokv_root_cause/kuairand_projected_lineage_retrain_theta7_theta8_20260808_v2.json"
)
CONFIG_V5 = "configs/evokv_root_cause/kuairand_projected_lineage_retrain_theta8_20260808_v5.json"
AUDIT_CONFIG = "configs/evokv_root_cause/kuairand_lineage_distribution_audit_20260809_v6.json"
AMPLIFIED_CONFIG = (
    "configs/evokv_root_cause/kuairand_amplified_fixed_theta1_theta10_20260809_v0.json"
)
NEXT1_CONFIG = (
    "configs/evokv_root_cause/kuairand_stationary_next1_theta5_theta10_20260809_v1.json"
)
SMOOTH_THETA3_CONFIG = (
    "configs/evokv_root_cause/kuairand_smooth_theta3_next1_20260809_v0.json"
)
EXTENSION_CONFIG = (
    "configs/evokv_root_cause/kuairand_theta11_theta13_extension_20260809_v2.json"
)
MEDIUM_REBUILD_CONFIG = (
    "configs/evokv_root_cause/kuairand_foundation_rebuild_medium_pooled4_20260809_v0.json"
)
LARGE_FOUNDATION_CONFIG = (
    "configs/evokv_root_cause/"
    "kuairand_foundation_rebuild_large_e4160_theta12_20260809_v7.json"
)
LARGE_THETA6_PROBE_CONFIG = (
    "configs/evokv_root_cause/"
    "kuairand_foundation_rebuild_large_theta6_probes_20260810_v1.json"
)


@dataclass(frozen=True)
class _TinyDenseConfig:
    width: int = 2


class _TinyDense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = _TinyDenseConfig()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))


class _TinyEmbedding:
    def __init__(self, rank: int, world_size: int = 2):
        self.num_embeddings = 7
        rows = (self.num_embeddings + world_size - 1 - rank) // world_size
        self.local_weight = torch.nn.Parameter(
            torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) + rank
        )
        self.projection_weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))

    @property
    def embedding_width(self):
        return self.local_weight.shape[1]

    @property
    def local_rows(self):
        return self.local_weight.shape[0]


def _sparse_delta_roundtrip_worker(rank: int, root_value: str, rendezvous: str):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    root = Path(root_value)
    document = {
        "checkpoint": {
            "versions": 2,
            "imported_prefix_versions": 1,
            "expected_global_parameter_bytes": 123,
            "embedding_storage": "sparse_delta_after_imported_prefix",
            "compatible_config_sha256": [],
        }
    }
    geometry = {"global_model_parameter_bytes": 123}
    dense = _TinyDense()
    embedding = _TinyEmbedding(rank)
    tracker = OptimizerActiveRowTracker(num_embeddings=7, rank=rank, world_size=2)
    persistent._save_checkpoint(
        root,
        1,
        dense,
        embedding,
        tracker,
        geometry,
        document,
        "tiny",
        {},
        rank,
        2,
    )
    rows = torch.tensor([1], dtype=torch.int64)
    with torch.no_grad():
        embedding.local_weight[1].add_(10 + rank)
        embedding.projection_weight.add_(2)
        dense.weight.add_(1)
    tracker.mark_local_rows(rows)
    expected_embedding = embedding.local_weight.detach().clone()
    expected_projection = embedding.projection_weight.detach().clone()
    expected_dense = dense.weight.detach().clone()
    persistent._save_checkpoint(
        root,
        2,
        dense,
        embedding,
        tracker,
        geometry,
        document,
        "tiny",
        {},
        rank,
        2,
        rows,
    )
    with torch.no_grad():
        embedding.local_weight.zero_()
        embedding.projection_weight.zero_()
        dense.weight.zero_()
    persistent._load_checkpoint(root, 2, dense, embedding, tracker, document, "tiny", rank, True)
    assert torch.equal(embedding.local_weight, expected_embedding)
    assert torch.equal(embedding.projection_weight, expected_projection)
    assert torch.equal(dense.weight, expected_dense)
    assert tracker.local_update_counts[1].item() == 1
    dist.destroy_process_group()


def _summary(value: float):
    metrics = {
        metric: {"relative_percent": value}
        for metric in (
            "candidate_cross_entropy",
            "hit_rate_at_5",
            "mrr",
            "ndcg_at_5",
        )
    }
    return {
        "endpoints": {
            "recompute": {
                "mrr": 0.2,
                "ndcg_at_5": 0.2,
                "hit_rate_at_5": 0.2,
            }
        },
        "comparisons": {
            "recompute_over_reuse": metrics,
            "fresh_update_value": metrics,
        },
        "sanity": {"passed": True},
    }


def test_lineage_retrain_config_and_gate():
    document = load_lineage_retrain_config(CONFIG)
    assert document["lineage_selection"]["versions"] == [7, 8]
    summaries = {source: _summary(4.0) for source in range(1, 7)}
    assert _lineage_gate(summaries, 7, document)["passed"]
    summaries[3] = _summary(-0.1)
    assert not _lineage_gate(summaries, 7, document)["passed"]


def test_lineage_temperature_sweep_config_is_separate():
    document = load_lineage_retrain_config(CONFIG_V1)
    candidate = document["training"]["candidate_ladder"][0]
    assert candidate["evaluation_temperatures"] == [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    source = {"training": {"temperature": 0.05}}
    updated = _temperature_document(source, 0.15)
    assert source["training"]["temperature"] == 0.05
    assert updated["training"]["temperature"] == 0.15


def test_lineage_primary_metric_gate_keeps_supporting_metrics_positive():
    document = load_lineage_retrain_config(CONFIG_V2)
    summaries = {source: _summary(4.0) for source in range(1, 7)}
    summaries[4]["comparisons"]["recompute_over_reuse"]["mrr"] = {"relative_percent": 0.1}
    assert _lineage_gate(summaries, 7, document)["passed"]
    summaries[4]["comparisons"]["recompute_over_reuse"]["mrr"] = {"relative_percent": -0.1}
    assert not _lineage_gate(summaries, 7, document)["passed"]


def test_lineage_gate_can_report_ce_without_using_it_for_admission():
    document = load_lineage_retrain_config(CONFIG_V2)
    document["lineage_selection"]["require_candidate_ce_positive"] = False
    summaries = {source: _summary(4.0) for source in range(1, 7)}
    summaries[2]["comparisons"]["recompute_over_reuse"]["candidate_cross_entropy"] = {
        "relative_percent": -1.0
    }
    gate = _lineage_gate(summaries, 7, document)
    assert not gate["all_lineage_candidate_ce_positive"]
    assert gate["candidate_ce_gate_pass"]
    assert gate["passed"]


def test_lineage_gate_can_report_fresh_update_without_using_it_for_admission():
    document = load_lineage_retrain_config(CONFIG_V2)
    document["lineage_selection"]["require_fresh_update_ranking_positive"] = False
    summaries = {source: _summary(4.0) for source in range(1, 7)}
    summaries[6]["comparisons"]["fresh_update_value"] = {
        metric: value.copy()
        for metric, value in summaries[6]["comparisons"]["fresh_update_value"].items()
    }
    summaries[6]["comparisons"]["fresh_update_value"]["ndcg_at_5"] = {"relative_percent": -1.0}
    gate = _lineage_gate(summaries, 7, document)
    assert not gate["fresh_update_ranking_positive"]
    assert gate["fresh_update_gate_pass"]
    assert gate["passed"]


def test_lineage_gate_can_allow_one_nonpositive_cell_when_frozen():
    document = load_lineage_retrain_config(CONFIG_V2)
    document["lineage_selection"]["minimum_lineage_positive_fraction"] = 0.8
    summaries = {source: _summary(4.0) for source in range(1, 7)}
    summaries[2]["comparisons"]["recompute_over_reuse"]["ndcg_at_5"] = {
        "relative_percent": -0.1
    }
    gate = _lineage_gate(summaries, 7, document)
    assert not gate["all_lineage_ranking_positive"]
    assert gate["lineage_ranking_positive_fraction"] == 17 / 18
    assert gate["lineage_ranking_fraction_pass"]
    assert gate["passed"]


def test_lineage_gate_can_reject_extreme_relative_loss_when_frozen():
    document = load_lineage_retrain_config(CONFIG_V2)
    document["lineage_selection"]["maximum_lineage_relative_percent"] = 100.0
    summaries = {source: _summary(4.0) for source in range(1, 7)}
    summaries[2]["comparisons"]["recompute_over_reuse"]["ndcg_at_5"] = {
        "relative_percent": 101.0
    }
    gate = _lineage_gate(summaries, 7, document)
    assert gate["maximum_observed_relative_percent"] == 101.0
    assert not gate["maximum_lineage_pass"]
    assert not gate["passed"]


def test_theta8_only_retrain_uses_imported_theta7_and_absolute_floor():
    document = load_lineage_retrain_config(CONFIG_V5)
    assert document["checkpoint"]["imported_prefix_versions"] == 7
    assert document["lineage_selection"]["versions"] == [8]
    summaries = {source: _summary(5.0) for source in range(1, 8)}
    assert _lineage_gate(summaries, 8, document)["passed"]
    summaries[7]["endpoints"]["recompute"]["ndcg_at_5"] = 0.12
    assert not _lineage_gate(summaries, 8, document)["passed"]


def test_distribution_audit_uses_eight_targets_without_training():
    document = persistent.load_persistent_config(AUDIT_CONFIG)
    assert document["evaluation"]["targets_per_user"] == 8
    assert document["checkpoint"]["imported_prefix_versions"] == 8
    assert document["lineage_selection"]["audit_only"] is True


def test_lineage_config_accepts_immediate_next_item():
    document = load_lineage_retrain_config(NEXT1_CONFIG)
    assert document["evaluation"]["targets_per_user"] == 1


def test_lineage_config_accepts_frozen_two_percent_bridge():
    document = load_lineage_retrain_config(SMOOTH_THETA3_CONFIG)
    assert document["lineage_selection"]["minimum_lineage_mean_relative_percent"] == 2.0


def test_persistent_selection_groups_support_ten_versions():
    assert persistent._selection_groups_valid([[2, 3], [4, 6], [7, 10]], 10)
    assert not persistent._selection_groups_valid([[2, 3], [5, 10]], 10)


def test_persistent_config_accepts_one_rank_medium_rebuild():
    document = persistent.load_persistent_config(MEDIUM_REBUILD_CONFIG)
    assert document["execution"]["world_size"] == 1
    assert document["transitions"][0]["update_dates"] == ["20220422", "20220423"]


def test_large_foundation_freezes_schedule_and_hybrid_storage():
    document = persistent.load_persistent_config(LARGE_FOUNDATION_CONFIG)
    assert document["execution"]["world_size"] == 2
    assert document["model"]["embedding_width"] == 4160
    assert document["checkpoint"]["full_checkpoint_from_version"] == 5
    assert persistent._embedding_storage_for_version(document, 4) == "sparse_delta"
    assert persistent._embedding_storage_for_version(document, 5) == "full"
    assert persistent._candidate_sequence(document, 10)[0]["name"] == "recent32_e2_kv005"
    assert persistent._candidate_sequence(document, 11)[0]["name"] == "recent32_e3_kv0075"


def test_large_theta6_probe_overlay_is_hash_bound_and_valid():
    document = persistent.load_candidate_probe_config(
        LARGE_THETA6_PROBE_CONFIG,
        LARGE_FOUNDATION_CONFIG,
    )
    assert len(document["candidates"]) == 8
    assert document["candidates"][0]["name"] == "recent32_e3_kv015"
    assert document["candidates"][-1]["name"] == "projection_dominant_n8192_e2"


def test_candidate_latest_update_dates_requires_sequential_training():
    candidate = {
        "name": "latest_one",
        "update_epochs": 2,
        "maximum_update_examples": 1024,
        "embedding_lr": 0.001,
        "projection_lr": 0.001,
        "dense_lr": 0.001,
        "temporal_training": "sequential_dates",
        "latest_update_dates": 1,
    }
    assert persistent._candidate_valid(candidate)
    candidate["temporal_training"] = "pooled"
    assert not persistent._candidate_valid(candidate)
    candidate["temporal_training"] = "sequential_dates"
    candidate["latest_update_dates"] = 0
    assert not persistent._candidate_valid(candidate)


def test_candidate_qkv_only_scope_is_explicit_and_validated():
    candidate = {
        "name": "qkv_only",
        "update_epochs": 2,
        "maximum_update_examples": 1024,
        "embedding_lr": 0.001,
        "projection_lr": 0.001,
        "dense_lr": 0.001,
        "kv_lr": 0.01,
        "dense_update_scope": "qkv_only",
    }
    assert persistent._candidate_valid(candidate)
    candidate["dense_update_scope"] = "attention"
    assert not persistent._candidate_valid(candidate)


def test_amplified_fixed_schedule_extends_to_theta10():
    document = persistent.load_persistent_config(AMPLIFIED_CONFIG)
    assert document["checkpoint"]["versions"] == 10
    assert document["checkpoint"]["imported_prefix_versions"] == 1
    assert document["training"]["checkpoint_policy"] == "fixed_schedule"
    assert document["training"]["candidate_ladder"][0]["kv_lr"] == 0.004
    assert document["transitions"][-1]["evaluation_date"] == "20220502"


def test_lineage_workload_limit_follows_checkpoint_versions():
    document = {"checkpoint": {"versions": 10}}
    assert _lineage_workload_limit(document) == 10
    assert _lineage_workload_limit(document, 7) == 7


def test_extension_uses_sparse_delta_checkpoints_after_theta10(monkeypatch, tmp_path):
    document = load_lineage_retrain_config(EXTENSION_CONFIG)
    checkpoint = document["checkpoint"]
    assert checkpoint["imported_prefix_versions"] == 10
    assert checkpoint["embedding_storage"] == "sparse_delta_after_imported_prefix"
    monkeypatch.setattr(
        persistent.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=100 << 30),
    )
    result = persistent._disk_preflight(document, tmp_path, 10)
    assert result["remaining_versions"] == 3
    assert result["required_remaining_bytes"] == 88 << 30


def test_sparse_delta_checkpoint_roundtrip_is_exact(tmp_path):
    mp.spawn(
        _sparse_delta_roundtrip_worker,
        args=(str(tmp_path / "checkpoints"), str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )


def test_sparse_warmup_full_suffix_storage_and_disk_preflight(monkeypatch, tmp_path):
    document = {
        "checkpoint": {
            "versions": 12,
            "imported_prefix_versions": 0,
            "expected_global_parameter_bytes": 100,
            "expected_checkpoint_bytes_per_version": 100,
            "expected_sparse_checkpoint_bytes_per_version": 10,
            "embedding_storage": "sparse_warmup_full_suffix",
            "full_checkpoint_from_version": 5,
            "minimum_free_bytes": 1,
            "write_reserve_bytes": 5,
        }
    }
    expected = [
        "full",
        "sparse_delta",
        "sparse_delta",
        "sparse_delta",
        "full",
        "full",
    ]
    assert [
        persistent._embedding_storage_for_version(document, version)
        for version in range(1, 7)
    ] == expected
    monkeypatch.setattr(
        persistent.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=10_000),
    )
    initial = persistent._disk_preflight(document, tmp_path, 0)
    resumed = persistent._disk_preflight(document, tmp_path, 5)
    assert initial["required_remaining_bytes"] == 935
    assert resumed["required_remaining_bytes"] == 705


def test_sparse_warmup_full_suffix_roundtrip_breaks_delta_chain(tmp_path):
    root = tmp_path / "checkpoints"
    document = {
        "execution": {"world_size": 1},
        "checkpoint": {
            "versions": 3,
            "imported_prefix_versions": 0,
            "expected_global_parameter_bytes": 123,
            "embedding_storage": "sparse_warmup_full_suffix",
            "full_checkpoint_from_version": 3,
            "compatible_config_sha256": [],
        },
    }
    geometry = {"global_model_parameter_bytes": 123}
    dense = _TinyDense()
    embedding = _TinyEmbedding(0, 1)
    tracker = OptimizerActiveRowTracker(num_embeddings=7, rank=0, world_size=1)
    persistent._save_checkpoint(
        root,
        1,
        dense,
        embedding,
        tracker,
        geometry,
        document,
        "tiny",
        {},
        0,
        1,
    )
    rows = torch.tensor([1], dtype=torch.int64)
    with torch.no_grad():
        embedding.local_weight[1].add_(10)
        embedding.projection_weight.add_(2)
        dense.weight.add_(1)
    tracker.mark_local_rows(rows)
    version2_embedding = embedding.local_weight.detach().clone()
    version2_projection = embedding.projection_weight.detach().clone()
    version2_dense = dense.weight.detach().clone()
    persistent._save_checkpoint(
        root,
        2,
        dense,
        embedding,
        tracker,
        geometry,
        document,
        "tiny",
        {},
        0,
        1,
        rows,
    )
    with torch.no_grad():
        embedding.local_weight[2].add_(20)
        embedding.projection_weight.add_(3)
        dense.weight.add_(4)
    tracker.mark_local_rows(torch.tensor([2], dtype=torch.int64))
    version3_embedding = embedding.local_weight.detach().clone()
    version3_projection = embedding.projection_weight.detach().clone()
    version3_dense = dense.weight.detach().clone()
    persistent._save_checkpoint(
        root,
        3,
        dense,
        embedding,
        tracker,
        geometry,
        document,
        "tiny",
        {},
        0,
        1,
    )
    with torch.no_grad():
        embedding.local_weight.zero_()
        embedding.projection_weight.zero_()
        dense.weight.zero_()
    persistent._load_checkpoint(root, 2, dense, embedding, tracker, document, "tiny", 0, True)
    assert torch.equal(embedding.local_weight, version2_embedding)
    assert torch.equal(embedding.projection_weight, version2_projection)
    assert torch.equal(dense.weight, version2_dense)
    with torch.no_grad():
        embedding.local_weight.zero_()
        embedding.projection_weight.zero_()
        dense.weight.zero_()
    persistent._load_checkpoint(root, 3, dense, embedding, tracker, document, "tiny", 0, True)
    assert torch.equal(embedding.local_weight, version3_embedding)
    assert torch.equal(embedding.projection_weight, version3_projection)
    assert torch.equal(dense.weight, version3_dense)


def test_candidate_schedule_selects_one_frozen_candidate_per_version():
    baseline = {"name": "baseline"}
    strong = {"name": "strong"}
    document = {
        "training": {
            "initial_candidate": {"name": "initial"},
            "candidate_ladder": [baseline, strong],
            "candidate_schedule": {"2": "baseline", "3": "strong"},
        },
        "transitions": [
            {"update_date": "20220422"},
            {"update_date": "20220423"},
            {"update_date": "20220424"},
        ],
    }
    assert persistent._candidate_sequence(document, 1)[0]["name"] == "initial"
    assert persistent._candidate_sequence(document, 2) == [baseline]
    assert persistent._candidate_sequence(document, 3) == [strong]


def test_single_rank_full_checkpoint_roundtrip_is_exact(tmp_path):
    root = tmp_path / "checkpoints"
    document = {
        "execution": {"world_size": 1},
        "checkpoint": {
            "versions": 1,
            "imported_prefix_versions": 0,
            "expected_global_parameter_bytes": 123,
            "embedding_storage": "full",
            "compatible_config_sha256": [],
        },
    }
    geometry = {"global_model_parameter_bytes": 123}
    dense = _TinyDense()
    embedding = _TinyEmbedding(0, 1)
    tracker = OptimizerActiveRowTracker(num_embeddings=7, rank=0, world_size=1)
    expected_embedding = embedding.local_weight.detach().clone()
    expected_projection = embedding.projection_weight.detach().clone()
    expected_dense = dense.weight.detach().clone()
    persistent._save_checkpoint(
        root,
        1,
        dense,
        embedding,
        tracker,
        geometry,
        document,
        "tiny",
        {},
        0,
        1,
    )
    with torch.no_grad():
        embedding.local_weight.zero_()
        embedding.projection_weight.zero_()
        dense.weight.zero_()
    persistent._load_checkpoint(root, 1, dense, embedding, tracker, document, "tiny", 0, True)
    assert torch.equal(embedding.local_weight, expected_embedding)
    assert torch.equal(embedding.projection_weight, expected_projection)
    assert torch.equal(dense.weight, expected_dense)


def test_lineage_holdout_summary_is_user_disjoint(monkeypatch):
    records = [{"user_id": user_id} for user_id in range(128)]
    expected = [
        record
        for record in records
        if int.from_bytes(
            hashlib.sha256(f"17:{record['user_id']}".encode()).digest()[:8],
            "little",
        )
        / float(1 << 64)
        >= 0.25
    ]
    monkeypatch.setattr(
        persistent,
        "_summary",
        lambda evaluation, document: {"seen": evaluation["records"]},
    )
    result = persistent._lineage_holdout_summary(
        {"records": records, "sanity": {"passed": True}},
        {"evaluation": {"bootstrap_samples": 2000}},
        17,
        0.25,
        200,
    )
    assert result["seen"] == expected
    assert result["partition"]["records"] == len(expected)
    assert result["partition"]["bootstrap_samples"] == 200


def test_lineage_tuning_and_holdout_are_complements(monkeypatch):
    records = [{"user_id": user_id} for user_id in range(128)]
    monkeypatch.setattr(
        persistent,
        "_summary",
        lambda evaluation, document: {"seen": evaluation["records"]},
    )
    evaluation = {"records": records, "sanity": {"passed": True}}
    document = {"evaluation": {"bootstrap_samples": 2000}}
    tuning = persistent._lineage_partition_summary(evaluation, document, 17, 0.25, 200, "tuning")
    holdout = persistent._lineage_partition_summary(evaluation, document, 17, 0.25, 200, "holdout")
    tuning_users = {record["user_id"] for record in tuning["seen"]}
    holdout_users = {record["user_id"] for record in holdout["seen"]}
    assert not tuning_users & holdout_users
    assert tuning_users | holdout_users == set(range(128))
