from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hstu_kvcache.data.qk_xp_edge_inputs import (
    array_sha256,
    artifact_sha256,
)
from hstu_kvcache.streaming.xp_multiversion import (
    XP_MULTIVERSION_PROTOCOL,
    XP_PREQUENTIAL_MULTIVERSION_PROTOCOL,
    XPLearningRateCandidate,
    build_window_batches,
    load_xp_multiversion_schedule,
    qualification_signal,
    select_learning_rate_candidate,
    validate_xp_multiversion_corpus,
)
from hstu_kvcache.streaming.xp_projected_edge import XPProjectedModelSpec
from hstu_kvcache.streaming.xp_version_training import file_sha256


def _spec() -> XPProjectedModelSpec:
    return XPProjectedModelSpec(
        num_embeddings=17,
        embedding_width=7,
        hidden_size=4,
        num_prediction_items=8,
        num_behaviors=3,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        max_seq_len=8,
    )


def _write_corpus(path: Path, summary_path: Path) -> None:
    role_users = {
        "theta01": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "theta12": np.asarray([20, 21, 22, 23], dtype=np.int64),
        "qualification": np.asarray([30, 31, 32, 33], dtype=np.int64),
    }
    records = 12
    length = 10
    offsets = np.arange(records + 1, dtype=np.int64) * length
    items = np.tile(
        np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 1, 2], dtype=np.uint32),
        records,
    )
    labels = np.ones(records * length, dtype=np.uint8)
    labels[np.arange(records, dtype=np.int64) * length] = 0
    arrays = {
        "record_user_ids": np.concatenate(list(role_users.values())),
        "record_role": np.repeat(
            np.arange(3, dtype=np.uint8),
            4,
        ),
        "record_offsets": offsets,
        "record_history_start": np.zeros(records, dtype=np.uint16),
        "record_history_end": np.full(records, 2, dtype=np.uint16),
        "record_update_start": np.full(records, 2, dtype=np.uint16),
        "record_update_end": np.full(records, length, dtype=np.uint16),
        "item_idx": items,
        "original_item_id": items.astype(np.int32),
        "behavior": np.ones(records * length, dtype=np.uint8),
        "action_mask": labels.copy(),
        "raw_label": labels.copy(),
        "label": labels.copy(),
        "raw_ordinal": np.tile(
            np.arange(length, dtype=np.uint16),
            records,
        ),
        "is_prediction_item": np.ones(records * length, dtype=np.uint8),
        "is_stream_only_fallback": np.zeros(
            records * length,
            dtype=np.uint8,
        ),
    }
    content_hash = artifact_sha256(arrays)
    metadata = {
        "protocol": "fixture_xp_multiversion",
        "scientific_result": False,
        "formal_result": False,
        "dataset": "tenrec-qk",
        "content_sha256": content_hash,
        "catalog": {
            "base_entity_rows": 16,
            "prediction_rows": 8,
        },
        "frozen_roles": {
            "included": ["theta01", "theta12", "qualification"],
            "excluded": ["fit", "profile", "final"],
            "included_user_ids_sha256": {
                role: array_sha256(users)
                for role, users in role_users.items()
            },
        },
        "boundaries": {
            role: {"history": [0, 2], "update": [2, 10]}
            for role in role_users
        },
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        **metadata,
        "status": "pass",
        "artifact": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "file_sha256": file_sha256(path),
            "content_sha256": content_hash,
        },
        "records": {role: len(users) for role, users in role_users.items()},
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _schedule_document() -> dict[str, object]:
    return {
        "protocol": XP_MULTIVERSION_PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "stack_identity": "fixture_base4_stack",
        "edge_inputs": "edges.npz",
        "edge_summary": "edges.json",
        "base_version": 0,
        "split_roles": {
            "train": "theta01",
            "tuning": "theta12",
            "quality": "qualification",
        },
        "quality_used_for_selection": False,
        "quality_controls_training": False,
        "updates": [
            {
                "source_version": index,
                "target_version": index + 1,
                "history_end": 2 + index * 2,
                "update_end": 4 + index * 2,
            }
            for index in range(4)
        ],
        "learning_rate_screen": {
            "edge_index": 0,
            "selection_role": "tuning",
            "primary_metric": "sampled_cross_entropy_reduction",
            "minimum_cross_entropy_reduction": 0.0,
            "quality_observed_during_screen": False,
            "candidates": [
                {
                    "name": "low",
                    "dense": 0.001,
                    "projection": 0.001,
                    "embedding": 0.01,
                },
                {
                    "name": "high",
                    "dense": 0.002,
                    "projection": 0.002,
                    "embedding": 0.02,
                },
            ],
        },
        "training": {
            "epochs_per_update": 1,
            "weight_decay": 0.0,
            "train_negatives": 2,
            "tuning_negatives": 3,
            "quality_negatives": 3,
            "seeds": {
                "training": 100,
                "tuning": 200,
                "quality": 300,
            },
        },
    }


def _write_fixture(tmp_path: Path) -> Path:
    _write_corpus(tmp_path / "edges.npz", tmp_path / "edges.json")
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(_schedule_document(), indent=2, sort_keys=True) + "\n"
    )
    return schedule_path


def _write_prequential_fixture(tmp_path: Path) -> Path:
    _write_corpus(tmp_path / "edges.npz", tmp_path / "edges.json")
    document = {
        "protocol": XP_PREQUENTIAL_MULTIVERSION_PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "stack_identity": "fixture_prequential3_stack",
        "edge_inputs": "edges.npz",
        "edge_summary": "edges.json",
        "base_version": 0,
        "split_roles": {
            "train": "theta01",
            "tuning": "theta12",
            "quality": "qualification",
        },
        "quality_used_for_selection": False,
        "quality_controls_training": False,
        "tuning_used_for_selection": False,
        "tuning_controls_training": False,
        "evaluation_semantics": "next_unseen_window",
        "checkpoint_admission": (
            "numerical_stability_and_nonzero_update"
        ),
        "updates": [
            {
                "source_version": index,
                "target_version": index + 1,
                "history_end": 2 + index * 2,
                "update_end": 4 + index * 2,
            }
            for index in range(3)
        ],
        "prequential_evaluations": [
            {
                "model_version": index,
                "history_end": 2 + index * 2,
                "evaluation_end": 4 + index * 2,
            }
            for index in range(4)
        ],
        "learning_rate_policy": {
            "mode": "predeclared_fixed",
            "selection_role": "none",
            "quality_observed_for_selection": False,
            "selected_candidate": "fixed",
            "candidates": [
                {
                    "name": "fixed",
                    "dense": 0.001,
                    "projection": 0.001,
                    "embedding": 0.01,
                }
            ],
        },
        "training": {
            "epochs_per_update": 3,
            "weight_decay": 0.0,
            "train_negatives": 2,
            "tuning_negatives": 3,
            "quality_negatives": 3,
            "seeds": {
                "training": 100,
                "tuning": 200,
                "quality": 300,
            },
        },
    }
    schedule_path = tmp_path / "prequential.json"
    schedule_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    return schedule_path


def test_schedule_corpus_isolation_and_window_batches(tmp_path: Path) -> None:
    schedule = load_xp_multiversion_schedule(_write_fixture(tmp_path))
    validated = validate_xp_multiversion_corpus(schedule, _spec())

    assert len(schedule.updates) == 4
    assert validated.audit["split_users_pairwise_disjoint"] is True
    assert validated.audit["legacy_two_edge_result_compatible"] is False
    batches, coverage = build_window_batches(
        validated.corpus,
        schedule.split_roles["train"],
        schedule.updates[-1],
        max_seq_len=8,
        batch_size_per_rank=1,
        rank=0,
        world_size=2,
    )
    assert len(batches) == 2
    assert coverage["history_end"] == 8
    assert coverage["update_end"] == 10
    assert coverage["physical_sequence_width"] == 8
    assert coverage["causal_window_start"] == 2
    assert coverage["local_targets"] == 4
    assert batches[0]["train_mask"].tolist() == [
        [False, False, False, False, False, False, True, True]
    ]


def test_prequential_schedule_binds_three_updates_and_four_windows(
    tmp_path: Path,
) -> None:
    schedule = load_xp_multiversion_schedule(
        _write_prequential_fixture(tmp_path)
    )
    validated = validate_xp_multiversion_corpus(schedule, _spec())

    assert schedule.protocol == XP_PREQUENTIAL_MULTIVERSION_PROTOCOL
    assert schedule.epochs_per_update == 3
    assert len(schedule.updates) == 3
    assert len(schedule.evaluation_windows) == 4
    assert schedule.evaluation_windows[-1].evaluation_end == 10
    assert validated.audit["evaluation_semantics"] == "next_unseen_window"


def test_schedule_rejects_overlapping_update_windows(tmp_path: Path) -> None:
    _write_corpus(tmp_path / "edges.npz", tmp_path / "edges.json")
    document = _schedule_document()
    document["updates"][1]["history_end"] = 3
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="not contiguous"):
        load_xp_multiversion_schedule(path)


def test_lr_selection_uses_only_predeclared_tuning_signal() -> None:
    candidates = (
        XPLearningRateCandidate("low", 0.001, 0.001, 0.01),
        XPLearningRateCandidate("high", 0.002, 0.002, 0.02),
    )
    before = {
        "sampled_cross_entropy": 1.0,
        "ndcg_at_10": 0.2,
        "hit_rate_at_10": 0.3,
        "mean_reciprocal_rank": 0.1,
    }
    weak = qualification_signal(
        before,
        {
            "sampled_cross_entropy": 0.99,
            "ndcg_at_10": 0.21,
            "hit_rate_at_10": 0.31,
            "mean_reciprocal_rank": 0.11,
        },
    )
    strong = qualification_signal(
        before,
        {
            "sampled_cross_entropy": 0.8,
            "ndcg_at_10": 0.22,
            "hit_rate_at_10": 0.32,
            "mean_reciprocal_rank": 0.12,
        },
    )
    reports = (
        {"candidate": "low", "tuning_signal": weak},
        {"candidate": "high", "tuning_signal": strong},
    )

    assert select_learning_rate_candidate(candidates, reports, 0.0) == (
        candidates[1]
    )
    assert select_learning_rate_candidate(candidates, reports, 0.25) is None


def test_two_rank_multiversion_batch_stops_or_commits_atomically(
    tmp_path: Path,
) -> None:
    schedule = _write_fixture(tmp_path)
    base = tmp_path / "base"
    checkpoint = tmp_path / "versions"
    output = tmp_path / "result.json"
    ledgers = tmp_path / "ledgers"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/run_evokv_xp_projected_checkpoint_canary.py",
            "--device",
            "cpu",
            "--checkpoint-dir",
            str(base),
            "--output",
            str(tmp_path / "base.json"),
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/train_evokv_xp_multiversion.py",
            "--device",
            "cpu",
            "--development-canary",
            "--schedule",
            str(schedule),
            "--base-checkpoint-root",
            str(base),
            "--checkpoint-root",
            str(checkpoint),
            "--output",
            str(output),
            "--ledger-dir",
            str(ledgers),
            "--progress-every",
            "100",
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["legacy_two_edge_result_compatible"] is False
    assert result["leakage_contract"]["quality_used_for_selection"] is False
    assert result["learning_rate_screen"][
        "quality_observed_during_screen"
    ] is False
    if result["status"] == "complete":
        assert result["downstream_d1_d2_gate_passed"] is True
        assert len(result["updates"]) == 4
        assert all(
            (checkpoint / f"theta_{version}" / "manifest.json").is_file()
            and (ledgers / f"version_{version:05d}.json").is_file()
            for version in range(1, 5)
        )
    else:
        assert result["downstream_d1_d2_gate_passed"] is False
        assert result["status"] in {
            "learning_rate_screen_positive_signal_gate_failed",
            "update_tuning_positive_signal_gate_failed",
        }


def test_two_rank_prequential_batch_commits_without_quality_gate(
    tmp_path: Path,
) -> None:
    schedule = _write_prequential_fixture(tmp_path)
    base = tmp_path / "base"
    checkpoint = tmp_path / "versions"
    output = tmp_path / "result.json"
    ledgers = tmp_path / "ledgers"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/run_evokv_xp_projected_checkpoint_canary.py",
            "--device",
            "cpu",
            "--checkpoint-dir",
            str(base),
            "--output",
            str(tmp_path / "base.json"),
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/train_evokv_xp_multiversion.py",
            "--device",
            "cpu",
            "--development-canary",
            "--schedule",
            str(schedule),
            "--base-checkpoint-root",
            str(base),
            "--checkpoint-root",
            str(checkpoint),
            "--output",
            str(output),
            "--ledger-dir",
            str(ledgers),
            "--progress-every",
            "100",
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["protocol"] == XP_PREQUENTIAL_MULTIVERSION_PROTOCOL
    assert result["status"] == "complete"
    assert result["checkpoint_admission"]["ranking_metrics_used"] is False
    assert len(result["updates"]) == 3
    assert len(result["prequential_evaluations"]) == 4
    assert all(
        update["checkpoint_admission"]["passed"] is True
        and update["target_checkpoint_committed"] is True
        for update in result["updates"]
    )
    assert all(
        (checkpoint / f"theta_{version}" / "manifest.json").is_file()
        and (ledgers / f"version_{version:05d}.json").is_file()
        for version in range(1, 4)
    )
