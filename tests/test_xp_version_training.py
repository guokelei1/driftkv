from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from hstu_kvcache.data.qk_xp_edge_inputs import (
    array_sha256,
    artifact_sha256,
)
from hstu_kvcache.streaming.xp_version_training import (
    build_role_batches,
    file_sha256,
    load_xp_fixed_edge_corpus,
)


def _write_edge_fixture(path: Path, summary_path: Path) -> None:
    role_users = {
        "theta01": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "theta12": np.asarray([20, 21, 22, 23], dtype=np.int64),
        "qualification": np.asarray(
            [30, 31, 32, 33],
            dtype=np.int64,
        ),
    }
    lengths = np.asarray(
        [6] * 4 + [8] * 4 + [6] * 4,
        dtype=np.uint16,
    )
    history = np.asarray(
        [4] * 4 + [6] * 4 + [4] * 4,
        dtype=np.uint16,
    )
    offsets = np.zeros(13, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    item_parts = []
    behavior_parts = []
    label_parts = []
    ordinal_parts = []
    for record, length in enumerate(lengths.tolist()):
        item_parts.append(
            (
                np.arange(length, dtype=np.uint32)
                + record
            )
            % 8
            + 1
        )
        behavior_parts.append(
            (
                np.arange(length, dtype=np.uint8)
                + record
            )
            % 3
            + 1
        )
        labels = np.ones(length, dtype=np.uint8)
        labels[0] = 0
        label_parts.append(labels)
        ordinal_parts.append(
            np.arange(length, dtype=np.uint16)
        )
    items = np.concatenate(item_parts)
    behaviors = np.concatenate(behavior_parts)
    labels = np.concatenate(label_parts)
    ordinals = np.concatenate(ordinal_parts)
    arrays = {
        "record_user_ids": np.concatenate(
            [
                role_users["theta01"],
                role_users["theta12"],
                role_users["qualification"],
            ]
        ),
        "record_role": np.concatenate(
            [
                np.full(4, code, dtype=np.uint8)
                for code in range(3)
            ]
        ),
        "record_offsets": offsets,
        "record_history_start": np.zeros(12, dtype=np.uint16),
        "record_history_end": history,
        "record_update_start": history.copy(),
        "record_update_end": lengths,
        "item_idx": items,
        "original_item_id": items.astype(np.int32),
        "behavior": behaviors,
        "action_mask": labels.copy(),
        "raw_label": labels.copy(),
        "label": labels.copy(),
        "raw_ordinal": ordinals,
        "is_prediction_item": np.ones(
            len(items),
            dtype=np.uint8,
        ),
        "is_stream_only_fallback": np.zeros(
            len(items),
            dtype=np.uint8,
        ),
    }
    content_hash = artifact_sha256(arrays)
    frozen_roles = {
        "included": ["theta01", "theta12", "qualification"],
        "excluded": ["fit", "profile", "final"],
        "file_sha256": "fixture-role-file",
        "hash_salt": "fixture",
        "source_protocol": "fixture",
        "included_user_ids_sha256": {
            role: array_sha256(users)
            for role, users in role_users.items()
        },
    }
    boundaries = {
        "theta01": {
            "history": [0, 4],
            "update": [4, 6],
        },
        "theta12": {
            "history": [0, 6],
            "update": [6, 8],
        },
        "qualification": {
            "history": [0, 4],
            "update": [4, 6],
        },
    }
    metadata = {
        "protocol": "fixture_xp_edges",
        "scientific_result": False,
        "formal_result": False,
        "dataset": "tenrec-qk",
        "content_sha256": content_hash,
        "catalog": {
            "base_entity_rows": 16,
            "prediction_rows": 8,
        },
        "frozen_roles": frozen_roles,
        "boundaries": boundaries,
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
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
        "records": {
            role: len(users)
            for role, users in role_users.items()
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def test_fixed_edge_loader_and_causal_batches(tmp_path: Path) -> None:
    path = tmp_path / "edges.npz"
    summary_path = tmp_path / "edges.json"
    _write_edge_fixture(path, summary_path)

    corpus = load_xp_fixed_edge_corpus(
        path,
        summary_path,
        num_embeddings=17,
        num_prediction_items=8,
        num_behaviors=3,
    )
    batches, coverage = build_role_batches(
        corpus,
        "theta12",
        max_seq_len=6,
        batch_size_per_rank=1,
        rank=0,
        world_size=2,
    )

    assert len(batches) == 2
    assert coverage["global_records"] == 4
    assert coverage["physical_sequence_width"] == 6
    assert coverage["causal_window_start"] == 2
    assert coverage["local_targets"] == 4
    first = batches[0]
    assert first["lengths"].tolist() == [6]
    assert first["window_starts"].tolist() == [2]
    assert first["time_deltas"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    assert first["train_mask"].tolist() == [
        [False, False, False, False, True, True]
    ]


def test_two_rank_cpu_real_edge_training_canary(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint_canary = tmp_path / "checkpoint_canary.json"
    edge_inputs = tmp_path / "edges.npz"
    edge_summary = tmp_path / "edges.json"
    output = tmp_path / "result.json"
    _write_edge_fixture(edge_inputs, edge_summary)
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
            str(checkpoint),
            "--output",
            str(checkpoint_canary),
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
            "scripts/train_evokv_xp_two_edges.py",
            "--device",
            "cpu",
            "--development-canary",
            "--edge-inputs",
            str(edge_inputs),
            "--edge-summary",
            str(edge_summary),
            "--checkpoint-root",
            str(checkpoint),
            "--output",
            str(output),
            "--train-negatives",
            "2",
            "--qualification-negatives",
            "3",
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
    assert completed.returncode == 0, (
        completed.stdout + completed.stderr
    )
    result = json.loads(output.read_text())
    assert result["status"].startswith("complete")
    assert result["execution"]["world_size"] == 2
    assert result["execution"]["device"] == "cpu"
    assert result["coverage"]["theta01"]["global_targets"] == 8
    assert result["coverage"]["theta12"]["global_targets"] == 8
    assert (
        result["optimizer_active_delta"]["theta0_to_theta1"][
            "global_updated_rows"
        ]
        > 0
    )
    assert (
        result["optimizer_active_delta"]["theta1_to_theta2"][
            "global_updated_rows"
        ]
        > 0
    )
    assert result["leakage_contract"][
        "qualification_used_by_optimizer"
    ] is False
    assert (
        result["independent_qualification"]["theta0"][
            "positive_targets"
        ]
        == 8
    )
    assert (
        checkpoint / "theta_1" / "manifest.json"
    ).is_file()
    assert (
        checkpoint / "theta_2" / "manifest.json"
    ).is_file()
