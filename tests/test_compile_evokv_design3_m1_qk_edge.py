from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compile_evokv_design3_m1_qk_edge.py"
SPEC = importlib.util.spec_from_file_location(
    "compile_evokv_design3_m1_qk_edge",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_tiny_prepared(
    path: Path,
    cohort_path: Path,
    invert_labels: bool = False,
) -> None:
    users = 12
    history = 512
    slide = 32
    total = history + slide
    user_idx = np.repeat(
        np.arange(1, users + 1, dtype=np.int64),
        total,
    )
    positions = np.tile(np.arange(total, dtype=np.int64), users)
    item_idx = ((user_idx * 17 + positions * 7) % 97 + 1).astype(np.int64)
    behavior = ((positions + user_idx) % 5).astype(np.int64)
    label = ((positions + user_idx) % 3 == 0).astype(np.int64)
    if invert_labels:
        label = 1 - label
    time_ms = (positions * 1_000).astype(np.int64)
    window_index = np.where(positions < history, -1, 0).astype(np.int8)
    original_user_ids = np.arange(10_000, 10_000 + users, dtype=np.int64)
    metadata = {
        "protocol": "tiny_qk_m1_fixture_v0",
        "dataset": "tenrec-qk",
        "selected_users": users,
        "fitted_items": 97,
        "num_prediction_items": 97,
        "context_hash_buckets": 0,
        "num_behaviors": 5,
        "window_count": 1,
        "history_length": history,
        "slide": slide,
        "old_window_filtered_positions": [0, history],
        "target_window_filtered_positions": [slide, slide + history],
    }
    np.savez_compressed(
        path,
        user_idx=user_idx,
        item_idx=item_idx,
        behavior=behavior,
        label=label,
        time_ms=time_ms,
        window_index=window_index,
        original_user_ids=original_user_ids,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    cohort = {
        "protocol": "tiny_qk_m1_fixture_v0",
        "fit_calibration_user_ids": original_user_ids[:2].tolist(),
        "benchmark_user_ids": original_user_ids[2:].tolist(),
    }
    cohort_path.write_text(json.dumps(cohort))


def test_snapshot_has_adjacent_extents_balanced_actions_and_stable_hash(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared.npz"
    cohorts = tmp_path / "cohorts.json"
    write_tiny_prepared(prepared, cohorts)

    first, fit_samples = MODULE.build_development_snapshot(
        prepared,
        cohorts,
        fit_users=2,
        horizon=5,
    )
    second, _ = MODULE.build_development_snapshot(
        prepared,
        cohorts,
        fit_users=2,
        horizon=5,
    )

    assert first["owner_independent_plan_sha256"] == second["owner_independent_plan_sha256"]
    assert first["counts"] == {
        "records": 10,
        "scheduled_exact": 2,
        "natural_exact": 0,
        "exact": 2,
        "compiled": 8,
        "scheduled_exact_fraction": 0.2,
    }
    assert first["scheduler"]["scheduled_exact_ids"] == [0, 5]
    assert first["scheduler"]["migrate_ids"] == [
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
    ]
    assert len(fit_samples) == 2
    assert all(len(value["history"]["item_ids"]) == 512 for value in fit_samples)
    for record in first["records"]:
        assert record["old_tokens"] == 512
        assert record["retained_start"] == 32
        assert record["retained_tokens"] == 480
        assert record["delta_start"] == 480
        assert record["delta_tokens"] == 31
        assert record["target_prefix_tokens"] == 511
        assert record["latest_tokens"] == 1
        assert record["final_tokens"] == 512
        assert not any("owner" in key for key in record)
    assert first["labels_used"] is False
    assert first["scheduler"]["labels_used"] is False


def test_snapshot_identity_and_routing_are_label_free(
    tmp_path: Path,
) -> None:
    cohorts = tmp_path / "cohorts.json"
    original = tmp_path / "original.npz"
    inverted = tmp_path / "inverted.npz"
    write_tiny_prepared(original, cohorts)
    write_tiny_prepared(inverted, cohorts, invert_labels=True)

    first, _ = MODULE.build_development_snapshot(
        original,
        cohorts,
        fit_users=2,
        horizon=5,
    )
    second, _ = MODULE.build_development_snapshot(
        inverted,
        cohorts,
        fit_users=2,
        horizon=5,
    )

    identity_fields = (
        "old_history_sha256",
        "target_history_sha256",
        "retained_identity_sha256",
        "delta_identity_sha256",
        "target_prefix_identity_sha256",
        "latest_identity_sha256",
    )
    assert [tuple(record[name] for name in identity_fields) for record in first["records"]] == [
        tuple(record[name] for name in identity_fields) for record in second["records"]
    ]
    assert (
        first["scheduler"]["action_partition_sha256"]
        == second["scheduler"]["action_partition_sha256"]
    )
    assert first["roles"]["fit"]["selection_sha256"] == second["roles"]["fit"]["selection_sha256"]
    assert first["prepared_data_sha256"] != second["prepared_data_sha256"]


def test_plan_only_needs_no_training_checkpoint_or_gpu(
    tmp_path: Path,
    capsys,
) -> None:
    prepared = tmp_path / "prepared.npz"
    cohorts = tmp_path / "cohorts.json"
    output = tmp_path / "actions.json"
    write_tiny_prepared(prepared, cohorts)

    MODULE.main(
        [
            "--prepared-data",
            str(prepared),
            "--cohort-ids",
            str(cohorts),
            "--fit-users",
            "2",
            "--action-output",
            str(output),
            "--training-result",
            str(tmp_path / "missing-training.json"),
            "--checkpoint-dir",
            str(tmp_path / "missing-checkpoints"),
            "--device",
            "cpu",
            "--plan-only",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    snapshot = json.loads(output.read_text())
    assert printed["status"] == "plan_only"
    assert snapshot["counts"]["scheduled_exact"] == 2
    assert (
        snapshot["owner_independent_plan_sha256"]
        == printed["action_snapshot"]["owner_independent_plan_sha256"]
    )


def test_fit_item_compaction_preserves_histories_and_uses_sorted_rows() -> None:
    samples = [
        {
            "history": {
                "item_ids": np.asarray([9, 2, 9], dtype=np.int64),
                "behaviors": np.asarray([1, 2, 1], dtype=np.int64),
            },
            "pos_items": [],
        },
        {
            "history": {
                "item_ids": np.asarray([4, 2], dtype=np.int64),
                "behaviors": np.asarray([3, 1], dtype=np.int64),
            },
            "pos_items": [],
        },
    ]

    used, compact = MODULE.compact_fit_samples(samples)

    assert used == (0, 2, 4, 9)
    assert compact[0]["history"]["item_ids"].tolist() == [3, 1, 3]
    assert compact[1]["history"]["item_ids"].tolist() == [2, 1]
    assert np.array_equal(
        compact[0]["history"]["behaviors"],
        samples[0]["history"]["behaviors"],
    )
    assert samples[0]["history"]["item_ids"].tolist() == [9, 2, 9]


def test_primary_defaults_select_large_entity_boundary() -> None:
    args = MODULE.parse_args([])

    assert args.prepared_data.endswith("qk_entity_2560.npz")
    assert args.checkpoint_dir.endswith("qk_entity_h1536/seed0")
    assert args.cohort_ids.endswith("qk_entity_cohorts.json")
