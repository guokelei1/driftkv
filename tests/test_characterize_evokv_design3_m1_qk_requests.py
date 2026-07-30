from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "characterize_evokv_design3_m1_qk_requests.py"
)
SPEC = importlib.util.spec_from_file_location(
    "characterize_evokv_design3_m1_qk_requests",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_fixture(
    prepared: Path,
    snapshot_path: Path,
) -> None:
    users = 4
    history_tokens = 8
    slide = 2
    total = history_tokens + slide
    user_idx = np.repeat(
        np.arange(1, users + 1, dtype=np.int64),
        total,
    )
    base_items = np.asarray(
        [1, 2, 9, 4, 10, 6, 11, 8, 3, 12],
        dtype=np.int64,
    )
    item_idx = np.tile(base_items, users)
    behavior = np.tile(
        np.arange(total, dtype=np.int64) % 3 + 1,
        users,
    )
    label = (item_idx <= 8).astype(np.int64)
    time_ms = np.tile(
        np.arange(total, dtype=np.int64) * 1_000,
        users,
    )
    window_index = np.tile(
        np.asarray(
            [-1] * history_tokens + [0] * slide,
            dtype=np.int8,
        ),
        users,
    )
    original_user_ids = np.asarray(
        [1001, 1002, 1003, 1004],
        dtype=np.int64,
    )
    metadata = {
        "protocol": "tiny_m1_request_fixture_v0",
        "dataset": "tenrec-qk",
        "selected_users": users,
        "fitted_items": 12,
        "num_prediction_items": 8,
        "context_hash_buckets": 4,
        "num_behaviors": 3,
        "window_count": 1,
        "history_length": history_tokens,
        "slide": slide,
    }
    np.savez_compressed(
        prepared,
        user_idx=user_idx,
        item_idx=item_idx,
        behavior=behavior,
        label=label,
        time_ms=time_ms,
        window_index=window_index,
        original_user_ids=original_user_ids,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    target_items = base_items[slide:]
    target_behaviors = (
        np.arange(total, dtype=np.int64) % 3 + 1
    )[slide:]
    target_times = np.arange(
        total,
        dtype=np.int64,
    )[slide:] * 1_000
    target_time_deltas = np.ones(
        history_tokens,
        dtype=np.float32,
    )
    target = {
        "item_ids": target_items,
        "behaviors": target_behaviors,
        "time_deltas": target_time_deltas,
        "timestamps": target_times,
    }
    target_sha256 = MODULE.history_identity_sha256(
        target,
        0,
        history_tokens,
    )
    layout = {
        "history_tokens": history_tokens,
        "old_filtered_start": 0,
        "old_filtered_stop": history_tokens,
        "target_filtered_start": slide,
        "target_filtered_stop": total,
        "retained_start": slide,
        "retained_tokens": history_tokens - slide,
        "delta_start": history_tokens - slide,
        "delta_tokens": slide - 1,
        "target_prefix_tokens": history_tokens - 1,
        "latest_tokens": 1,
        "final_tokens": history_tokens,
    }
    records = [
        {
            "record_id": record_id,
            "prepared_user_id": record_id + 1,
            "original_user_id": int(
                original_user_ids[record_id]
            ),
            "requested_action": (
                "exact" if record_id % 2 == 0 else "compiled"
            ),
            "old_tokens": history_tokens,
            "final_tokens": history_tokens,
            "target_history_sha256": target_sha256,
        }
        for record_id in range(users)
    ]
    plan = {
        "protocol": (
            "evokv_design3_m1_qk_adjacent_action_snapshot_dev_v0"
        ),
        "status": "development_snapshot",
        "scientific_result": False,
        "formal_design3": False,
        "prepared_data_sha256": MODULE.file_sha256(prepared),
        "layout": layout,
        "counts": {
            "records": users,
            "exact": 2,
            "compiled": 2,
        },
        "records": records,
    }
    snapshot = {
        **plan,
        "owner_independent_plan_sha256": MODULE.canonical_sha256(
            plan
        ),
        "bindings": {"prepared_data": str(prepared)},
    }
    snapshot_path.write_text(
        json.dumps(snapshot, sort_keys=True)
    )


def test_characterization_binds_inputs_and_splits_requests(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared.npz"
    snapshot = tmp_path / "actions.json"
    write_fixture(prepared, snapshot)

    result = MODULE.build_result(
        prepared,
        snapshot,
        expected_records=4,
        expected_history_tokens=8,
    )

    assert result["bindings"]["prepared_data"][
        "snapshot_hash_verified"
    ]
    assert result["bindings"]["action_snapshot"][
        "content_hash_verified"
    ]
    target_binding = result["bindings"]["target_histories"]
    assert target_binding["records_verified"] == 4
    assert len(target_binding["aggregate_target_history_sha256"]) == 64
    assert target_binding["all_record_hashes_verified"]
    assert result["configuration"][
        "record_owner_equivalent_to"
    ] == "record_id_modulo_2"
    assert result["methods"]["all_exact"]["request_tokens"] == 32
    assert result["methods"]["mixed"]["request_tokens"] == 20
    assert result["methods"]["mixed"]["by_item_role"][
        "prediction"
    ]["request_tokens"] == 10
    assert result["methods"]["mixed"]["by_item_role"]["context"][
        "request_tokens"
    ] == 10
    assert result["methods"]["mixed"]["requester_ranks"][0][
        "request_tokens"
    ] == 16
    assert result["methods"]["mixed"]["requester_ranks"][1][
        "request_tokens"
    ] == 4
    assert result["timing_scope"]["old_source_materialization"][
        "classification"
    ] == "setup_only_excluded_from_primary_timer"


def test_main_writes_compact_result(tmp_path: Path, capsys) -> None:
    prepared = tmp_path / "prepared.npz"
    snapshot = tmp_path / "actions.json"
    output = tmp_path / "result.json"
    write_fixture(prepared, snapshot)

    MODULE.main(
        [
            "--prepared-data",
            str(prepared),
            "--action-snapshot",
            str(snapshot),
            "--output",
            str(output),
            "--expected-records",
            "4",
            "--expected-history-tokens",
            "8",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed["status"] == "complete"
    assert written["scientific_result"] is False
    assert "records" not in written
    assert output.stat().st_size < 50_000


def test_rejects_target_history_hash_mismatch(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared.npz"
    snapshot_path = tmp_path / "actions.json"
    write_fixture(prepared, snapshot_path)
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["records"][0]["target_history_sha256"] = "0" * 64
    plan = {
        key: value
        for key, value in snapshot.items()
        if key not in {"owner_independent_plan_sha256", "bindings"}
    }
    snapshot["owner_independent_plan_sha256"] = (
        MODULE.canonical_sha256(plan)
    )
    snapshot_path.write_text(json.dumps(snapshot, sort_keys=True))

    with pytest.raises(
        ValueError,
        match="target history hash differs",
    ):
        MODULE.build_result(
            prepared,
            snapshot_path,
            expected_records=4,
            expected_history_tokens=8,
        )
