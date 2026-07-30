from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration.design2_plan import canonical_sha256
from hstu_kvcache.migration.design3_store import (
    PageableDramExtentStore,
)
from hstu_kvcache.models import HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_evokv_design3_m1.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_evokv_design3_m1",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(record_id: int, exact: bool) -> dict[str, object]:
    return {
        "record_id": record_id,
        "prepared_user_id": record_id + 1,
        "requested_action": "exact" if exact else "compiled",
        "requested_reason": "scheduled_exact" if exact else "migrate",
        "old_tokens": 4,
        "retained_start": 1,
        "retained_tokens": 3,
        "delta_start": 3,
        "delta_tokens": 0,
        "target_prefix_tokens": 3,
        "latest_tokens": 1,
        "final_tokens": 4,
    }


def _snapshot() -> dict[str, object]:
    records = [_record(index, index % 3 == 0) for index in range(12)]
    payload = {
        "protocol": "tiny_m1",
        "status": "development_snapshot",
        "scientific_result": False,
        "formal_design3": False,
        "source_version": "theta0",
        "target_version": "theta1",
        "labels_used": False,
        "future_history_used": False,
        "prepared_data_sha256": "0" * 64,
        "counts": {
            "records": 12,
            "compiled": 8,
            "exact": 4,
        },
        "records": records,
    }
    return {
        **payload,
        "owner_independent_plan_sha256": canonical_sha256(payload),
        "bindings": {},
    }


def test_action_snapshot_owner_and_group_plan_are_stable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.json"
    path.write_text(json.dumps(_snapshot()))
    snapshot, actions = MODULE.load_action_snapshot(path)
    owners = MODULE.build_owner_map(actions, 2)
    selected = MODULE.select_actions(
        actions,
        owners,
        2,
        "canary",
        1,
    )
    groups = MODULE.build_s0_groups(
        selected,
        owners,
        2,
        1,
    )

    assert snapshot["owner_independent_plan_sha256"]
    assert len(selected) == 4
    assert {value.route for value in selected} == {
        "compiled",
        "exact",
    }
    assert len(groups) == 2
    assert [value.route for value in groups] == [
        "compiled",
        "exact",
    ]
    observed = [
        record_id
        for group in groups
        for values in group.record_ids_by_rank
        for record_id in values
    ]
    assert set(observed) == {value.record_id for value in selected}
    assert MODULE.group_plan_sha256(groups) == (
        MODULE.group_plan_sha256(groups)
    )


def test_capacity_projection_counts_full_old_and_private_target() -> None:
    actions = tuple(
        MODULE.M1Action.from_dict(value)
        for value in _snapshot()["records"]
    )
    owners = MODULE.build_owner_map(actions, 2)
    cfg = HSTUConfig(
        num_items=100,
        num_prediction_items=80,
        num_behaviors=5,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        max_seq_len=4,
    )

    projection = MODULE.capacity_projection(actions, owners, 2, cfg)
    bytes_per_record = 2 * 2 * 4 * 8 * 2

    assert projection["old_store_payload_bytes"] == 12 * bytes_per_record
    assert projection["target_store_payload_bytes"] == 12 * bytes_per_record
    assert projection["combined_store_payload_bytes"] == 24 * bytes_per_record
    assert [value["records"] for value in projection["ranks"]] == [6, 6]


def test_segmented_and_exact_publication_complete_target_store(
    tmp_path: Path,
) -> None:
    actions = (
        MODULE.M1Action.from_dict(_record(0, False)),
        MODULE.M1Action.from_dict(_record(1, True)),
    )
    with PageableDramExtentStore.create(
        tmp_path / "target.bin",
        (0, 1),
        (4, 4),
        num_layers=1,
        width=2,
    ) as store:
        retained_k = torch.arange(
            6,
            dtype=torch.float16,
        ).view(1, 3, 2)
        retained_v = retained_k + 10
        suffix_k = torch.full((1, 1, 2), 20, dtype=torch.float16)
        suffix_v = suffix_k + 10
        exact_k = torch.full((1, 4, 2), 40, dtype=torch.float16)
        exact_v = exact_k + 10

        compiled_bytes = MODULE.publish_output_segments(
            store,
            (actions[0],),
            "compiled",
            ((retained_k, retained_v), (suffix_k, suffix_v)),
        )
        exact_bytes = MODULE.publish_output_segments(
            store,
            (actions[1],),
            "exact",
            ((exact_k, exact_v),),
        )

        ledger = store.ledger()
        read_k = torch.empty((1, 4, 2), dtype=torch.float16)
        read_v = torch.empty_like(read_k)
        store.read_record_into(0, read_k, read_v)
        assert compiled_bytes == 2 * 1 * 4 * 2 * 2
        assert exact_bytes == compiled_bytes
        assert ledger.complete_records == 2
        assert ledger.partial_records == 0
        assert ledger.missing_records == 0
        assert torch.equal(read_k[:, :3], retained_k)
        assert torch.equal(read_k[:, 3:], suffix_k)


def test_packed_slot_views_remain_contiguous_for_short_batches() -> None:
    storage = torch.empty((2, 20, 3), dtype=torch.float16)
    first = MODULE._packed_slot_kv_view(storage, 4)
    second = MODULE._packed_slot_kv_view(storage, 2, 4)

    assert first.shape == (2, 4, 3)
    assert second.shape == (2, 2, 3)
    assert first.is_contiguous()
    assert second.is_contiguous()
    assert (
        first.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )


def test_old_store_reuse_is_bound_to_model_data_plan_and_owner(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    action = tmp_path / "action"
    training = tmp_path / "training"
    checkpoint = tmp_path / "theta0"
    for index, path in enumerate(
        (prepared, action, training, checkpoint)
    ):
        path.write_bytes(bytes((index,)))
    snapshot = {"owner_independent_plan_sha256": "a" * 64}
    store_path = tmp_path / "rank0.old.bin"
    with PageableDramExtentStore.create(
        store_path,
        (0,),
        (4,),
        num_layers=1,
        width=2,
    ) as store:
        binding = MODULE._old_store_binding(
            store,
            0,
            prepared,
            action,
            snapshot,
            training,
            checkpoint,
            "b" * 64,
        )
        MODULE._write_old_store_binding(store_path, binding)
        MODULE._validate_old_store_binding(store_path, binding)
        changed = {**binding, "owner_map_sha256": "c" * 64}
        with pytest.raises(ValueError):
            MODULE._validate_old_store_binding(store_path, changed)
