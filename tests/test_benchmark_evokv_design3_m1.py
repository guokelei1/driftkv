from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


def test_primary_runtime_defaults_select_large_entity_boundary() -> None:
    args = MODULE.parse_args([])

    assert args.prepared_data.endswith("qk_entity_2560.npz")
    assert args.checkpoint_dir.endswith("qk_entity_h1536/seed0")
    assert args.run_id == "qk_entity_h1536_seed0"


def _tiny_slot() -> MODULE.M1PinnedSlot:
    values = [torch.empty(1) for _ in range(10)]
    return MODULE.M1PinnedSlot(*values)


def test_s1_mode_and_disjoint_slot_contract() -> None:
    args = MODULE.parse_args(["--mode", "s1"])
    proposed = MODULE.parse_args(["--mode", "d3"])
    first = _tiny_slot()
    second = _tiny_slot()

    MODULE._validate_s1_slots((first, second))
    first.output_k = first.source_k
    with pytest.raises(ValueError):
        MODULE._validate_s1_slots((first, second))

    assert args.mode == "s1"
    assert proposed.mode == "d3"


def test_s1_input_edge_reports_overlap_and_exposed_tail() -> None:
    staged = MODULE.M1S1StagedGroup(
        group=MODULE.M1Group(
            ordinal=1,
            route="compiled",
            record_ids_by_rank=((0,), ()),
        ),
        actions=(),
        local_microbatches=(),
        micro_steps=0,
        source=None,
        device_histories=(),
        oldkv_read_bytes=0,
        h2d_bytes=0,
        pageable_to_pinned_seconds=0.0,
        h2d_seconds=0.0,
        staging_started_at=2.0,
        staging_finished_at=5.0,
        slot_index=1,
    )

    edge = MODULE._s1_input_edge(
        0,
        1.0,
        4.0,
        staged,
        0.75,
    )

    assert edge["overlap_interval_seconds"] == pytest.approx(2.0)
    assert edge["staging_tail_after_producer_seconds"] == pytest.approx(1.0)
    assert edge["measured_boundary_wait_seconds"] == pytest.approx(0.75)
    assert edge["overlap_fraction"] == pytest.approx(2.0 / 3.0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for segmented input staging",
)
def test_d3_segmented_input_preserves_source_and_history(
    tmp_path: Path,
) -> None:
    actions = tuple(
        MODULE.M1Action.from_dict(_record(index, False))
        for index in range(3)
    )
    group = MODULE.M1Group(
        ordinal=0,
        route="compiled",
        record_ids_by_rank=((0, 1, 2), ()),
    )
    cfg = HSTUConfig(
        num_items=100,
        num_prediction_items=80,
        num_behaviors=5,
        hidden_size=2,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        max_seq_len=4,
    )
    target = {
        action.record_id: {
            "item_ids": np.arange(4, dtype=np.int64)
            + action.record_id * 10,
            "behaviors": np.arange(4, dtype=np.int64),
            "time_deltas": np.arange(4, dtype=np.float32),
        }
        for action in actions
    }
    expected = []
    with PageableDramExtentStore.create(
        tmp_path / "old.bin",
        tuple(value.record_id for value in actions),
        (4, 4, 4),
        num_layers=1,
        width=2,
    ) as store:
        for action in actions:
            k = (
                torch.arange(8, dtype=torch.float16)
                .view(1, 4, 2)
                .add(action.record_id * 100)
            )
            store.write_record(action.record_id, k, k + 1000)
            expected.append(k[:, 1:4])
        slots = (
            MODULE._allocate_slot(cfg, 1, 3, 4, 4),
            MODULE._allocate_slot(cfg, 1, 3, 4, 4),
        )
        device = torch.device("cuda:0")
        stream = torch.cuda.Stream(device=device)
        staged = MODULE._stage_d3_group_input(
            group,
            0,
            {value.record_id: value for value in actions},
            MODULE.M1Histories(old={}, target=target),
            store,
            slots,
            0,
            device,
            stream,
            1,
        )

    assert staged.input_pipeline == "segmented_microbatch_pingpong"
    assert staged.input_segments == 3
    assert staged.source is not None
    assert torch.equal(
        staged.source.k.cpu(),
        torch.cat(expected, dim=1),
    )
    assert [
        int(value.item_ids[0, 0].item())
        for value in staged.device_histories
    ] == [3, 13, 23]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for segmented output draining",
)
def test_d3_segmented_output_preserves_order_and_lifetime(
    tmp_path: Path,
) -> None:
    actions = tuple(
        MODULE.M1Action.from_dict(_record(index, False))
        for index in range(3)
    )
    cfg = HSTUConfig(
        num_items=100,
        num_prediction_items=80,
        num_behaviors=5,
        hidden_size=2,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        max_seq_len=4,
    )
    device = torch.device("cuda:0")
    retained_rows = [
        torch.arange(6, dtype=torch.float16)
        .view(1, 3, 2)
        .add(index * 100)
        for index in range(3)
    ]
    suffix_rows = [
        torch.full(
            (1, 1, 2),
            50 + index * 100,
            dtype=torch.float16,
        )
        for index in range(3)
    ]
    retained_k = torch.cat(retained_rows, dim=1).to(device)
    suffix_k = torch.cat(suffix_rows, dim=1).to(device)
    target_retained = MODULE._device_jagged(
        actions,
        retained_k,
        retained_k + 1000,
        "retained_tokens",
        "theta0",
        "theta1",
    )
    target_suffix = MODULE._device_jagged(
        actions,
        suffix_k,
        suffix_k + 1000,
        "suffix_tokens",
        "theta1",
        "theta1",
    )
    ready = torch.cuda.Event()
    ready.record()
    computed = MODULE.M1S1ComputedGroup(
        group=MODULE.M1Group(
            ordinal=0,
            route="compiled",
            record_ids_by_rank=((0, 1, 2), ()),
        ),
        actions=actions,
        target_retained=target_retained,
        target_suffix=target_suffix,
        target_exact=None,
        report={"ordinal": 0, "route": "compiled"},
        lookup_metrics={},
        execution_started_at=0.0,
        execution_finished_at=1.0,
        ready_event=ready,
        slot_index=0,
    )
    slots = (
        MODULE._allocate_slot(cfg, 1, 3, 4, 4),
        MODULE._allocate_slot(cfg, 1, 3, 4, 4),
    )
    with PageableDramExtentStore.create(
        tmp_path / "target.bin",
        (0, 1, 2),
        (4, 4, 4),
        num_layers=1,
        width=2,
    ) as store:
        result = MODULE._drain_d3_computed_group(
            computed,
            store,
            slots,
            device,
            torch.cuda.Stream(device=device),
            1,
        )
        observed = []
        for index in range(3):
            k = torch.empty((1, 4, 2), dtype=torch.float16)
            v = torch.empty_like(k)
            store.read_record_into(index, k, v)
            observed.append(k)
        ledger = store.ledger()

    report = result["group_report"]
    assert report["output_pipeline"] == (
        "segmented_microbatch_pingpong"
    )
    assert report["output_segments"] == 3
    assert result["observed_record_ids"] == (0, 1, 2)
    assert ledger.complete_records == 3
    assert all(
        torch.equal(
            observed[index],
            torch.cat(
                (retained_rows[index], suffix_rows[index]),
                dim=1,
            ),
        )
        for index in range(3)
    )
