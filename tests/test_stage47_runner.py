import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hstu_kvcache.migration import (
    CacheLifecycleState,
    JaggedMigratedKVBatch,
)
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "run_cohortkv_stage4_7_organic_chain.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage4_7_organic_chain",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass
class FakeHistory:
    timestamps: np.ndarray
    item_ids: np.ndarray
    behaviors: np.ndarray
    time_deltas: np.ndarray

    def __len__(self) -> int:
        return len(self.item_ids)


def history(
    timestamps: list[int],
    items: list[int],
    behaviors: list[int],
) -> FakeHistory:
    return FakeHistory(
        timestamps=np.asarray(timestamps, dtype=np.int64),
        item_ids=np.asarray(items, dtype=np.int64),
        behaviors=np.asarray(behaviors, dtype=np.int64),
        time_deltas=np.zeros(len(items), dtype=np.float32),
    )


def record(
    value: FakeHistory | None,
    digest: str | None,
    identities: tuple[str, ...] | None = None,
):
    prepared_identities = (
        ()
        if value is None
        else tuple(
            f"{timestamp}:{item}:{behavior}"
            for timestamp, item, behavior in zip(
                value.timestamps,
                value.item_ids,
                value.behaviors,
                strict=True,
            )
        )
    )
    return SimpleNamespace(
        history=value,
        history_sha256=digest,
        history_event_identities=(
            prepared_identities if identities is None else identities
        ),
        engaged_positive_item_ids=(),
    )


def exact_state(record_id: int, version: int = 0) -> CacheLifecycleState:
    return CacheLifecycleState.exact(record_id, version)


def migrated_state(
    record_id: int,
    served_version: int,
    depth: int,
) -> CacheLifecycleState:
    return CacheLifecycleState(
        record_id=record_id,
        served_version=served_version,
        last_exact_version=served_version - depth,
        migration_depth=depth,
        risk_score=0.0,
        state_kind="migrated",
    )


def action_map(decisions) -> dict[int, str]:
    return {
        decision.record_id: decision.action for decision in decisions
    }


def test_norm_shift_selector_uses_fixed_quota_and_current_shift() -> None:
    states = tuple(exact_state(record_id) for record_id in range(10))
    shifts = {record_id: record_id / 10 for record_id in range(10)}
    decisions = MODULE.select_norm_shift_decisions(
        states,
        1,
        shifts,
    )
    selected = {
        decision.record_id
        for decision in decisions
        if decision.action == "exact"
    }

    assert selected == {8, 9}
    assert len(selected) == 2
    assert all(decision.candidate_evaluated for decision in decisions)
    assert all(
        decision.reason
        in {"norm_shift_exact_quota", "norm_shift_migrate"}
        for decision in decisions
    )


def test_norm_shift_selector_prioritizes_deadline_then_age() -> None:
    states = (
        migrated_state(0, 4, 4),
        migrated_state(1, 4, 2),
        migrated_state(2, 4, 1),
        exact_state(3, 4),
        exact_state(4, 4),
    )
    shifts = {
        0: 0.0,
        1: 0.01,
        2: 100.0,
        3: 200.0,
        4: 300.0,
    }
    decisions = MODULE.select_norm_shift_decisions(
        states,
        5,
        shifts,
        exact_fraction=0.4,
    )
    selected = {
        decision.record_id
        for decision in decisions
        if decision.action == "exact"
    }
    by_id = {decision.record_id: decision for decision in decisions}

    assert selected == {0, 1}
    assert by_id[0].reason == "depth_deadline_after_probe"
    assert by_id[0].candidate_evaluated
    assert by_id[1].reason == "norm_shift_exact_quota"
    assert by_id[2].action == "migrate"
    audit = MODULE.selector_audit(
        states,
        decisions,
        shifts,
        exact_fraction=0.4,
    )
    assert audit["mandatory_depth_exact_records"] == 1
    assert audit["norm_shift_quota_exact_records"] == 1
    assert audit["realized_exact_fraction"] == 0.4


def test_norm_shift_selector_is_deterministic_with_hash_only_as_tie() -> None:
    states = tuple(exact_state(record_id) for record_id in range(20))
    shifts = {record_id: 0.5 for record_id in range(20)}
    first = MODULE.select_norm_shift_decisions(states, 1, shifts)
    second = MODULE.select_norm_shift_decisions(
        tuple(reversed(states)),
        1,
        shifts,
    )

    assert action_map(first) == action_map(second)
    assert sum(value.action == "exact" for value in first) == 4
    assert MODULE.constant_policy().edge_severities == (0.0,) * 11
    audit = MODULE.selector_audit(states, first, shifts)
    assert audit["norm_shift_unique_count"] == 1
    assert audit["norm_shift_quota_exact_records"] == 4
    assert audit["cutoff_tie_records"] == 20
    assert audit["cutoff_tie_selected_records"] == 4
    assert audit["sha256_boundary_tie_used"]


def test_transition_descriptor_covers_continue_zero_cold_and_expire() -> None:
    old = record(
        history([1, 2, 3], [10, 20, 30], [1, 2, 3]),
        "old",
    )
    continued = record(
        history([2, 3, 4], [20, 30, 40], [2, 3, 4]),
        "continued",
    )
    disjoint = record(
        history([4, 5], [40, 50], [4, 1]),
        "disjoint",
    )
    absent = record(None, None)

    overlap = MODULE.transition_descriptor(old, continued, True)
    zero = MODULE.transition_descriptor(old, disjoint, True)
    cold = MODULE.transition_descriptor(old, continued, False)
    expired = MODULE.transition_descriptor(old, absent, True)

    assert overlap.status == "continued"
    assert overlap.overlap == 2
    assert overlap.evicted == 1
    assert overlap.appended == 0
    assert overlap.new_length == 2
    assert overlap.previous_actual_consumed
    assert zero.status == "zero_overlap"
    assert zero.overlap == 0
    assert not zero.previous_actual_consumed
    assert cold.status == "cold"
    assert cold.appended == 2
    assert not cold.previous_actual_consumed
    assert expired.status == "expired"
    assert expired.new_history_hash is None


def test_overlap_is_label_free_and_handles_duplicate_event_keys() -> None:
    old = record(
        history([1], [10], [1]),
        "old",
        ("same-event-label0",),
    )
    changed_label = record(
        history([1, 2], [10, 20], [1, 1]),
        "new",
        ("same-event-label1", "latest"),
    )
    transition = MODULE.transition_descriptor(
        old,
        changed_label,
        True,
    )

    assert transition.status == "continued"
    assert transition.overlap == 1
    changed_behavior = record(
        history([1, 2], [10, 20], [2, 1]),
        "changed",
    )
    changed_transition = MODULE.transition_descriptor(
        old,
        changed_behavior,
        True,
    )
    assert changed_transition.status == "zero_overlap"
    assert changed_transition.overlap == 0
    assert (
        MODULE._suffix_prefix_identity_overlap(
            ("a", "b", "a", "b"),
            ("a", "b", "c"),
        )
        == 2
    )


def test_foreground_append_excludes_target_latest_token() -> None:
    old = record(
        history([1, 2], [10, 20], [1, 1]),
        "old",
    )
    new = record(
        history([1, 2, 3, 4], [10, 20, 30, 40], [1, 1, 1, 1]),
        "new",
    )
    transition = MODULE.transition_descriptor(old, new, True)
    appended = MODULE._appended_batch(
        [new],
        [transition],
        torch.device("cpu"),
    )

    assert transition.new_length == 3
    assert transition.appended == 1
    assert appended["lengths"].tolist() == [1]
    assert appended["item_ids"].tolist() == [[30]]


def test_source_prefix_preparation_bypasses_cold_zero_and_short(
    monkeypatch,
) -> None:
    model = HSTU(
        HSTUConfig(
            num_items=50,
            num_behaviors=4,
            hidden_size=8,
            num_layers=2,
            num_heads=1,
            head_dim=8,
            max_seq_len=8,
            input_dropout=0.0,
        )
    ).eval()
    old_records = {
        1: record(history([1, 2], [1, 2], [1, 1]), "old-1"),
        2: record(None, None),
        3: record(history([1], [6], [1]), "old-3"),
        4: record(history([1], [9], [1]), "old-4"),
    }
    target_records = {
        1: record(history([1, 2, 3], [1, 2, 3], [1, 1, 1]), "new-1"),
        2: record(history([1, 2], [4, 5], [1, 1]), "new-2"),
        3: record(history([2, 3], [7, 8], [1, 1]), "new-3"),
        4: record(history([1], [9], [1]), "new-4"),
    }
    group = tuple(
        {"record_id": index, "user_id": index + 1}
        for index in range(4)
    )
    previous = {}
    for record_id in (0, 2, 3):
        source = old_records[record_id + 1].history
        items = torch.as_tensor(source.item_ids).unsqueeze(0)
        behaviors = torch.as_tensor(source.behaviors).unsqueeze(0)
        deltas = torch.as_tensor(source.time_deltas).unsqueeze(0)
        lengths = torch.tensor([len(source)])
        cache = model.compute_kv(
            items,
            behaviors,
            deltas,
            lengths=lengths,
        )
        previous[record_id] = MODULE.pack_padded_cache(
            cache,
            lengths,
            (record_id,),
            "theta0",
            "theta0",
        )
    states = {
        record_id: exact_state(record_id)
        for record_id in previous
    }
    monkeypatch.setattr(
        MODULE,
        "timed_cuda",
        lambda action, device: (action(), 0.0),
    )
    costs = {
        "foreground_evict": 0.0,
        "foreground_incremental_append": 0.0,
        "candidate_transform": 0.0,
        "router_probe": 0.0,
        "exact_refresh": 0.0,
        "natural_direct_exact": 0.0,
        "publication": 0.0,
        "common_latest": 0.0,
        "common_publication": 0.0,
    }
    source, source_states, direct, transitions, peak = (
        MODULE._prepare_source_prefix(
            model,
            SimpleNamespace(records=old_records),
            SimpleNamespace(records=target_records),
            (group,),
            previous,
            states,
            torch.device("cpu"),
            costs,
        )
    )

    assert previous == {}
    assert set(source) == {0}
    assert source[0].lengths.tolist() == [2]
    assert set(source_states) == {0}
    assert direct == {1, 2, 3}
    assert transitions[1].status == "cold"
    assert transitions[2].status == "zero_overlap"
    assert transitions[3].status == "short_no_prefix"
    assert peak > 0


def test_publication_partition_keeps_short_after_source_consumption() -> None:
    group = (
        {"record_id": 0, "user_id": 1},
        {"record_id": 1, "user_id": 2},
        {"record_id": 2, "user_id": 3},
    )
    record_by_id = {value["record_id"]: value for value in group}
    target_window = SimpleNamespace(
        records={
            1: SimpleNamespace(history=history([1, 2], [1, 2], [1, 1])),
            2: SimpleNamespace(history=history([1], [1], [1])),
            3: SimpleNamespace(history=None),
        }
    )
    descriptors, eligible, direct_prefix, short = (
        MODULE._publication_partition(
        group,
        target_window,
        record_by_id,
        {0: exact_state(0)},
        {1},
        {0: object()},
        )
    )

    assert len(descriptors) == 3
    assert [value["record_id"] for value in eligible] == [0]
    assert direct_prefix == []
    assert [value["record_id"] for value in short] == [1]


def test_exact_reference_partition_combines_reuse_and_natural_rows() -> None:
    descriptors = [
        {"record_id": 7, "user_id": 10},
        {"record_id": 2, "user_id": 20},
        {"record_id": 9, "user_id": 30},
        {"record_id": 4, "user_id": 40},
    ]
    target_window = SimpleNamespace(
        records={
            10: record(history([1, 2, 3], [1, 2, 3], [1, 1, 1]), "a"),
            20: record(history([1], [1], [1]), "b"),
            30: record(None, None),
            40: record(history([1, 2], [4, 5], [1, 1]), "c"),
        }
    )

    selected = MODULE._exact_reference_descriptors(
        descriptors,
        target_window,
    )

    assert [value["record_id"] for value in selected] == [7, 4]


def test_theta0_anchor_supports_nonidentical_record_and_user_ids(
    monkeypatch,
) -> None:
    cfg = HSTUConfig(
        num_items=50,
        num_behaviors=4,
        hidden_size=8,
        num_layers=2,
        num_heads=1,
        head_dim=8,
        max_seq_len=8,
        input_dropout=0.0,
    )
    model = HSTU(cfg).eval()
    descriptor = {
        "record_id": 7,
        "user_id": 10,
        "evaluation_role": "fit",
    }
    window = SimpleNamespace(
        target_date="d0",
        records={
            10: record(
                history([1, 2], [3, 4], [1, 1]),
                "history",
            )
        },
    )
    monkeypatch.setattr(
        MODULE,
        "load_checkpoint_model",
        lambda cfg, checkpoint_dir, version, device: model,
    )
    monkeypatch.setattr(
        MODULE,
        "timed_cuda",
        lambda action, device: (action(), 0.0),
    )

    caches, states, endpoint_records, endpoint = MODULE._initialize_theta0(
        cfg,
        "unused",
        window,
        ((descriptor,),),
        {10: descriptor},
        torch.device("cpu"),
        torch.arange(1, 11),
    )

    assert set(caches) == {7}
    assert set(states) == {7}
    assert endpoint_records[0]["record_id"] == 7
    assert endpoint_records[0]["user_id"] == 10
    assert endpoint["anchor"]["layout_matches_history"]


def test_cost_summary_has_primary_and_update_only_boundaries() -> None:
    summary = MODULE.cost_summary(
        {
            "foreground_evict": 1.0,
            "foreground_incremental_append": 2.0,
            "candidate_transform": 4.0,
            "router_probe": 0.5,
            "exact_refresh": 5.0,
            "natural_direct_exact": 2.0,
            "publication": 1.5,
            "common_latest": 7.0,
            "common_publication": 3.0,
        },
        20.0,
    )

    assert summary["foreground_ms"] == 3.0
    assert summary["update_only_ms"] == 11.0
    assert summary["primary_update_only_ratio"] == pytest.approx(0.55)
    assert summary["symmetric_lifecycle_ratio"] == pytest.approx(
        14.0 / 23.0
    )
    assert summary["common_inclusive_ratio"] == pytest.approx(
        24.0 / 33.0
    )
    assert summary["conservative_asymmetric_ratio"] == pytest.approx(
        0.7
    )
    assert summary["update_only_ratio"] == pytest.approx(0.55)
    assert summary["migration_ms"] == 4.0
    assert summary["primary_and_symmetric_exclude_common_latest"]
    assert summary["primary_and_symmetric_exclude_common_publication"]
    with pytest.raises(ValueError, match="ledger"):
        MODULE.cost_summary({}, 20.0)


def test_task_summary_keeps_full_exact_pairing_and_subset_reuse() -> None:
    def values(base: float) -> dict[str, float]:
        return {
            "mean_rank": base,
            "catalog_auc": base,
            "ndcg@100": base,
            "hit@100": base,
        }

    summary = MODULE.summarize_organic_tasks(
        [
            {
                "evaluation_role": "final_test",
                "task": {
                    "mixed": values(2.0),
                    "all_exact": values(3.0),
                    "reuse": values(1.0),
                },
            },
            {
                "evaluation_role": "final_test",
                "task": {
                    "mixed": values(4.0),
                    "all_exact": values(5.0),
                },
            },
        ]
    )

    assert summary["records"] == 2
    assert summary["reuse_coverage_records"] == 1
    assert summary["reuse_coverage_fraction"] == 0.5
    assert summary["mixed"]["catalog_auc"] == 3.0
    assert summary["all_exact"]["catalog_auc"] == 4.0
    assert (
        summary["reuse_continued_subset"]["reuse"]["catalog_auc"]
        == 1.0
    )


def test_posthoc_selector_diagnostic_reuses_frozen_actions() -> None:
    states = tuple(exact_state(record_id) for record_id in range(10))
    shifts = {record_id: float(record_id) for record_id in range(10)}
    decisions = MODULE.select_norm_shift_decisions(states, 1, shifts)
    errors = {record_id: float(record_id) for record_id in range(10)}
    before = action_map(decisions)
    diagnostic = MODULE.posthoc_selector_diagnostics(
        shifts,
        errors,
        decisions,
    )

    assert diagnostic["posthoc_only"]
    assert not diagnostic["actions_changed"]
    assert diagnostic[
        "spearman_norm_shift_vs_candidate_error"
    ] == pytest.approx(1.0)
    assert diagnostic["selected_exact_candidate_error"]["mean"] == 8.5
    assert diagnostic["migrated_candidate_error"]["mean"] == 3.5
    assert diagnostic["selected_vs_oracle_overlap_fraction"] == 1.0
    assert action_map(decisions) == before


def fake_manifest() -> dict:
    return {
        "content_sha256": "manifest",
        "records": [
            {
                "record_id": 0,
                "user_id": 1,
                "evaluation_role": "fit",
            },
            {
                "record_id": 1,
                "user_id": 2,
                "evaluation_role": "final_test",
            },
        ],
        "timeline": {
            "target_dates": [f"d{version}" for version in range(12)]
        },
    }


def fake_windows():
    output = []
    known_by_user = {
        user_id: [f"base-{user_id}-0", f"base-{user_id}-1"]
        for user_id in (1, 2)
    }
    for version in range(12):
        records = {}
        for user_id in (1, 2):
            new_identities = (
                f"target-{version}-{user_id}-0",
                f"target-{version}-{user_id}-1",
            )
            records[user_id] = SimpleNamespace(
                user_id=user_id,
                as_of_timestamp_ms=100,
                history=history([10, 20], [1, 2], [1, 1]),
                history_sha256=f"history-{version}-{user_id}",
                history_event_identities=tuple(known_by_user[user_id][-2:]),
                new_event_identities=new_identities,
                new_events=FakeHistory(
                    timestamps=np.asarray([100, 110], dtype=np.int64),
                    item_ids=np.asarray([3, 4], dtype=np.int64),
                    behaviors=np.asarray([1, 1], dtype=np.int64),
                    time_deltas=np.zeros(2, dtype=np.float32),
                ),
            )
            known_by_user[user_id].extend(new_identities)
        output.append(
            SimpleNamespace(
                version=version,
                target_date=f"d{version}",
                records=records,
                content_sha256=f"window{version}",
            )
        )
    return tuple(output)


def fake_compiler(windows) -> dict:
    checkpoints = [
        {
            "version": f"theta{version}",
            "path": f"theta_{version}.pt",
            "sha256": f"sha-{version}",
            "bytes": version + 1,
        }
        for version in range(12)
    ]
    return {
        "protocol": MODULE.COMPILER_PROTOCOL,
        "experiment_protocol": MODULE.EXPERIMENT_PROTOCOL,
        "status": "complete",
        "inputs": {"checkpoints": checkpoints},
        "manifest": {"content_sha256": "manifest"},
        "windows": [
            {
                "version": window.version,
                "target_date": window.target_date,
                "history_view_sha256": MODULE.history_view_sha256(
                    window,
                    (1, 2),
                ),
            }
            for window in windows
        ],
        "pairs": [
            {
                "source_version": f"theta{source}",
                "target_version": f"theta{source + 1}",
                "history_target_date": windows[source + 1].target_date,
                "history_view_sha256": MODULE.history_view_sha256(
                    windows[source + 1],
                    (1,),
                ),
                "direct_program": {},
                "load_validation": {
                    "passed": True,
                    "provenance": {
                        "labels_used": False,
                        "future_history_used": False,
                        "history_version": f"theta{source + 1}",
                        "history_view_sha256": (
                            MODULE.history_view_sha256(
                                windows[source + 1],
                                (1,),
                            )
                        ),
                        "source_checkpoint_sha256": f"sha-{source}",
                        "target_checkpoint_sha256": f"sha-{source + 1}",
                    }
                },
            }
            for source in range(11)
        ],
    }


def test_window_and_compiler_validation_are_causal() -> None:
    manifest = fake_manifest()
    windows = fake_windows()
    window_checks = MODULE.validate_windows(windows, manifest)
    compiler_checks = MODULE.validate_compiler_payload(
        fake_compiler(windows),
        manifest,
        windows,
        fake_compiler(windows)["inputs"]["checkpoints"],
    )

    assert all(window_checks.values())
    assert all(compiler_checks.values())
    history_hash = MODULE.history_view_sha256(windows[0], (1, 2))
    windows[0].records[1].new_events = FakeHistory(
        timestamps=np.asarray([999], dtype=np.int64),
        item_ids=np.asarray([999], dtype=np.int64),
        behaviors=np.asarray([9], dtype=np.int64),
        time_deltas=np.zeros(1, dtype=np.float32),
    )
    windows[0].records[1].engaged_positive_item_ids = (999,)
    assert MODULE.history_view_sha256(windows[0], (1, 2)) == history_hash
    windows[0].records[1].history_sha256 = "changed-history"
    assert MODULE.history_view_sha256(windows[0], (1, 2)) != history_hash
    windows = fake_windows()
    broken = fake_compiler(windows)
    broken["pairs"][0]["load_validation"]["provenance"][
        "future_history_used"
    ] = True
    with pytest.raises(ValueError, match="compiler"):
        MODULE.validate_compiler_payload(
            broken,
            manifest,
            windows,
        )
    broken_checkpoint = fake_compiler(windows)
    checkpoints = broken_checkpoint["inputs"]["checkpoints"]
    broken_checkpoint["pairs"][0]["load_validation"]["provenance"][
        "source_checkpoint_sha256"
    ] = "replaced"
    with pytest.raises(ValueError, match="compiler"):
        MODULE.validate_compiler_payload(
            broken_checkpoint,
            manifest,
            windows,
            checkpoints,
        )


def test_future_date_partition_identity_in_history_fails_causality() -> None:
    windows = fake_windows()
    contaminated = windows[4].records[1]
    identities = list(contaminated.history_event_identities)
    identities[-1] = contaminated.new_event_identities[0]
    contaminated.history_event_identities = tuple(identities)

    with pytest.raises(ValueError, match="causality"):
        MODULE.validate_windows(windows, fake_manifest())


def test_raw_timestamp_overlap_is_diagnostic_not_gate() -> None:
    windows = fake_windows()
    windows[3].records[1].history.timestamps[-1] = 150

    checks = MODULE.validate_windows(windows, fake_manifest())
    diagnostic = MODULE.raw_timestamp_overlap_diagnostic(windows)

    assert all(checks.values())
    assert "history_strictly_precedes_target" not in checks
    assert checks["history_is_suffix_of_prior_date_partitions"]
    assert not diagnostic["validity_gate"]
    assert diagnostic["overlap_record_windows"] == 1
    assert diagnostic["overlap_history_tokens"] == 1
    assert (
        diagnostic["active_record_windows"]
        + diagnostic["inactive_record_windows"]
        == diagnostic["record_windows"]
    )
    assert (
        diagnostic["active_resident_record_windows"]
        + diagnostic["inactive_resident_record_windows"]
        == diagnostic["resident_record_windows"]
    )
    assert diagnostic["active_overlap_record_windows"] == 1
    assert diagnostic["inactive_overlap_record_windows"] == 0
    assert diagnostic["maximum_lead_ms"] == 50
    assert diagnostic["maximum_lead_seconds"] == pytest.approx(0.05)
    assert diagnostic["by_version"][3]["overlap_record_windows"] == 1


def test_fixed_groups_identity_slice_and_smoke() -> None:
    assert hasattr(MODULE._initialize_theta0, "__wrapped__")
    assert hasattr(MODULE._run_edge, "__wrapped__")
    manifest = {
        "records": [
            {"record_id": index, "user_id": index + 1}
            for index in range(6)
        ]
    }
    groups = MODULE.fixed_record_groups(manifest, 4)
    assert [len(group) for group in groups] == [4, 2]

    k = torch.arange(12, dtype=torch.float16).reshape(1, 6, 2)
    cache = JaggedMigratedKVBatch(
        record_ids=(0, 1),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=k,
        v=(k + 1).contiguous(),
        lengths=torch.tensor([2, 4]),
        offsets=torch.tensor([0, 2, 6]),
    )
    sliced = MODULE.identity_jagged_slice(cache)
    assert sliced.cache is cache
    assert sliced.lengths == (2, 4)
    assert sliced.empty_rows == ()
    cache_by_record = MODULE._split_cache(cache)
    assert MODULE.resident_cache_bytes(cache_by_record) == 48
    consumed = MODULE._consume_record_caches((0,), cache_by_record)
    assert consumed.record_ids == (0,)
    assert set(cache_by_record) == {1}

    smoke = MODULE.smoke_payload()
    assert smoke["status"] == "smoke_passed"
    assert smoke["exact_actions"] == 4
    assert smoke["migrate_actions"] == 16
    assert (
        smoke["memory_preflight"][
            "maximum_simultaneous_full_cohort_dicts"
        ]
        == 1
    )


def test_global_depth_check_is_computed_from_lineage() -> None:
    manifest = {"records": [{"record_id": 0, "user_id": 10}]}
    windows = tuple(
        SimpleNamespace(
            records={
                10: SimpleNamespace(history_sha256=f"history-{version}")
            }
        )
        for version in range(12)
    )
    endpoints = [
        {"anchor": {"layout_matches_history": True}},
        *({} for _ in range(11)),
    ]
    steps = [
        {
            "source_version": version,
            "target_version": version + 1,
            "selector": {
                "configured_exact_fraction": MODULE.EXACT_FRACTION
            },
            "checks": {"passed": True},
            "lineage": [
                {
                    "user_id": 10,
                    "new_history_hash": f"history-{version + 1}",
                    "foreground_status": "continued",
                    "overlap_tokens": 1,
                    "previous_actual_consumed": True,
                    "candidate_evaluated": True,
                    "decision": {"reason": "norm_shift_migrate"},
                    "state_after": {"migration_depth": version % 5},
                }
            ],
        }
        for version in range(11)
    ]

    passing = MODULE._global_checks(
        windows,
        endpoints,
        steps,
        manifest,
    )
    assert passing["maximum_depth_four"]
    steps[-1]["lineage"][0]["state_after"]["migration_depth"] = 5
    failing = MODULE._global_checks(
        windows,
        endpoints,
        steps,
        manifest,
    )
    assert not failing["maximum_depth_four"]
