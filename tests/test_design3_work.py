from dataclasses import replace
from pathlib import Path

import pytest

from hstu_kvcache.migration import (
    D2ActionPlan,
    D3GroupPlan,
    D3WorkManifest,
    audit_d3_group_plan,
    audit_d3_work_manifest,
    build_d2_record_owner_map,
    build_d3_byte_bounded_groups,
    build_d3_work_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ACTION_PLAN = (
    ROOT
    / "configs/cohortkv_d2"
    / "action_plan_theta1_theta2_staggered_renewal_h12.json"
)


def _manifest() -> tuple[
    D2ActionPlan,
    dict[int, int],
    D3WorkManifest,
]:
    plan = D2ActionPlan.load(ACTION_PLAN)
    owner_map = build_d2_record_owner_map(
        plan,
        2,
        "strict_cow_lpt",
    )
    manifest = build_d3_work_manifest(
        plan,
        owner_map,
        world_size=2,
        stack_revision="h12_w2_m0_r0",
        action_plan_ref=str(ACTION_PLAN),
        program_ref="theta1_to_theta2_direct_oldkv",
        extra={"capacity_emulation": True},
    )
    return plan, owner_map, manifest


def test_real_h12_w2_manifest_and_roundtrip(tmp_path: Path) -> None:
    plan, owner_map, manifest = _manifest()
    audit_d3_work_manifest(manifest, plan, owner_map)
    assert len(manifest.records) == 682
    assert [
        sum(record.owner_rank == rank for record in manifest.records)
        for rank in range(2)
    ] == [341, 341]
    assert {
        pool: sum(record.pool == pool for record in manifest.records)
        for pool in ("compiled", "exact")
    } == {"compiled": 548, "exact": 134}
    assert sum(
        record.bytes.old_kv_allocated
        for record in manifest.records
    ) == 28_383_969_280
    assert sum(
        record.bytes.target_write for record in manifest.records
    ) == 30_635_360_256
    assert sum(
        record.bytes.old_kv_read for record in manifest.records
    ) == 19_262_832_640
    assert all(
        record.old_kv_ref is not None
        for record in manifest.records
        if record.bytes.old_kv_allocated
    )
    assert all(
        record.bytes.old_kv_read == 0
        for record in manifest.records
        if record.pool == "exact"
    )
    output = tmp_path / "manifest.json"
    manifest.write(output)
    assert D3WorkManifest.load(output) == manifest


def test_byte_bounded_groups_are_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    _, _, manifest = _manifest()
    budget = (1024**3, 1024**3)
    first = build_d3_byte_bounded_groups(manifest, budget)
    second = build_d3_byte_bounded_groups(manifest, budget)
    audit_d3_group_plan(manifest, first)
    assert first == second
    assert first.dev_sha256 == second.dev_sha256
    observed = [
        record_id
        for group in first.groups
        for record_ids in group.record_ids_by_rank
        for record_id in record_ids
    ]
    assert len(observed) == 682
    assert set(observed) == {
        record.record_id for record in manifest.records
    }
    assert all(
        bytes_used <= budget[rank]
        for group in first.groups
        for rank, bytes_used in enumerate(group.estimated_bytes_by_rank)
    )
    assert {group.pool for group in first.groups} == {
        "compiled",
        "exact",
    }
    output = tmp_path / "groups.json"
    first.write(output)
    assert D3GroupPlan.load(output) == first


def test_grouping_supports_empty_rank_and_custom_estimator() -> None:
    _, _, manifest = _manifest()
    selected = tuple(
        record
        for record in manifest.records
        if record.owner_rank == 0
    )[:8]
    rank_zero_only = replace(manifest, records=selected)
    default = build_d3_byte_bounded_groups(
        rank_zero_only,
        (1024**3, 1024**3),
    )
    assert all(
        not group.record_ids_by_rank[1] for group in default.groups
    )

    def one_record_per_group(
        rank: int,
        records,
    ) -> int:
        del rank
        return len(records) * 10

    custom = build_d3_byte_bounded_groups(
        rank_zero_only,
        (10, 10),
        estimator=one_record_per_group,
    )
    assert len(custom.groups) == len(selected)


def test_grouping_rejects_oversize_record_and_audit_is_separate() -> None:
    plan, owner_map, manifest = _manifest()
    required = manifest.records[0].bytes.grouping_estimate
    with pytest.raises(
        ValueError,
        match="record=.*rank=.*required=.*cap=",
    ):
        build_d3_byte_bounded_groups(
            manifest,
            (required - 1, required - 1),
        )
    changed = replace(
        manifest,
        records=(
            replace(
                manifest.records[0],
                retained_start=manifest.records[0].retained_start + 1,
            ),
            *manifest.records[1:],
        ),
    )
    with pytest.raises(
        ValueError,
        match="work record differs",
    ):
        audit_d3_work_manifest(changed, plan, owner_map)


def test_group_plan_cross_audit_rejects_wrong_binding() -> None:
    _, _, manifest = _manifest()
    groups = build_d3_byte_bounded_groups(
        manifest,
        (1024**3, 1024**3),
    )
    with pytest.raises(
        ValueError,
        match="differs from its work manifest",
    ):
        audit_d3_group_plan(
            manifest,
            replace(groups, work_manifest_sha256="0" * 64),
        )
