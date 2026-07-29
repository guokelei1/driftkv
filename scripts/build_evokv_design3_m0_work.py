from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from hstu_kvcache.migration import (
    D2ActionPlan,
    audit_d3_group_plan,
    audit_d3_work_manifest,
    build_d2_record_owner_map,
    build_d3_byte_bounded_groups,
    build_d3_work_manifest,
    canonical_json_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_PROGRAM = (
    "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
    "single_config_v1/stage4_7_organic_runtime/"
    "theta1_to_theta2_direct_oldkv_fp16.pt"
)
DEFAULT_OUTPUT_DIR = "configs/evokv_d3/development"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stack-revision", default="h12_w2_m0_r0")
    parser.add_argument("--group-budget-gib", type=float, default=1.0)
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.group_budget_gib <= 0:
        raise ValueError("M0 group budget must be positive")
    action_path = _path(args.action_plan)
    program_path = _path(args.program)
    output_dir = _path(args.output_dir)
    action_plan = D2ActionPlan.load(action_path)
    owner_map = build_d2_record_owner_map(
        action_plan,
        2,
        "strict_cow_lpt",
    )
    manifest = build_d3_work_manifest(
        action_plan,
        owner_map,
        world_size=2,
        stack_revision=args.stack_revision,
        action_plan_ref=str(action_path),
        program_ref=str(program_path),
        extra={
            "capacity_emulation": True,
            "physical_oversubscription": False,
            "logical_gpu_pair": [0, 1],
        },
    )
    audit_d3_work_manifest(manifest, action_plan, owner_map)
    group_budget_bytes = int(args.group_budget_gib * 1024**3)
    group_plan = build_d3_byte_bounded_groups(
        manifest,
        (group_budget_bytes, group_budget_bytes),
        extra={
            "capacity_emulation": True,
            "budget_kind": "logical_payload_estimate",
            "logical_gpu_pair": [0, 1],
        },
    )
    audit_d3_group_plan(manifest, group_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "h12_w2_m0_work_manifest.json"
    groups_path = output_dir / "h12_w2_m0_group_plan.json"
    summary_path = output_dir / "h12_w2_m0_foundation.json"
    manifest.write(manifest_path)
    group_plan.write(groups_path)
    summary = {
        "status": "complete",
        "artifact_role": "m0_input_build",
        "scientific_result": False,
        "formal_design3": False,
        "capacity_emulation": True,
        "physical_oversubscription": False,
        "stack_revision": args.stack_revision,
        "logical_gpu_pair": [0, 1],
        "world_size": 2,
        "action_plan": str(action_path),
        "work_manifest": str(manifest_path),
        "work_manifest_sha256": manifest.dev_sha256,
        "group_plan": str(groups_path),
        "group_plan_sha256": group_plan.dev_sha256,
        "group_budget_bytes_by_rank": list(
            group_plan.group_budget_bytes_by_rank
        ),
        "group_budget_kind": "logical_payload_estimate",
        "records": len(manifest.records),
        "records_by_rank": [
            sum(record.owner_rank == rank for record in manifest.records)
            for rank in range(2)
        ],
        "pools": dict(Counter(record.pool for record in manifest.records)),
        "pools_by_rank": [
            dict(
                Counter(
                    record.pool
                    for record in manifest.records
                    if record.owner_rank == rank
                )
            )
            for rank in range(2)
        ],
        "allocated_old_kv_bytes": sum(
            record.bytes.old_kv_allocated
            for record in manifest.records
        ),
        "old_kv_read_bytes": sum(
            record.bytes.old_kv_read for record in manifest.records
        ),
        "history_read_bytes": sum(
            record.bytes.history_read for record in manifest.records
        ),
        "target_write_bytes": sum(
            record.bytes.target_write for record in manifest.records
        ),
        "groups": len(group_plan.groups),
        "groups_by_pool": dict(
            Counter(group.pool for group in group_plan.groups)
        ),
        "maximum_estimated_group_bytes_by_rank": [
            max(
                group.estimated_bytes_by_rank[rank]
                for group in group_plan.groups
            )
            for rank in range(2)
        ],
        "next_at_creation": (
            "implement GPU0/GPU1 pageable-DRAM sequential executor"
        ),
    }
    summary_path.write_bytes(canonical_json_bytes(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    summary = run(parse_args(argv))
    print(
        f"records={summary['records']} groups={summary['groups']} "
        f"manifest={summary['work_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
