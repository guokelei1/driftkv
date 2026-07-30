from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from hstu_kvcache.migration.design2_plan import (
    canonical_json_bytes,
    file_sha256,
)
from hstu_kvcache.migration.design3_residency import (
    D3ResidencyGroup,
    D3RouteGranularity,
    D3RouteStageProfile,
    d3_stack_revision_sha256,
    select_residency_plan,
)

CALIBRATION_RESULT_PROTOCOL = (
    "evokv_design3_m1_qk_decoupled_io_d3_development_v2"
)
RUNNER_PROTOCOL = (
    "evokv_design3_m1_qk_decoupled_residency_runner_development_v2"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--hbm-total-bytes", type=int, action="append")
    parser.add_argument(
        "--hbm-margin-bytes",
        type=int,
        default=2 * 1024**3,
    )
    parser.add_argument(
        "--pinned-limit-bytes",
        type=int,
        default=8 * 1024**3,
    )
    parser.add_argument("--selector-tie-fraction", type=float, default=0.03)
    parser.add_argument("--steady-groups", type=int, default=3)
    parser.add_argument("--max-search-states", type=int, default=4096)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def stack_revision_sha256(report: Mapping[str, object]) -> str:
    rank_reports = report["rank_reports"]
    development_identity = report.get("development_identity")
    if not isinstance(development_identity, Mapping):
        raise ValueError("D3 calibration execution identity is missing")
    devices = []
    for value in rank_reports:
        device_identity = value.get("device_identity")
        if not isinstance(device_identity, Mapping):
            raise ValueError("D3 calibration device identity is missing")
        devices.append(
            {
                **dict(device_identity),
                "source_checkpoint_sha256": value[
                    "source_checkpoint"
                ]["sha256"],
                "target_checkpoint_sha256": value[
                    "target_checkpoint"
                ]["sha256"],
            }
        )
    return d3_stack_revision_sha256(
        bindings=report["bindings"],
        world_size=int(report["execution"]["world_size"]),
        group_records_per_rank=int(
            report["group_records_per_rank"]
        ),
        records=int(report["records"]),
        counts=report["counts"],
        embedding_scale_role=str(report["embedding_scale_role"]),
        device_names=tuple(
            value["device_name"] for value in rank_reports
        ),
        source_checkpoint_sha256=tuple(
            value["source_checkpoint"]["sha256"]
            for value in rank_reports
        ),
        target_checkpoint_sha256=tuple(
            value["target_checkpoint"]["sha256"]
            for value in rank_reports
        ),
        capacity_groups=report["capacity"]["capacity_groups"],
        execution_identity={
            **dict(development_identity),
            "devices": devices,
        },
    )


def _granularity(
    report: Mapping[str, object],
    route: str,
) -> D3RouteGranularity:
    value = report.get("execution_granularity")
    if isinstance(value, Mapping):
        route_value = value.get(route)
        if isinstance(route_value, Mapping):
            return D3RouteGranularity.from_dict(route_value)
    legacy = int(report["micro_batch_records"])
    return D3RouteGranularity(legacy, legacy, legacy)


def _compute_seconds(group: Mapping[str, object]) -> float:
    value = group.get("execution_wall_seconds")
    if value is not None:
        return float(value)
    return sum(
        float(group[name])
        for name in (
            "d2d_transform_seconds",
            "d2d_assemble_seconds",
            "lookup_exchange_seconds",
            "compute_excluding_lookup_seconds",
        )
    )


def _robust_service(values: Sequence[float]) -> float:
    if not values or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise ValueError("D3 calibration service samples are invalid")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median + max(2.0 * mad, 0.05 * median)


def profile_result(
    path: Path,
    report: Mapping[str, object],
    route: str,
    steady_groups: int,
) -> D3RouteStageProfile:
    group_records = int(report["group_records_per_rank"])
    capacity_groups = report["capacity"]["capacity_groups"]["groups"]
    full_ordinals = [
        int(value["ordinal"])
        for value in capacity_groups
        if value["route"] == route
        and max(int(item) for item in value["records_by_rank"])
        == group_records
    ]
    if len(full_ordinals) >= 3:
        full_ordinals = full_ordinals[1 : 1 + steady_groups]
    else:
        full_ordinals = full_ordinals[:steady_groups]
    if not full_ordinals:
        raise ValueError(f"no full {route} calibration group in {path}")
    rank_reports = report["rank_reports"]
    by_rank = []
    for rank_report in rank_reports:
        groups = rank_report["d3"]["group_reports"]
        by_rank.append(
            {int(value["ordinal"]): value for value in groups}
        )
    input_samples = []
    compute_samples = []
    output_samples = []
    for ordinal in full_ordinals:
        reports = [value[ordinal] for value in by_rank]
        input_samples.append(
            max(
                float(value["input_staging_wall_seconds"])
                for value in reports
            )
        )
        compute_samples.append(
            max(_compute_seconds(value) for value in reports)
        )
        output_samples.append(
            max(float(value["drain_wall_seconds"]) for value in reports)
        )
    world_size = len(rank_reports)
    input_seconds = _robust_service(input_samples)
    compute_seconds = _robust_service(compute_samples)
    output_seconds = _robust_service(output_samples)
    return D3RouteStageProfile(
        route=route,
        granularity=_granularity(report, route),
        reference_records=group_records,
        input_seconds_by_rank=(input_seconds,) * world_size,
        compute_seconds_by_rank=(compute_seconds,) * world_size,
        output_seconds_by_rank=(output_seconds,) * world_size,
        peak_hbm_reserved_bytes_by_rank=tuple(
            int(value["d3"]["peak_hbm_reserved_bytes"])
            for value in rank_reports
        ),
        pinned_bytes_by_rank=tuple(
            int(value["d3"]["pinned_slot_bytes"])
            for value in rank_reports
        ),
        sample_groups=len(full_ordinals),
        source_sha256=file_sha256(path),
    )


def _groups(report: Mapping[str, object]) -> tuple[D3ResidencyGroup, ...]:
    return tuple(
        D3ResidencyGroup.from_dict(value)
        for value in report["capacity"]["capacity_groups"]["groups"]
    )


def _hbm_totals(
    requested: Sequence[int] | None,
    world_size: int,
) -> tuple[int, ...]:
    if requested:
        values = tuple(int(value) for value in requested)
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise ValueError(
                "HBM totals are required when the planned GPUs are unavailable"
            )
        values = tuple(
            torch.cuda.get_device_properties(rank).total_memory
            for rank in range(world_size)
        )
    if len(values) == 1:
        values = values * world_size
    if len(values) != world_size:
        raise ValueError("HBM total count differs from world size")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if (
        args.hbm_margin_bytes < 0
        or args.steady_groups < 1
        or args.pinned_limit_bytes < 1
        or args.max_search_states < 1
        or not 0 <= args.selector_tie_fraction < 1
    ):
        raise ValueError("D3 planner arguments are invalid")
    paths = tuple(Path(value).resolve() for value in args.result)
    reports = tuple(_load_json(path) for path in paths)
    for path, report in zip(paths, reports, strict=True):
        development_identity = report.get("development_identity")
        if (
            report.get("status") != "complete"
            or report.get("mode") != "d3"
            or report.get("scope") != "full"
            or not report.get("exactly_once_pass")
            or report.get("protocol") != CALIBRATION_RESULT_PROTOCOL
            or report.get("residency_plan") is not None
            or report.get("scientific_result") is not False
            or report.get("formal_design3") is not False
            or not isinstance(development_identity, Mapping)
            or development_identity.get("runner_protocol")
            != RUNNER_PROTOCOL
        ):
            raise ValueError(f"D3 calibration result is invalid: {path}")
    stack_hashes = tuple(
        stack_revision_sha256(report) for report in reports
    )
    if len(set(stack_hashes)) != 1:
        raise ValueError("D3 calibration stack revisions differ")
    group_hashes = tuple(
        str(report["bindings"]["group_plan_sha256"])
        for report in reports
    )
    if len(set(group_hashes)) != 1:
        raise ValueError("D3 calibration group plans differ")
    groups = _groups(reports[0])
    if any(_groups(report) != groups for report in reports[1:]):
        raise ValueError("D3 calibration capacity groups differ")
    profiles = {
        route: tuple(
            profile_result(
                path,
                report,
                route,
                args.steady_groups,
            )
            for path, report in zip(paths, reports, strict=True)
        )
        for route in ("compiled", "exact")
    }
    world_size = int(reports[0]["execution"]["world_size"])
    hbm_totals = _hbm_totals(args.hbm_total_bytes, world_size)
    pinned_limits = (args.pinned_limit_bytes,) * world_size
    plan = select_residency_plan(
        groups,
        profiles,
        hbm_totals,
        group_plan_sha256=group_hashes[0],
        stack_revision_sha256=stack_hashes[0],
        hbm_margin_bytes=args.hbm_margin_bytes,
        pinned_limit_bytes_by_rank=pinned_limits,
        selector_tie_fraction=args.selector_tie_fraction,
        max_interleavings=args.max_search_states,
    )
    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    for route, values in profiles.items():
        for profile in values:
            output = profile_dir / (
                f"{route}_{profile.content_sha256}.json"
            )
            output.write_bytes(canonical_json_bytes(profile.to_dict()))
    output_plan = Path(args.output_plan)
    plan.write(output_plan)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_plan": str(output_plan.resolve()),
                "plan_sha256": plan.content_sha256,
                "predicted_makespan_seconds": (
                    plan.predicted_makespan_seconds
                ),
                "compiled": plan.compiled.to_dict(),
                "exact": plan.exact.to_dict(),
                "launch_order": list(plan.launch_order),
                "stack_revision_sha256": plan.stack_revision_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
