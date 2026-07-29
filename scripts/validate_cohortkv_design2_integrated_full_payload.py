from __future__ import annotations

import argparse
import gc
import json
import os
import uuid
from pathlib import Path

import benchmark_cohortkv_design2_integrated_w3 as benchmark
import torch
import torch.distributed as dist

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration.design2_distributed import (
    close_d2_distributed_runtime,
    init_d2_distributed_runtime,
)
from hstu_kvcache.migration.design2_embedding import (
    build_modulo_sharded_hstu_from_cpu,
)
from hstu_kvcache.migration.design2_integrated import (
    INTEGRATED_ROUTES,
    build_integrated_schedule,
    integrated_route,
    integrated_sharded_append,
    integrated_sharded_append_only,
    integrated_sharded_exact,
    select_integrated_records,
    slice_integrated_jagged_ranges,
)
from hstu_kvcache.migration.design2_payload_validation import (
    D2PayloadComparisonAccumulator,
    compare_jagged_payloads,
    compare_jagged_to_append_only,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionPlan,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
    file_sha256,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    load_direct_oldkv_program,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.streaming import (
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_design2_integrated_full_payload_validation_v1"
DEFAULT_OUTPUT = (
    "results/system/"
    "cohortkv_design2_integrated_full_payload_development_v1/"
    "full682.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action-plan",
        default=benchmark.DEFAULT_ACTION_PLAN,
    )
    parser.add_argument(
        "--stage-a-summary",
        default=benchmark.DEFAULT_STAGE_A_SUMMARY,
    )
    parser.add_argument(
        "--training-result",
        default=benchmark.DEFAULT_TRAINING,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=benchmark.DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument("--extent-size", type=int, default=16)
    parser.add_argument(
        "--compiled-order",
        choices=("final_length", "suffix_retained"),
        default="suffix_retained",
    )
    parser.add_argument("--kv-atol", type=float, default=2e-2)
    parser.add_argument("--kv-rtol", type=float, default=2e-2)
    parser.add_argument("--hidden-atol", type=float, default=2e-5)
    parser.add_argument("--hidden-rtol", type=float, default=2e-5)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--expected-visible-devices", default="0,1,3")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _file_binding(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"input hash differs: {path}")
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "verified": expected_sha256 is None or actual == expected_sha256,
    }


def _extent_summary(
    route: str,
    ordinal: int,
    reports: tuple[dict[str, object], ...],
) -> dict[str, object]:
    elements = sum(int(value["elements"]) for value in reports)
    absolute_error_sum = sum(
        float(value["mean_absolute_error"]) * int(value["elements"])
        for value in reports
    )
    return {
        "route": route,
        "ordinal": ordinal,
        "records": len(reports),
        "record_ids": [int(value["record_id"]) for value in reports],
        "tokens": sum(int(value["tokens"]) for value in reports),
        "elements": elements,
        "allclose": all(bool(value["allclose"]) for value in reports),
        "bitwise_equal": all(
            bool(value["bitwise_equal"]) for value in reports
        ),
        "finite": all(bool(value["finite"]) for value in reports),
        "max_absolute_error": max(
            (
                float(value["max_absolute_error"])
                for value in reports
            ),
            default=0.0,
        ),
        "mean_absolute_error": (
            absolute_error_sum / elements if elements else 0.0
        ),
        "comparison_sha256": canonical_sha256(list(reports)),
    }


def _input_history_bindings(
    selected,
    source_version: str,
    target_version: str,
) -> dict[str, object]:
    return {
        "source_version": source_version,
        "target_version": target_version,
        "source_records_sha256": canonical_sha256(
            [
                {
                    "history_sha256": value.old_history_sha256,
                    "record_id": value.record_id,
                    "tokens": value.old_tokens,
                }
                for value in selected
            ]
        ),
        "target_records_sha256": canonical_sha256(
            [
                {
                    "delta_start": value.delta_start,
                    "final_tokens": value.final_tokens,
                    "history_sha256": value.target_history_sha256,
                    "record_id": value.record_id,
                }
                for value in selected
            ]
        ),
    }


def _validate_visible_devices(
    expected: str,
) -> tuple[str, tuple[str, ...]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = tuple(
        value.strip() for value in visible.split(",") if value.strip()
    )
    expected_tokens = tuple(
        value.strip() for value in expected.split(",") if value.strip()
    )
    if visible_tokens != expected_tokens or len(visible_tokens) != 3:
        raise RuntimeError(
            "full-payload CUDA_VISIBLE_DEVICES differs from expected W3"
        )
    return visible, visible_tokens


def _write_json_atomic(
    path: Path,
    value: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object] | None:
    if args.extent_size < 1:
        raise ValueError("full-payload extent size must be positive")
    runtime = init_d2_distributed_runtime(
        timeout_seconds=args.timeout_seconds
    )
    try:
        if (
            runtime.world_size != 3
            or runtime.backend != "nccl"
            or runtime.device.type != "cuda"
        ):
            raise RuntimeError(
                "full-payload validation requires three NCCL CUDA ranks"
            )
        visible_devices, visible_tokens = _validate_visible_devices(
            args.expected_visible_devices
        )
        action_path = _path(args.action_plan)
        stage_a_path = _path(args.stage_a_summary)
        training_path = _path(args.training_result)
        checkpoint_dir = _path(args.checkpoint_dir)
        action_plan = D2ActionPlan.load(action_path)
        stage_a = json.loads(stage_a_path.read_text())
        training = json.loads(training_path.read_text())
        if (
            stage_a["status"] != "complete"
            or stage_a["action_plan"]["content_sha256"]
            != action_plan.content_sha256
        ):
            raise ValueError("full-payload Stage A binding differs")
        owner_map = build_d2_record_owner_map(
            action_plan,
            runtime.world_size,
            "strict_cow_lpt",
        )
        selected = select_integrated_records(
            action_plan,
            owner_map,
            runtime.world_size,
            "full682",
        )
        actions_by_id = {value.record_id: value for value in selected}
        local_actions = tuple(
            value
            for value in selected
            if owner_map[value.record_id] == runtime.rank
        )
        schedule = build_integrated_schedule(
            selected,
            owner_map,
            runtime.world_size,
            args.extent_size,
            route_major=True,
            compiled_order=args.compiled_order,
        )
        prepared_path = _path(action_plan.provenance.prepared_data)
        data_plan, prepared_metadata = load_prepared_kuairand_plan(
            prepared_path
        )
        validate_long_context_plan(data_plan, prepared_metadata, 4)
        windows = reconstruct_organic_windows(
            data_plan,
            (value.prepared_user_id for value in selected),
        )
        source_index = int(
            action_plan.source_version.removeprefix("theta")
        )
        target_index = int(
            action_plan.target_version.removeprefix("theta")
        )
        source_window = windows[source_index]
        target_window = windows[target_index]
        for action in selected:
            source_record = source_window.records[action.prepared_user_id]
            target_record = target_window.records[action.prepared_user_id]
            if (
                source_record.history is None
                or source_record.history_sha256
                != action.old_history_sha256
                or len(source_record.history) != action.old_tokens
                or target_record.history is None
                or target_record.history_sha256
                != action.target_history_sha256
                or len(target_record.history) != action.final_tokens
            ):
                raise ValueError(
                    "full-payload reconstructed history differs"
                )
        cfg = HSTUConfig(**training["model"])
        source_checkpoint = benchmark._checkpoint_path(
            checkpoint_dir,
            action_plan.source_version,
        )
        target_checkpoint = benchmark._checkpoint_path(
            checkpoint_dir,
            action_plan.target_version,
        )
        source_descriptor = benchmark._checkpoint_descriptor(
            action_plan,
            action_plan.source_version,
        )
        target_descriptor = benchmark._checkpoint_descriptor(
            action_plan,
            action_plan.target_version,
        )
        if (
            _path(source_descriptor["path"]).resolve()
            != source_checkpoint.resolve()
            or _path(target_descriptor["path"]).resolve()
            != target_checkpoint.resolve()
        ):
            raise ValueError("full-payload checkpoint path differs")
        stage_a_program = benchmark._program_descriptor(stage_a)
        program_path = _path(stage_a_program["path"])
        input_bindings = {
            "action_plan": _file_binding(action_path),
            "stage_a_summary": _file_binding(stage_a_path),
            "training_result": _file_binding(training_path),
            "prepared_data": _file_binding(
                prepared_path,
                expected_sha256=(
                    action_plan.provenance.prepared_data_sha256
                ),
            ),
            "source_checkpoint": _file_binding(
                source_checkpoint,
                expected_sha256=source_descriptor["sha256"],
            ),
            "target_checkpoint": _file_binding(
                target_checkpoint,
                expected_sha256=target_descriptor["sha256"],
            ),
            "program": _file_binding(
                program_path,
                expected_sha256=stage_a_program["sha256"],
            ),
            "reconstructed_histories": _input_history_bindings(
                selected,
                action_plan.source_version,
                action_plan.target_version,
            ),
        }
        source_cpu = benchmark._load_cpu_model(
            cfg,
            source_checkpoint,
        )
        source_model = build_modulo_sharded_hstu_from_cpu(
            source_cpu,
            runtime.rank,
            runtime.world_size,
            runtime.device,
        )
        del source_cpu
        target_cpu = benchmark._load_cpu_model(
            cfg,
            target_checkpoint,
        )
        target_model = build_modulo_sharded_hstu_from_cpu(
            target_cpu,
            runtime.rank,
            runtime.world_size,
            runtime.device,
        )
        del target_cpu
        program_cpu, loaded_program = load_direct_oldkv_program(
            program_path,
            expected_sha256=stage_a_program["sha256"],
            expected_source_version=action_plan.source_version,
            expected_target_version=action_plan.target_version,
            expected_num_layers=cfg.num_layers,
            expected_kv_width=cfg.hidden_size,
        )
        operator = DirectOldKVFusedOperator()
        program = operator.prepare_program(
            program_cpu,
            runtime.device,
        )
        del program_cpu
        accumulator = D2PayloadComparisonAccumulator(
            kv_atol=args.kv_atol,
            kv_rtol=args.kv_rtol,
            hidden_atol=args.hidden_atol,
            hidden_rtol=args.hidden_rtol,
        )
        extent_reports = []
        compiled_extent_count = sum(
            value.route == "compiled" for value in schedule
        )
        completed_compiled_extents = 0
        for extent_index, extent in enumerate(schedule):
            actions = benchmark._actions_for_extent(
                extent,
                runtime.rank,
                actions_by_id,
            )
            if extent.route == "compiled":
                source_batch = benchmark._history_batch(
                    actions,
                    source_window,
                    (0,) * len(actions),
                    tuple(value.old_tokens for value in actions),
                    action_plan.source_version,
                    runtime.device,
                )
                source_exact = integrated_sharded_exact(
                    source_model,
                    source_batch,
                    action_plan.source_version,
                )
                suffix = benchmark._history_batch(
                    actions,
                    target_window,
                    tuple(value.delta_start for value in actions),
                    tuple(value.final_tokens for value in actions),
                    action_plan.target_version,
                    runtime.device,
                )
                if source_exact.fragment is None:
                    retained = None
                else:
                    source_retained = slice_integrated_jagged_ranges(
                        source_exact.fragment,
                        tuple(
                            value.retained_start for value in actions
                        ),
                        tuple(value.old_tokens for value in actions),
                    )
                    retained = operator.execute_into(
                        program,
                        source_retained,
                        benchmark._direct_destination(
                            source_retained,
                            action_plan.target_version,
                        ),
                    )
                    del source_retained
                del source_exact
                del source_batch
                contiguous = integrated_sharded_append(
                    target_model,
                    retained,
                    suffix,
                    action_plan.target_version,
                )
                segmented = integrated_sharded_append_only(
                    target_model,
                    retained,
                    suffix,
                    action_plan.target_version,
                )
                if (
                    contiguous.fragment is None
                    or segmented.fragment is None
                ):
                    if actions:
                        raise RuntimeError(
                            "compiled payload output is missing"
                        )
                    record_reports: tuple[
                        dict[str, object], ...
                    ] = ()
                else:
                    record_reports = compare_jagged_to_append_only(
                        accumulator,
                        route=extent.route,
                        left=contiguous.fragment,
                        left_last_hidden=contiguous.last_hidden,
                        right=segmented.fragment,
                        right_last_hidden=segmented.last_hidden,
                    )
                del contiguous
                del segmented
                del retained
                del suffix
                completed_compiled_extents += 1
                if completed_compiled_extents == compiled_extent_count:
                    del source_model
            else:
                target_batch = benchmark._history_batch(
                    actions,
                    target_window,
                    (0,) * len(actions),
                    tuple(value.final_tokens for value in actions),
                    action_plan.target_version,
                    runtime.device,
                )
                contiguous = integrated_sharded_exact(
                    target_model,
                    target_batch,
                    action_plan.target_version,
                )
                segmented_workflow = integrated_sharded_exact(
                    target_model,
                    target_batch,
                    action_plan.target_version,
                )
                if (
                    contiguous.fragment is None
                    or segmented_workflow.fragment is None
                ):
                    if actions:
                        raise RuntimeError("exact payload output is missing")
                    record_reports = ()
                else:
                    record_reports = compare_jagged_payloads(
                        accumulator,
                        route=extent.route,
                        left=contiguous.fragment,
                        left_last_hidden=contiguous.last_hidden,
                        right=segmented_workflow.fragment,
                        right_last_hidden=(
                            segmented_workflow.last_hidden
                        ),
                    )
                del contiguous
                del segmented_workflow
                del target_batch
            extent_reports.append(
                _extent_summary(
                    extent.route,
                    extent.ordinal,
                    record_reports,
                )
            )
            torch.cuda.synchronize(runtime.device)
            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()
            if runtime.is_primary and (
                (extent_index + 1) % 8 == 0
                or extent_index + 1 == len(schedule)
            ):
                print(
                    json.dumps(
                        {
                            "event": "full_payload_progress",
                            "completed_extents": extent_index + 1,
                            "total_extents": len(schedule),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        payload_report = accumulator.report()
        expected_tokens = sum(
            value.final_tokens for value in local_actions
        )
        expected_k_elements = (
            cfg.num_layers * expected_tokens * cfg.hidden_size
        )
        expected_hidden_elements = len(local_actions) * cfg.hidden_size
        local_checks = {
            "record_coverage": (
                payload_report["record_ids"]
                == sorted(value.record_id for value in local_actions)
            ),
            "token_coverage": payload_report["tokens"] == expected_tokens,
            "element_coverage": (
                payload_report["components"]["k"]["elements"]
                == expected_k_elements
                and payload_report["components"]["v"]["elements"]
                == expected_k_elements
                and payload_report["components"]["last_hidden"][
                    "elements"
                ]
                == expected_hidden_elements
            ),
            "all_payloads_allclose": payload_report["allclose"],
            "all_payloads_finite": payload_report["finite"],
            "all_extents_allclose": all(
                value["allclose"] for value in extent_reports
            ),
            "exact_routes_allclose": all(
                value["allclose"]
                for value in payload_report["record_reports"]
                if value["route"]
                in {"scheduled_exact", "natural_exact"}
            ),
            "compiled_route_allclose": all(
                value["allclose"]
                for value in payload_report["record_reports"]
                if value["route"] == "compiled"
            ),
        }
        local_report = {
            "rank": runtime.rank,
            "local_rank": runtime.local_rank,
            "logical_cuda_index": torch.cuda.current_device(),
            "physical_visible_token": visible_tokens[runtime.local_rank],
            "device_name": torch.cuda.get_device_name(runtime.device),
            "record_ids": [value.record_id for value in local_actions],
            "expected_tokens": expected_tokens,
            "expected_k_elements": expected_k_elements,
            "expected_hidden_elements": expected_hidden_elements,
            "route_counts": {
                route: sum(
                    integrated_route(value) == route
                    for value in local_actions
                )
                for route in INTEGRATED_ROUTES
            },
            "checks": local_checks,
            "payload_comparison": payload_report,
            "extents": extent_reports,
        }
        gathered: list[dict[str, object] | None] = [
            None for _ in range(runtime.world_size)
        ]
        dist.all_gather_object(gathered, local_report)
        if not runtime.is_primary:
            return None
        ranks = [value for value in gathered if value is not None]
        checks = {
            "rank_count": len(ranks) == runtime.world_size,
            "owner_coverage": (
                sorted(
                    record_id
                    for rank_report in ranks
                    for record_id in rank_report["record_ids"]
                )
                == [value.record_id for value in selected]
            ),
            "all_rank_checks": all(
                all(rank_report["checks"].values())
                for rank_report in ranks
            ),
            "full_682_records_compared": sum(
                rank_report["payload_comparison"]["records"]
                for rank_report in ranks
            )
            == 682,
            "all_valid_tokens_compared": sum(
                rank_report["payload_comparison"]["tokens"]
                for rank_report in ranks
            )
            == sum(value.final_tokens for value in selected),
            "route_coverage": {
                route: sum(
                    rank_report["route_counts"][route]
                    for rank_report in ranks
                )
                for route in INTEGRATED_ROUTES
            }
            == {
                route: sum(
                    integrated_route(value) == route
                    for value in selected
                )
                for route in INTEGRATED_ROUTES
            },
            "input_hashes_verified": all(
                not isinstance(value, dict)
                or "verified" not in value
                or value["verified"]
                for value in input_bindings.values()
            ),
        }
        left_rank_hashes = [
            {
                "rank": value["rank"],
                "sha256": value["payload_comparison"]["left_sha256"],
            }
            for value in ranks
        ]
        right_rank_hashes = [
            {
                "rank": value["rank"],
                "sha256": value["payload_comparison"]["right_sha256"],
            }
            for value in ranks
        ]
        artifact: dict[str, object] = {
            "protocol": PROTOCOL,
            "status": "complete" if all(checks.values()) else "failed",
            "scientific_result": False,
            "performance_result": False,
            "scope": {
                "development_only": True,
                "full_payload_correctness_only": True,
                "cohort": "full682",
                "physical_w3": True,
                "timed_region": False,
                "performance_claim": False,
                "per_extent_execution": True,
                "per_record_segment_materialization": True,
                "full_contiguous_segmented_coexistence_avoided": True,
                "exact_routes_independently_replayed": True,
                "compiled_retained_input_shared_between_finalizers": True,
            },
            "configuration": {
                "world_size": runtime.world_size,
                "backend": runtime.backend,
                "cuda_visible_devices": visible_devices,
                "expected_visible_devices": (
                    args.expected_visible_devices
                ),
                "records": len(selected),
                "extent_size": args.extent_size,
                "compiled_order": args.compiled_order,
                "owner_strategy": "strict_cow_lpt",
                "owner_map_sha256": d2_record_owner_map_sha256(
                    owner_map
                ),
                "schedule_sha256": canonical_sha256(
                    [
                        {
                            "ordinal": value.ordinal,
                            "record_ids_by_rank": (
                                value.record_ids_by_rank
                            ),
                            "route": value.route,
                        }
                        for value in schedule
                    ]
                ),
                "route_counts": {
                    route: sum(
                        integrated_route(value) == route
                        for value in selected
                    )
                    for route in INTEGRATED_ROUTES
                },
                "tolerances": {
                    "kv_atol": args.kv_atol,
                    "kv_rtol": args.kv_rtol,
                    "hidden_atol": args.hidden_atol,
                    "hidden_rtol": args.hidden_rtol,
                },
            },
            "inputs": {
                **input_bindings,
                "action_plan_content_sha256": (
                    action_plan.content_sha256
                ),
                "loaded_program": loaded_program,
            },
            "checks": checks,
            "payload_hashes": {
                "protocol": payload_report["hash_protocol"],
                "left_rank_hashes": left_rank_hashes,
                "right_rank_hashes": right_rank_hashes,
                "left_rank_ordered_merkle_sha256": canonical_sha256(
                    left_rank_hashes
                ),
                "right_rank_ordered_merkle_sha256": canonical_sha256(
                    right_rank_hashes
                ),
                "rank_hashes_match": all(
                    left["sha256"] == right["sha256"]
                    for left, right in zip(
                        left_rank_hashes,
                        right_rank_hashes,
                        strict=True,
                    )
                ),
            },
            "rank_reports": ranks,
        }
        artifact["content_sha256"] = canonical_sha256(artifact)
        return artifact
    finally:
        close_d2_distributed_runtime(runtime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = run(args)
    if artifact is None:
        return
    output_path = _path(args.output)
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                "protocol": artifact["protocol"],
                "status": artifact["status"],
                "output": _relative(output_path),
                "checks": artifact["checks"],
                "payload_hashes": artifact["payload_hashes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
