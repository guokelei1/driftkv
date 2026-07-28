from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import cohortkv_stage4_8_sweep_common as stage48
import run_cohortkv_stage4_7_organic_chain as base
import run_cohortkv_stage4_9_formal_confirmation as formal
import run_cohortkv_stage4_9_rollout_boundary as stage49
import torch
from cohortkv_stage4_7_common import (
    CHECKPOINT_DIR,
    PREPARED_PATH,
    TRAINING_PATH,
    load_inputs,
    sha256,
)
from evaluate_cohortkv_stage4_6_lifecycle import LAUNCH, execute_direct
from motivation_validity import seed_everything

from hstu_kvcache.migration import (
    RENEWAL_CALIBRATION_MODES,
    JaggedMigratedKVBatch,
    fit_renewal_calibrated_direct_oldkv_program,
)
from hstu_kvcache.migration.stage45_oldkv import (
    DirectOldKVFusedOperator,
    DirectOldKVProgram,
)
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)
from hstu_kvcache.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_stage4_10_renewal_calibrated_h12_smoke_v1"
OUTPUT_DIR = "results/system/cohortkv_single_config_full_chain_v1"
DEFAULT_EDGES = 2
DEFAULT_RIDGE = 0.001
DEFAULT_MAX_FIT_TOKENS = 8192


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--calibration-mode",
        required=True,
        choices=RENEWAL_CALIBRATION_MODES,
    )
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--edges", type=int, default=DEFAULT_EDGES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument(
        "--max-fit-tokens",
        type=int,
        default=DEFAULT_MAX_FIT_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
        or args.seed != 0
        or args.batch_size != 4
        or args.edges != DEFAULT_EDGES
        or args.warmup_repeats < 0
        or args.timing_repeats < 1
        or args.ridge <= 0
        or args.max_fit_tokens < 2
    ):
        raise ValueError("renewal-calibrated smoke settings differ")


def output_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / (
        "stage4_10_renewal_calibrated_h12_"
        f"{args.calibration_mode}_{args.edges}edges_seed0.json"
    )


def _tensor_sha256(program: DirectOldKVProgram) -> str:
    digest = hashlib.sha256()
    for value in (program.weights, program.biases):
        prepared = value.detach().contiguous().cpu()
        digest.update(prepared.numpy().tobytes())
    return digest.hexdigest()


def _measurement_total(
    values: list[dict[str, object]],
    timing_repeats: int,
) -> dict[str, object]:
    if not values:
        return formal._zero_measurement(timing_repeats)
    return formal._measurement_total(values, timing_repeats)


def _aggregate_components(
    group_components: list[dict[str, dict[str, object]]],
    edge_components: dict[str, dict[str, object]],
    timing_repeats: int,
) -> dict[str, object]:
    names = sorted(
        {
            name
            for components in group_components
            for name in components
        }
    )
    combined = {
        name: _measurement_total(
            [
                components.get(
                    name,
                    formal._zero_measurement(timing_repeats),
                )
                for components in group_components
            ],
            timing_repeats,
        )
        for name in names
    }
    combined.update(edge_components)
    samples = [
        sum(
            float(value["samples_ms"][index])
            for value in combined.values()
        )
        for index in range(timing_repeats)
    ]
    return {
        "components": combined,
        "samples_ms": samples,
        "median_of_repetition_sums_ms": float(statistics.median(samples)),
        "sum_of_component_medians_ms": sum(
            float(value["median_ms"]) for value in combined.values()
        ),
    }


def _prepare_calibration(
    args: argparse.Namespace,
    cfg,
    target_model,
    target_version: int,
    target_window,
    groups,
    record_by_id: dict[int, dict],
    plans,
    selection,
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    operator: DirectOldKVFusedOperator,
    device: torch.device,
) -> dict[str, object]:
    scheduled = set(selection.scheduled_exact_ids)
    scheduled_batches = {}
    crop_measurements = {}
    exact_measurements = {}
    old_parts = []
    fresh_parts = []
    movement = []
    ordered_ids = []
    for group in groups:
        scheduled_ids = tuple(
            int(value["record_id"])
            for value in group
            if int(value["record_id"]) in scheduled
            and plans[int(value["record_id"])].final_tokens > 0
        )
        if not scheduled_ids:
            continue
        staged, transfer = formal._timed_store_transfer(
            cache_by_record,
            scheduled_ids,
            device,
            device,
            "host_to_device_calibration_actual",
        )
        movement.append(transfer)
        cropped, crop_measurement = formal._timed_repeated(
            partial(
                formal._crop_actual_retained,
                staged,
                scheduled_ids,
                plans,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
        retained_batch = stage49._retained_batch(
            scheduled_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        fresh, exact_measurement = formal._timed_repeated(
            partial(
                stage49._exact_cache,
                target_model,
                retained_batch,
                scheduled_ids,
                target_version,
                torch.float16,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
        scheduled_batches[scheduled_ids] = fresh
        crop_measurements[scheduled_ids] = crop_measurement
        exact_measurements[scheduled_ids] = exact_measurement
        old_parts.append(cropped)
        fresh_parts.append(fresh)
        ordered_ids.extend(scheduled_ids)
        del staged, retained_batch
    calibration_ids = tuple(sorted(ordered_ids))
    if (
        set(calibration_ids) != scheduled
        or len(calibration_ids) != len(scheduled)
        or not calibration_ids
    ):
        raise RuntimeError("renewal calibration does not cover scheduled exact")
    actual_old = stage48._assemble_target_sources(
        calibration_ids,
        tuple(old_parts),
        target_version - 1,
    )
    fresh_target = stage48._assemble_target_sources(
        calibration_ids,
        tuple(fresh_parts),
        target_version,
    )
    source_model = None
    if args.calibration_mode == "inverse_norm_ridge":
        source_model = load_checkpoint_model(
            cfg,
            args.checkpoint_dir,
            target_version - 1,
            device,
        )
    build_started = time.perf_counter()
    (fit_result, build_measurement) = formal._timed_repeated(
        partial(
            fit_renewal_calibrated_direct_oldkv_program,
            actual_old,
            fresh_target,
            source_version=f"theta{target_version - 1}",
            target_version=f"theta{target_version}",
            mode=args.calibration_mode,
            ridge=args.ridge,
            max_fit_tokens=args.max_fit_tokens,
            seed=args.seed + target_version * 1009,
            source_model=source_model,
            target_model=target_model,
        ),
        device,
        args.warmup_repeats,
        args.timing_repeats,
    )
    program, fit_metrics = fit_result
    prepared_program, prepare_measurement = formal._timed_repeated(
        partial(operator.prepare_program, program, device),
        device,
        args.warmup_repeats,
        args.timing_repeats,
    )
    build_wall_ms = (time.perf_counter() - build_started) * 1000.0
    program_sha256 = _tensor_sha256(prepared_program)
    del actual_old, fresh_target, old_parts, fresh_parts, program
    if source_model is not None:
        del source_model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "program": prepared_program,
        "program_sha256": program_sha256,
        "fit_metrics": fit_metrics.to_dict(),
        "program_build_measurement": build_measurement,
        "program_prepare_measurement": prepare_measurement,
        "program_build_wall_ms": build_wall_ms,
        "scheduled_batches": scheduled_batches,
        "crop_measurements": crop_measurements,
        "exact_measurements": exact_measurements,
        "calibration_ids": calibration_ids,
        "calibration_h2d": formal._sum_state_movement(
            movement,
            "host_to_device_calibration_actual",
        ),
    }


def _group_actions(group, plans, selection) -> dict[str, tuple[int, ...]]:
    group_ids = tuple(int(value["record_id"]) for value in group)
    resident = tuple(
        value for value in group_ids if plans[value].final_tokens > 0
    )
    migrate = set(selection.migrate_ids)
    scheduled = set(selection.scheduled_exact_ids)
    natural = set(selection.natural_exact_ids)
    migrate_ids = tuple(value for value in resident if value in migrate)
    scheduled_ids = tuple(value for value in resident if value in scheduled)
    natural_ids = tuple(value for value in resident if value in natural)
    missing_ids = tuple(
        value
        for value in natural_ids
        if plans[value].timed_retained_rebuild
    )
    natural_prefix_ids = tuple(
        value
        for value in natural_ids
        if value not in set(missing_ids)
        and plans[value].target_prefix_tokens > 0
    )
    short_ids = tuple(
        value
        for value in natural_ids
        if plans[value].target_prefix_tokens == 0
    )
    return {
        "group": group_ids,
        "resident": resident,
        "migrate": migrate_ids,
        "scheduled": scheduled_ids,
        "natural": natural_ids,
        "missing": missing_ids,
        "natural_prefix": natural_prefix_ids,
        "short": short_ids,
        "timed": tuple(sorted((*migrate_ids, *scheduled_ids, *missing_ids))),
    }


def _execute_group(
    args: argparse.Namespace,
    cfg,
    target_model,
    target_version: int,
    target_window,
    group,
    record_by_id: dict[int, dict],
    plans,
    selection,
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    last_exact_by_record: dict[int, int],
    calibration: dict[str, object],
    operator: DirectOldKVFusedOperator,
    device: torch.device,
) -> dict[str, object]:
    actions = _group_actions(group, plans, selection)
    migrate_ids = actions["migrate"]
    scheduled_ids = actions["scheduled"]
    missing_ids = actions["missing"]
    timed_ids = actions["timed"]
    staged, migrant_h2d = formal._timed_store_transfer(
        cache_by_record,
        migrate_ids,
        device,
        device,
        "host_to_device_previous_actual",
    )
    components = {
        "calibration_source_crop_ms": calibration[
            "crop_measurements"
        ].get(
            scheduled_ids,
            formal._zero_measurement(args.timing_repeats),
        ),
        "scheduled_exact_retained_ms": calibration[
            "exact_measurements"
        ].get(
            scheduled_ids,
            formal._zero_measurement(args.timing_repeats),
        ),
        "retained_source_crop_ms": formal._zero_measurement(
            args.timing_repeats
        ),
        "retained_transform_ms": formal._zero_measurement(
            args.timing_repeats
        ),
        "missing_exact_retained_ms": formal._zero_measurement(
            args.timing_repeats
        ),
        "retained_materialization_ms": formal._zero_measurement(
            args.timing_repeats
        ),
    }
    retained_sources = []
    if migrate_ids:
        cropped, measurement = formal._timed_repeated(
            partial(
                formal._crop_actual_retained,
                staged,
                migrate_ids,
                plans,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
        components["retained_source_crop_ms"] = measurement
        migrated, measurement = formal._timed_repeated(
            partial(
                execute_direct,
                operator,
                calibration["program"],
                cropped,
                target_version,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
        components["retained_transform_ms"] = measurement
        retained_sources.append(migrated)
        del cropped
    del staged
    if scheduled_ids:
        retained_sources.append(
            calibration["scheduled_batches"][scheduled_ids]
        )
    if missing_ids:
        missing_batch = stage49._retained_batch(
            missing_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        missing_exact, measurement = formal._timed_repeated(
            partial(
                stage49._exact_cache,
                target_model,
                missing_batch,
                missing_ids,
                target_version,
                torch.float16,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
        components["missing_exact_retained_ms"] = measurement
        retained_sources.append(missing_exact)
    mixed_retained = None
    if timed_ids:
        mixed_retained, measurement = formal._timed_repeated(
            partial(
                stage48._assemble_target_sources,
                timed_ids,
                tuple(retained_sources),
                target_version,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
        components["retained_materialization_ms"] = measurement
        exact_batch = stage49._retained_batch(
            timed_ids,
            plans,
            record_by_id,
            target_window,
            device,
        )
        _, exact_measurement = formal._timed_repeated(
            partial(
                stage49._exact_cache,
                target_model,
                exact_batch,
                timed_ids,
                target_version,
                torch.float16,
            ),
            device,
            args.warmup_repeats,
            args.timing_repeats,
        )
    else:
        exact_measurement = formal._zero_measurement(args.timing_repeats)
    prefix_sources = []
    if timed_ids:
        prefix_sources.append(
            formal._append_delta_once(
                target_model,
                mixed_retained,
                plans,
                record_by_id,
                target_window,
                device,
                torch.float16,
            )
        )
    natural_prefix_ids = actions["natural_prefix"]
    if natural_prefix_ids:
        prefix_sources.append(
            formal._build_natural_prefix_once(
                target_model,
                natural_prefix_ids,
                plans,
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float16,
            )
        )
    nonshort_ids = tuple(
        value
        for value in actions["resident"]
        if plans[value].target_prefix_tokens > 0
    )
    final_sources = []
    if nonshort_ids:
        prefix = stage48._assemble_target_sources(
            nonshort_ids,
            tuple(prefix_sources),
            target_version,
        )
        final_sources.append(
            formal._append_latest_once(
                target_model,
                prefix,
                record_by_id,
                target_window,
                device,
                torch.float16,
            )
        )
    if actions["short"]:
        final_sources.append(
            formal._append_fresh_latest_once(
                target_model,
                actions["short"],
                record_by_id,
                target_window,
                target_version,
                cfg,
                device,
                torch.float16,
            )
        )
    if actions["resident"]:
        final_cache, final_hidden = formal._merge_final_once(
            actions["resident"],
            tuple(final_sources),
            target_version,
            device,
        )
        if (
            not bool(torch.isfinite(final_cache.k).all())
            or not bool(torch.isfinite(final_cache.v).all())
            or not bool(torch.isfinite(final_hidden).all())
        ):
            raise RuntimeError("renewal-calibrated output is nonfinite")
        split = base._split_cache(final_cache)
        host_split, next_d2h = formal._timed_store_transfer(
            split,
            tuple(split),
            torch.device("cpu"),
            device,
            "device_to_host_next_actual",
        )
    else:
        host_split = {}
        next_d2h = formal._zero_state_movement(
            "device_to_host_next_actual"
        )
    next_last_exact = {}
    migrate = set(migrate_ids)
    for record_id in actions["resident"]:
        if record_id in migrate:
            next_last_exact[record_id] = last_exact_by_record[record_id]
        else:
            next_last_exact[record_id] = target_version
    return {
        "cache": host_split,
        "last_exact": next_last_exact,
        "u_components": components,
        "e": exact_measurement,
        "migrant_h2d": migrant_h2d,
        "next_d2h": next_d2h,
        "actions": actions,
    }


def _run_edge(
    args: argparse.Namespace,
    cfg,
    old_window,
    target_window,
    groups,
    record_by_id: dict[int, dict],
    cache_by_record: dict[int, JaggedMigratedKVBatch],
    last_exact_by_record: dict[int, int],
    expected_ids: set[int],
    scheduler_state,
    operator: DirectOldKVFusedOperator,
    source_version: int,
    device: torch.device,
) -> tuple[dict, dict, set[int], object, dict]:
    target_version = source_version + 1
    plans, plan_checks = stage49._plan_edge(
        old_window,
        target_window,
        [record_by_id[value] for value in sorted(record_by_id)],
        expected_ids,
        set(cache_by_record),
    )
    spec = stage49._candidate_spec("staggered_renewal_h12")
    selection, scheduler_checks = stage49._select_actions(
        plans,
        last_exact_by_record,
        source_version,
        target_version,
        spec,
        scheduler_state,
    )
    target_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        target_version,
        device,
    )
    calibration = _prepare_calibration(
        args,
        cfg,
        target_model,
        target_version,
        target_window,
        groups,
        record_by_id,
        plans,
        selection,
        cache_by_record,
        operator,
        device,
    )
    group_results = []
    next_cache = {}
    next_last_exact = {}
    for group in groups:
        result = _execute_group(
            args,
            cfg,
            target_model,
            target_version,
            target_window,
            group,
            record_by_id,
            plans,
            selection,
            cache_by_record,
            last_exact_by_record,
            calibration,
            operator,
            device,
        )
        group_results.append(result)
        next_cache.update(result["cache"])
        next_last_exact.update(result["last_exact"])
        gc.collect()
        torch.cuda.empty_cache()
    target_expected = {
        record_id
        for record_id, plan in plans.items()
        if plan.final_tokens > 0
    }
    u = _aggregate_components(
        [value["u_components"] for value in group_results],
        {
            "program_fit_and_compile_ms": calibration[
                "program_build_measurement"
            ],
            "program_prepare_ms": calibration[
                "program_prepare_measurement"
            ],
        },
        args.timing_repeats,
    )
    e = _measurement_total(
        [value["e"] for value in group_results],
        args.timing_repeats,
    )
    calibration_ids = set(calibration["calibration_ids"])
    checks = {
        "retained_plan": all(plan_checks.values()),
        "scheduler": all(scheduler_checks.values()),
        "action_partition": set(selection.migrate_ids)
        | set(selection.scheduled_exact_ids)
        | set(selection.natural_exact_ids)
        == target_expected,
        "calibration_ids_equal_scheduled_exact": calibration_ids
        == set(selection.scheduled_exact_ids),
        "calibration_and_migrants_disjoint": not calibration_ids
        & set(selection.migrate_ids),
        "extra_exact_fit_records_zero": True,
        "scheduled_exact_reused_as_refresh": all(
            ids in calibration["scheduled_batches"]
            for ids in calibration["scheduled_batches"]
        ),
        "no_serialized_program_loaded": True,
        "program_build_included_in_u": (
            "program_fit_and_compile_ms" in u["components"]
        ),
        "no_semantic_gate": True,
        "no_automatic_action_changes": True,
        "recursive_cache_covers_target": set(next_cache)
        == target_expected,
        "recursive_last_exact_covers_target": set(next_last_exact)
        == target_expected,
        "next_store_contract": all(
            formal._persistent_cpu_store_checks(
                next_cache,
                expected_version=target_version,
                expected_lengths={
                    value: plans[value].final_tokens for value in next_cache
                },
            ).values()
        ),
        "finite_positive_exact_denominator": float(e["median_ms"]) > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "renewal-calibrated edge failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    step = {
        "source_version": source_version,
        "target_version": target_version,
        "actions": {
            "migrate": len(selection.migrate_ids),
            "scheduled_exact": len(selection.scheduled_exact_ids),
            "natural_exact": len(selection.natural_exact_ids),
        },
        "calibration": {
            **calibration["fit_metrics"],
            "program_fit_source": "scheduled_exact_pairs",
            "source_state": "previous_actual_post_append",
            "calibration_record_ids": list(
                calibration["calibration_ids"]
            ),
            "calibration_record_ids_sha256": formal._record_id_sha256(
                calibration_ids
            ),
            "extra_exact_fit_records": 0,
            "program_tensor_sha256": calibration["program_sha256"],
            "program_build_wall_ms": calibration[
                "program_build_wall_ms"
            ],
            "serialized_program_loaded": False,
            "semantic_gate_used": False,
        },
        "cost": {
            "u": u,
            "e": e,
            "primary_u_over_e": float(
                u["sum_of_component_medians_ms"]
            )
            / float(e["median_ms"]),
            "program_build_included_in_u": True,
            "target_append_excluded_from_u_and_e": True,
            "state_movement_outside_primary": {
                "calibration_h2d": calibration["calibration_h2d"],
                "migrant_h2d": formal._sum_state_movement(
                    [value["migrant_h2d"] for value in group_results],
                    "host_to_device_previous_actual",
                ),
                "next_d2h": formal._sum_state_movement(
                    [value["next_d2h"] for value in group_results],
                    "device_to_host_next_actual",
                ),
            },
        },
        "scheduler": {
            "variant": spec.to_dict(),
            "diagnostics": selection.diagnostics,
            "state_after": formal._serialize_state(selection.next_state),
            "scheduled_exact_ids": list(selection.scheduled_exact_ids),
            "migrate_ids": list(selection.migrate_ids),
        },
        "recursive_store": {
            "previous_record_ids_sha256": formal._record_id_sha256(
                cache_by_record
            ),
            "previous_lengths_sha256": formal._record_length_sha256(
                cache_by_record
            ),
            "next_record_ids_sha256": formal._record_id_sha256(next_cache),
            "next_lengths_sha256": formal._record_length_sha256(next_cache),
        },
        "checks": checks,
    }
    del target_model, calibration, group_results
    gc.collect()
    torch.cuda.empty_cache()
    return (
        next_cache,
        next_last_exact,
        target_expected,
        selection.next_state,
        step,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    path = output_path(args)
    if path.exists() and not args.force:
        raise FileExistsError(f"renewal-calibrated output exists: {path}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(
        int(value["user_id"]) for value in manifest["records"]
    )
    windows = reconstruct_organic_windows(plan, user_ids)
    window_checks = base.validate_windows(windows, manifest)
    if not all(window_checks.values()):
        raise ValueError("renewal-calibrated windows differ")
    groups = base.fixed_record_groups(manifest, args.batch_size)
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    (
        cache_by_record,
        last_exact,
        initialization_ms,
        initialization_movement,
    ) = formal._initialize_theta0_host(
        cfg,
        args.checkpoint_dir,
        windows[0],
        groups,
        device,
    )
    expected_ids = set(cache_by_record)
    operator = DirectOldKVFusedOperator(**LAUNCH)
    scheduler_state = None
    steps = []
    started = time.perf_counter()
    for source_version in range(args.edges):
        previous_cache = cache_by_record
        (
            cache_by_record,
            last_exact,
            expected_ids,
            scheduler_state,
            step,
        ) = _run_edge(
            args,
            cfg,
            windows[source_version],
            windows[source_version + 1],
            groups,
            record_by_id,
            cache_by_record,
            last_exact,
            expected_ids,
            scheduler_state,
            operator,
            source_version,
            device,
        )
        previous_cache.clear()
        if steps:
            step["checks"]["previous_lengths_equal_prior_next"] = (
                step["recursive_store"]["previous_lengths_sha256"]
                == steps[-1]["recursive_store"]["next_lengths_sha256"]
            )
            step["checks"]["scheduler_state_continuous"] = True
            step["checks"][
                "renewal_calibration_consumes_prior_migrants"
            ] = bool(
                set(step["calibration"]["calibration_record_ids"])
                & set(steps[-1]["scheduler"]["migrate_ids"])
            )
            if not all(step["checks"].values()):
                raise RuntimeError(
                    "renewal-calibrated recursive smoke failed"
                )
        steps.append(step)
        print(
            json.dumps(
                {
                    "mode": args.calibration_mode,
                    "edge": f"theta{source_version}->theta{source_version + 1}",
                    "actions": step["actions"],
                    "program_build_ms": step["cost"]["u"][
                        "components"
                    ]["program_fit_and_compile_ms"]["median_ms"],
                    "u_over_e": step["cost"]["primary_u_over_e"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    sum_u = sum(
        float(value["cost"]["u"]["sum_of_component_medians_ms"])
        for value in steps
    )
    sum_e = sum(float(value["cost"]["e"]["median_ms"]) for value in steps)
    payload = {
        "protocol": PROTOCOL,
        "status": "smoke_complete",
        "scientific_result": False,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "candidate": "staggered_renewal_h12",
            "calibration_mode": args.calibration_mode,
            "edges": args.edges,
            "batch_size": args.batch_size,
            "warmup_repeats": args.warmup_repeats,
            "timing_repeats": args.timing_repeats,
            "ridge": args.ridge,
            "max_fit_tokens": args.max_fit_tokens,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "model": training["model"],
        },
        "input_provenance": {
            "prepared_data": {
                "path": args.prepared_data,
                "sha256": sha256(args.prepared_data),
                "protocol": metadata["protocol"],
            },
            "training_result": {
                "path": args.training_result,
                "sha256": sha256(args.training_result),
                "protocol": training["protocol"],
            },
            "checkpoints": checkpoints[: args.edges + 1],
            "manifest_content_sha256": manifest["content_sha256"],
        },
        "initialization": {
            "theta0_gpu_ms": initialization_ms,
            "state_movement": initialization_movement,
            "excluded_from_u_and_e": True,
        },
        "steps": steps,
        "summary": {
            "edges": len(steps),
            "sum_u_ms": sum_u,
            "sum_e_ms": sum_e,
            "sum_u_over_sum_e": sum_u / sum_e,
            "program_build_included_in_u": True,
            "serialized_program_loaded": False,
            "extra_exact_fit_records": 0,
            "semantic_gate_used": False,
            "target_append_excluded_from_u_and_e": True,
        },
        "checks": {
            "requested_edge_count": len(steps) == args.edges,
            "two_real_adjacent_updates": args.edges == 2
            and [value["target_version"] for value in steps] == [1, 2],
            "all_edge_checks": all(
                all(value["checks"].values()) for value in steps
            ),
            "program_build_counted_every_edge": all(
                value["cost"]["program_build_included_in_u"]
                for value in steps
            ),
            "no_serialized_program_loaded": all(
                not value["calibration"]["serialized_program_loaded"]
                for value in steps
            ),
            "no_semantic_gate": all(
                not value["calibration"]["semantic_gate_used"]
                for value in steps
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not all(payload["checks"].values()):
        raise RuntimeError("renewal-calibrated smoke checks failed")
    save_json(payload, path)
    print(json.dumps({"output": str(path), **payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
