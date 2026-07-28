from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import cohortkv_stage4_8_sweep_common as stage48
import run_cohortkv_stage4_7_organic_chain as base
import torch
from cohortkv_stage4_7_common import (
    CHECKPOINT_DIR,
    COMPILER_OUTPUT,
    PREPARED_PATH,
    RUNTIME_DIR,
    TRAINING_PATH,
    load_inputs,
    sha256,
)
from evaluate_cohortkv_stage4_6_lifecycle import (
    LAUNCH,
    execute_direct,
    timed_cuda,
)
from motivation_validity import seed_everything

from hstu_kvcache.migration import (
    ROLLOUT_BOUNDARY_PROTOCOL,
    JaggedMigratedKVBatch,
    RetainedPrefixPlan,
    append_jagged_suffix,
    pack_padded_cache,
    plan_retained_prefix,
    retained_population_sha256,
    tail_slice_jagged_cache,
)
from hstu_kvcache.migration.organic_schedulers import SchedulerRecord
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVFusedOperator
from hstu_kvcache.streaming import (
    load_checkpoint_model,
    reconstruct_organic_windows,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROLLOUT_BOUNDARY_PROTOCOL
SMOKE_PROTOCOL = "cohortkv_single_config_stage4_9_smoke_v1"
BATCH_SIZE = 4
OUTPUT_DIR = "results/system/cohortkv_single_config_full_chain_v1"
CANDIDATES = {
    "token_debt_total10": ("token_debt", 0),
    "staggered_renewal_h12": ("staggered_renewal", 2),
}
IMPLEMENTATION_PATHS = {
    "runner": ROOT / "scripts/run_cohortkv_stage4_9_rollout_boundary.py",
    "rollout_abi": ROOT / "src/hstu_kvcache/migration/rollout.py",
    "organic_migration": ROOT / "src/hstu_kvcache/migration/organic.py",
    "organic_schedulers": (
        ROOT / "src/hstu_kvcache/migration/organic_schedulers.py"
    ),
    "stage4_8_worker": ROOT / "scripts/cohortkv_stage4_8_sweep_common.py",
    "stage4_7_chain": (
        ROOT / "scripts/run_cohortkv_stage4_7_organic_chain.py"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=tuple(CANDIDATES),
        default="token_debt_total10",
    )
    parser.add_argument("--device")
    parser.add_argument("--prepared-data", default=PREPARED_PATH)
    parser.add_argument("--training-result", default=TRAINING_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--compiler-result", default=COMPILER_OUTPUT)
    parser.add_argument("--runtime-dir", default=RUNTIME_DIR)
    parser.add_argument("--baseline", default=stage48.BASELINE_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--runtime-smoke-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test == args.runtime_smoke_test:
        raise ValueError(
            "Stage 4.9 minimal runner requires exactly one smoke mode"
        )
    if args.seed != 0 or args.batch_size != BATCH_SIZE:
        raise ValueError("Stage 4.9 freezes seed 0 and batch size 4")
    if args.runtime_smoke_test:
        if args.device is None:
            raise ValueError("Stage 4.9 runtime smoke requires --device")
        device = torch.device(args.device)
        if (
            device.type != "cuda"
            or device.index is None
            or device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                "Stage 4.9 runtime smoke requires an available explicit CUDA index"
            )
    elif args.device is not None:
        raise ValueError("Stage 4.9 static smoke does not accept --device")


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _candidate_spec(name: str):
    scheme, index = CANDIDATES[name]
    return stage48.variant_specs(scheme)[index]


def implementation_snapshot() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for name, path in IMPLEMENTATION_PATHS.items()
    }


def _label_free_identities(record) -> tuple[str, ...]:
    history = record.history
    if history is None:
        return ()
    return tuple(
        f"{int(timestamp)}:{int(item)}:{int(behavior)}"
        for timestamp, item, behavior in zip(
            history.timestamps,
            history.item_ids,
            history.behaviors,
            strict=True,
        )
    )


def _plan_edge(
    old_window,
    target_window,
    manifest_records: list[dict],
    previous_expected_ids: set[int],
    previous_present_ids: set[int],
) -> tuple[dict[int, RetainedPrefixPlan], dict[str, bool]]:
    plans = {}
    legacy_matches = []
    for descriptor in manifest_records:
        record_id = int(descriptor["record_id"])
        user_id = int(descriptor["user_id"])
        old_record = old_window.records[user_id]
        target_record = target_window.records[user_id]
        plan = plan_retained_prefix(
            record_id,
            user_id,
            _label_free_identities(old_record),
            _label_free_identities(target_record),
            old_record.history_sha256,
            target_record.history_sha256,
            record_id in previous_expected_ids,
            record_id in previous_present_ids,
        )
        plans[record_id] = plan
        legacy = base.transition_descriptor(
            old_record,
            target_record,
            record_id in previous_present_ids,
        )
        legacy_matches.append(
            plan.target_prefix_tokens == legacy.new_length
            and (
                not plan.migration_eligible
                or (
                    plan.retained_tokens == legacy.overlap
                    and plan.delta_tokens == legacy.appended
                    and plan.retained_start == legacy.retained_old_start
                    and plan.delta_start == legacy.appended_new_start
                )
            )
        )
    checks = {
        "manifest_coverage": set(plans)
        == {int(value["record_id"]) for value in manifest_records},
        "legacy_target_prefix_arithmetic": all(legacy_matches),
        "label_free_planning": True,
        "latest_is_separate": all(
            value.target_prefix_tokens + value.latest_tokens
            == value.final_tokens
            for value in plans.values()
        ),
        "reusable_has_retained_tokens": all(
            not value.migration_eligible or value.retained_tokens > 0
            for value in plans.values()
        ),
        "missing_cache_is_timed_retained_rebuild": all(
            not value.missing_expected_cache
            or (
                value.timed_retained_rebuild
                and value.retained_tokens > 0
                and not value.migration_eligible
            )
            for value in plans.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 4.9 retained plan differs: {checks}")
    return plans, checks


def _select_actions(
    plans: dict[int, RetainedPrefixPlan],
    last_exact_by_record: dict[int, int],
    source_version: int,
    target_version: int,
    spec,
    scheduler_state,
):
    records = tuple(
        SchedulerRecord(
            record_id=record_id,
            prefix_tokens=plan.target_prefix_tokens,
            migration_age=(
                source_version - int(last_exact_by_record[record_id])
                if plan.migration_eligible
                else 0
            ),
            natural_exact=not plan.migration_eligible,
        )
        for record_id, plan in sorted(plans.items())
        if plan.final_tokens > 0
    )
    selection = stage48._select_actions(
        records,
        spec,
        target_version,
        None,
        scheduler_state,
    )
    natural_ids = {
        value.record_id for value in records if value.natural_exact
    }
    reusable_ids = {
        value.record_id for value in records if not value.natural_exact
    }
    scheduled_ids = set(selection.scheduled_exact_ids)
    migrate_ids = set(selection.migrate_ids)
    checks = {
        "natural_coverage": set(selection.natural_exact_ids) == natural_ids,
        "reusable_coverage": scheduled_ids | migrate_ids == reusable_ids,
        "action_disjointness": not scheduled_ids & migrate_ids,
        "labels_not_used": selection.diagnostics.get("labels_used") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 4.9 scheduler differs: {checks}")
    return selection, checks


def _pick_one(
    record_ids: set[int],
    plans: dict[int, RetainedPrefixPlan],
    predicate,
) -> int:
    matching = sorted(
        record_id
        for record_id in record_ids
        if predicate(plans[record_id])
    )
    if not matching:
        raise RuntimeError("Stage 4.9 smoke category has no qualifying record")
    return matching[0]


def _choose_smoke_ids(
    plans: dict[int, RetainedPrefixPlan],
    selection,
) -> tuple[int, ...]:
    migrate_ids = set(selection.migrate_ids)
    scheduled_ids = set(selection.scheduled_exact_ids)
    natural_ids = set(selection.natural_exact_ids)
    chosen = [
        _pick_one(
            migrate_ids,
            plans,
            lambda value: value.delta_tokens > 0
            and value.evicted_tokens > 0,
        ),
        _pick_one(
            scheduled_ids,
            plans,
            lambda value: value.delta_tokens > 0,
        ),
        _pick_one(
            natural_ids,
            plans,
            lambda value: value.target_prefix_tokens > 0
            and not value.timed_retained_rebuild,
        ),
        _pick_one(
            natural_ids,
            plans,
            lambda value: value.timed_retained_rebuild,
        ),
    ]
    remaining = sorted(
        record_id
        for record_id in migrate_ids
        if record_id not in set(chosen)
        and plans[record_id].retained_tokens > 0
    )
    if not remaining:
        raise RuntimeError("Stage 4.9 smoke lacks a second migrant")
    chosen.append(remaining[0])
    short = sorted(
        record_id
        for record_id in natural_ids
        if plans[record_id].status == "short_no_prefix"
    )
    if short:
        chosen.append(short[0])
    return tuple(sorted(chosen))


def _segment_batch(
    records: list,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(records) != len(starts) or len(records) != len(stops):
        raise ValueError("Stage 4.9 segment rows differ")
    lengths = tuple(stop - start for start, stop in zip(starts, stops, strict=True))
    if any(length < 0 for length in lengths):
        raise ValueError("Stage 4.9 segment range is invalid")
    width = max(lengths, default=0)
    item_ids = torch.zeros(
        (len(records), width),
        dtype=torch.long,
        device=device,
    )
    behaviors = torch.zeros_like(item_ids)
    time_deltas = torch.zeros(
        (len(records), width),
        dtype=torch.float32,
        device=device,
    )
    for row, (record, start, stop) in enumerate(
        zip(records, starts, stops, strict=True)
    ):
        if record.history is None or stop > len(record.history):
            raise ValueError("Stage 4.9 segment exceeds target history")
        length = stop - start
        if length:
            item_ids[row, :length] = torch.tensor(
                record.history.item_ids[start:stop].copy(),
                dtype=torch.long,
                device=device,
            )
            behaviors[row, :length] = torch.tensor(
                record.history.behaviors[start:stop].copy(),
                dtype=torch.long,
                device=device,
            )
            time_deltas[row, :length] = torch.tensor(
                record.history.time_deltas[start:stop].copy(),
                dtype=torch.float32,
                device=device,
            )
    return {
        "item_ids": item_ids,
        "behaviors": behaviors,
        "time_deltas": time_deltas,
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }


@torch.inference_mode()
def _exact_cache(
    model,
    batch: dict[str, torch.Tensor],
    record_ids: tuple[int, ...],
    version: int,
    dtype: torch.dtype,
) -> JaggedMigratedKVBatch:
    cache = model.compute_kv(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        lengths=batch["lengths"],
    )
    return pack_padded_cache(
        cache,
        batch["lengths"],
        record_ids,
        f"theta{version}",
        f"theta{version}",
        dtype=dtype,
    )


@torch.inference_mode()
def _exact_full(
    model,
    batch: dict[str, torch.Tensor],
    record_ids: tuple[int, ...],
    version: int,
    dtype: torch.dtype,
) -> tuple[JaggedMigratedKVBatch, torch.Tensor]:
    hidden, cache = model(
        batch["item_ids"],
        batch["behaviors"],
        batch["time_deltas"],
        return_kv=True,
        lengths=batch["lengths"],
    )
    if cache is None:
        raise RuntimeError("Stage 4.9 exact full replay returned no K/V")
    return (
        pack_padded_cache(
            cache,
            batch["lengths"],
            record_ids,
            f"theta{version}",
            f"theta{version}",
            dtype=dtype,
        ),
        model.last_hidden(hidden, batch["lengths"]),
    )


def _records_for_ids(
    record_ids: tuple[int, ...],
    record_by_id: dict[int, dict],
    window,
) -> list:
    return [
        window.records[int(record_by_id[record_id]["user_id"])]
        for record_id in record_ids
    ]


def _retained_batch(
    record_ids: tuple[int, ...],
    plans: dict[int, RetainedPrefixPlan],
    record_by_id: dict[int, dict],
    target_window,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    records = _records_for_ids(record_ids, record_by_id, target_window)
    return _segment_batch(
        records,
        (0,) * len(records),
        tuple(plans[value].retained_tokens for value in record_ids),
        device,
    )


def _append_delta(
    model,
    cache: JaggedMigratedKVBatch,
    plans: dict[int, RetainedPrefixPlan],
    record_by_id: dict[int, dict],
    target_window,
    device: torch.device,
    dtype: torch.dtype,
):
    records = _records_for_ids(cache.record_ids, record_by_id, target_window)
    batch = _segment_batch(
        records,
        tuple(plans[value].delta_start for value in cache.record_ids),
        tuple(plans[value].target_prefix_tokens for value in cache.record_ids),
        device,
    )
    result, elapsed = timed_cuda(
        partial(
            append_jagged_suffix,
            model,
            base.identity_jagged_slice(cache),
            batch["item_ids"],
            batch["behaviors"],
            batch["time_deltas"],
            batch["lengths"],
            dtype=dtype,
        ),
        device,
    )
    return result.cache, elapsed


def _build_natural_prefix(
    model,
    record_ids: tuple[int, ...],
    plans: dict[int, RetainedPrefixPlan],
    record_by_id: dict[int, dict],
    target_window,
    target_version: int,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
):
    records = _records_for_ids(record_ids, record_by_id, target_window)
    batch = _segment_batch(
        records,
        (0,) * len(records),
        tuple(plans[value].target_prefix_tokens for value in record_ids),
        device,
    )
    if bool(torch.any(batch["lengths"] < 1)):
        raise RuntimeError("Stage 4.9 smoke natural prefix is empty")
    result, elapsed = timed_cuda(
        partial(
            append_jagged_suffix,
            model,
            base.empty_jagged_slice(
                record_ids,
                target_version,
                cfg.num_layers,
                cfg.num_heads * cfg.head_dim,
                dtype,
                device,
            ),
            batch["item_ids"],
            batch["behaviors"],
            batch["time_deltas"],
            batch["lengths"],
            dtype=dtype,
        ),
        device,
    )
    return result.cache, elapsed


def _append_latest(
    model,
    prefix: JaggedMigratedKVBatch,
    record_by_id: dict[int, dict],
    target_window,
    device: torch.device,
    dtype: torch.dtype,
):
    records = _records_for_ids(prefix.record_ids, record_by_id, target_window)
    suffix = base._suffix_batch(records, device)
    result, elapsed = timed_cuda(
        partial(
            append_jagged_suffix,
            model,
            base.identity_jagged_slice(prefix),
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
            suffix["lengths"],
            dtype=dtype,
        ),
        device,
    )
    if result.last_appended_hidden is None:
        raise RuntimeError("Stage 4.9 latest append returned no hidden")
    return result.cache, result.last_appended_hidden, elapsed


def _append_fresh_latest(
    model,
    record_ids: tuple[int, ...],
    record_by_id: dict[int, dict],
    target_window,
    target_version: int,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
):
    records = _records_for_ids(record_ids, record_by_id, target_window)
    suffix = base._suffix_batch(records, device)
    result, elapsed = timed_cuda(
        partial(
            append_jagged_suffix,
            model,
            base.empty_jagged_slice(
                record_ids,
                target_version,
                cfg.num_layers,
                cfg.num_heads * cfg.head_dim,
                dtype,
                device,
            ),
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
            suffix["lengths"],
            dtype=dtype,
        ),
        device,
    )
    if result.last_appended_hidden is None:
        raise RuntimeError("Stage 4.9 fresh latest append returned no hidden")
    return result.cache, result.last_appended_hidden, elapsed


def _merge_final_outputs(
    record_ids: tuple[int, ...],
    sources: tuple[tuple[JaggedMigratedKVBatch, torch.Tensor], ...],
    target_version: int,
    device: torch.device,
):
    cache, elapsed = stage48._target_prefix(
        record_ids,
        tuple(value[0] for value in sources),
        target_version,
        device,
    )
    hidden_by_id = {
        record_id: hidden[row]
        for source, hidden in sources
        for row, record_id in enumerate(source.record_ids)
    }
    if set(hidden_by_id) != set(record_ids):
        raise RuntimeError("Stage 4.9 final hidden coverage differs")
    hidden = torch.stack([hidden_by_id[value] for value in record_ids])
    return cache, hidden, elapsed


def _tensor_hash(cache: JaggedMigratedKVBatch) -> str:
    digest = hashlib.sha256()
    digest.update(cache.k.detach().cpu().contiguous().numpy().tobytes())
    digest.update(cache.v.detach().cpu().contiguous().numpy().tobytes())
    digest.update(cache.lengths.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    return float((left.float() - right.float()).abs().max().item())


def rollout_cost_summary(
    mixed_components: dict[str, float],
    paired_exact_retained_ms: float,
    excluded_components: dict[str, float],
    one_shot_final_exact_ms: float,
) -> dict[str, object]:
    required_mixed = {
        "retained_crop_ms",
        "retained_transform_ms",
        "scheduled_exact_retained_ms",
        "missing_exact_retained_ms",
        "retained_materialization_ms",
    }
    required_excluded = {
        "mixed_target_delta_append_ms",
        "mixed_target_prefix_assembly_ms",
        "mixed_latest_append_ms",
        "mixed_final_split_ms",
        "exact_target_delta_append_ms",
        "exact_target_prefix_assembly_ms",
        "exact_latest_append_ms",
        "mixed_natural_target_prefix_build_ms",
        "exact_natural_target_prefix_build_ms",
    }
    values = [
        *mixed_components.values(),
        paired_exact_retained_ms,
        *excluded_components.values(),
        one_shot_final_exact_ms,
    ]
    if (
        set(mixed_components) != required_mixed
        or set(excluded_components) != required_excluded
        or any(not math.isfinite(value) or value < 0 for value in values)
        or paired_exact_retained_ms <= 0
        or one_shot_final_exact_ms <= 0
    ):
        raise ValueError("Stage 4.9 rollout cost ledger is invalid")
    mixed_u = sum(mixed_components.values())
    mixed_append = sum(
        value
        for name, value in excluded_components.items()
        if name.startswith("mixed_")
    )
    return {
        "timed_retained_repair": {
            **mixed_components,
            "mixed_u_ms": mixed_u,
            "paired_exact_e_ms": paired_exact_retained_ms,
            "primary_u_over_e": mixed_u / paired_exact_retained_ms,
        },
        "outside_rollout_timer": excluded_components,
        "target_append_excluded_from_u_and_e": True,
        "source_model_append_in_rollout": False,
        "diagnostic_final_state_ready": {
            "mixed_ms": mixed_u + mixed_append,
            "best_one_shot_exact_ms": one_shot_final_exact_ms,
            "mixed_over_one_shot_exact": (
                mixed_u + mixed_append
            )
            / one_shot_final_exact_ms,
            "is_migration_speedup": False,
        },
    }


def smoke_payload(args: argparse.Namespace) -> dict[str, object]:
    baseline = stage48.load_exact_baseline(args.baseline)
    reusable = plan_retained_prefix(
        0,
        1,
        ("A", "B", "C", "D"),
        ("C", "D", "E", "F"),
        "old",
        "target",
        True,
        True,
    )
    cold = plan_retained_prefix(
        1,
        2,
        ("A", "B"),
        ("A", "B", "C"),
        "old",
        "target",
        False,
        False,
    )
    missing = plan_retained_prefix(
        2,
        3,
        ("A", "B", "C", "D"),
        ("C", "D", "E", "F"),
        "old",
        "target",
        True,
        False,
    )
    short = plan_retained_prefix(
        3,
        4,
        (),
        ("A",),
        None,
        "target",
        False,
        False,
    )
    cost = rollout_cost_summary(
        {
            "retained_crop_ms": 1.0,
            "retained_transform_ms": 2.0,
            "scheduled_exact_retained_ms": 3.0,
            "missing_exact_retained_ms": 0.0,
            "retained_materialization_ms": 0.5,
        },
        10.0,
        {
            "mixed_target_delta_append_ms": 20.0,
            "mixed_target_prefix_assembly_ms": 1.0,
            "mixed_latest_append_ms": 2.0,
            "mixed_final_split_ms": 0.5,
            "exact_target_delta_append_ms": 21.0,
            "exact_target_prefix_assembly_ms": 1.0,
            "exact_latest_append_ms": 2.0,
            "mixed_natural_target_prefix_build_ms": 4.0,
            "exact_natural_target_prefix_build_ms": 4.5,
        },
        30.0,
    )
    checks = {
        "reusable_plan": reusable.migration_eligible,
        "retained_then_delta_then_latest": (
            reusable.retained_tokens,
            reusable.delta_tokens,
            reusable.latest_tokens,
        )
        == (2, 1, 1),
        "cold_has_no_retained_source": not cold.migration_eligible
        and cold.retained_tokens == 0,
        "missing_cache_requires_timed_retained_rebuild": (
            missing.status == "missing_cache"
            and missing.timed_retained_rebuild
            and missing.retained_tokens == 2
            and missing.delta_tokens == 1
        ),
        "latest_only_has_no_prefix": (
            short.status == "short_no_prefix"
            and short.target_prefix_tokens == 0
            and short.latest_tokens == 1
        ),
        "append_excluded_from_primary": cost[
            "target_append_excluded_from_u_and_e"
        ]
        is True
        and cost["timed_retained_repair"]["mixed_u_ms"] == 6.5,
        "formal_execution_disabled": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 4.9 static smoke failed: {checks}")
    return {
        "protocol": SMOKE_PROTOCOL,
        "status": "smoke_passed",
        "scientific_result": False,
        "formal_result_written": False,
        "candidate": _candidate_spec(args.candidate).to_dict(),
        "implementation": implementation_snapshot(),
        "baseline": {
            "path": args.baseline,
            "sha256": sha256(_repo_path(args.baseline)),
            "used_for_provenance_only": True,
            "old_exact_denominator_reused": False,
            "source_artifacts_verified": baseline["status"] == "complete",
        },
        "rollout_abi": reusable.protocol,
        "cost_contract": cost,
        "checks": checks,
    }


@torch.inference_mode()
def runtime_smoke_payload(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    baseline = stage48.load_exact_baseline(args.baseline)
    plan, metadata, training, cfg, manifest, checkpoints = load_inputs(
        args.prepared_data,
        args.training_result,
        args.checkpoint_dir,
    )
    user_ids = tuple(int(value["user_id"]) for value in manifest["records"])
    windows = reconstruct_organic_windows(plan, user_ids)
    compiler = json.loads(_repo_path(args.compiler_result).read_text())
    window_checks = base.validate_windows(windows, manifest)
    compiler_checks = base.validate_compiler_payload(
        compiler,
        manifest,
        windows,
        checkpoints,
    )
    provenance_checks = stage48.validate_runtime_provenance(
        args,
        baseline,
        metadata,
        training,
        manifest,
        checkpoints,
        windows,
        compiler,
    )
    if torch.cuda.get_device_name(device) != baseline["configuration"][
        "device_class"
    ]:
        raise ValueError("Stage 4.9 runtime smoke device differs")
    old_window = windows[0]
    target_window = windows[1]
    record_by_id = {
        int(value["record_id"]): value for value in manifest["records"]
    }
    previous_expected_ids = {
        record_id
        for record_id, descriptor in record_by_id.items()
        if old_window.records[int(descriptor["user_id"])].history is not None
    }
    previous_present_ids = set(previous_expected_ids)
    provisional_plans, _ = _plan_edge(
        old_window,
        target_window,
        manifest["records"],
        previous_expected_ids,
        previous_present_ids,
    )
    injected_missing_id = min(
        record_id
        for record_id, value in provisional_plans.items()
        if value.migration_eligible
    )
    previous_present_ids.remove(injected_missing_id)
    plans, plan_checks = _plan_edge(
        old_window,
        target_window,
        manifest["records"],
        previous_expected_ids,
        previous_present_ids,
    )
    last_exact = {record_id: 0 for record_id in previous_present_ids}
    spec = _candidate_spec(args.candidate)
    selection, scheduler_checks = _select_actions(
        plans,
        last_exact,
        0,
        1,
        spec,
        None,
    )
    selected_ids = _choose_smoke_ids(plans, selection)
    selected_records = _records_for_ids(
        selected_ids,
        record_by_id,
        target_window,
    )
    migrate_ids = tuple(
        value for value in selected_ids if value in set(selection.migrate_ids)
    )
    scheduled_ids = tuple(
        value
        for value in selected_ids
        if value in set(selection.scheduled_exact_ids)
    )
    natural_ids = tuple(
        value
        for value in selected_ids
        if value in set(selection.natural_exact_ids)
    )
    missing_ids = tuple(
        value for value in natural_ids if plans[value].timed_retained_rebuild
    )
    short_ids = tuple(
        value
        for value in natural_ids
        if plans[value].status == "short_no_prefix"
    )
    natural_prefix_ids = tuple(
        value
        for value in natural_ids
        if value not in set(missing_ids)
        and value not in set(short_ids)
    )
    source_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        0,
        device,
    )
    migrate_old_records = _records_for_ids(
        migrate_ids,
        record_by_id,
        old_window,
    )
    old_batch = base._history_batch(
        migrate_old_records,
        cfg.max_seq_len,
        device,
        prefix=False,
    )
    (old_migrant_cache, _), initialization_ms = timed_cuda(
        partial(
            base._exact_full_batch,
            source_model,
            old_batch,
            migrate_ids,
            0,
        ),
        device,
    )
    source_hash = _tensor_hash(old_migrant_cache)
    del source_model, old_batch
    gc.collect()
    torch.cuda.empty_cache()
    operator = DirectOldKVFusedOperator(**LAUNCH)
    program, program_descriptor, program_cpu = base._load_program(
        args,
        cfg,
        compiler,
        0,
        device,
        operator,
    )
    del program_cpu
    target_model = load_checkpoint_model(
        cfg,
        args.checkpoint_dir,
        1,
        device,
    )
    sliced, crop_ms = timed_cuda(
        partial(
            tail_slice_jagged_cache,
            old_migrant_cache,
            tuple(plans[value].retained_tokens for value in migrate_ids),
        ),
        device,
    )
    if sliced.cache is None:
        raise RuntimeError("Stage 4.9 smoke migrant retained prefix is empty")
    migrated, transform_ms = timed_cuda(
        partial(
            execute_direct,
            operator,
            program,
            sliced.cache,
            1,
        ),
        device,
    )
    scheduled_batch = _retained_batch(
        scheduled_ids,
        plans,
        record_by_id,
        target_window,
        device,
    )
    scheduled_exact, scheduled_exact_ms = timed_cuda(
        partial(
            _exact_cache,
            target_model,
            scheduled_batch,
            scheduled_ids,
            1,
            torch.float16,
        ),
        device,
    )
    missing_batch = _retained_batch(
        missing_ids,
        plans,
        record_by_id,
        target_window,
        device,
    )
    missing_exact, missing_exact_ms = timed_cuda(
        partial(
            _exact_cache,
            target_model,
            missing_batch,
            missing_ids,
            1,
            torch.float16,
        ),
        device,
    )
    timed_retained_ids = tuple(
        sorted((*migrate_ids, *scheduled_ids, *missing_ids))
    )
    exact_retained_batch = _retained_batch(
        timed_retained_ids,
        plans,
        record_by_id,
        target_window,
        device,
    )
    paired_exact_retained, paired_exact_ms = timed_cuda(
        partial(
            _exact_cache,
            target_model,
            exact_retained_batch,
            timed_retained_ids,
            1,
            torch.float16,
        ),
        device,
    )
    parity_exact_retained, parity_retained_ms = timed_cuda(
        partial(
            _exact_cache,
            target_model,
            exact_retained_batch,
            timed_retained_ids,
            1,
            torch.float32,
        ),
        device,
    )
    migrated_delta, migrated_delta_ms = _append_delta(
        target_model,
        migrated,
        plans,
        record_by_id,
        target_window,
        device,
        torch.float16,
    )
    scheduled_delta, scheduled_delta_ms = _append_delta(
        target_model,
        scheduled_exact,
        plans,
        record_by_id,
        target_window,
        device,
        torch.float16,
    )
    missing_delta, missing_delta_ms = _append_delta(
        target_model,
        missing_exact,
        plans,
        record_by_id,
        target_window,
        device,
        torch.float16,
    )
    natural_prefix, natural_prefix_ms = _build_natural_prefix(
        target_model,
        natural_prefix_ids,
        plans,
        record_by_id,
        target_window,
        1,
        cfg,
        device,
        torch.float16,
    )
    nonshort_ids = tuple(
        value for value in selected_ids if value not in set(short_ids)
    )
    mixed_prefix, mixed_assembly_ms = stage48._target_prefix(
        nonshort_ids,
        (
            migrated_delta,
            scheduled_delta,
            missing_delta,
            natural_prefix,
        ),
        1,
        device,
    )
    mixed_nonshort, mixed_nonshort_hidden, mixed_latest_ms = _append_latest(
        target_model,
        mixed_prefix,
        record_by_id,
        target_window,
        device,
        torch.float16,
    )
    mixed_final_merge_ms = 0.0
    mixed_short_latest_ms = 0.0
    if short_ids:
        (
            mixed_short,
            mixed_short_hidden,
            mixed_short_latest_ms,
        ) = _append_fresh_latest(
            target_model,
            short_ids,
            record_by_id,
            target_window,
            1,
            cfg,
            device,
            torch.float16,
        )
        mixed_full, mixed_hidden, mixed_final_merge_ms = (
            _merge_final_outputs(
                selected_ids,
                (
                    (mixed_nonshort, mixed_nonshort_hidden),
                    (mixed_short, mixed_short_hidden),
                ),
                1,
                device,
            )
        )
    else:
        mixed_full = mixed_nonshort
        mixed_hidden = mixed_nonshort_hidden
    mixed_split, mixed_split_ms = timed_cuda(
        partial(base._split_cache, mixed_full),
        device,
    )
    exact_delta, exact_delta_ms = _append_delta(
        target_model,
        paired_exact_retained,
        plans,
        record_by_id,
        target_window,
        device,
        torch.float16,
    )
    exact_natural_prefix, exact_natural_ms = _build_natural_prefix(
        target_model,
        natural_prefix_ids,
        plans,
        record_by_id,
        target_window,
        1,
        cfg,
        device,
        torch.float16,
    )
    exact_prefix, exact_assembly_ms = stage48._target_prefix(
        nonshort_ids,
        (exact_delta, exact_natural_prefix),
        1,
        device,
    )
    (
        exact_cost_nonshort,
        exact_cost_nonshort_hidden,
        exact_latest_ms,
    ) = _append_latest(
        target_model,
        exact_prefix,
        record_by_id,
        target_window,
        device,
        torch.float16,
    )
    exact_final_merge_ms = 0.0
    exact_short_latest_ms = 0.0
    if short_ids:
        (
            exact_cost_short,
            exact_cost_short_hidden,
            exact_short_latest_ms,
        ) = _append_fresh_latest(
            target_model,
            short_ids,
            record_by_id,
            target_window,
            1,
            cfg,
            device,
            torch.float16,
        )
        exact_cost_full, _, exact_final_merge_ms = _merge_final_outputs(
            selected_ids,
            (
                (exact_cost_nonshort, exact_cost_nonshort_hidden),
                (exact_cost_short, exact_cost_short_hidden),
            ),
            1,
            device,
        )
    else:
        exact_cost_full = exact_cost_nonshort
    parity_delta, parity_delta_ms = _append_delta(
        target_model,
        parity_exact_retained,
        plans,
        record_by_id,
        target_window,
        device,
        torch.float32,
    )
    parity_natural_prefix, parity_natural_ms = _build_natural_prefix(
        target_model,
        natural_prefix_ids,
        plans,
        record_by_id,
        target_window,
        1,
        cfg,
        device,
        torch.float32,
    )
    parity_prefix, parity_assembly_ms = stage48._target_prefix(
        nonshort_ids,
        (parity_delta, parity_natural_prefix),
        1,
        device,
    )
    (
        parity_nonshort,
        parity_nonshort_hidden,
        parity_latest_ms,
    ) = _append_latest(
        target_model,
        parity_prefix,
        record_by_id,
        target_window,
        device,
        torch.float32,
    )
    parity_final_merge_ms = 0.0
    parity_short_latest_ms = 0.0
    if short_ids:
        (
            parity_short,
            parity_short_hidden,
            parity_short_latest_ms,
        ) = _append_fresh_latest(
            target_model,
            short_ids,
            record_by_id,
            target_window,
            1,
            cfg,
            device,
            torch.float32,
        )
        exact_two_stage, exact_hidden, parity_final_merge_ms = (
            _merge_final_outputs(
                selected_ids,
                (
                    (parity_nonshort, parity_nonshort_hidden),
                    (parity_short, parity_short_hidden),
                ),
                1,
                device,
            )
        )
    else:
        exact_two_stage = parity_nonshort
        exact_hidden = parity_nonshort_hidden
    full_batch = base._history_batch(
        selected_records,
        cfg.max_seq_len,
        device,
        prefix=False,
    )
    (exact_one_shot_cost, _), one_shot_ms = timed_cuda(
        partial(
            _exact_full,
            target_model,
            full_batch,
            selected_ids,
            1,
            torch.float16,
        ),
        device,
    )
    exact_cost_split, one_shot_split_ms = timed_cuda(
        partial(base._split_cache, exact_one_shot_cost),
        device,
    )
    (exact_one_shot, one_shot_hidden), parity_one_shot_ms = timed_cuda(
        partial(
            _exact_full,
            target_model,
            full_batch,
            selected_ids,
            1,
            torch.float32,
        ),
        device,
    )
    all_items = torch.arange(
        1,
        cfg.num_prediction_items + 1,
        device=device,
    )
    exact_two_stage_scores = target_model.item_emb.score(
        exact_hidden,
        all_items,
    )
    one_shot_scores = target_model.item_emb.score(
        one_shot_hidden,
        all_items,
    )
    mixed_scores = target_model.item_emb.score(mixed_hidden, all_items)
    topk = min(100, all_items.numel())
    exact_cache_k_max = _max_abs(exact_two_stage.k, exact_one_shot.k)
    exact_cache_v_max = _max_abs(exact_two_stage.v, exact_one_shot.v)
    exact_hidden_max = _max_abs(exact_hidden, one_shot_hidden)
    exact_score_max = _max_abs(exact_two_stage_scores, one_shot_scores)
    population_sha = retained_population_sha256(
        [plans[value] for value in timed_retained_ids]
    )
    actual_retained_lengths = {
        record_id: int(source.lengths[row])
        for source in (migrated, scheduled_exact, missing_exact)
        for row, record_id in enumerate(source.record_ids)
    }
    mixed_components = {
        "retained_crop_ms": crop_ms,
        "retained_transform_ms": transform_ms,
        "scheduled_exact_retained_ms": scheduled_exact_ms,
        "missing_exact_retained_ms": missing_exact_ms,
        "retained_materialization_ms": 0.0,
    }
    excluded_components = {
        "mixed_target_delta_append_ms": (
            migrated_delta_ms + scheduled_delta_ms + missing_delta_ms
        ),
        "mixed_target_prefix_assembly_ms": mixed_assembly_ms,
        "mixed_latest_append_ms": (
            mixed_latest_ms + mixed_short_latest_ms
        ),
        "mixed_final_split_ms": mixed_split_ms + mixed_final_merge_ms,
        "exact_target_delta_append_ms": exact_delta_ms,
        "exact_target_prefix_assembly_ms": (
            exact_assembly_ms + exact_final_merge_ms
        ),
        "exact_latest_append_ms": exact_latest_ms + exact_short_latest_ms,
        "mixed_natural_target_prefix_build_ms": natural_prefix_ms,
        "exact_natural_target_prefix_build_ms": exact_natural_ms,
    }
    cost = rollout_cost_summary(
        mixed_components,
        paired_exact_ms,
        excluded_components,
        one_shot_ms + one_shot_split_ms,
    )
    selected_actions = {
        record_id: (
            "migrate"
            if record_id in migrate_ids
            else (
                "scheduled_exact"
                if record_id in scheduled_ids
                else (
                    "missing_cache_exact"
                    if record_id in missing_ids
                    else "natural_exact"
                )
            )
        )
        for record_id in selected_ids
    }
    lineage = [
        {
            **plans[record_id].to_dict(),
            "action": selected_actions[record_id],
            "previous_overlap_available": (
                plans[record_id].potential_overlap_tokens > 0
            ),
            "previous_kv_used_for_retained_repair": record_id in migrate_ids,
            "previous_kv_discarded_for_exact": record_id in scheduled_ids,
            "append_model_version": 1,
            "last_exact_version_before": 0,
            "last_exact_version_after": (
                0 if record_id in migrate_ids else 1
            ),
            "state_kind_after": (
                "migrated_retained_plus_target_delta"
                if record_id in migrate_ids
                else "exact"
            ),
        }
        for record_id in selected_ids
    ]
    checks = {
        "causality": all(window_checks.values()),
        "compiler": all(compiler_checks.values()),
        "provenance": all(provenance_checks.values()),
        "retained_plan": all(plan_checks.values()),
        "scheduler": all(scheduler_checks.values()),
        "covers_migrate_scheduled_natural": bool(migrate_ids)
        and bool(scheduled_ids)
        and bool(natural_ids),
        "covers_injected_missing_cache": missing_ids
        == (injected_missing_id,),
        "source_model_append_calls_zero": True,
        "all_append_models_are_target": all(
            value["append_model_version"] == 1 for value in lineage
        ),
        "timed_population_matches": (
            set(actual_retained_lengths) == set(timed_retained_ids)
            and all(
                actual_retained_lengths[value]
                == plans[value].retained_tokens
                for value in timed_retained_ids
            )
            and paired_exact_retained.record_ids == timed_retained_ids
            and tuple(int(value) for value in paired_exact_retained.lengths)
            == tuple(
                plans[value].retained_tokens
                for value in timed_retained_ids
            )
            and population_sha
            == retained_population_sha256(
                [plans[value] for value in actual_retained_lengths]
            )
        ),
        "retained_endpoint_dtype_matches": all(
            value.k.dtype == torch.float16
            and value.v.dtype == torch.float16
            for value in (
                migrated,
                scheduled_exact,
                missing_exact,
                paired_exact_retained,
            )
        ),
        "missing_rebuild_charged_to_u": missing_exact_ms > 0
        and mixed_components["missing_exact_retained_ms"]
        == missing_exact_ms,
        "exact_two_stage_k_matches_one_shot": exact_cache_k_max <= 1e-4,
        "exact_two_stage_v_matches_one_shot": exact_cache_v_max <= 1e-4,
        "exact_two_stage_hidden_matches_one_shot": exact_hidden_max <= 1e-4,
        "exact_two_stage_scores_match_one_shot": exact_score_max <= 1e-4,
        "exact_top100_matches_one_shot": torch.equal(
            torch.topk(exact_two_stage_scores, k=topk, dim=1).indices,
            torch.topk(one_shot_scores, k=topk, dim=1).indices,
        ),
        "mixed_cache_covers_selected": set(mixed_split) == set(selected_ids),
        "mixed_lengths_match_target_history": all(
            int(mixed_split[record_id].lengths[0])
            == plans[record_id].final_tokens
            for record_id in selected_ids
        ),
        "mixed_cache_finite": bool(torch.isfinite(mixed_full.k).all())
        and bool(torch.isfinite(mixed_full.v).all())
        and bool(torch.isfinite(mixed_hidden).all())
        and bool(torch.isfinite(mixed_scores).all()),
        "previous_actual_migrant_hash_bound": source_hash
        == _tensor_hash(old_migrant_cache)
        and old_migrant_cache.record_ids == migrate_ids,
        "fp16_final_cost_endpoint_matches": (
            exact_cost_full.k.dtype
            == exact_one_shot_cost.k.dtype
            == torch.float16
            and exact_cost_full.v.dtype
            == exact_one_shot_cost.v.dtype
            == torch.float16
            and exact_cost_full.record_ids
            == exact_one_shot_cost.record_ids
            == selected_ids
            and set(exact_cost_split) == set(selected_ids)
        ),
        "final_state_publication_matches": set(mixed_split)
        == set(exact_cost_split)
        == set(selected_ids),
        "target_append_excluded_from_primary": cost[
            "target_append_excluded_from_u_and_e"
        ]
        is True,
        "old_exact_denominator_not_reused": True,
        "formal_result_not_written": True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage 4.9 runtime smoke failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "runtime_smoke_passed",
        "scientific_result": False,
        "formal_result_written": False,
        "candidate": spec.to_dict(),
        "configuration": {
            "dataset": "KuaiRand-1K",
            "split": "4+12",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "source_version": 0,
            "target_version": 1,
            "selected_records": len(selected_ids),
        },
        "implementation": implementation_snapshot(),
        "input_provenance": {
            "baseline_path": args.baseline,
            "baseline_sha256": sha256(_repo_path(args.baseline)),
            "baseline_used_for_provenance_only": True,
            "old_exact_denominator_reused": False,
            "compiler_sha256": sha256(_repo_path(args.compiler_result)),
            "program_sha256": program_descriptor["sha256"],
            "target_window_sha256": target_window.content_sha256,
        },
        "population": {
            "selected_record_ids": list(selected_ids),
            "migrate_ids": list(migrate_ids),
            "scheduled_exact_ids": list(scheduled_ids),
            "natural_exact_ids": list(natural_ids),
            "missing_cache_exact_ids": list(missing_ids),
            "short_no_prefix_ids": list(short_ids),
            "injected_missing_cache_record_id": injected_missing_id,
            "retained_population_sha256": population_sha,
            "retained_tokens": sum(
                plans[value].retained_tokens
                for value in timed_retained_ids
            ),
            "delta_tokens": sum(
                plans[value].delta_tokens for value in selected_ids
            ),
            "latest_tokens": sum(
                plans[value].latest_tokens for value in selected_ids
            ),
            "full_cohort_reusable_records": sum(
                value.migration_eligible for value in plans.values()
            ),
            "full_cohort_missing_expected_cache_records": sum(
                value.missing_expected_cache for value in plans.values()
            ),
            "full_cohort_resident_records": sum(
                value.final_tokens > 0 for value in plans.values()
            ),
        },
        "timing_boundary": {
            "timed_state": "retained_prefix",
            "recursive_state": "post_append_full_cache",
            "append_order": "post_migration_target_model",
            "guard_hook": "post_retained_prefix_pre_append",
            "transaction_commit_state": "post_append_full_cache",
            "source_model_append_calls": 0,
            "retained_endpoint_dtype": "torch.float16",
            "retained_materialization": "integrated_in_action_timers",
            "final_state_publication": "per_record_split_on_both_sides",
            "kernel_warmup_repetitions": 0,
            "timing_is_scientific": False,
        },
        "cost": cost,
        "diagnostics": {
            "theta0_selected_initialization_ms": initialization_ms,
            "source_migrant_cache_sha256": source_hash,
            "exact_parity_only_ms": {
                "retained_exact": parity_retained_ms,
                "target_delta_append": parity_delta_ms,
                "natural_target_prefix_build": parity_natural_ms,
                "target_prefix_assembly": (
                    parity_assembly_ms + parity_final_merge_ms
                ),
                "latest_append": (
                    parity_latest_ms + parity_short_latest_ms
                ),
                "one_shot_exact": parity_one_shot_ms,
            },
            "one_shot_cost_publication_ms": one_shot_split_ms,
            "exact_two_stage_vs_one_shot": {
                "k_max_abs": exact_cache_k_max,
                "v_max_abs": exact_cache_v_max,
                "hidden_max_abs": exact_hidden_max,
                "score_max_abs": exact_score_max,
                "top100_equal": checks["exact_top100_matches_one_shot"],
            },
            "mixed_vs_one_shot": {
                "k_max_abs": _max_abs(mixed_full.k, exact_one_shot.k),
                "v_max_abs": _max_abs(mixed_full.v, exact_one_shot.v),
                "hidden_max_abs": _max_abs(mixed_hidden, one_shot_hidden),
                "score_max_abs": _max_abs(mixed_scores, one_shot_scores),
            },
        },
        "lineage": lineage,
        "checks": checks,
    }
    del (
        old_migrant_cache,
        sliced,
        migrated,
        scheduled_exact,
        paired_exact_retained,
        migrated_delta,
        scheduled_delta,
        natural_prefix,
        mixed_prefix,
        mixed_full,
        mixed_hidden,
        mixed_split,
        exact_delta,
        exact_natural_prefix,
        exact_prefix,
        exact_two_stage,
        exact_hidden,
        exact_one_shot,
        one_shot_hidden,
        exact_two_stage_scores,
        one_shot_scores,
        mixed_scores,
        all_items,
        program,
        target_model,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.smoke_test:
        print(json.dumps(smoke_payload(args), indent=2))
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    print(json.dumps(runtime_smoke_payload(args), indent=2))


if __name__ == "__main__":
    main()
