from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hstu_kvcache.data import load_prepared_exposure_plan
from hstu_kvcache.migration.design2_plan import (
    canonical_sha256,
    file_sha256,
)
from hstu_kvcache.migration.design3_checkpoint import (
    D3VersionCheckpoint,
    load_compact_hstu,
    resolve_version_checkpoint,
    training_model_config,
)
from hstu_kvcache.migration.organic_schedulers import (
    SchedulerRecord,
    select_work_balanced_staggered_renewal,
)
from hstu_kvcache.models import HSTUConfig
from hstu_kvcache.utils import save_json

PROTOCOL = "evokv_design3_m1_qk_adjacent_compiler_dev_v0"
ACTION_PROTOCOL = "evokv_design3_m1_qk_adjacent_action_snapshot_dev_v0"
DEFAULT_PREPARED_DATA = "data/processed/evokv_d3_m1_qk_entity_2560.npz"
DEFAULT_TRAINING_RESULT = (
    "results/system/evokv_design3_m1/"
    "qk_entity_h1536_sharded_two_version_training_seed0.json"
)
DEFAULT_CHECKPOINT_DIR = "checkpoints/evokv_design3_m1_qk_entity_h1536/seed0"
DEFAULT_COHORT_IDS = "configs/evokv_d3/m1/qk_entity_cohorts.json"
DEFAULT_ACTION_OUTPUT = (
    "configs/evokv_d3/m1/qk_entity_adjacent_action_snapshot.json"
)
DEFAULT_PROGRAM_OUTPUT = (
    "checkpoints/evokv_design3_m1_qk_entity_h1536/seed0/"
    "theta0_to_theta1_direct_oldkv_fp16.pt"
)
DEFAULT_OUTPUT = (
    "results/system/evokv_design3_m1/"
    "qk_entity_h1536_adjacent_compiler_seed0.json"
)
HISTORY_FIELDS = (
    "item_ids",
    "behaviors",
    "time_deltas",
    "timestamps",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED_DATA)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cohort-ids", default=DEFAULT_COHORT_IDS)
    parser.add_argument("--action-output", default=DEFAULT_ACTION_OUTPUT)
    parser.add_argument("--program-output", default=DEFAULT_PROGRAM_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-users", type=int, default=40)
    parser.add_argument("--benchmark-users", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--attention-weight-cap", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.fit_users < 1:
        raise ValueError("fit user count must be positive")
    if args.benchmark_users < 0:
        raise ValueError("benchmark user count cannot be negative")
    if args.horizon < 1:
        raise ValueError("scheduler horizon must be positive")
    if args.batch_size < 1 or args.max_fit_tokens < 1:
        raise ValueError("fit batch size and token cap must be positive")
    if args.attention_weight_cap <= 0 or args.ridge <= 0:
        raise ValueError("fit cap and ridge must be positive")


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def typed_array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def history_identity_sha256(
    history: dict[str, np.ndarray],
    start: int = 0,
    stop: int | None = None,
) -> str:
    stop = len(history["item_ids"]) if stop is None else stop
    if not 0 <= start <= stop <= len(history["item_ids"]):
        raise ValueError("history identity extent is invalid")
    return canonical_sha256(
        {
            "tokens": stop - start,
            "item_ids_sha256": typed_array_sha256(
                history["item_ids"][start:stop],
                "<i8",
            ),
            "behaviors_sha256": typed_array_sha256(
                history["behaviors"][start:stop],
                "<i8",
            ),
            "time_deltas_sha256": typed_array_sha256(
                history["time_deltas"][start:stop],
                "<f4",
            ),
            "timestamps_sha256": typed_array_sha256(
                history["timestamps"][start:stop],
                "<i8",
            ),
        }
    )


def copy_history(
    history: dict[str, np.ndarray],
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    output = {
        name: np.asarray(history[name][start:stop]).copy() for name in (*HISTORY_FIELDS, "labels")
    }
    lengths = {len(value) for value in output.values()}
    if lengths != {stop - start}:
        raise ValueError("history fields have inconsistent lengths")
    return output


def histories_match(
    left: dict[str, np.ndarray],
    left_start: int,
    left_stop: int,
    right: dict[str, np.ndarray],
    right_start: int,
    right_stop: int,
) -> bool:
    return all(
        np.array_equal(
            left[name][left_start:left_stop],
            right[name][right_start:right_stop],
        )
        for name in HISTORY_FIELDS
    )


def prepared_header(
    path: str | Path,
) -> tuple[dict[str, object], np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        original_user_ids = source["original_user_ids"].astype(
            np.int64,
            copy=True,
        )
    if not isinstance(metadata, dict):
        raise ValueError("prepared metadata must be a JSON object")
    selected_users = int(metadata.get("selected_users", 0))
    if selected_users < 1 or len(original_user_ids) != selected_users:
        raise ValueError("prepared user identity mapping differs")
    if len(set(original_user_ids.tolist())) != len(original_user_ids):
        raise ValueError("prepared original user identities are not unique")
    return metadata, original_user_ids


def adjacent_layout(metadata: dict[str, object]) -> dict[str, object]:
    history_length = int(metadata.get("history_length", 0))
    slide = int(metadata.get("slide", 0))
    old = list(
        metadata.get(
            "old_window_filtered_positions",
            [0, history_length],
        )
    )
    target = list(
        metadata.get(
            "target_window_filtered_positions",
            [slide, slide + history_length],
        )
    )
    if (
        history_length < 2
        or not 0 < slide < history_length
        or old != [0, history_length]
        or target != [slide, slide + history_length]
    ):
        raise ValueError("prepared adjacent history layout differs")
    retained_tokens = history_length - slide
    delta_tokens = slide - 1
    return {
        "history_tokens": history_length,
        "old_filtered_start": 0,
        "old_filtered_stop": history_length,
        "target_filtered_start": slide,
        "target_filtered_stop": slide + history_length,
        "retained_start": slide,
        "retained_tokens": retained_tokens,
        "delta_start": retained_tokens,
        "delta_tokens": delta_tokens,
        "target_prefix_tokens": history_length - 1,
        "latest_tokens": 1,
        "final_tokens": history_length,
    }


def selected_roles(
    cohort: dict[str, object],
    original_user_ids: np.ndarray,
    fit_users: int,
    benchmark_users: int,
) -> dict[str, object]:
    fit_raw = tuple(int(value) for value in cohort.get("fit_calibration_user_ids", []))
    benchmark_raw = tuple(int(value) for value in cohort.get("benchmark_user_ids", []))
    if fit_users > len(fit_raw):
        raise ValueError("requested fit users exceed the fit cohort")
    if benchmark_users == 0:
        benchmark_users = len(benchmark_raw)
    if benchmark_users > len(benchmark_raw):
        raise ValueError("requested benchmark users exceed the benchmark cohort")
    fit_raw = fit_raw[:fit_users]
    benchmark_raw = benchmark_raw[:benchmark_users]
    if set(fit_raw) & set(benchmark_raw):
        raise ValueError("fit and benchmark cohorts overlap")
    raw_to_prepared = {int(original): index + 1 for index, original in enumerate(original_user_ids)}
    missing = sorted((set(fit_raw) | set(benchmark_raw)) - set(raw_to_prepared))
    if missing:
        raise ValueError(f"cohort users are absent from prepared data: {missing[:5]}")
    return {
        "fit_original_ids": fit_raw,
        "fit_prepared_ids": tuple(raw_to_prepared[value] for value in fit_raw),
        "benchmark_original_ids": benchmark_raw,
        "benchmark_prepared_ids": tuple(raw_to_prepared[value] for value in benchmark_raw),
    }


def materialize_role_histories(
    prepared_data: str | Path,
    layout: dict[str, object],
    roles: dict[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    history_tokens = int(layout["history_tokens"])
    slide = int(layout["target_filtered_start"])
    wanted = tuple(
        dict.fromkeys(
            (
                *roles["fit_prepared_ids"],
                *roles["benchmark_prepared_ids"],
            )
        )
    )
    plan, _ = load_prepared_exposure_plan(
        prepared_data,
        max_seq_len=history_tokens,
    )
    if plan.base_dates != ["base"] or plan.stream_dates[:1] != ["window_0"]:
        raise ValueError("prepared stream lacks the adjacent base/window_0 edge")
    plan.init_base()
    old_histories: dict[int, dict[str, np.ndarray]] = {}
    for user_id in wanted:
        history = plan.user_histories[int(user_id)]
        if len(history["item_ids"]) != history_tokens:
            raise ValueError("base history does not match the old extent")
        old_histories[int(user_id)] = copy_history(
            history,
            0,
            history_tokens,
        )
    plan.ingest_day("window_0")
    target_histories: dict[int, dict[str, np.ndarray]] = {}
    for user_id in wanted:
        history = plan.user_histories[int(user_id)]
        if len(history["item_ids"]) != history_tokens + slide:
            raise ValueError("updated history does not match the target extent")
        target_histories[int(user_id)] = copy_history(
            history,
            slide,
            slide + history_tokens,
        )
    fit_samples = []
    for user_id in roles["fit_prepared_ids"]:
        history = target_histories[int(user_id)]
        fit_samples.append(
            {
                "history": {
                    **history,
                    "user_id": int(user_id),
                    "available_length_before_token_cap": history_tokens,
                    "token_truncated": False,
                },
                "pos_items": [],
            }
        )
    benchmark_histories = []
    for original_id, prepared_id in zip(
        roles["benchmark_original_ids"],
        roles["benchmark_prepared_ids"],
        strict=True,
    ):
        benchmark_histories.append(
            {
                "original_user_id": int(original_id),
                "prepared_user_id": int(prepared_id),
                "old": old_histories[int(prepared_id)],
                "target": target_histories[int(prepared_id)],
            }
        )
    return fit_samples, benchmark_histories


def build_action_records(
    benchmark_histories: list[dict[str, object]],
    layout: dict[str, object],
    horizon: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    retained_start = int(layout["retained_start"])
    retained_tokens = int(layout["retained_tokens"])
    delta_start = int(layout["delta_start"])
    delta_tokens = int(layout["delta_tokens"])
    target_prefix_tokens = int(layout["target_prefix_tokens"])
    history_tokens = int(layout["history_tokens"])
    scheduler_records = [
        SchedulerRecord(
            record_id=record_id,
            prefix_tokens=retained_tokens,
            migration_age=0,
            natural_exact=False,
        )
        for record_id in range(len(benchmark_histories))
    ]
    selection = select_work_balanced_staggered_renewal(
        scheduler_records,
        target_version=1,
        horizon=horizon,
    )
    scheduled = set(selection.scheduled_exact_ids)
    records = []
    for record_id, source in enumerate(benchmark_histories):
        old = source["old"]
        target = source["target"]
        if not histories_match(
            old,
            retained_start,
            history_tokens,
            target,
            0,
            retained_tokens,
        ):
            raise ValueError("old and target retained identities differ")
        exact = record_id in scheduled
        records.append(
            {
                "record_id": record_id,
                "prepared_user_id": int(source["prepared_user_id"]),
                "original_user_id": int(source["original_user_id"]),
                "requested_action": "exact" if exact else "compiled",
                "requested_reason": ("scheduled_exact" if exact else "migrate"),
                "old_tokens": history_tokens,
                "old_filtered_start": int(layout["old_filtered_start"]),
                "old_filtered_stop": int(layout["old_filtered_stop"]),
                "target_filtered_start": int(layout["target_filtered_start"]),
                "target_filtered_stop": int(layout["target_filtered_stop"]),
                "retained_start": retained_start,
                "retained_tokens": retained_tokens,
                "delta_start": delta_start,
                "delta_tokens": delta_tokens,
                "target_prefix_tokens": target_prefix_tokens,
                "latest_tokens": int(layout["latest_tokens"]),
                "final_tokens": int(layout["final_tokens"]),
                "last_exact_version": "theta0",
                "migration_depth_before": 0,
                "migration_depth_after": 0 if exact else 1,
                "previous_cache_expected": True,
                "previous_cache_present": True,
                "old_raw_ordinal_first": int(old["timestamps"][0] // 1000),
                "old_raw_ordinal_last": int(old["timestamps"][-1] // 1000),
                "target_raw_ordinal_first": int(target["timestamps"][0] // 1000),
                "target_raw_ordinal_last": int(target["timestamps"][-1] // 1000),
                "old_history_sha256": history_identity_sha256(old),
                "target_history_sha256": history_identity_sha256(target),
                "retained_identity_sha256": history_identity_sha256(
                    old,
                    retained_start,
                    history_tokens,
                ),
                "delta_identity_sha256": history_identity_sha256(
                    target,
                    delta_start,
                    target_prefix_tokens,
                ),
                "target_prefix_identity_sha256": (
                    history_identity_sha256(
                        target,
                        0,
                        target_prefix_tokens,
                    )
                ),
                "latest_identity_sha256": history_identity_sha256(
                    target,
                    target_prefix_tokens,
                    history_tokens,
                ),
            }
        )
    action_partition = [
        {
            "record_id": value["record_id"],
            "requested_action": value["requested_action"],
            "requested_reason": value["requested_reason"],
        }
        for value in records
    ]
    scheduler = {
        "family": "work_balanced_staggered_renewal",
        "horizon": horizon,
        "target_version": 1,
        "scheduled_exact_ids": list(selection.scheduled_exact_ids),
        "natural_exact_ids": list(selection.natural_exact_ids),
        "migrate_ids": list(selection.migrate_ids),
        "next_state": asdict(selection.next_state),
        "diagnostics": selection.diagnostics,
        "action_partition_sha256": canonical_sha256(action_partition),
        "labels_used": False,
    }
    return records, scheduler


def fit_role_summary(
    fit_samples: list[dict[str, object]],
    roles: dict[str, object],
) -> dict[str, object]:
    identities = [
        {
            "prepared_user_id": int(prepared_id),
            "original_user_id": int(original_id),
            "target_history_sha256": history_identity_sha256(sample["history"]),
        }
        for sample, original_id, prepared_id in zip(
            fit_samples,
            roles["fit_original_ids"],
            roles["fit_prepared_ids"],
            strict=True,
        )
    ]
    return {
        "records": len(identities),
        "history_tokens_per_record": (
            len(fit_samples[0]["history"]["item_ids"]) if fit_samples else 0
        ),
        "selection_sha256": canonical_sha256(identities),
        "identities": identities,
        "labels_used": False,
    }


def build_development_snapshot(
    prepared_data: str | Path,
    cohort_ids: str | Path,
    fit_users: int = 40,
    benchmark_users: int = 0,
    horizon: int = 5,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    metadata, original_user_ids = prepared_header(prepared_data)
    cohort = load_json(cohort_ids)
    layout = adjacent_layout(metadata)
    roles = selected_roles(
        cohort,
        original_user_ids,
        fit_users,
        benchmark_users,
    )
    fit_samples, benchmark_histories = materialize_role_histories(
        prepared_data,
        layout,
        roles,
    )
    records, scheduler = build_action_records(
        benchmark_histories,
        layout,
        horizon,
    )
    scheduled_exact = len(scheduler["scheduled_exact_ids"])
    natural_exact = len(scheduler["natural_exact_ids"])
    compiled = len(scheduler["migrate_ids"])
    plan = {
        "protocol": ACTION_PROTOCOL,
        "status": "development_snapshot",
        "scientific_result": False,
        "formal_design3": False,
        "artifact_role": "owner_independent_adjacent_d1_action_snapshot",
        "source_version": "theta0",
        "target_version": "theta1",
        "labels_used": False,
        "future_history_used": False,
        "prepared_data_sha256": file_sha256(prepared_data),
        "cohort_content_sha256": canonical_sha256(cohort),
        "prepared_protocol": metadata.get("protocol"),
        "cohort_protocol": cohort.get("protocol"),
        "layout": layout,
        "roles": {
            "fit": fit_role_summary(fit_samples, roles),
            "benchmark": {
                "records": len(records),
                "original_user_ids_sha256": typed_array_sha256(
                    np.asarray(roles["benchmark_original_ids"]),
                    "<i8",
                ),
                "prepared_user_ids_sha256": typed_array_sha256(
                    np.asarray(roles["benchmark_prepared_ids"]),
                    "<i8",
                ),
            },
            "disjoint": True,
        },
        "scheduler": scheduler,
        "counts": {
            "records": len(records),
            "scheduled_exact": scheduled_exact,
            "natural_exact": natural_exact,
            "exact": scheduled_exact + natural_exact,
            "compiled": compiled,
            "scheduled_exact_fraction": (scheduled_exact / len(records) if records else 0.0),
        },
        "records": records,
    }
    snapshot = {
        **plan,
        "owner_independent_plan_sha256": canonical_sha256(plan),
        "bindings": {
            "prepared_data": str(prepared_data),
            "cohort_ids": str(cohort_ids),
        },
    }
    return snapshot, fit_samples


def write_action_snapshot(
    snapshot: dict[str, object],
    path: str | Path,
    force: bool,
) -> str:
    output = Path(path)
    if output.exists():
        existing = load_json(output)
        if (
            existing.get("owner_independent_plan_sha256")
            == snapshot["owner_independent_plan_sha256"]
        ):
            return "reused"
        if not force:
            raise FileExistsError(f"action snapshot differs; pass --force to replace: {output}")
    save_json(snapshot, output)
    return "written"


def training_inputs(
    training_result: str | Path,
    checkpoint_dir: str | Path,
    prepared_data_sha256: str,
) -> tuple[
    dict[str, object],
    HSTUConfig,
    dict[str, D3VersionCheckpoint],
]:
    training = load_json(training_result)
    if training.get("status") != "complete":
        raise ValueError("two-version training result is incomplete")
    prepared = training.get("prepared_data")
    if (
        not isinstance(prepared, dict)
        or prepared.get("sha256") != prepared_data_sha256
    ):
        raise ValueError("two-version training boundary differs")
    cfg = training_model_config(training)
    checkpoints = {
        f"theta{version}": resolve_version_checkpoint(
            training,
            checkpoint_dir,
            version,
        )
        for version in (0, 1)
    }
    return (
        training,
        cfg,
        checkpoints,
    )


def compact_fit_samples(
    fit_samples: list[dict[str, object]],
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    arrays = [
        np.asarray(value["history"]["item_ids"], dtype=np.int64)
        for value in fit_samples
    ]
    if not arrays:
        raise ValueError("fit cohort is empty")
    used = np.unique(
        np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                *arrays,
            ]
        )
    )
    if used[0] != 0 or np.any(used < 0):
        raise ValueError("fit histories contain invalid item IDs")
    remapped = []
    for sample, item_ids in zip(fit_samples, arrays, strict=True):
        compact = np.searchsorted(used, item_ids)
        if not np.array_equal(used[compact], item_ids):
            raise RuntimeError("fit item compaction failed")
        remapped.append(
            {
                **sample,
                "history": {
                    **sample["history"],
                    "item_ids": compact.astype(np.int64, copy=False),
                },
            }
        )
    return tuple(int(value) for value in used), remapped


def compile_adjacent_program(
    args: argparse.Namespace,
    snapshot: dict[str, object],
    fit_samples: list[dict[str, object]],
) -> dict[str, object]:
    import torch

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from search_kuairand_long_context_attention_weighted import (
        fit_attention_family,
        mix_name,
    )

    from hstu_kvcache.migration import MigrationProgram
    from hstu_kvcache.migration.stage45_oldkv import (
        compile_direct_oldkv_program,
        load_direct_oldkv_program,
        write_direct_oldkv_program,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("adjacent program compilation requires CUDA")
    if Path(args.program_output).exists() and not args.force:
        raise FileExistsError(f"program exists; pass --force to replace: {args.program_output}")
    if Path(args.output).exists() and not args.force:
        raise FileExistsError(f"result exists; pass --force to replace: {args.output}")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    training, cfg, checkpoints = training_inputs(
        args.training_result,
        args.checkpoint_dir,
        str(snapshot["prepared_data_sha256"]),
    )
    layout = snapshot["layout"]
    if (
        cfg.max_seq_len != int(layout["history_tokens"])
        or cfg.head_dim is None
        or cfg.num_heads * cfg.head_dim != cfg.hidden_size
    ):
        raise ValueError("M1 QK model and adjacent snapshot shapes differ")
    used_item_ids, compact_samples = compact_fit_samples(
        fit_samples
    )
    source_model = load_compact_hstu(
        cfg,
        checkpoints["theta0"],
        used_item_ids,
        device,
    )
    target_model = load_compact_hstu(
        cfg,
        checkpoints["theta1"],
        used_item_ids,
        device,
    )
    fit_args = argparse.Namespace(
        seq_len=cfg.max_seq_len,
        batch_size=args.batch_size,
        max_fit_tokens=args.max_fit_tokens,
        attention_mixes=[1.0],
        attention_weight_cap=args.attention_weight_cap,
        ridge=args.ridge,
        seed=args.seed,
    )
    started = time.perf_counter()
    family, fit = fit_attention_family(
        target_model,
        source_model,
        compact_samples,
        fit_args,
        device,
    )
    parent = MigrationProgram(
        source_version="theta0",
        target_version="theta1",
        adapter=family[mix_name(1.0)],
    )
    direct, compile_metrics = compile_direct_oldkv_program(
        source_model,
        parent,
    )
    action_snapshot_sha256 = file_sha256(args.action_output)
    descriptor = write_direct_oldkv_program(
        direct,
        args.program_output,
        {
            "experiment_protocol": PROTOCOL,
            "action_snapshot_protocol": ACTION_PROTOCOL,
            "owner_independent_plan_sha256": snapshot["owner_independent_plan_sha256"],
            "action_snapshot_sha256": action_snapshot_sha256,
            "prepared_data_sha256": snapshot["prepared_data_sha256"],
            "cohort_content_sha256": snapshot["cohort_content_sha256"],
            "training_result_sha256": file_sha256(args.training_result),
            "training_protocol": training.get("protocol"),
            "source_checkpoint_sha256": (
                checkpoints["theta0"].identity_sha256
            ),
            "target_checkpoint_sha256": (
                checkpoints["theta1"].identity_sha256
            ),
            "fit_selection_sha256": snapshot["roles"]["fit"]["selection_sha256"],
            "fit_records": len(fit_samples),
            "fit_unique_embedding_rows_including_padding": len(
                used_item_ids
            ),
            "fit_embedding_loading": (
                "only rows touched by fit histories are extracted from "
                "memory-mapped checkpoint shards"
            ),
            "attention_mix": 1.0,
            "attention_weight_cap": args.attention_weight_cap,
            "max_fit_tokens_per_layer": args.max_fit_tokens,
            "ridge": args.ridge,
            "labels_used": False,
            "future_history_used": False,
            "derivation": (
                "theta0-to-theta1 label-free attention-weighted residual "
                "fit composed through the theta0 stacked K/V projection"
            ),
        },
        compile_metrics,
    )
    loaded, loaded_descriptor = load_direct_oldkv_program(
        descriptor["path"],
        expected_sha256=descriptor["sha256"],
        expected_source_version="theta0",
        expected_target_version="theta1",
        expected_num_layers=cfg.num_layers,
        expected_kv_width=cfg.num_heads * int(cfg.head_dim),
    )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "formal_design3": False,
        "artifact_role": "m1_adjacent_direct_oldkv_program",
        "action_snapshot": {
            "path": args.action_output,
            "sha256": action_snapshot_sha256,
            "owner_independent_plan_sha256": snapshot["owner_independent_plan_sha256"],
        },
        "training_result": {
            "path": args.training_result,
            "sha256": file_sha256(args.training_result),
            "protocol": training.get("protocol"),
        },
        "checkpoints": {
            name: value.descriptor()
            for name, value in checkpoints.items()
        },
        "fit": fit,
        "program": loaded_descriptor,
        "runtime_seconds": time.perf_counter() - started,
    }
    save_json(result, args.output)
    del loaded, direct, parent, family, target_model, source_model
    torch.cuda.empty_cache()
    return result


def plan_only_summary(
    args: argparse.Namespace,
    snapshot: dict[str, object],
    write_status: str,
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "status": "plan_only",
        "scientific_result": False,
        "formal_design3": False,
        "action_snapshot": {
            "path": args.action_output,
            "write_status": write_status,
            "sha256": file_sha256(args.action_output),
            "owner_independent_plan_sha256": snapshot["owner_independent_plan_sha256"],
        },
        "counts": snapshot["counts"],
        "layout": snapshot["layout"],
        "next_step": (
            "compile theta0-to-theta1 direct-old-K/V program after the "
            "two-version checkpoints are available"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    snapshot, fit_samples = build_development_snapshot(
        args.prepared_data,
        args.cohort_ids,
        fit_users=args.fit_users,
        benchmark_users=args.benchmark_users,
        horizon=args.horizon,
    )
    write_status = write_action_snapshot(
        snapshot,
        args.action_output,
        args.force,
    )
    if args.plan_only:
        print(
            json.dumps(
                plan_only_summary(args, snapshot, write_status),
                indent=2,
            ),
            flush=True,
        )
        return
    result = compile_adjacent_program(args, snapshot, fit_samples)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": args.output,
                "program": result["program"]["path"],
                "program_sha256": result["program"]["sha256"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
