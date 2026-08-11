from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

PROTOCOL = "evokv_a40_population_card_hours_v0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_estimation"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or int(document.get("population_users", 0)) < 1
        or len(document.get("configurations", [])) != 5
    ):
        raise ValueError("A40 population card-hour config differs")
    source = document.get("source_result", {})
    source_path = Path(source.get("path", ""))
    if (
        not source_path.is_file()
        or file_sha256(source_path) != source.get("sha256")
    ):
        raise ValueError("A40 population card-hour source result differs")
    implementation = document.get("implementation", {})
    implementation_path = Path(implementation.get("path", ""))
    if (
        not implementation_path.is_file()
        or file_sha256(implementation_path) != implementation.get("sha256")
    ):
        raise ValueError("A40 population card-hour implementation differs")
    assumptions = document.get("assumptions", {})
    if (
        assumptions.get("kv_storage_dtype") != "fp16"
        or int(assumptions.get("kv_element_bytes", 0)) != 2
        or assumptions.get("active_kv_consumption_dtype") != "fp32"
        or int(assumptions.get("active_kv_element_bytes", 0)) != 4
        or assumptions.get("source_target_double_buffer") is not True
        or int(assumptions.get("calibration_world_size", 0)) != 2
        or int(assumptions.get("calibration_local_batch", 0)) != 8
        or not 0.0 < float(assumptions.get("sustained_efficiency", 0.0)) <= 1.0
        or not 0.0 <= float(assumptions.get("cache_headroom_fraction", -1.0)) < 1.0
    ):
        raise ValueError("A40 population card-hour assumptions differ")
    budgets = [int(value) for value in assumptions.get("cache_budgets_gib", [])]
    if budgets != sorted(set(budgets)) or budgets != [10, 20]:
        raise ValueError("A40 population card-hour budgets differ")
    identifiers = [str(value.get("id", "")) for value in document["configurations"]]
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        raise ValueError("A40 population card-hour configuration identity differs")
    return document


def select_row(
    source: dict[str, Any], matcher: dict[str, Any]
) -> dict[str, Any]:
    family = str(matcher["family"])
    candidates = source["tables"][family]
    fields = ("layers", "hidden_size", "sequence_tokens")
    selected = [
        row
        for row in candidates
        if all(int(row[field]) == int(matcher[field]) for field in fields)
    ]
    if len(selected) != 1:
        raise ValueError("A40 population card-hour source row differs")
    row = selected[0]
    if int(row["global_batch_size"]) != 16:
        raise ValueError("A40 population card-hour calibration batch differs")
    return row


def measured_timing(row: dict[str, Any]) -> dict[str, float]:
    return {
        "reuse_ms_per_global_request": float(
            row["reuse_median_ms_per_request_amortized"]
        ),
        "recompute_ms_per_global_request": float(
            row["recompute_median_ms_per_request_amortized"]
        ),
        "core_parameter_bytes": float(row["timed_core_parameter_bytes"]),
    }


def factorized_timing(
    source: dict[str, Any], estimator: dict[str, Any]
) -> dict[str, float]:
    base = measured_timing(select_row(source, estimator["base"]))
    width_from = measured_timing(select_row(source, estimator["width_from"]))
    width_to = measured_timing(select_row(source, estimator["width_to"]))
    reuse_factor = (
        width_to["reuse_ms_per_global_request"]
        / width_from["reuse_ms_per_global_request"]
    )
    recompute_factor = (
        width_to["recompute_ms_per_global_request"]
        / width_from["recompute_ms_per_global_request"]
    )
    core_factor = (
        width_to["core_parameter_bytes"] / width_from["core_parameter_bytes"]
    )
    return {
        "reuse_ms_per_global_request": base["reuse_ms_per_global_request"]
        * reuse_factor,
        "recompute_ms_per_global_request": base[
            "recompute_ms_per_global_request"
        ]
        * recompute_factor,
        "core_parameter_bytes": base["core_parameter_bytes"] * core_factor,
        "reuse_width_factor": reuse_factor,
        "recompute_width_factor": recompute_factor,
        "core_width_factor": core_factor,
    }


def effective_batch(capacity: int, saturation: int) -> int:
    admitted = min(capacity, saturation)
    if admitted < 1:
        return 0
    return 1 << int(math.floor(math.log2(admitted)))


def estimate_row(
    configuration: dict[str, Any],
    source: dict[str, Any],
    assumptions: dict[str, Any],
    population: int,
) -> dict[str, Any]:
    layers = int(configuration["layers"])
    hidden = int(configuration["hidden_size"])
    sequence = int(configuration["sequence_tokens"])
    if configuration["provenance"] == "measured":
        timing = measured_timing(select_row(source, configuration["source"]))
        estimator_detail = None
    elif configuration["provenance"] == "factorized_estimate":
        timing = factorized_timing(source, configuration["estimator"])
        estimator_detail = {
            key: timing[key]
            for key in (
                "reuse_width_factor",
                "recompute_width_factor",
                "core_width_factor",
            )
        }
    else:
        raise ValueError("A40 population card-hour provenance differs")
    element_bytes = int(assumptions["kv_element_bytes"])
    active_element_bytes = int(assumptions["active_kv_element_bytes"])
    kv_bytes = 2 * layers * sequence * hidden * element_bytes
    active_kv_bytes = 2 * layers * sequence * hidden * active_element_bytes
    world_size = int(assumptions["calibration_world_size"])
    calibration_batch = int(assumptions["calibration_local_batch"])
    headroom = float(assumptions["cache_headroom_fraction"])
    efficiency = float(assumptions["sustained_efficiency"])
    penalties = {
        int(key): float(value)
        for key, value in assumptions["small_batch_penalty"].items()
    }
    budgets = {}
    for budget_gib in assumptions["cache_budgets_gib"]:
        budget_bytes = int(budget_gib) * (1 << 30)
        resident = budget_bytes // kv_bytes
        double_buffer_capacity = int(
            budget_bytes * (1.0 - headroom)
        ) // (2 * active_kv_bytes)
        local_batch = effective_batch(double_buffer_capacity, calibration_batch)
        if local_batch not in penalties:
            raise ValueError("A40 population card-hour batch penalty differs")
        batch_penalty = penalties[local_batch]
        method_rows = {}
        for method in ("reuse", "recompute"):
            global_request_ms = timing[f"{method}_ms_per_global_request"]
            gpu_ms_per_user = global_request_ms * world_size * batch_penalty
            ideal_gpu_hours = population * gpu_ms_per_user / 3_600_000.0
            method_rows[method] = {
                "calibrated_gpu_ms_per_user": gpu_ms_per_user,
                "ideal_hot_kernel_gpu_hours": ideal_gpu_hours,
                "planning_gpu_hours": ideal_gpu_hours / efficiency,
            }
        budgets[str(budget_gib)] = {
            "budget_bytes": budget_bytes,
            "single_version_resident_users": resident,
            "double_buffer_admitted_users_with_headroom": double_buffer_capacity,
            "effective_local_batch": local_batch,
            "small_batch_penalty": batch_penalty,
            "admission_bottleneck": (
                "hbm_capacity"
                if local_batch < calibration_batch
                else "calibrated_kernel_throughput_at_local_batch_8"
            ),
            "methods": method_rows,
            "recompute_over_reuse_ratio": (
                method_rows["recompute"]["planning_gpu_hours"]
                / method_rows["reuse"]["planning_gpu_hours"]
            ),
            "planning_gpu_hours_saved_by_reuse": (
                method_rows["recompute"]["planning_gpu_hours"]
                - method_rows["reuse"]["planning_gpu_hours"]
            ),
        }
    result = {
        "id": configuration["id"],
        "label": configuration["label"],
        "provenance": configuration["provenance"],
        "geometry": {
            "layers": layers,
            "hidden_size": hidden,
            "heads": hidden // int(configuration["head_dim"]),
            "head_dim": int(configuration["head_dim"]),
            "sequence_tokens": sequence,
            "suffix_tokens": 2,
            "timed_core_parameter_bytes": int(round(timing["core_parameter_bytes"])),
        },
        "kv_capacity": {
            "formula": "2_kv_x_layers_x_sequence_tokens_x_hidden_size_x_element_bytes",
            "fp16_bytes_per_user": kv_bytes,
            "fp16_mib_per_user": kv_bytes / (1 << 20),
            "active_fp32_bytes_per_user": active_kv_bytes,
            "active_fp32_mib_per_user": active_kv_bytes / (1 << 20),
            "population_single_version_tib": population * kv_bytes / (1 << 40),
        },
        "calibration": {
            "global_batch_size": calibration_batch * world_size,
            "local_batch_size": calibration_batch,
            "reuse_ms_per_global_request": timing[
                "reuse_ms_per_global_request"
            ],
            "recompute_ms_per_global_request": timing[
                "recompute_ms_per_global_request"
            ],
        },
        "budget_scenarios": budgets,
        "method_bottleneck": {
            "reuse": (
                "launch_and_materialization"
                if sequence <= 256
                else "kv_read_concat_and_output_memory_traffic"
            ),
            "recompute": (
                "short_sequence_launch_and_dense_compute"
                if sequence <= 256
                else "full_sequence_hstu_attention_compute"
            ),
        },
    }
    if estimator_detail is not None:
        result["factorized_estimator"] = estimator_detail
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    source_path = Path(config["source_result"]["path"])
    source = json.loads(source_path.read_text())
    if (
        source.get("protocol") != "evokv_kuairand_large_hotkv_scaling_v0"
        or source.get("status") != "complete"
        or source.get("profile") != "full"
    ):
        raise ValueError("A40 population card-hour source protocol differs")
    rows = [
        estimate_row(
            configuration,
            source,
            config["assumptions"],
            int(config["population_users"]),
        )
        for configuration in config["configurations"]
    ]
    document = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "source_result": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "population_users": int(config["population_users"]),
        "hardware": config["hardware"],
        "assumptions": config["assumptions"],
        "interpretation": {
            "gpu_hour_formula": "population_users_x_gpu_ms_per_user_div_3600000",
            "gpu_ms_per_user_formula": "two_rank_wall_ms_per_global_request_x_world_size_x_small_batch_penalty",
            "planning_formula": "ideal_hot_kernel_gpu_hours_div_sustained_efficiency",
            "card_hour_not_wall_hour": True,
            "wall_hours_with_n_identical_cards": "planning_gpu_hours_div_n",
            "ten_million_caches_are_processed_in_hbm_cohorts": True,
            "host_or_storage_ingress_timed": False,
        },
        "rows": rows,
    }
    output = Path(args.output or config["output"])
    if output.is_file():
        existing = json.loads(output.read_text())
        if existing != document:
            raise ValueError("A40 population card-hour output already differs")
    else:
        atomic_json(output, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
