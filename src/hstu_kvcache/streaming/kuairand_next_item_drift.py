from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_next_item_chain import _effective_document, _lag_pass, _source_root
from .kuairand_root_cause import (
    PROTOCOL,
    _atomic_json,
    _evaluate_edge,
    _load_checkpoint,
    _popularity_ranks,
    file_sha256,
    load_plan,
    make_model,
)

NEXT_ITEM_DRIFT_PROTOCOL = "evokv_kuairand_next_item_drift_curve_v0"


def load_next_item_drift_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    bindings = [
        source.get("config", {}),
        source.get("resume", {}),
        source.get("theta2", {}),
    ]
    if (
        document.get("protocol") != NEXT_ITEM_DRIFT_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("source_version") != 2
        or document.get("current_version") != 7
        or document.get("total_num_days") != 22
        or document.get("update_date_indices") != [16, 17, 18, 19, 20]
        or document.get("evaluation_date_index") != 21
        or document.get("source_versions") != [6, 5, 4, 3, 2]
        or document.get("evaluation_methods")
        != ["fresh_full_a", "fresh_full_b", "stale_previous", "no_prefix"]
        or int(document.get("record_limit_per_rank", 0)) < 1
        or not isinstance(document.get("cap_user_limit_to_eligible", False), bool)
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand next-item drift config differs")
    return document


def run_next_item_drift_evaluation(config_path: str | Path) -> dict[str, Any] | None:
    config = load_next_item_drift_config(config_path)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != 2:
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand next-item drift evaluation requires two ranks")
    output = Path(config["evaluation_result"])
    try:
        if output.is_file():
            result = json.loads(output.read_text())
            return result if runtime.is_primary else None
        document = _effective_document(config)
        torch.set_float32_matmul_precision("high")
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        for date_index in [14, 15, *config["update_date_indices"]]:
            plan.ingest_day(dates[int(date_index)])
        current = make_model(document, plan, runtime.device)
        _load_checkpoint(current, Path(config["checkpoint_root"]), config["current_version"])
        current.eval()
        popularity = _popularity_ranks(plan)
        started = time.perf_counter()
        lags = []
        for source_version in config["source_versions"]:
            source = make_model(document, plan, runtime.device)
            _load_checkpoint(source, _source_root(config, int(source_version)), int(source_version))
            source.eval()
            edge = _evaluate_edge(
                document,
                plan,
                source,
                current,
                int(config["current_version"]),
                dates[int(config["update_date_indices"][-1])],
                dates[int(config["evaluation_date_index"])],
                popularity,
                runtime,
            )
            if runtime.is_primary:
                lags.append(
                    {
                        "source_version": source_version,
                        "lag": int(config["current_version"]) - int(source_version),
                        "edge": edge,
                        "passed": _lag_pass(edge),
                    }
                )
            del source
            torch.cuda.empty_cache()
        if not runtime.is_primary:
            dist.barrier()
            return None
        passed_lags = [value["lag"] for value in lags if value["passed"]]
        result = {
            "protocol": NEXT_ITEM_DRIFT_PROTOCOL,
            "source_protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data": metadata,
            "lags": lags,
            "decision": {
                "passed_lags": passed_lags,
                "first_passing_lag": min(passed_lags) if passed_lags else None,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)


def validate_next_item_drift_result(result: dict[str, Any]) -> None:
    lags = result.get("lags", [])
    selected_hashes = {value["edge"]["selected_user_ids_sha256"] for value in lags}
    fresh_endpoints = {
        json.dumps(value["edge"]["endpoints"]["fresh_full_a"], sort_keys=True)
        for value in lags
    }
    if (
        result.get("protocol") != NEXT_ITEM_DRIFT_PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or [value.get("lag") for value in lags] != [1, 2, 3, 4, 5]
        or not all(value["edge"]["sanity"]["implementation_passed"] for value in lags)
        or len(selected_hashes) != 1
        or len(fresh_endpoints) != 1
    ):
        raise ValueError("KuaiRand next-item drift result differs")
