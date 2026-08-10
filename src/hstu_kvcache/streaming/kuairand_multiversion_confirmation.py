from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .kuairand_engagement import (
    PROTOCOL,
    _atomic_json,
    _evaluate_edge,
    _load_checkpoint,
    file_sha256,
    load_engagement_config,
    make_model,
)
from .kuairand_multiversion_staleness import _prepare_plan, _source_root

CONFIRMATION_PROTOCOL = "evokv_kuairand_multiversion_confirmation_v0"


def load_confirmation_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    bindings = [
        document.get("data_config", {}),
        document.get("source_round_config", {}),
        document.get("chain_round_config", {}),
    ]
    if (
        document.get("protocol") != CONFIRMATION_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("training_seed") != 14929
        or document.get("current_version") != 7
        or document.get("source_versions") != [6, 2]
        or document.get("evaluation_date_index") != 20
        or document.get("horizons") != [4, 8]
        or any(file_sha256(binding.get("path", "")) != binding.get("sha256") for binding in bindings)
    ):
        raise ValueError("KuaiRand multiversion confirmation config differs")
    return document


def _locked_pass(lag5, lag1):
    for horizon in (4, 8):
        current = next(value for value in lag5["values"] if value["max_exposures"] == horizon)
        control = next(value for value in lag1["values"] if value["max_exposures"] == horizon)
        stale = current["comparisons"]["recompute_over_reuse"]
        control_stale = control["comparisons"]["recompute_over_reuse"]
        if not (
            stale["average_precision"]["positive_direction_with_ci"]
            and stale["ndcg_at_10"]["positive_direction_with_ci"]
            and stale["ndcg_at_50"]["positive_direction_with_ci"]
            and stale["average_precision"]["relative_percent"] >= 5.0
            and stale["average_precision"]["relative_percent"]
            > control_stale["average_precision"]["relative_percent"]
        ):
            return False
    return True


def run_confirmation(config_path: str | Path):
    config = load_confirmation_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(config["data_config"]["path"])
    source_document = json.loads(Path(config["source_round_config"]["path"]).read_text())
    chain_document = json.loads(Path(config["chain_round_config"]["path"]).read_text())
    plan, metadata, source_date, boundary, update_key, evaluation_key = _prepare_plan(
        document,
        int(config["evaluation_date_index"]),
        float(config["split_fraction"]),
    )
    device = torch.device("cuda:0")
    source_root = Path(source_document["checkpoint_root"])
    chain_root = Path(chain_document["checkpoint_root"])
    current = make_model(document, plan, device)
    _load_checkpoint(current, chain_root, int(config["current_version"]))
    current.eval()
    lags = []
    started = time.monotonic()
    for source_version in config["source_versions"]:
        source = make_model(document, plan, device)
        _load_checkpoint(
            source,
            _source_root(int(source_version), source_root, chain_root),
            int(source_version),
        )
        source.eval()
        values = []
        for horizon in config["horizons"]:
            print(
                f"phase=multiversion_confirmation source={source_version} horizon={horizon}",
                flush=True,
            )
            values.append(
                _evaluate_edge(
                    document,
                    plan,
                    source,
                    current,
                    update_key,
                    evaluation_key,
                    int(config["current_version"]),
                    device,
                    max_exposures=int(horizon),
                )
            )
        lags.append(
            {
                "source_version": source_version,
                "lag": int(config["current_version"]) - int(source_version),
                "values": values,
            }
        )
    lag1 = next(value for value in lags if value["lag"] == 1)
    lag5 = next(value for value in lags if value["lag"] == 5)
    result = {
        "protocol": CONFIRMATION_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_confirmation",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "training_seed": config["training_seed"],
        "data": metadata,
        "source_date": source_date,
        "boundary_time_ms": boundary,
        "current_version": config["current_version"],
        "lags": lags,
        "decision": {"locked_confirmation_passed": _locked_pass(lag5, lag1)},
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_confirmation(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != CONFIRMATION_PROTOCOL
        or result.get("status") != "complete_development_confirmation"
        or result.get("scientific_result") is not False
        or [lag.get("source_version") for lag in result.get("lags", [])] != [6, 2]
        or not all(len(lag.get("values", [])) == 2 for lag in result["lags"])
        or not all(
            value.get("same_model_sanity_passed")
            for lag in result["lags"]
            for value in lag["values"]
        )
    ):
        raise ValueError("KuaiRand multiversion confirmation result differs")
