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
    load_plan,
    make_model,
)

CUMULATIVE_PROTOCOL = "evokv_kuairand_cumulative_staleness_v0"


def load_cumulative_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    base = document.get("base_config", {})
    if (
        document.get("protocol") != CUMULATIVE_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("source_version") != 0
        or document.get("current_version") != 2
        or document.get("evaluation_date_index") != 16
        or document.get("horizons") != [4, 8, 16, 32, 64]
        or file_sha256(base.get("path", "")) != base.get("sha256")
    ):
        raise ValueError("KuaiRand cumulative-staleness config differs")
    return document


def run_cumulative_staleness(config_path: str | Path):
    config = load_cumulative_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(config["base_config"]["path"])
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for update_index in document["evaluation"]["update_date_indices"]:
        plan.ingest_day(dates[int(update_index)])
    device = torch.device("cuda:0")
    source = make_model(document, plan, device)
    current = make_model(document, plan, device)
    root = Path(document["outputs"]["checkpoint_root"])
    _load_checkpoint(source, root, int(config["source_version"]))
    _load_checkpoint(current, root, int(config["current_version"]))
    update_date = dates[int(document["evaluation"]["update_date_indices"][-1])]
    eval_date = dates[int(config["evaluation_date_index"])]
    values = []
    started = time.monotonic()
    for horizon in config["horizons"]:
        print(f"phase=cumulative_staleness horizon={horizon}", flush=True)
        values.append(
            _evaluate_edge(
                document,
                plan,
                source,
                current,
                update_date,
                eval_date,
                int(config["current_version"]),
                device,
                max_exposures=int(horizon),
            )
        )
    result = {
        "protocol": CUMULATIVE_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "base_config": config["base_config"],
        "data": metadata,
        "source_version": config["source_version"],
        "current_version": config["current_version"],
        "values": values,
        "decision": {
            "positive_horizons": [value["max_exposures"] for value in values if value["gate"]["passed"]]
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_cumulative_staleness(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != CUMULATIVE_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or len(result.get("values", [])) != 5
        or not all(value.get("same_model_sanity_passed") for value in result["values"])
    ):
        raise ValueError("KuaiRand cumulative-staleness result differs")
