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

SWEEP_PROTOCOL = "evokv_kuairand_engagement_horizon_sweep_v0"


def load_horizon_sweep_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    base = document.get("base_config", {})
    if (
        document.get("protocol") != SWEEP_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("horizons") != [4, 8, 16, 32, 64]
        or file_sha256(base.get("path", "")) != base.get("sha256")
    ):
        raise ValueError("KuaiRand horizon sweep config differs")
    return document


def run_horizon_sweep(config_path: str | Path):
    sweep = load_horizon_sweep_config(config_path)
    output = Path(sweep["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(sweep["base_config"]["path"])
    plan, metadata = load_plan(document)
    plan.init_base()
    device = torch.device("cuda:0")
    previous = make_model(document, plan, device)
    current = make_model(document, plan, device)
    root = Path(document["outputs"]["checkpoint_root"])
    dates = plan.base_dates + plan.stream_dates
    edges = []
    started = time.monotonic()
    for edge, (update_index, eval_index) in enumerate(
        zip(
            document["evaluation"]["update_date_indices"],
            document["evaluation"]["evaluation_date_indices"],
            strict=True,
        ),
        start=1,
    ):
        update_date = dates[int(update_index)]
        eval_date = dates[int(eval_index)]
        plan.ingest_day(update_date)
        _load_checkpoint(previous, root, edge - 1)
        _load_checkpoint(current, root, edge)
        values = []
        for horizon in sweep["horizons"]:
            print(f"phase=engagement_horizon_sweep edge={edge} horizon={horizon}", flush=True)
            values.append(
                _evaluate_edge(
                    document,
                    plan,
                    previous,
                    current,
                    update_date,
                    eval_date,
                    edge,
                    device,
                    max_exposures=int(horizon),
                )
            )
        edges.append({"edge": edge, "values": values})
    common_positive_horizons = [
        horizon
        for horizon in sweep["horizons"]
        if all(
            next(
                value
                for value in edge["values"]
                if value["max_exposures"] == horizon
            )["gate"]["passed"]
            for edge in edges
        )
    ]
    result = {
        "protocol": SWEEP_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "base_config": sweep["base_config"],
        "data": metadata,
        "edges": edges,
        "decision": {"common_positive_horizons": common_positive_horizons},
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_horizon_sweep(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != SWEEP_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or len(result.get("edges", [])) != 2
        or not all(len(edge.get("values", [])) == 5 for edge in result["edges"])
        or not all(
            value.get("same_model_sanity_passed")
            for edge in result["edges"]
            for value in edge["values"]
        )
    ):
        raise ValueError("KuaiRand horizon sweep result differs")
