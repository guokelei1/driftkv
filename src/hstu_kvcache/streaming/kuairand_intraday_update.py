from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kuairand_engagement import (
    PROTOCOL,
    _atomic_json,
    _evaluate_edge,
    _load_checkpoint,
    _save_checkpoint,
    _train_epoch,
    file_sha256,
    load_engagement_config,
    load_plan,
    make_model,
)

INTRADAY_PROTOCOL = "evokv_kuairand_intraday_update_v0"


def load_intraday_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    base = document.get("base_config", {})
    training = document.get("training", {})
    if (
        document.get("protocol") != INTRADAY_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("split_fraction") != 0.5
        or document.get("horizons") != [4, 8, 16, 32, 64]
        or document.get("source_date_indices") != [14, 15]
        or training.get("update_epochs") != 2
        or training.get("update_lr") != 0.0002
        or training.get("batch_size") not in (8, 16)
        or file_sha256(base.get("path", "")) != base.get("sha256")
    ):
        raise ValueError("KuaiRand intraday-update config differs")
    return document


def _split_day(frame, fraction):
    ordered = frame.sort_values(["time_ms", "user_idx"]).reset_index(drop=True)
    boundary = int(ordered.iloc[int(np.floor(fraction * len(ordered)))]["time_ms"])
    update = ordered[ordered["time_ms"] < boundary].copy()
    evaluation = ordered[ordered["time_ms"] >= boundary].copy()
    return boundary, update, evaluation


def run_intraday_update(config_path: str | Path):
    config = load_intraday_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    base = load_engagement_config(config["base_config"]["path"])
    effective = json.loads(json.dumps(base))
    effective["training"].update(config["training"])
    plan, metadata = load_plan(base)
    plan.init_base()
    device = torch.device("cuda:0")
    current = make_model(base, plan, device)
    base_root = Path(base["outputs"]["checkpoint_root"])
    root = Path(config["checkpoint_root"])
    _load_checkpoint(current, base_root, 0)
    optimizer = torch.optim.AdamW(
        current.parameters(),
        lr=float(config["training"]["update_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    dates = plan.base_dates + plan.stream_dates
    edges = []
    checkpoints = []
    started = time.monotonic()
    for edge, date_index in enumerate(config["source_date_indices"], start=1):
        source_date = dates[int(date_index)]
        boundary, update_frame, evaluation_frame = _split_day(
            plan.daily_segments[source_date],
            float(config["split_fraction"]),
        )
        update_key = f"intraday_{edge}_update"
        evaluation_key = f"intraday_{edge}_evaluation"
        plan.daily_segments[update_key] = update_frame
        plan.daily_segments[evaluation_key] = evaluation_frame
        previous = make_model(base, plan, device)
        previous_root = base_root if edge == 1 else root
        _load_checkpoint(previous, previous_root, edge - 1)
        plan.ingest_day(update_key)
        epochs = []
        for epoch in range(int(config["training"]["update_epochs"])):
            np.random.seed(int(config["training"]["seed"]) + edge * 1009 + epoch)
            batches = plan.iter_train_batches(
                update_key,
                int(config["training"]["batch_size"]),
                all_chunks=True,
                bucket_by_length=True,
                pad_to_max_seq_len=False,
            )
            epochs.append(
                _train_epoch(
                    current,
                    optimizer,
                    batches,
                    effective,
                    device,
                    f"intraday_theta{edge}_e{epoch + 1}",
                )
            )
        checkpoints.append(
            _save_checkpoint(current, root, edge, config_path, metadata, epochs)
        )
        previous.eval()
        current.eval()
        values = []
        for horizon in config["horizons"]:
            print(f"phase=intraday_eval edge={edge} horizon={horizon}", flush=True)
            values.append(
                _evaluate_edge(
                    base,
                    plan,
                    previous,
                    current,
                    update_key,
                    evaluation_key,
                    edge,
                    device,
                    max_exposures=int(horizon),
                )
            )
        plan.ingest_day(evaluation_key)
        edges.append(
            {
                "edge": edge,
                "source_date": source_date,
                "boundary_time_ms": boundary,
                "update_rows": len(update_frame),
                "evaluation_rows": len(evaluation_frame),
                "training": epochs,
                "values": values,
            }
        )
    common_positive_horizons = [
        horizon
        for horizon in config["horizons"]
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
        "protocol": INTRADAY_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "base_config": config["base_config"],
        "data": metadata,
        "split_fraction": config["split_fraction"],
        "checkpoints": checkpoints,
        "edges": edges,
        "decision": {"common_positive_horizons": common_positive_horizons},
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_intraday_update(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != INTRADAY_PROTOCOL
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
        raise ValueError("KuaiRand intraday-update result differs")
