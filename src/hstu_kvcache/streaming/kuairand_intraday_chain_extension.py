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
from .kuairand_intraday_update import _split_day

EXTENSION_PROTOCOL = "evokv_kuairand_intraday_chain_extension_v0"


def load_extension_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data_config", {})
    source = document.get("source_round_config", {})
    if (
        document.get("protocol") != EXTENSION_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("source_version") != 2
        or document.get("source_date_indices") != [16, 17, 18, 19, 20]
        or document.get("horizons") != [4, 8, 16]
        or document.get("split_fraction") != 0.5
        or file_sha256(data.get("path", "")) != data.get("sha256")
        or file_sha256(source.get("path", "")) != source.get("sha256")
    ):
        raise ValueError("KuaiRand intraday chain-extension config differs")
    return document


def _reconstruct_source_history(plan, dates, split_fraction):
    for date_index in (14, 15):
        _, update, evaluation = _split_day(
            plan.daily_segments[dates[date_index]], split_fraction
        )
        for frame in (update, evaluation):
            for user, group in frame.groupby("user_idx"):
                plan._append_day_to_history(int(user), group)


def _edge_positive(value):
    stale = value["comparisons"]["recompute_over_reuse"]
    return (
        stale["average_precision"]["positive_direction_with_ci"]
        and stale["ndcg_at_10"]["positive_direction_with_ci"]
        and max(
            stale["average_precision"]["relative_percent"],
            stale["ndcg_at_10"]["relative_percent"],
        )
        >= 5.0
    )


def run_chain_extension(config_path: str | Path):
    config = load_extension_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    data_document = load_engagement_config(config["data_config"]["path"])
    source_document = json.loads(Path(config["source_round_config"]["path"]).read_text())
    plan, metadata = load_plan(data_document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    _reconstruct_source_history(plan, dates, float(config["split_fraction"]))
    device = torch.device("cuda:0")
    current = make_model(data_document, plan, device)
    source_root = Path(source_document["checkpoint_root"])
    root = Path(config["checkpoint_root"])
    _load_checkpoint(current, source_root, int(config["source_version"]))
    effective = json.loads(json.dumps(data_document))
    effective["training"].update(config["training"])
    optimizer = torch.optim.AdamW(
        current.parameters(),
        lr=float(config["training"]["update_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    edges = []
    checkpoints = []
    started = time.monotonic()
    for offset, date_index in enumerate(config["source_date_indices"], start=1):
        version = int(config["source_version"]) + offset
        source_date = dates[int(date_index)]
        boundary, update_frame, evaluation_frame = _split_day(
            plan.daily_segments[source_date],
            float(config["split_fraction"]),
        )
        update_key = f"extension_{version}_update"
        evaluation_key = f"extension_{version}_evaluation"
        plan.daily_segments[update_key] = update_frame
        plan.daily_segments[evaluation_key] = evaluation_frame
        previous = make_model(data_document, plan, device)
        previous_root = source_root if version == 3 else root
        _load_checkpoint(previous, previous_root, version - 1)
        plan.ingest_day(update_key)
        epochs = []
        for epoch in range(int(config["training"]["update_epochs"])):
            np.random.seed(int(config["training"]["seed"]) + version * 1009 + epoch)
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
                    f"extension_theta{version}_e{epoch + 1}",
                )
            )
        checkpoints.append(
            _save_checkpoint(
                current,
                root,
                version,
                config_path,
                metadata,
                epochs,
            )
        )
        previous.eval()
        current.eval()
        values = []
        for horizon in config["horizons"]:
            print(f"phase=chain_extension version={version} horizon={horizon}", flush=True)
            values.append(
                _evaluate_edge(
                    data_document,
                    plan,
                    previous,
                    current,
                    update_key,
                    evaluation_key,
                    version,
                    device,
                    max_exposures=int(horizon),
                )
            )
        plan.ingest_day(evaluation_key)
        edges.append(
            {
                "version": version,
                "source_date": source_date,
                "boundary_time_ms": boundary,
                "training": epochs,
                "values": values,
                "positive_horizons": [
                    value["max_exposures"] for value in values if _edge_positive(value)
                ],
            }
        )
    result = {
        "protocol": EXTENSION_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data_config": config["data_config"],
        "source_round_config": config["source_round_config"],
        "data": metadata,
        "checkpoints": checkpoints,
        "edges": edges,
        "decision": {
            "positive_versions": [edge["version"] for edge in edges if edge["positive_horizons"]],
            "all_versions_have_positive_horizon": all(edge["positive_horizons"] for edge in edges),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_chain_extension(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != EXTENSION_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or [edge.get("version") for edge in result.get("edges", [])] != [3, 4, 5, 6, 7]
        or not all(len(edge.get("values", [])) == 3 for edge in result["edges"])
        or not all(
            value.get("same_model_sanity_passed")
            for edge in result["edges"]
            for value in edge["values"]
        )
    ):
        raise ValueError("KuaiRand intraday chain-extension result differs")
