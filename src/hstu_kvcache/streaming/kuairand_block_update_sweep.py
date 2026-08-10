from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

BLOCK_UPDATE_PROTOCOL = "evokv_kuairand_block_update_sweep_v0"


def load_block_update_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data_config", {})
    base = document.get("base_round_config", {})
    candidates = document.get("candidates", [])
    if (
        document.get("protocol") != BLOCK_UPDATE_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("update_date_blocks") != [[14, 15, 16], [17, 18, 19]]
        or document.get("evaluation_date_indices") != [17, 20]
        or document.get("horizons") != [4, 8, 16]
        or [value.get("name") for value in candidates]
        != ["block3_e1_lr200", "block3_e2_lr200", "block3_e2_lr500"]
        or file_sha256(data.get("path", "")) != data.get("sha256")
        or file_sha256(base.get("path", "")) != base.get("sha256")
    ):
        raise ValueError("KuaiRand block-update config differs")
    return document


def _candidate_pass(edges: list[dict[str, Any]], horizon: int) -> bool:
    values = [
        next(value for value in edge["values"] if value["max_exposures"] == horizon)
        for edge in edges
    ]
    return all(
        value["comparisons"]["history_value"]["average_precision"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["fresh_update_value"]["average_precision"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["recompute_over_reuse"]["average_precision"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["recompute_over_reuse"]["ndcg_at_10"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["recompute_over_reuse"]["average_precision"][
            "relative_percent"
        ]
        >= 5.0
        for value in values
    )


def run_block_update_sweep(config_path: str | Path) -> dict[str, Any]:
    config = load_block_update_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(config["data_config"]["path"])
    base_document = json.loads(Path(config["base_round_config"]["path"]).read_text())
    base_root = Path(base_document["outputs"]["checkpoint_root"])
    device = torch.device("cuda:0")
    results = []
    started = time.monotonic()
    for candidate in config["candidates"]:
        seed = int(candidate["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        plan, metadata = load_plan(document)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        current = make_model(document, plan, device)
        _load_checkpoint(current, base_root, 0)
        optimizer = torch.optim.AdamW(
            current.parameters(),
            lr=float(candidate["update_lr"]),
            weight_decay=float(candidate["weight_decay"]),
        )
        effective = json.loads(json.dumps(document))
        effective["training"].update(candidate)
        root = Path(config["checkpoint_root"]) / candidate["name"]
        edges = []
        checkpoints = []
        for edge, (block, evaluation_index) in enumerate(
            zip(config["update_date_blocks"], config["evaluation_date_indices"], strict=True),
            start=1,
        ):
            update_key = f"{candidate['name']}_edge{edge}_update"
            frames = [plan.daily_segments[dates[int(index)]] for index in block]
            plan.daily_segments[update_key] = (
                pd.concat(frames, ignore_index=True)
                .sort_values(["time_ms", "user_idx"])
                .reset_index(drop=True)
            )
            evaluation_key = dates[int(evaluation_index)]
            previous = make_model(document, plan, device)
            previous_root = base_root if edge == 1 else root
            _load_checkpoint(previous, previous_root, edge - 1)
            plan.ingest_day(update_key)
            epochs = []
            for epoch in range(int(candidate["update_epochs"])):
                np.random.seed(seed + edge * 1009 + epoch)
                batches = plan.iter_train_batches(
                    update_key,
                    int(candidate["batch_size"]),
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
                        f"{candidate['name']}_theta{edge}_e{epoch + 1}",
                    )
                )
            checkpoints.append(
                _save_checkpoint(current, root, edge, config_path, metadata, epochs)
            )
            previous.eval()
            current.eval()
            values = []
            for horizon in config["horizons"]:
                print(
                    f"phase=block_update candidate={candidate['name']} "
                    f"edge={edge} horizon={horizon}",
                    flush=True,
                )
                values.append(
                    _evaluate_edge(
                        document,
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
            edges.append(
                {
                    "edge": edge,
                    "update_date_indices": block,
                    "evaluation_date_index": evaluation_index,
                    "training": epochs,
                    "values": values,
                }
            )
        common = [
            horizon
            for horizon in config["horizons"]
            if _candidate_pass(edges, int(horizon))
        ]
        results.append(
            {
                "name": candidate["name"],
                "candidate": candidate,
                "checkpoints": checkpoints,
                "edges": edges,
                "common_positive_horizons": common,
            }
        )
        del current, optimizer, plan
        torch.cuda.empty_cache()
    result = {
        "protocol": BLOCK_UPDATE_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "candidates": results,
        "decision": {
            "admitted_candidates": [
                value["name"] for value in results if value["common_positive_horizons"]
            ]
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_block_update_sweep(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != BLOCK_UPDATE_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or [value.get("name") for value in result.get("candidates", [])]
        != ["block3_e1_lr200", "block3_e2_lr200", "block3_e2_lr500"]
        or not all(len(value.get("edges", [])) == 2 for value in result["candidates"])
        or not all(
            metric.get("same_model_sanity_passed")
            for candidate in result["candidates"]
            for edge in candidate["edges"]
            for metric in edge["values"]
        )
    ):
        raise ValueError("KuaiRand block-update result differs")
