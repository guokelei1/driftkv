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

SWEEP_PROTOCOL = "evokv_kuairand_intraday_theta2_sweep_v0"


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    base = document.get("base_config", {})
    source = document.get("source_round_config", {})
    names = [candidate.get("name") for candidate in document.get("candidates", [])]
    if (
        document.get("protocol") != SWEEP_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("horizons") != [4, 8, 16, 32, 64]
        or names != ["lr3e4_e2", "lr5e4_e2", "lr2e4_e4", "lr5e4_e4"]
        or file_sha256(base.get("path", "")) != base.get("sha256")
        or file_sha256(source.get("path", "")) != source.get("sha256")
    ):
        raise ValueError("KuaiRand intraday theta2 sweep config differs")
    return document


def _prepare_plan(base, split_fraction):
    plan, metadata = load_plan(base)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    first_date = dates[14]
    _, first_update, first_evaluation = _split_day(
        plan.daily_segments[first_date], split_fraction
    )
    for frame in (first_update, first_evaluation):
        for user, group in frame.groupby("user_idx"):
            plan._append_day_to_history(int(user), group)
    second_date = dates[15]
    boundary, update, evaluation = _split_day(
        plan.daily_segments[second_date], split_fraction
    )
    update_key = "theta2_sweep_update"
    evaluation_key = "theta2_sweep_evaluation"
    plan.daily_segments[update_key] = update
    plan.daily_segments[evaluation_key] = evaluation
    plan.ingest_day(update_key)
    return plan, metadata, second_date, boundary, update_key, evaluation_key


def _candidate_admitted(values):
    primary = next(value for value in values if value["max_exposures"] == 4)
    stale = primary["comparisons"]["recompute_over_reuse"]
    return (
        primary["comparisons"]["history_value"]["average_precision"][
            "positive_direction_with_ci"
        ]
        and stale["average_precision"]["positive_direction_with_ci"]
        and stale["ndcg_at_10"]["positive_direction_with_ci"]
        and stale["ndcg_at_50"]["positive_direction_with_ci"]
        and max(
            stale["average_precision"]["relative_percent"],
            stale["ndcg_at_10"]["relative_percent"],
        )
        >= 5.0
    )


def run_theta2_sweep(config_path: str | Path):
    config = load_sweep_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    base = load_engagement_config(config["base_config"]["path"])
    source_config = json.loads(Path(config["source_round_config"]["path"]).read_text())
    source_root = Path(source_config["checkpoint_root"])
    plan, metadata, source_date, boundary, update_key, evaluation_key = _prepare_plan(
        base,
        float(source_config["split_fraction"]),
    )
    device = torch.device("cuda:0")
    previous = make_model(base, plan, device)
    _load_checkpoint(previous, source_root, 1)
    previous.eval()
    candidates = []
    started = time.monotonic()
    for candidate in config["candidates"]:
        current = make_model(base, plan, device)
        _load_checkpoint(current, source_root, 1)
        effective = json.loads(json.dumps(base))
        effective["training"].update(candidate)
        optimizer = torch.optim.AdamW(
            current.parameters(),
            lr=float(candidate["update_lr"]),
            weight_decay=float(candidate["weight_decay"]),
        )
        epochs = []
        for epoch in range(int(candidate["update_epochs"])):
            np.random.seed(int(candidate["seed"]) + epoch)
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
                    f"theta2_{candidate['name']}_e{epoch + 1}",
                )
            )
        candidate_root = Path(config["checkpoint_root"]) / candidate["name"]
        checkpoint = _save_checkpoint(
            current,
            candidate_root,
            2,
            config_path,
            metadata,
            epochs,
        )
        current.eval()
        values = []
        for horizon in config["horizons"]:
            print(
                f"phase=theta2_sweep candidate={candidate['name']} horizon={horizon}",
                flush=True,
            )
            values.append(
                _evaluate_edge(
                    base,
                    plan,
                    previous,
                    current,
                    update_key,
                    evaluation_key,
                    2,
                    device,
                    max_exposures=int(horizon),
                )
            )
        candidates.append(
            {
                "name": candidate["name"],
                "training": epochs,
                "checkpoint": checkpoint,
                "values": values,
                "admitted": _candidate_admitted(values),
            }
        )
    admitted = [candidate for candidate in candidates if candidate["admitted"]]
    selected = None
    if admitted:
        selected = max(
            admitted,
            key=lambda candidate: next(
                value for value in candidate["values"] if value["max_exposures"] == 4
            )["comparisons"]["recompute_over_reuse"]["average_precision"][
                "relative_percent"
            ],
        )["name"]
    result = {
        "protocol": SWEEP_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "base_config": config["base_config"],
        "source_round_config": config["source_round_config"],
        "data": metadata,
        "source_date": source_date,
        "boundary_time_ms": boundary,
        "candidates": candidates,
        "decision": {
            "admitted_candidates": [candidate["name"] for candidate in admitted],
            "selected_candidate": selected,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_theta2_sweep(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != SWEEP_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or len(result.get("candidates", [])) != 4
        or not all(len(candidate.get("values", [])) == 5 for candidate in result["candidates"])
        or not all(
            value.get("same_model_sanity_passed")
            for candidate in result["candidates"]
            for value in candidate["values"]
        )
    ):
        raise ValueError("KuaiRand intraday theta2 sweep result differs")
